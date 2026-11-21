from pathlib import Path
import copy
import importlib.util
import time

import torch
from PIL import Image


PROJECT = Path(
    "/root/autodl-tmp/FineVLA_bundle_20260827_v1/"
    "FineVLA_ablation_improved_eager/FineVLA-Policy"
)

STAGE2_PATH = (
    PROJECT
    / "experiments/delta_grounding/"
      "teacher_stage2_fast_local_bradley_terry_v1.py"
)

VIEW_ORDER = (
    "cam_high",
    "cam_left_wrist",
    "cam_right_wrist",
)


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(
        name,
        path,
    )

    module = importlib.util.module_from_spec(
        spec
    )

    spec.loader.exec_module(
        module
    )

    return module


stage2 = load_module(
    "production_crossview_bt_stage2",
    STAGE2_PATH,
)


def cuda_sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def run_crossview_bt(
    multiview_result,
    model,
    processor,
    output_dir,
    topk=3,
    temperature=1.0,
):
    """
    Production Cross-view Bradley-Terry.

    Input:
        multiview_result:
            output from Multi-view Teacher pool.

        model / processor:
            MUST be the already-loaded shared Qwen from
            teacher_per_view.py.

    Cross-view stage:
        high TopK
        left TopK
        right TopK
            ↓
        one unified tripartite BT graph
            ↓
        candidate reliability
            ↓
        view reliability

    Important:
        - No model reload.
        - No main camera.
        - No bbox IoU across cameras.
        - No A-H visibility classification.
        - Top4~Top6 evidence is still preserved.
    """

    if topk < 1:
        raise ValueError(
            "topk must be >= 1"
        )

    if temperature <= 0:
        raise ValueError(
            "temperature must be > 0"
        )

    output_dir = Path(
        output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    result = copy.deepcopy(
        multiview_result
    )

    target_text = str(
        result[
            "target_text"
        ]
    ).strip()

    stage2.screen.TARGET_TEXT = (
        target_text
    )

    print()
    print("=" * 120)
    print("PRODUCTION CROSS-VIEW BT")
    print("=" * 120)

    print(
        "TARGET =",
        target_text,
    )

    print(
        "TOPK / VIEW =",
        topk,
    )


    # ========================================================
    # Build unified candidate pool.
    # ========================================================

    candidate_meta = []
    candidate_views = {}

    position = 0

    for view_id in VIEW_ORDER:

        if (
            view_id
            not in result["views"]
        ):
            raise RuntimeError(
                f"Missing view: {view_id}"
            )

        view = result[
            "views"
        ][
            view_id
        ]

        image = (
            Image.open(
                view[
                    "image_path"
                ]
            )
            .convert("RGB")
        )

        top_items = (
            view[
                "topk"
            ][
                :topk
            ]
        )

        if (
            len(top_items)
            != topk
        ):
            raise RuntimeError(
                f"{view_id}: expected "
                f"{topk} candidates, "
                f"got {len(top_items)}"
            )

        for view_rank, item in enumerate(
            top_items,
            start=1,
        ):

            candidate_meta.append(
                {
                    "position":
                        position,

                    "view_id":
                        view_id,

                    "view_rank":
                        view_rank,

                    "original_index":
                        int(
                            item[
                                "candidate_index"
                            ]
                        ),

                    "box":
                        item[
                            "box"
                        ],

                    "single_view_bt":
                        (
                            float(
                                item[
                                    "bt_score"
                                ]
                            )
                            if item.get(
                                "bt_score"
                            )
                            is not None
                            else None
                        ),
                }
            )

            candidate_views[
                position
            ] = (
                stage2.screen
                .make_candidate_view(
                    image,
                    item[
                        "box"
                    ],
                )
            )

            position += 1


    candidate_count = (
        len(
            candidate_meta
        )
    )

    expected_count = (
        len(
            VIEW_ORDER
        )
        *
        topk
    )

    if (
        candidate_count
        != expected_count
    ):
        raise RuntimeError(
            f"Candidate count mismatch: "
            f"{candidate_count} "
            f"!= {expected_count}"
        )


    print()
    print(
        "CANDIDATE COUNT =",
        candidate_count,
    )

    for item in candidate_meta:

        print(
            "POS=",
            item[
                "position"
            ],
            "VIEW=",
            item[
                "view_id"
            ],
            "RANK=",
            item[
                "view_rank"
            ],
            "INDEX=",
            item[
                "original_index"
            ],
        )


    # ========================================================
    # Complete tripartite graph.
    #
    # Only cross-camera comparisons.
    # Never repeat same-camera BT.
    # ========================================================

    pairs = []

    degree = [
        0
        for _ in range(
            candidate_count
        )
    ]

    for i in range(
        candidate_count
    ):

        for j in range(
            i + 1,
            candidate_count,
        ):

            if (
                candidate_meta[
                    i
                ][
                    "view_id"
                ]
                ==
                candidate_meta[
                    j
                ][
                    "view_id"
                ]
            ):
                continue

            pairs.append(
                (
                    i,
                    j,
                )
            )

            degree[
                i
            ] += 1

            degree[
                j
            ] += 1


    if (
        len(
            set(
                degree
            )
        )
        != 1
    ):
        raise RuntimeError(
            "Cross-view graph "
            f"is not balanced: {degree}"
        )


    print()
    print(
        "PAIR COUNT =",
        len(
            pairs
        ),
    )

    print(
        "PAIR DEGREE =",
        degree[
            0
        ],
    )


    # ========================================================
    # Pair template helpers.
    # ========================================================

    def build_pair_teacher_inputs(
        group,
        label,
    ):

        content = (
            stage2.screen
            .build_content(
                group,
                candidate_views,
            )
        )

        messages = [[
            {
                "role":
                    "user",

                "content":
                    content,
            },
            {
                "role":
                    "assistant",

                "content": [
                    {
                        "type":
                            "text",

                        "text":
                            label,
                    }
                ],
            },
        ]]

        return (
            processor
            .apply_chat_template(
                messages,
                tokenize=True,
                padding=True,
                add_generation_prompt=False,
                return_dict=True,
                return_tensors="pt",
            )
        )


    def build_pair_template(
        group,
    ):

        content = (
            stage2.screen
            .build_content(
                group,
                candidate_views,
            )
        )

        prompt_messages = [[
            {
                "role":
                    "user",

                "content":
                    content,
            }
        ]]

        prompt_inputs = (
            processor
            .apply_chat_template(
                prompt_messages,
                tokenize=True,
                padding=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            )
        )

        input_a = (
            build_pair_teacher_inputs(
                group,
                "A",
            )
        )

        input_b = (
            build_pair_teacher_inputs(
                group,
                "B",
            )
        )

        prompt_ids = (
            prompt_inputs[
                "input_ids"
            ][0]
            .tolist()
        )

        ids_a = (
            input_a[
                "input_ids"
            ][0]
            .tolist()
        )

        ids_b = (
            input_b[
                "input_ids"
            ][0]
            .tolist()
        )

        (
            pos_a,
            token_a,
        ) = (
            stage2.screen
            .find_answer_token(
                processor.tokenizer,
                prompt_ids,
                ids_a,
                "A",
            )
        )

        (
            pos_b,
            token_b,
        ) = (
            stage2.screen
            .find_answer_token(
                processor.tokenizer,
                prompt_ids,
                ids_b,
                "B",
            )
        )

        if (
            pos_a
            !=
            pos_b
        ):
            raise RuntimeError(
                "A/B answer positions differ."
            )

        if (
            ids_a[
                :pos_a
            ]
            !=
            ids_b[
                :pos_b
            ]
        ):
            raise RuntimeError(
                "A/B causal prefixes differ."
            )

        return (
            input_a,
            int(
                pos_a
            ),
            int(
                token_a
            ),
            int(
                token_b
            ),
        )


    # ========================================================
    # Build language template once.
    # ========================================================

    template_start = (
        time.perf_counter()
    )

    template_group = [
        pairs[
            0
        ][
            0
        ],
        pairs[
            0
        ][
            1
        ],
    ]

    (
        template_inputs,
        answer_position,
        token_a,
        token_b,
    ) = (
        build_pair_template(
            template_group
        )
    )

    template_input_ids = (
        template_inputs[
            "input_ids"
        ]
        .to(
            "cuda"
        )
    )

    template_attention_mask = (
        template_inputs[
            "attention_mask"
        ]
        .to(
            "cuda"
        )
    )

    template_image_grid_thw = (
        template_inputs[
            "image_grid_thw"
        ]
        .to(
            "cuda"
        )
    )

    with torch.inference_mode():

        base_inputs_embeds = (
            model.model
            .get_input_embeddings()(
                template_input_ids
            )
        )

        (
            image_mask,
            _,
        ) = (
            model.model
            .get_placeholder_mask(
                template_input_ids,
                inputs_embeds=
                    base_inputs_embeds,
            )
        )

        (
            position_ids,
            _,
        ) = (
            model.model
            .get_rope_index(
                input_ids=
                    template_input_ids,

                image_grid_thw=
                    template_image_grid_thw,

                attention_mask=
                    template_attention_mask,
            )
        )


    template_sec = (
        time.perf_counter()
        -
        template_start
    )


    print(
        "CROSSVIEW_TEMPLATE=PASS"
    )

    print(
        "TEMPLATE_SEC =",
        f"{template_sec:.4f}",
    )


    # ========================================================
    # Visual encoder.
    # ========================================================

    @torch.inference_mode()
    def encode_two_candidates(
        index_a,
        index_b,
    ):

        pair_inputs = (
            build_pair_teacher_inputs(
                [
                    index_a,
                    index_b,
                ],
                "A",
            )
        )

        pixel_values = (
            pair_inputs[
                "pixel_values"
            ]
            .to(
                "cuda"
            )
        )

        image_grid_thw = (
            pair_inputs[
                "image_grid_thw"
            ]
            .to(
                "cuda"
            )
        )

        image_outputs = (
            model
            .get_image_features(
                pixel_values=
                    pixel_values,

                image_grid_thw=
                    image_grid_thw,

                return_dict=True,
            )
        )

        features = (
            image_outputs
            .pooler_output
        )

        if (
            len(
                features
            )
            != 2
        ):
            raise RuntimeError(
                "Expected exactly "
                "two image features."
            )

        return (
            features[
                0
            ].detach(),
            features[
                1
            ].detach(),
        )


    # ========================================================
    # Vision shape warmup.
    # ========================================================

    cuda_sync()

    vision_warmup_start = (
        time.perf_counter()
    )

    warm_a, warm_b = (
        encode_two_candidates(
            pairs[
                0
            ][
                0
            ],
            pairs[
                0
            ][
                1
            ],
        )
    )

    cuda_sync()

    vision_warmup_sec = (
        time.perf_counter()
        -
        vision_warmup_start
    )

    del warm_a
    del warm_b


    # ========================================================
    # Build visual cache.
    # ========================================================

    visual_cache = {}

    all_indices = list(
        range(
            candidate_count
        )
    )

    cuda_sync()

    visual_cache_start = (
        time.perf_counter()
    )

    visual_cache_forwards = 0

    for start in range(
        0,
        candidate_count,
        2,
    ):

        index_a = (
            all_indices[
                start
            ]
        )

        if (
            start + 1
            <
            candidate_count
        ):

            index_b = (
                all_indices[
                    start + 1
                ]
            )

            keep_b = True

        else:

            index_b = (
                all_indices[
                    0
                ]
            )

            keep_b = False


        (
            feature_a,
            feature_b,
        ) = (
            encode_two_candidates(
                index_a,
                index_b,
            )
        )

        visual_cache_forwards += 1

        visual_cache[
            index_a
        ] = (
            feature_a
        )

        if keep_b:

            visual_cache[
                index_b
            ] = (
                feature_b
            )


    cuda_sync()

    visual_cache_sec = (
        time.perf_counter()
        -
        visual_cache_start
    )


    if (
        len(
            visual_cache
        )
        != candidate_count
    ):
        raise RuntimeError(
            "Visual cache incomplete."
        )


    print(
        "CROSSVIEW_VISUAL_CACHE=PASS"
    )

    print(
        "VISUAL CACHE COUNT =",
        len(
            visual_cache
        ),
    )

    print(
        "VISUAL CACHE FORWARDS =",
        visual_cache_forwards,
    )

    print(
        "VISUAL CACHE SEC =",
        f"{visual_cache_sec:.4f}",
    )


    # ========================================================
    # Cached language orientation.
    # ========================================================

    @torch.inference_mode()
    def cached_orientation(
        first_index,
        second_index,
    ):

        image_embeds = torch.cat(
            [
                visual_cache[
                    first_index
                ].to(
                    device=
                        base_inputs_embeds.device,

                    dtype=
                        base_inputs_embeds.dtype,
                ),

                visual_cache[
                    second_index
                ].to(
                    device=
                        base_inputs_embeds.device,

                    dtype=
                        base_inputs_embeds.dtype,
                ),
            ],
            dim=0,
        )

        inputs_embeds = (
            base_inputs_embeds
            .masked_scatter(
                image_mask,
                image_embeds,
            )
        )

        output = model(
            input_ids=None,

            inputs_embeds=
                inputs_embeds,

            attention_mask=
                template_attention_mask,

            position_ids=
                position_ids,

            pixel_values=None,

            image_grid_thw=None,

            use_cache=False,

            return_dict=True,
        )

        logits = (
            output.logits[
                0,
                answer_position - 1,
            ]
            .float()
        )

        log_probs = (
            torch.log_softmax(
                logits,
                dim=-1,
            )
        )

        score_a = float(
            log_probs[
                token_a
            ].item()
        )

        score_b = float(
            log_probs[
                token_b
            ].item()
        )

        two_scores = torch.tensor(
            [
                score_a,
                score_b,
            ],
            dtype=torch.float64,
        )

        probs = torch.softmax(
            two_scores,
            dim=0,
        )

        del output

        return {
            "p_a":
                float(
                    probs[
                        0
                    ].item()
                ),

            "p_b":
                float(
                    probs[
                        1
                    ].item()
                ),
        }


    # ========================================================
    # Language shape warmup.
    # ========================================================

    cuda_sync()

    language_warmup_start = (
        time.perf_counter()
    )

    _ = cached_orientation(
        pairs[
            0
        ][
            0
        ],
        pairs[
            0
        ][
            1
        ],
    )

    cuda_sync()

    language_warmup_sec = (
        time.perf_counter()
        -
        language_warmup_start
    )


    print(
        "LANGUAGE WARMUP SEC =",
        f"{language_warmup_sec:.4f}",
    )


    # ========================================================
    # Pairwise AB / BA scoring.
    # ========================================================

    pair_probs = {
        i: []
        for i in range(
            candidate_count
        )
    }

    hard_wins = {
        i: []
        for i in range(
            candidate_count
        )
    }

    matches = []


    cuda_sync()

    pairwise_start = (
        time.perf_counter()
    )


    for match_number, (
        i_pos,
        j_pos,
    ) in enumerate(
        pairs,
        start=1,
    ):

        ab = (
            cached_orientation(
                i_pos,
                j_pos,
            )
        )

        ba = (
            cached_orientation(
                j_pos,
                i_pos,
            )
        )

        p_i_ab = float(
            ab[
                "p_a"
            ]
        )

        p_i_ba = float(
            ba[
                "p_b"
            ]
        )

        p_i = (
            0.5
            *
            (
                p_i_ab
                +
                p_i_ba
            )
        )

        p_j = (
            1.0
            -
            p_i
        )

        pair_probs[
            i_pos
        ].append(
            p_i
        )

        pair_probs[
            j_pos
        ].append(
            p_j
        )

        hard_wins[
            i_pos
        ].append(
            float(
                p_i
                >
                0.5
            )
        )

        hard_wins[
            j_pos
        ].append(
            float(
                p_j
                >
                0.5
            )
        )

        matches.append(
            {
                "i_pos":
                    i_pos,

                "j_pos":
                    j_pos,

                "p_i":
                    p_i,

                "p_i_ab":
                    p_i_ab,

                "p_i_ba":
                    p_i_ba,
            }
        )

        if (
            match_number == 1
            or
            match_number % 5 == 0
            or
            match_number
            ==
            len(
                pairs
            )
        ):

            print(
                "CROSSVIEW MATCH",
                match_number,
                "/",
                len(
                    pairs
                ),
                "P_I=",
                f"{p_i:.4f}",
            )


    cuda_sync()

    pairwise_sec = (
        time.perf_counter()
        -
        pairwise_start
    )


    # ========================================================
    # Unified Bradley-Terry fit.
    # ========================================================

    bt_fit_start = (
        time.perf_counter()
    )

    theta = (
        stage2
        .fit_bradley_terry(
            candidate_count,
            matches,
        )
    )

    bt_fit_sec = (
        time.perf_counter()
        -
        bt_fit_start
    )


    # ========================================================
    # Candidate reliability:
    #
    # softmax(theta / temperature)
    #
    # View reliability:
    #
    # sum candidate probability
    # belonging to each view.
    # ========================================================

    candidate_prob_tensor = (
        torch.softmax(
            theta.double()
            /
            float(
                temperature
            ),
            dim=0,
        )
    )

    ranking = []

    position_info = {}


    for pos, meta in enumerate(
        candidate_meta
    ):

        probs = (
            pair_probs[
                pos
            ]
        )

        wins = (
            hard_wins[
                pos
            ]
        )

        item = {
            **meta,

            "crossview_bt_score":
                float(
                    theta[
                        pos
                    ].item()
                ),

            "crossview_pair_mean":
                float(
                    sum(
                        probs
                    )
                    /
                    len(
                        probs
                    )
                ),

            "crossview_hard_win_rate":
                float(
                    sum(
                        wins
                    )
                    /
                    len(
                        wins
                    )
                ),

            "candidate_reliability":
                float(
                    candidate_prob_tensor[
                        pos
                    ].item()
                ),
        }

        ranking.append(
            item
        )

        position_info[
            pos
        ] = (
            item
        )


    ranking.sort(
        key=lambda item:
            item[
                "crossview_bt_score"
            ],
        reverse=True,
    )


    for global_rank, item in enumerate(
        ranking,
        start=1,
    ):

        item[
            "global_rank"
        ] = (
            global_rank
        )

        position_info[
            item[
                "position"
            ]
        ][
            "global_rank"
        ] = (
            global_rank
        )


    view_reliability = {
        view_id:
            0.0
        for view_id in VIEW_ORDER
    }


    for pos, meta in enumerate(
        candidate_meta
    ):

        view_reliability[
            meta[
                "view_id"
            ]
        ] += float(
            candidate_prob_tensor[
                pos
            ].item()
        )


    # ========================================================
    # Write reliability back into all per-view TopK evidence.
    #
    # Top1~Top3:
    #   direct candidate-level Cross-view BT.
    #
    # Top4~Top6:
    #   kept untouched as evidence, but inherit
    #   view-level reliability only.
    # ========================================================

    lookup = {}

    for item in ranking:

        lookup[
            (
                item[
                    "view_id"
                ],
                item[
                    "view_rank"
                ],
            )
        ] = (
            item
        )


    for view_id in VIEW_ORDER:

        result[
            "views"
        ][
            view_id
        ][
            "reliability"
        ] = float(
            view_reliability[
                view_id
            ]
        )

        result[
            "views"
        ][
            view_id
        ][
            "visibility"
        ] = None


        for evidence in (
            result[
                "views"
            ][
                view_id
            ][
                "topk"
            ]
        ):

            view_rank = int(
                evidence[
                    "rank"
                ]
            )

            evidence[
                "view_reliability"
            ] = float(
                view_reliability[
                    view_id
                ]
            )

            key = (
                view_id,
                view_rank,
            )

            if key in lookup:

                cv = (
                    lookup[
                        key
                    ]
                )

                evidence[
                    "crossview_bt_score"
                ] = (
                    cv[
                        "crossview_bt_score"
                    ]
                )

                evidence[
                    "crossview_global_rank"
                ] = (
                    cv[
                        "global_rank"
                    ]
                )

                evidence[
                    "crossview_pair_mean"
                ] = (
                    cv[
                        "crossview_pair_mean"
                    ]
                )

                evidence[
                    "crossview_hard_win_rate"
                ] = (
                    cv[
                        "crossview_hard_win_rate"
                    ]
                )

                evidence[
                    "candidate_reliability"
                ] = (
                    cv[
                        "candidate_reliability"
                    ]
                )

            else:

                evidence[
                    "crossview_bt_score"
                ] = None

                evidence[
                    "crossview_global_rank"
                ] = None

                evidence[
                    "crossview_pair_mean"
                ] = None

                evidence[
                    "crossview_hard_win_rate"
                ] = None

                evidence[
                    "candidate_reliability"
                ] = None


    # candidate_pool is also kept synchronized.

    for evidence in (
        result[
            "candidate_pool"
        ]
    ):

        view_id = (
            evidence[
                "view_id"
            ]
        )

        view_rank = int(
            evidence[
                "rank"
            ]
        )

        evidence[
            "view_reliability"
        ] = float(
            view_reliability[
                view_id
            ]
        )

        key = (
            view_id,
            view_rank,
        )

        if key in lookup:

            cv = (
                lookup[
                    key
                ]
            )

            evidence[
                "crossview_bt_score"
            ] = (
                cv[
                    "crossview_bt_score"
                ]
            )

            evidence[
                "crossview_global_rank"
            ] = (
                cv[
                    "global_rank"
                ]
            )

            evidence[
                "crossview_pair_mean"
            ] = (
                cv[
                    "crossview_pair_mean"
                ]
            )

            evidence[
                "crossview_hard_win_rate"
            ] = (
                cv[
                    "crossview_hard_win_rate"
                ]
            )

            evidence[
                "candidate_reliability"
            ] = (
                cv[
                    "candidate_reliability"
                ]
            )

        else:

            evidence[
                "crossview_bt_score"
            ] = None

            evidence[
                "crossview_global_rank"
            ] = None

            evidence[
                "crossview_pair_mean"
            ] = None

            evidence[
                "crossview_hard_win_rate"
            ] = None

            evidence[
                "candidate_reliability"
            ] = None


    result[
        "cross_view_method"
    ] = (
        "unified_crossview_bradley_terry"
    )

    result[
        "cross_view_topk"
    ] = (
        int(
            topk
        )
    )

    result[
        "cross_view_temperature"
    ] = (
        float(
            temperature
        )
    )

    result[
        "cross_view_reliability"
    ] = (
        view_reliability
    )

    result[
        "final_multiview_ranking"
    ] = (
        ranking
    )

    result[
        "cross_view_pairs"
    ] = (
        pairs
    )

    result[
        "cross_view_degree"
    ] = (
        degree
    )

    result[
        "cross_view_timing"
    ] = {
        "template_sec":
            template_sec,

        "vision_warmup_sec":
            vision_warmup_sec,

        "visual_cache_sec":
            visual_cache_sec,

        "language_warmup_sec":
            language_warmup_sec,

        "pairwise_sec":
            pairwise_sec,

        "bt_fit_sec":
            bt_fit_sec,

        "steady_crossview_total_sec":
            (
                template_sec
                +
                visual_cache_sec
                +
                pairwise_sec
                +
                bt_fit_sec
            ),
    }


    # ========================================================
    # Report.
    # ========================================================

    print()
    print("=" * 120)
    print("CROSS-VIEW GLOBAL RANKING")
    print("=" * 120)

    for item in ranking:

        print(
            "GLOBAL_RANK=",
            item[
                "global_rank"
            ],

            "VIEW=",
            item[
                "view_id"
            ],

            "VIEW_RANK=",
            item[
                "view_rank"
            ],

            "INDEX=",
            item[
                "original_index"
            ],

            "BT=",
            f"{item['crossview_bt_score']:.4f}",

            "REL=",
            f"{item['candidate_reliability']:.4f}",
        )


    print()
    print("=" * 120)
    print("VIEW RELIABILITY")
    print("=" * 120)

    for view_id in sorted(
        VIEW_ORDER,
        key=lambda v:
            view_reliability[
                v
            ],
        reverse=True,
    ):

        print(
            view_id,
            "=",
            f"{view_reliability[view_id]:.6f}",
        )


    best_view = max(
        VIEW_ORDER,
        key=lambda view_id:
            view_reliability[
                view_id
            ],
    )


    result[
        "best_view"
    ] = (
        best_view
    )


    result_path = (
        output_dir
        / "multiview_results_bt_v1.pt"
    )

    torch.save(
        result,
        result_path,
    )


    print()
    print(
        "BEST VIEW =",
        best_view,
    )

    print(
        "RESULT =",
        result_path,
    )

    print(
        "TEACHER_CROSSVIEW_BT=PASS"
    )

    return (
        result,
        result_path,
    )
