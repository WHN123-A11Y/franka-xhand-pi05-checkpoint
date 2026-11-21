from pathlib import Path
import argparse
import importlib.util
import time

import torch
from PIL import Image


# ============================================================
# Persistent multi-scene state.
# ============================================================

_SHARED_PROCESSOR = None
_SHARED_MODEL = None

# Main coarse-stage warmup is target-text dependent.
_SHARED_MAIN_WARMED_TARGETS = set()

# Vision cache always has the same 2-image execution shape.
_SHARED_BT_VISION_WARMED = False

# Language prompt shape may depend on target text.
_SHARED_BT_LANGUAGE_WARMED_TARGETS = set()


ROOT = Path(
    "/root/autodl-tmp/"
    "FineVLA_bundle_20260827_v1/"
    "FineVLA_ablation_improved_eager/"
    "FineVLA-Policy"
)

STAGE1_PATH = (
    ROOT
    / "experiments/delta_grounding/"
      "FROZEN_TEACHER_1691_V1_20260816/"
      "teacher_stage1_1691_phaseaware_twoschedule.py"
)

STAGE2_PATH = (
    ROOT
    / "experiments/delta_grounding/"
      "teacher_stage2_fast_local_bradley_terry_v1.py"
)


MARGIN = 0.10

CENTER_RATIOS = [
    (0.50, 0.50),
    (0.40, 0.50),
    (0.60, 0.50),
    (0.50, 0.40),
    (0.50, 0.60),
]

SIZE_RATIOS = [
    0.50,
    0.75,
    1.00,
]


def load_module(
    name,
    path,
):
    spec = (
        importlib.util
        .spec_from_file_location(
            name,
            path,
        )
    )

    module = (
        importlib.util
        .module_from_spec(
            spec
        )
    )

    spec.loader.exec_module(
        module
    )

    return module


def cuda_sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def measure(
    function,
):
    cuda_sync()

    start = (
        time.perf_counter()
    )

    result = function()

    cuda_sync()

    elapsed = (
        time.perf_counter()
        - start
    )

    return (
        result,
        elapsed,
    )


def sliding_boxes(
    box_w,
    box_h,
    stride_x,
    stride_y,
):
    boxes = []

    y = 0.0

    while (
        y + box_h
        <= 1.00001
    ):
        x = 0.0

        while (
            x + box_w
            <= 1.00001
        ):
            boxes.append(
                (
                    round(
                        x,
                        6,
                    ),
                    round(
                        y,
                        6,
                    ),
                    round(
                        x + box_w,
                        6,
                    ),
                    round(
                        y + box_h,
                        6,
                    ),
                )
            )

            x += stride_x

        y += stride_y

    return boxes


def build_coarse_boxes():
    boxes = []

    boxes += sliding_boxes(
        box_w=0.50,
        box_h=0.50,
        stride_x=0.25,
        stride_y=0.25,
    )

    boxes += sliding_boxes(
        box_w=0.40,
        box_h=0.50,
        stride_x=0.20,
        stride_y=0.25,
    )

    boxes += sliding_boxes(
        box_w=0.50,
        box_h=0.40,
        stride_x=0.25,
        stride_y=0.20,
    )

    boxes = sorted(
        set(
            boxes
        )
    )

    return torch.tensor(
        boxes,
        dtype=torch.float32,
    )


def expand_box(
    box,
    margin,
):
    x1, y1, x2, y2 = [
        float(x)
        for x in box
    ]

    return (
        max(
            0.0,
            x1 - margin,
        ),
        max(
            0.0,
            y1 - margin,
        ),
        min(
            1.0,
            x2 + margin,
        ),
        min(
            1.0,
            y2 + margin,
        ),
    )


def build_local_candidates(
    coarse_boxes,
    coarse_survivors,
):
    candidate_set = set()

    for coarse_index in (
        coarse_survivors
    ):
        rx1, ry1, rx2, ry2 = (
            expand_box(
                coarse_boxes[
                    coarse_index
                ],
                MARGIN,
            )
        )

        roi_w = (
            rx2 - rx1
        )

        roi_h = (
            ry2 - ry1
        )

        for crx, cry in (
            CENTER_RATIOS
        ):
            center_x = (
                rx1
                + crx
                * roi_w
            )

            center_y = (
                ry1
                + cry
                * roi_h
            )

            for wr in (
                SIZE_RATIOS
            ):
                for hr in (
                    SIZE_RATIOS
                ):
                    width = (
                        wr
                        * roi_w
                    )

                    height = (
                        hr
                        * roi_h
                    )

                    x1 = max(
                        rx1,
                        center_x
                        - width * 0.5,
                    )

                    y1 = max(
                        ry1,
                        center_y
                        - height * 0.5,
                    )

                    x2 = min(
                        rx2,
                        center_x
                        + width * 0.5,
                    )

                    y2 = min(
                        ry2,
                        center_y
                        + height * 0.5,
                    )

                    candidate_set.add(
                        (
                            round(
                                x1,
                                6,
                            ),
                            round(
                                y1,
                                6,
                            ),
                            round(
                                x2,
                                6,
                            ),
                            round(
                                y2,
                                6,
                            ),
                        )
                    )

    boxes = sorted(
        candidate_set
    )

    if not boxes:
        raise RuntimeError(
            "Local candidate bank is empty."
        )

    return torch.tensor(
        boxes,
        dtype=torch.float32,
    )


def box_iou(
    a,
    b,
):
    ax1, ay1, ax2, ay2 = [
        float(x)
        for x in a
    ]

    bx1, by1, bx2, by2 = [
        float(x)
        for x in b
    ]

    ix1 = max(
        ax1,
        bx1,
    )

    iy1 = max(
        ay1,
        by1,
    )

    ix2 = min(
        ax2,
        bx2,
    )

    iy2 = min(
        ay2,
        by2,
    )

    inter = (
        max(
            0.0,
            ix2 - ix1,
        )
        *
        max(
            0.0,
            iy2 - iy1,
        )
    )

    area_a = (
        ax2 - ax1
    ) * (
        ay2 - ay1
    )

    area_b = (
        bx2 - bx1
    ) * (
        by2 - by1
    )

    union = (
        area_a
        + area_b
        - inter
    )

    return (
        inter
        / max(
            union,
            1e-12,
        )
    )


def main():
    parser = (
        argparse.ArgumentParser()
    )

    parser.add_argument(
        "--image",
        required=True,
    )

    parser.add_argument(
        "--target",
        required=True,
    )

    parser.add_argument(
        "--output-dir",
        required=True,
    )

    parser.add_argument(
        "--reference-stage2",
        default="",
    )

    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA required."
        )

    image_path = Path(
        args.image
    )

    output_dir = Path(
        args.output_dir
    )

    target_text = str(
        args.target
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    stage1 = load_module(
        "fast_teacher_stage1",
        STAGE1_PATH,
    )

    stage2 = load_module(
        "fast_teacher_stage2",
        STAGE2_PATH,
    )

    stage1.TARGET_TEXT = (
        target_text
    )

    stage2.screen.TARGET_TEXT = (
        target_text
    )

    image = (
        Image.open(
            image_path
        )
        .convert(
            "RGB"
        )
    )

    # ========================================================
    # Coarse proposal bank
    # ========================================================

    coarse_boxes = (
        build_coarse_boxes()
    )

    coarse_views = {
        index:
            stage1.make_candidate_view(
                image,
                coarse_boxes[
                    index
                ],
            )
        for index in range(
            coarse_boxes.shape[0]
        )
    }

    print("=" * 100)
    print(
        "FAST TEACHER COARSE-TO-FINE V6 CACHED"
    )
    print("=" * 100)

    print(
        "IMAGE =",
        image_path,
    )

    print(
        "TARGET =",
        target_text,
    )

    print(
        "COARSE BANK =",
        int(
            coarse_boxes.shape[0]
        ),
    )

    # ========================================================
    # Persistent model: load exactly once for all scenes.
    # ========================================================

    global _SHARED_PROCESSOR
    global _SHARED_MODEL
    global _SHARED_MAIN_WARMED_TARGETS
    global _SHARED_BT_VISION_WARMED
    global _SHARED_BT_LANGUAGE_WARMED_TARGETS

    print()
    print("=" * 100)
    print(
        "LOAD / REUSE MODEL"
    )
    print("=" * 100)

    if _SHARED_MODEL is None:

        cuda_sync()

        model_load_start = (
            time.perf_counter()
        )

        processor = (
            stage1.AutoProcessor
            .from_pretrained(
                stage1.MODEL_PATH,
                local_files_only=True,
            )
        )

        model = (
            stage1
            .Qwen3_5ForConditionalGeneration
            .from_pretrained(
                stage1.MODEL_PATH,
                torch_dtype=torch.bfloat16,
                attn_implementation="sdpa",
                local_files_only=True,
            )
            .to(
                "cuda"
            )
            .eval()
        )

        cuda_sync()

        model_load_sec = (
            time.perf_counter()
            - model_load_start
        )

        _SHARED_PROCESSOR = (
            processor
        )

        _SHARED_MODEL = (
            model
        )

        print(
            "MODEL_LOAD=PASS"
        )

    else:

        processor = (
            _SHARED_PROCESSOR
        )

        model = (
            _SHARED_MODEL
        )

        model_load_sec = 0.0

        print(
            "MODEL_REUSE=PASS"
        )

    print(
        "MODEL_LOAD_SEC =",
        f"{model_load_sec:.3f}",
    )

    # ========================================================
    # Main Qwen warmup once per target-text family.
    # ========================================================

    print()
    print("=" * 100)
    print(
        "ONE-TIME TARGET WARMUP"
    )
    print("=" * 100)

    if (
        target_text
        not in
        _SHARED_MAIN_WARMED_TARGETS
    ):

        warm_group = list(
            range(
                min(
                    6,
                    coarse_boxes.shape[0],
                )
            )
        )

        cuda_sync()

        warmup_start = (
            time.perf_counter()
        )

        stage1.choose_one(
            model,
            processor,
            warm_group,
            coarse_views,
        )

        cuda_sync()

        warmup_sec = (
            time.perf_counter()
            - warmup_start
        )

        _SHARED_MAIN_WARMED_TARGETS.add(
            target_text
        )

        print(
            "WARMUP=PASS"
        )

    else:

        warmup_sec = 0.0

        print(
            "WARMUP=REUSED"
        )

    print(
        "WARMUP_SEC =",
        f"{warmup_sec:.3f}",
    )

    # ========================================================
    # Instrument Qwen calls
    # ========================================================

    active_stage = {
        "name": "",
    }

    call_counts = {
        "coarse1": 0,
        "coarse2": 0,
        "local": 0,
        "prebt": 0,
        "bt": 0,
    }

    call_time = {
        key: 0.0
        for key in (
            call_counts
        )
    }

    original_stage1_choose = (
        stage1.choose_one
    )

    def timed_stage1_choose(
        model,
        processor,
        group,
        candidate_views,
    ):
        cuda_sync()

        start = (
            time.perf_counter()
        )

        result = (
            original_stage1_choose(
                model,
                processor,
                group,
                candidate_views,
            )
        )

        cuda_sync()

        elapsed = (
            time.perf_counter()
            - start
        )

        name = (
            active_stage[
                "name"
            ]
        )

        call_counts[
            name
        ] += 1

        call_time[
            name
        ] += elapsed

        return result

    stage1.choose_one = (
        timed_stage1_choose
    )

    original_stage2_choose = (
        stage2
        .screen
        .choose_one
    )

    def timed_stage2_choose(
        model,
        processor,
        group,
        candidate_views,
    ):
        cuda_sync()

        start = (
            time.perf_counter()
        )

        result = (
            original_stage2_choose(
                model,
                processor,
                group,
                candidate_views,
            )
        )

        cuda_sync()

        elapsed = (
            time.perf_counter()
            - start
        )

        call_counts[
            "bt"
        ] += 1

        call_time[
            "bt"
        ] += elapsed

        return result

    stage2.screen.choose_one = (
        timed_stage2_choose
    )

    # ========================================================
    # Persistent image pipeline starts here
    # ========================================================

    cuda_sync()

    pipeline_start = (
        time.perf_counter()
    )

    # ========================================================
    # Coarse Round 1
    # ========================================================

    print()
    print("=" * 100)
    print(
        "COARSE ROUND 1"
    )
    print("=" * 100)

    active_stage[
        "name"
    ] = "coarse1"

    (
        coarse1_result,
        coarse1_sec,
    ) = measure(
        lambda:
            stage1.run_round(
                model,
                processor,
                list(
                    range(
                        coarse_boxes.shape[
                            0
                        ]
                    )
                ),
                coarse_views,
                round_index=1,
            )
    )

    coarse1, coarse1_record = (
        coarse1_result
    )

    # ========================================================
    # Coarse Round 2
    # ========================================================

    print()
    print("=" * 100)
    print(
        "COARSE ROUND 2"
    )
    print("=" * 100)

    active_stage[
        "name"
    ] = "coarse2"

    (
        coarse2_result,
        coarse2_sec,
    ) = measure(
        lambda:
            stage1.run_round(
                model,
                processor,
                coarse1,
                coarse_views,
                round_index=2,
            )
    )

    coarse2, coarse2_record = (
        coarse2_result
    )

    if not coarse2:
        raise RuntimeError(
            "Coarse screening produced no survivors."
        )

    # ========================================================
    # Dynamic local proposal generation
    # ========================================================

    local_gen_start = (
        time.perf_counter()
    )

    local_boxes = (
        build_local_candidates(
            coarse_boxes,
            coarse2,
        )
    )

    local_views = {
        index:
            stage1.make_candidate_view(
                image,
                local_boxes[
                    index
                ],
            )
        for index in range(
            local_boxes.shape[0]
        )
    }

    local_gen_sec = (
        time.perf_counter()
        - local_gen_start
    )


    # ========================================================
    # SpeedOpt v2:
    # reusable 6-image visual cache for Local + Pre-BT.
    #
    # Exact-equivalence audit already established:
    #
    #   1. Cached reconstruction reproduces original Stage1
    #      label scores exactly.
    #
    #   2. A candidate pooler_output is invariant across
    #      different SIX-IMAGE contexts and positions.
    #
    # Therefore:
    #   - all original schedules are preserved;
    #   - GROUP_SIZE=6 is preserved;
    #   - winner rule is preserved;
    #   - only repeated Vision Encoder work is removed.
    # ========================================================

    print()
    print("=" * 100)
    print(
        "BUILD LOCAL VISUAL CACHE"
    )
    print("=" * 100)

    if (
        stage1.GROUP_SIZE
        != 6
    ):
        raise RuntimeError(
            "VisualCache v2 requires "
            "GROUP_SIZE == 6."
        )

    local_visual_cache = {}

    local_visual_cache_forwards = 0

    # --------------------------------------------------------
    # Exact Stage1 teacher input builder.
    # --------------------------------------------------------

    def build_stage1_teacher_inputs(
        group,
        label,
    ):
        content = (
            stage1.build_content(
                group,
                local_views,
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
            processor.apply_chat_template(
                messages,
                tokenize=True,
                padding=True,
                add_generation_prompt=False,
                return_dict=True,
                return_tensors="pt",
            )
        )

    # --------------------------------------------------------
    # Build one fixed SIX-IMAGE Stage1 language template.
    #
    # Every Local candidate view has the same image size.
    # Images differ only in their visual contents / red box.
    # --------------------------------------------------------

    template_group_stage1 = list(
        range(
            stage1.GROUP_SIZE
        )
    )

    template_content = (
        stage1.build_content(
            template_group_stage1,
            local_views,
        )
    )

    prompt_messages = [[
        {
            "role":
                "user",

            "content":
                template_content,
        }
    ]]

    template_prompt_inputs = (
        processor.apply_chat_template(
            prompt_messages,
            tokenize=True,
            padding=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
    )

    template_prompt_ids = (
        template_prompt_inputs[
            "input_ids"
        ][0]
        .tolist()
    )

    stage1_valid_labels = (
        stage1.LABELS[
            :stage1.GROUP_SIZE
        ]
    )

    stage1_positions = {}
    stage1_token_ids = {}
    stage1_prefixes = {}
    stage1_reference_inputs = None

    for label in stage1_valid_labels:

        teacher_inputs = (
            build_stage1_teacher_inputs(
                template_group_stage1,
                label,
            )
        )

        teacher_ids = (
            teacher_inputs[
                "input_ids"
            ][0]
            .tolist()
        )

        (
            target_position,
            target_token_id,
        ) = (
            stage1.find_answer_token(
                processor.tokenizer,
                template_prompt_ids,
                teacher_ids,
                label,
            )
        )

        stage1_positions[
            label
        ] = int(
            target_position
        )

        stage1_token_ids[
            label
        ] = int(
            target_token_id
        )

        stage1_prefixes[
            label
        ] = teacher_ids[
            :target_position
        ]

        if (
            label
            == stage1_valid_labels[0]
        ):
            stage1_reference_inputs = (
                teacher_inputs
            )

    stage1_answer_position = (
        stage1_positions[
            stage1_valid_labels[0]
        ]
    )

    stage1_reference_prefix = (
        stage1_prefixes[
            stage1_valid_labels[0]
        ]
    )

    for label in stage1_valid_labels:

        if (
            stage1_positions[
                label
            ]
            !=
            stage1_answer_position
        ):
            raise RuntimeError(
                "Stage1 answer-position mismatch."
            )

        if (
            stage1_prefixes[
                label
            ]
            !=
            stage1_reference_prefix
        ):
            raise RuntimeError(
                "Stage1 causal-prefix mismatch."
            )

    with torch.inference_mode():

        stage1_template_input_ids = (
            stage1_reference_inputs[
                "input_ids"
            ]
            .to(
                "cuda"
            )
        )

        stage1_template_attention_mask = (
            stage1_reference_inputs[
                "attention_mask"
            ]
            .to(
                "cuda"
            )
        )

        stage1_template_grid = (
            stage1_reference_inputs[
                "image_grid_thw"
            ]
            .to(
                "cuda"
            )
        )

        stage1_base_inputs_embeds = (
            model.model
            .get_input_embeddings()(
                stage1_template_input_ids
            )
        )

        (
            stage1_image_mask,
            _,
        ) = (
            model.model
            .get_placeholder_mask(
                stage1_template_input_ids,

                inputs_embeds=
                    stage1_base_inputs_embeds,
            )
        )

        (
            stage1_position_ids,
            _,
        ) = (
            model.model
            .get_rope_index(
                input_ids=
                    stage1_template_input_ids,

                image_grid_thw=
                    stage1_template_grid,

                attention_mask=
                    stage1_template_attention_mask,
            )
        )

    # --------------------------------------------------------
    # Encode all 135 candidates exactly once.
    #
    # Cache construction always uses SIX-IMAGE context.
    # The final incomplete bank group is padded with other
    # distinct candidates, and only its real candidate features
    # are retained.
    # --------------------------------------------------------

    cuda_sync()

    local_visual_cache_start = (
        time.perf_counter()
    )

    all_local_indices = list(
        range(
            local_boxes.shape[
                0
            ]
        )
    )

    for start in range(
        0,
        len(
            all_local_indices
        ),
        stage1.GROUP_SIZE,
    ):

        actual_group = (
            all_local_indices[
                start:
                start
                + stage1.GROUP_SIZE
            ]
        )

        encode_group = list(
            actual_group
        )

        if (
            len(
                encode_group
            )
            <
            stage1.GROUP_SIZE
        ):

            for candidate_index in (
                all_local_indices
            ):

                if (
                    candidate_index
                    in encode_group
                ):
                    continue

                encode_group.append(
                    candidate_index
                )

                if (
                    len(
                        encode_group
                    )
                    ==
                    stage1.GROUP_SIZE
                ):
                    break

        if (
            len(
                encode_group
            )
            !=
            stage1.GROUP_SIZE
        ):
            raise RuntimeError(
                "Unable to construct full "
                "six-image cache context."
            )

        visual_inputs = (
            build_stage1_teacher_inputs(
                encode_group,
                "A",
            )
        )

        pixel_values = (
            visual_inputs[
                "pixel_values"
            ]
            .to(
                "cuda"
            )
        )

        image_grid_thw = (
            visual_inputs[
                "image_grid_thw"
            ]
            .to(
                "cuda"
            )
        )

        with torch.inference_mode():

            image_outputs = (
                model.get_image_features(
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
            !=
            stage1.GROUP_SIZE
        ):
            raise RuntimeError(
                "Stage1 cache expected "
                "six visual features."
            )

        for position, candidate_index in (
            enumerate(
                actual_group
            )
        ):

            local_visual_cache[
                int(
                    candidate_index
                )
            ] = (
                features[
                    position
                ]
                .detach()
            )

        local_visual_cache_forwards += 1

        del image_outputs
        del features

    cuda_sync()

    local_visual_cache_sec = (
        time.perf_counter()
        -
        local_visual_cache_start
    )

    if (
        len(
            local_visual_cache
        )
        !=
        local_boxes.shape[
            0
        ]
    ):
        raise RuntimeError(
            "Local visual cache incomplete."
        )

    print(
        "LOCAL_VISUAL_CACHE=PASS"
    )

    print(
        "LOCAL VISUAL CACHE COUNT =",
        len(
            local_visual_cache
        ),
    )

    print(
        "LOCAL VISUAL CACHE FORWARDS =",
        local_visual_cache_forwards,
    )

    print(
        "LOCAL VISUAL CACHE SEC =",
        f"{local_visual_cache_sec:.3f}",
    )

    # --------------------------------------------------------
    # Exact cached Stage1 chooser for a FULL six-image group.
    #
    # Partial final groups deliberately fall back to original
    # choose_one(), so no unverified cross-group-size cache
    # assumption is introduced.
    # --------------------------------------------------------

    @torch.inference_mode()
    def cached_stage1_choose_full6(
        group,
    ):

        if (
            len(
                group
            )
            !=
            stage1.GROUP_SIZE
        ):
            raise RuntimeError(
                "cached_stage1_choose_full6 "
                "requires exactly six candidates."
            )

        image_embeds = torch.cat(
            [
                local_visual_cache[
                    int(
                        candidate_index
                    )
                ].to(
                    device=
                        stage1_base_inputs_embeds.device,

                    dtype=
                        stage1_base_inputs_embeds.dtype,
                )

                for candidate_index
                in group
            ],
            dim=0,
        )

        inputs_embeds = (
            stage1_base_inputs_embeds
            .masked_scatter(
                stage1_image_mask,
                image_embeds,
            )
        )

        cuda_sync()

        cached_forward_start = (
            time.perf_counter()
        )

        output = model(
            input_ids=None,

            inputs_embeds=
                inputs_embeds,

            attention_mask=
                stage1_template_attention_mask,

            position_ids=
                stage1_position_ids,

            pixel_values=None,

            image_grid_thw=None,

            use_cache=False,

            return_dict=True,
        )

        cuda_sync()

        cached_forward_sec = (
            time.perf_counter()
            -
            cached_forward_start
        )

        current_stage = (
            active_stage[
                "name"
            ]
        )

        if (
            current_stage
            in call_counts
        ):
            call_counts[
                current_stage
            ] += 1

            call_time[
                current_stage
            ] += (
                cached_forward_sec
            )

        logits = (
            output.logits[
                0,
                stage1_answer_position
                - 1,
            ]
            .float()
        )

        log_probs = torch.log_softmax(
            logits,
            dim=-1,
        )

        label_scores = {}

        for label in (
            stage1_valid_labels
        ):

            label_scores[
                label
            ] = float(
                log_probs[
                    stage1_token_ids[
                        label
                    ]
                ].item()
            )

        winner_label = max(
            stage1_valid_labels,

            key=lambda label:
                label_scores[
                    label
                ],
        )

        winner_position = (
            stage1.LABELS.index(
                winner_label
            )
        )

        winner = int(
            group[
                winner_position
            ]
        )

        del output

        return {
            "status":
                "PASS",

            "answer":
                winner_label,

            "winner":
                winner,

            "label_scores":
                label_scores,
        }

    # --------------------------------------------------------
    # Exact original scheduling logic with cached full groups.
    #
    # Same:
    #   seeds
    #   shuffle
    #   GROUP_SIZE
    #   winner rule
    #   union survivor rule
    # --------------------------------------------------------

    def run_stage1_round_visual_cached(
        candidates,
        round_index,
    ):

        print()
        print("=" * 100)
        print(
            "SCREEN ROUND:",
            round_index,
            "INPUT COUNT:",
            len(
                candidates
            ),
        )
        print("=" * 100)

        print(
            "VISUAL CACHE MODE = ON"
        )

        union_survivors = set()

        round_record = {
            "round":
                round_index,

            "input_candidates":
                list(
                    candidates
                ),

            "schedules": [],
        }

        for (
            schedule_index,
            base_seed,
        ) in enumerate(
            stage1.SCHEDULE_SEEDS,
            start=1,
        ):

            seed = (
                base_seed
                +
                round_index
                * 10000
            )

            schedule = (
                stage1.make_schedule(
                    candidates,
                    seed,
                )
            )

            schedule_survivors = set()

            groups = [
                schedule[
                    start:
                    start
                    + stage1.GROUP_SIZE
                ]

                for start in range(
                    0,
                    len(
                        schedule
                    ),
                    stage1.GROUP_SIZE,
                )
            ]

            print()
            print(
                "SCHEDULE:",
                schedule_index,

                "SEED:",
                seed,

                "GROUP COUNT:",
                len(
                    groups
                ),
            )

            schedule_record = {
                "schedule":
                    schedule_index,

                "seed":
                    seed,

                "groups": [],
            }

            for (
                group_index,
                group,
            ) in enumerate(
                groups,
                start=1,
            ):

                if (
                    len(
                        group
                    )
                    == 1
                ):

                    result = {
                        "status":
                            "SINGLE",

                        "answer":
                            "",

                        "winner":
                            int(
                                group[
                                    0
                                ]
                            ),
                    }

                elif (
                    len(
                        group
                    )
                    ==
                    stage1.GROUP_SIZE
                ):

                    result = (
                        cached_stage1_choose_full6(
                            group
                        )
                    )

                else:

                    # Exact original fallback for an incomplete
                    # final group.
                    result = (
                        stage1.choose_one(
                            model,
                            processor,
                            group,
                            local_views,
                        )
                    )

                if (
                    result[
                        "winner"
                    ]
                    is not None
                ):

                    schedule_survivors.add(
                        int(
                            result[
                                "winner"
                            ]
                        )
                    )

                else:

                    for candidate_index in (
                        group
                    ):

                        schedule_survivors.add(
                            int(
                                candidate_index
                            )
                        )

                schedule_record[
                    "groups"
                ].append(
                    {
                        "group":
                            list(
                                group
                            ),

                        "status":
                            result[
                                "status"
                            ],

                        "answer":
                            result[
                                "answer"
                            ],

                        "winner":
                            result[
                                "winner"
                            ],
                    }
                )

                if (
                    group_index
                    == 1

                    or
                    group_index
                    % 10
                    == 0

                    or
                    group_index
                    ==
                    len(
                        groups
                    )
                ):

                    print(
                        "GROUP:",
                        group_index,
                        "/",
                        len(
                            groups
                        ),

                        "STATUS:",
                        result[
                            "status"
                        ],

                        "WINNER:",
                        result[
                            "winner"
                        ],

                        "SURVIVORS SO FAR:",
                        len(
                            schedule_survivors
                        ),
                    )

            print(
                "SCHEDULE SURVIVORS:",
                len(
                    schedule_survivors
                ),
            )

            union_survivors.update(
                schedule_survivors
            )

            schedule_record[
                "survivors"
            ] = sorted(
                schedule_survivors
            )

            round_record[
                "schedules"
            ].append(
                schedule_record
            )

        survivors = sorted(
            union_survivors
        )

        print()
        print(
            "ROUND UNION SURVIVORS:",
            len(
                survivors
            ),
        )

        round_record[
            "output_candidates"
        ] = survivors

        return (
            survivors,
            round_record,
        )

    # ========================================================
    # Local screening
    # ========================================================

    print()
    print("=" * 100)
    print(
        "LOCAL FINE SCREEN"
    )
    print("=" * 100)

    active_stage[
        "name"
    ] = "local"

    (
        local_result,
        local_screen_sec,
    ) = measure(
        lambda:
            run_stage1_round_visual_cached(
                list(
                    range(
                        local_boxes.shape[
                            0
                        ]
                    )
                ),
                round_index=3,
            )

    )

    local_survivors, local_record = (
        local_result
    )

    # ========================================================
    # Pre-BT pruning:
    # use 3 schedules.
    # ========================================================

    print()
    print("=" * 100)
    print(
        "PRE-BT PRUNE"
    )
    print("=" * 100)

    old_schedule_seeds = (
        stage1.SCHEDULE_SEEDS
    )

    stage1.SCHEDULE_SEEDS = (
        1301,
        2903,
        4709,
    )

    active_stage[
        "name"
    ] = "prebt"

    (
        prebt_result,
        prebt_sec,
    ) = measure(
        lambda:
            run_stage1_round_visual_cached(
                local_survivors,
                round_index=4,
            )

    )

    stage1.SCHEDULE_SEEDS = (
        old_schedule_seeds
    )

    prebt_survivors, prebt_record = (
        prebt_result
    )

    if len(
        prebt_survivors
    ) < 2:
        raise RuntimeError(
            "Pre-BT pruning left fewer than 2 candidates."
        )

    # ========================================================
    # BT candidate views
    #
    # Use Stage2's original view construction.
    # ========================================================

    bt_view_start = (
        time.perf_counter()
    )

    bt_views = {
        index:
            stage2.screen.make_candidate_view(
                image,
                local_boxes[
                    index
                ],
            )
        for index in (
            prebt_survivors
        )
    }

    bt_view_sec = (
        time.perf_counter()
        - bt_view_start
    )

    # ========================================================
    # Build original balanced pair graph.
    # ========================================================

    pairs, degree = (
        stage2.build_balanced_pairs(
            len(
                prebt_survivors
            )
        )
    )

    print()
    print("=" * 100)
    print(
        "BT PAIRWISE SCORING"
    )
    print("=" * 100)

    print(
        "BT CANDIDATES =",
        len(
            prebt_survivors
        ),
    )

    print(
        "PAIR COUNT =",
        len(
            pairs
        ),
    )

    print(
        "PAIR DEGREE VALUES =",
        sorted(
            set(
                degree
            )
        ),
    )

    pair_probs = {
        position: []
        for position in range(
            len(
                prebt_survivors
            )
        )
    }

    hard_wins = {
        position: []
        for position in range(
            len(
                prebt_survivors
            )
        )
    }

    matches = []

    active_stage[
        "name"
    ] = "bt"

    # ========================================================
    # Exact cached Stage2.
    #
    # Proven properties:
    #
    # 1. Every Stage2 pair has identical token/template layout.
    #
    # 2. Candidate pooler_output is identical across different
    #    TWO-IMAGE pair contexts.
    #
    # 3. Manually scattering cached pooler_output and supplying
    #    exact multimodal position_ids reproduces the original
    #    Stage2 pair probabilities exactly.
    # ========================================================

    bt_vision_cache_forwards = (
        (
            len(
                prebt_survivors
            )
            + 1
        )
        // 2
    )

    bt_language_forwards = (
        len(
            pairs
        )
        * 2
    )

    # --------------------------------------------------------
    # Build one exact teacher-forced pair input.
    # --------------------------------------------------------

    def build_pair_teacher_inputs(
        group,
        label,
    ):
        content = (
            stage2.screen.build_content(
                group,
                bt_views,
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
            processor.apply_chat_template(
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
            stage2.screen.build_content(
                group,
                bt_views,
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
            processor.apply_chat_template(
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
            stage2.screen.find_answer_token(
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
            stage2.screen.find_answer_token(
                processor.tokenizer,
                prompt_ids,
                ids_b,
                "B",
            )
        )

        if (
            pos_a
            != pos_b
        ):
            raise RuntimeError(
                "A/B answer positions differ."
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

    # --------------------------------------------------------
    # Fixed Stage2 template.
    # --------------------------------------------------------

    bt_template_start = (
        time.perf_counter()
    )

    first_i_pos, first_j_pos = (
        pairs[
            0
        ]
    )

    template_group = [
        prebt_survivors[
            first_i_pos
        ],
        prebt_survivors[
            first_j_pos
        ],
    ]

    (
        template_inputs,
        answer_position,
        token_a,
        token_b,
    ) = build_pair_template(
        template_group
    )

    with torch.inference_mode():

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

        base_inputs_embeds = (
            model.model
            .get_input_embeddings()(
                template_input_ids
            )
        )

        image_mask, _ = (
            model.model
            .get_placeholder_mask(
                template_input_ids,
                inputs_embeds=
                    base_inputs_embeds,
            )
        )

        (
            position_ids,
            template_rope_deltas,
        ) = (
            model.model.get_rope_index(
                input_ids=
                    template_input_ids,

                image_grid_thw=
                    template_image_grid_thw,

                attention_mask=
                    template_attention_mask,
            )
        )

    cuda_sync()

    bt_template_sec = (
        time.perf_counter()
        - bt_template_start
    )

    print(
        "BT_PAIR_TEMPLATE=PASS"
    )

    # --------------------------------------------------------
    # Two-image visual encoder.
    #
    # IMPORTANT:
    # Do NOT encode a candidate alone.
    # Exact equivalence was established for TWO-IMAGE context.
    # --------------------------------------------------------

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
            model.get_image_features(
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
                "Expected exactly two image features."
            )

        return (
            features[
                0
            ].detach(),
            features[
                1
            ].detach(),
        )

    # --------------------------------------------------------
    # TWO-IMAGE vision shape warmup:
    # pay once for the entire multi-scene process.
    # --------------------------------------------------------

    if not _SHARED_BT_VISION_WARMED:

        cuda_sync()

        bt_vision_warmup_start = (
            time.perf_counter()
        )

        warm_visual_a, warm_visual_b = (
            encode_two_candidates(
                prebt_survivors[
                    0
                ],
                prebt_survivors[
                    1
                ],
            )
        )

        cuda_sync()

        bt_vision_warmup_sec = (
            time.perf_counter()
            - bt_vision_warmup_start
        )

        del warm_visual_a
        del warm_visual_b

        torch.cuda.empty_cache()

        _SHARED_BT_VISION_WARMED = True

        print(
            "BT_VISION_SHAPE_WARMUP=PASS"
        )

    else:

        bt_vision_warmup_sec = 0.0

        print(
            "BT_VISION_SHAPE_WARMUP=REUSED"
        )

    # --------------------------------------------------------
    # Build complete reusable candidate visual cache.
    #
    # 16 candidates:
    #     8 two-image Vision forwards.
    # --------------------------------------------------------

    visual_cache = {}

    cuda_sync()

    bt_visual_cache_start = (
        time.perf_counter()
    )

    for start_position in range(
        0,
        len(
            prebt_survivors
        ),
        2,
    ):
        index_a = (
            prebt_survivors[
                start_position
            ]
        )

        if (
            start_position + 1
            <
            len(
                prebt_survivors
            )
        ):
            index_b = (
                prebt_survivors[
                    start_position + 1
                ]
            )
        else:
            index_b = (
                prebt_survivors[
                    0
                ]
            )

        (
            feature_a,
            feature_b,
        ) = (
            encode_two_candidates(
                index_a,
                index_b,
            )
        )

        visual_cache[
            index_a
        ] = (
            feature_a
        )

        if (
            start_position + 1
            <
            len(
                prebt_survivors
            )
        ):
            visual_cache[
                index_b
            ] = (
                feature_b
            )

    cuda_sync()

    bt_visual_cache_sec = (
        time.perf_counter()
        - bt_visual_cache_start
    )

    if (
        len(
            visual_cache
        )
        !=
        len(
            prebt_survivors
        )
    ):
        raise RuntimeError(
            "Visual cache incomplete."
        )

    print(
        "BT_VISUAL_CACHE=PASS"
    )

    print(
        "BT_VISUAL CACHE COUNT =",
        len(
            visual_cache
        ),
    )

    print(
        "BT_VISION CACHE FORWARDS =",
        bt_vision_cache_forwards,
    )

    # --------------------------------------------------------
    # Cached language-only orientation.
    #
    # first_index  -> Candidate A
    # second_index -> Candidate B
    # --------------------------------------------------------

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

        two_way_scores = torch.tensor(
            [
                score_a,
                score_b,
            ],
            dtype=torch.float64,
        )

        two_way_probs = (
            torch.softmax(
                two_way_scores,
                dim=0,
            )
        )

        p_a = float(
            two_way_probs[
                0
            ].item()
        )

        p_b = float(
            two_way_probs[
                1
            ].item()
        )

        del output

        return {
            "score_a":
                score_a,

            "score_b":
                score_b,

            "p_a":
                p_a,

            "p_b":
                p_b,

            "label_scores": {
                "A":
                    score_a,

                "B":
                    score_b,
            },
        }

    # --------------------------------------------------------
    # Cached-language execution-shape warmup:
    # once per target-text family.
    # --------------------------------------------------------

    if (
        target_text
        not in
        _SHARED_BT_LANGUAGE_WARMED_TARGETS
    ):

        cuda_sync()

        bt_language_warmup_start = (
            time.perf_counter()
        )

        warm_cached = (
            cached_orientation(
                template_group[
                    0
                ],
                template_group[
                    1
                ],
            )
        )

        cuda_sync()

        bt_language_warmup_sec = (
            time.perf_counter()
            - bt_language_warmup_start
        )

        del warm_cached

        torch.cuda.empty_cache()

        _SHARED_BT_LANGUAGE_WARMED_TARGETS.add(
            target_text
        )

        print(
            "BT_LANGUAGE_SHAPE_WARMUP=PASS"
        )

    else:

        bt_language_warmup_sec = 0.0

        print(
            "BT_LANGUAGE_SHAPE_WARMUP=REUSED"
        )

    # --------------------------------------------------------
    # Full exact 64-pair cached scoring.
    # --------------------------------------------------------

    cuda_sync()

    cached_pairwise_start = (
        time.perf_counter()
    )

    for match_number, (
        i_pos,
        j_pos,
    ) in enumerate(
        pairs,
        start=1,
    ):
        i_index = (
            prebt_survivors[
                i_pos
            ]
        )

        j_index = (
            prebt_survivors[
                j_pos
            ]
        )

        # AB:
        # i is Candidate A.
        ab = (
            cached_orientation(
                i_index,
                j_index,
            )
        )

        # BA:
        # i is Candidate B.
        ba = (
            cached_orientation(
                j_index,
                i_index,
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
            * (
                p_i_ab
                +
                p_i_ba
            )
        )

        p_j = (
            1.0
            - p_i
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
                p_i > 0.5
            )
        )

        hard_wins[
            j_pos
        ].append(
            float(
                p_j > 0.5
            )
        )

        matches.append(
            {
                "i_pos":
                    i_pos,

                "j_pos":
                    j_pos,

                "i_index":
                    i_index,

                "j_index":
                    j_index,

                "p_i":
                    p_i,

                "p_i_ab":
                    p_i_ab,

                "p_i_ba":
                    p_i_ba,

                "scores_ab":
                    ab[
                        "label_scores"
                    ],

                "scores_ba":
                    ba[
                        "label_scores"
                    ],
            }
        )

        if (
            match_number == 1
            or match_number % 10 == 0
            or match_number
            == len(
                pairs
            )
        ):
            print(
                "MATCH:",
                match_number,
                "/",
                len(
                    pairs
                ),
                "I:",
                i_index,
                "J:",
                j_index,
                "P_I:",
                f"{p_i:.4f}",
            )

    cuda_sync()

    cached_pairwise_sec = (
        time.perf_counter()
        - cached_pairwise_start
    )

    # Total steady-state BT scoring cost.
    #
    # Excludes the two one-time execution-shape warmups.
    pairwise_sec = (
        bt_template_sec
        +
        bt_visual_cache_sec
        +
        cached_pairwise_sec
    )

    # Existing instrumentation counts choose_one() calls.
    # Cached Stage2 bypasses choose_one(), so fill these values
    # manually for an informative report.
    call_counts[
        "bt"
    ] = (
        bt_language_forwards
    )

    call_time[
        "bt"
    ] = (
        cached_pairwise_sec
    )

    active_stage[
        "name"
    ] = ""

    # ========================================================
    # Bradley-Terry fit
    # ========================================================

    (
        theta,
        bt_fit_sec,
    ) = measure(
        lambda:
            stage2.fit_bradley_terry(
                len(
                    prebt_survivors
                ),
                matches,
            )
    )

    ranking = []

    for position, index in enumerate(
        prebt_survivors
    ):
        probabilities = (
            pair_probs[
                position
            ]
        )

        wins = (
            hard_wins[
                position
            ]
        )

        ranking.append(
            {
                "position":
                    position,

                "index":
                    int(
                        index
                    ),

                "bt_score":
                    float(
                        theta[
                            position
                        ].item()
                    ),

                "pair_mean":
                    float(
                        sum(
                            probabilities
                        )
                        /
                        len(
                            probabilities
                        )
                    ),

                "hard_win_rate":
                    float(
                        sum(
                            wins
                        )
                        /
                        len(
                            wins
                        )
                    ),

                "box":
                    local_boxes[
                        index
                    ].tolist(),
            }
        )

    ranking.sort(
        key=lambda item:
            item[
                "bt_score"
            ],
        reverse=True,
    )

    cuda_sync()

    raw_pipeline_sec = (
        time.perf_counter()
        - pipeline_start
    )

    pipeline_sec = (
        raw_pipeline_sec
        -
        bt_vision_warmup_sec
        -
        bt_language_warmup_sec
    )

    # ========================================================
    # Save results
    # ========================================================

    timing = {
        "model_load_sec":
            model_load_sec,

        "warmup_sec":
            warmup_sec,

        "coarse1_sec":
            coarse1_sec,

        "coarse2_sec":
            coarse2_sec,

        "local_generation_sec":
            local_gen_sec,

        "local_screen_sec":
            local_screen_sec,

        "prebt_sec":
            prebt_sec,

        "bt_view_sec":
            bt_view_sec,

        "bt_template_sec":
            bt_template_sec,

        "bt_vision_warmup_sec":
            bt_vision_warmup_sec,

        "bt_visual_cache_sec":
            bt_visual_cache_sec,

        "bt_language_warmup_sec":
            bt_language_warmup_sec,

        "cached_pairwise_sec":
            cached_pairwise_sec,

        "pairwise_sec":
            pairwise_sec,

        "bt_fit_sec":
            bt_fit_sec,

        "raw_pipeline_sec":
            raw_pipeline_sec,

        "persistent_pipeline_sec":
            pipeline_sec,

        "cold_model_plus_warmup_plus_pipeline_sec":
            (
                model_load_sec
                + warmup_sec
                + raw_pipeline_sec
            ),
    }

    total_qwen_calls = sum(
        call_counts.values()
    )

    result_path = (
        output_dir
        / "fast_teacher_results.pt"
    )

    torch.save(
        {
            "target_text":
                target_text,

            "image_path":
                str(
                    image_path
                ),

            "coarse_boxes":
                coarse_boxes,

            "coarse_round1":
                coarse1,

            "coarse_round2":
                coarse2,

            "local_boxes":
                local_boxes,

            "local_survivors":
                local_survivors,

            "prebt_candidates":
                prebt_survivors,

            "pair_offsets":
                stage2.PAIR_OFFSETS,

            "pair_degree":
                degree,

            "matches":
                matches,

            "bt_scores":
                theta.detach().cpu(),

            "ranking":
                ranking,

            "timing":
                timing,

            "qwen_call_counts":
                call_counts,

            "qwen_call_time":
                call_time,

            "total_qwen_calls":
                total_qwen_calls,
        },
        result_path,
    )

    coarse_contact = (
        output_dir
        / "coarse_final.png"
    )

    prebt_contact = (
        output_dir
        / "prebt_candidates.png"
    )

    top12_contact = (
        output_dir
        / "top12_final_bt.png"
    )

    stage1.save_contact_sheet(
        coarse2,
        coarse_views,
        coarse_contact,
    )

    stage1.save_contact_sheet(
        prebt_survivors,
        local_views,
        prebt_contact,
    )

    stage2.save_contact_sheet(
        ranking,
        bt_views,
        top12_contact,
        top_k=12,
    )

    # ========================================================
    # Optional reference comparison
    # ========================================================

    reference_iou = None

    if args.reference_stage2:
        reference_path = Path(
            args.reference_stage2
        )

        reference = torch.load(
            reference_path,
            map_location="cpu",
            weights_only=False,
        )

        reference_box = (
            reference[
                "ranking"
            ][0][
                "box"
            ]
        )

        reference_iou = (
            box_iou(
                reference_box,
                ranking[
                    0
                ][
                    "box"
                ],
            )
        )

    # ========================================================
    # Final report
    # ========================================================

    print()
    print("=" * 100)
    print(
        "FAST TEACHER FINAL REPORT"
    )
    print("=" * 100)

    print(
        "COARSE INPUT =",
        int(
            coarse_boxes.shape[0]
        ),
    )

    print(
        "COARSE ROUND1 =",
        len(
            coarse1
        ),
    )

    print(
        "COARSE ROUND2 =",
        len(
            coarse2
        ),
    )

    print(
        "LOCAL INPUT =",
        int(
            local_boxes.shape[0]
        ),
    )

    print(
        "LOCAL SURVIVORS =",
        len(
            local_survivors
        ),
    )

    print(
        "PREBT SURVIVORS =",
        len(
            prebt_survivors
        ),
    )

    print(
        "BT PAIRS =",
        len(
            pairs
        ),
    )

    print()

    print(
        "TOP1 INDEX =",
        ranking[
            0
        ][
            "index"
        ],
    )

    print(
        "TOP1 BOX =",
        ranking[
            0
        ][
            "box"
        ],
    )

    print(
        "TOP1 BT SCORE =",
        ranking[
            0
        ][
            "bt_score"
        ],
    )

    print(
        "TOP3 INDICES =",
        [
            item[
                "index"
            ]
            for item in ranking[
                :3
            ]
        ],
    )

    if reference_iou is not None:
        print(
            "REFERENCE TOP1 IOU =",
            f"{reference_iou:.6f}",
        )

    print()
    print("-" * 100)

    for name in [
        "coarse1",
        "coarse2",
        "local",
        "prebt",
        "bt",
    ]:
        print(
            f"{name.upper()} QWEN CALLS =",
            call_counts[
                name
            ],
            "QWEN_SEC =",
            f"{call_time[name]:.3f}",
        )

    print()

    print(
        "TOTAL QWEN CALLS =",
        total_qwen_calls,
    )

    print(
        "BT VISION CACHE FORWARDS =",
        bt_vision_cache_forwards,
    )

    print(
        "BT CACHED LANGUAGE FORWARDS =",
        bt_language_forwards,
    )

    print(
        "TOTAL MODEL PASSES INCLUDING "
        "VISION-CACHE BUILDS =",
        total_qwen_calls
        + bt_vision_cache_forwards,
    )

    print()
    print("-" * 100)

    print(
        "MODEL LOAD SEC =",
        f"{model_load_sec:.3f}",
    )

    print(
        "ONE-TIME WARMUP SEC =",
        f"{warmup_sec:.3f}",
    )

    print(
        "COARSE1 SEC =",
        f"{coarse1_sec:.3f}",
    )

    print(
        "COARSE2 SEC =",
        f"{coarse2_sec:.3f}",
    )

    print(
        "LOCAL GENERATION SEC =",
        f"{local_gen_sec:.3f}",
    )

    print(
        "LOCAL SCREEN SEC =",
        f"{local_screen_sec:.3f}",
    )

    print(
        "PREBT SEC =",
        f"{prebt_sec:.3f}",
    )

    print(
        "BT TEMPLATE SEC =",
        f"{bt_template_sec:.3f}",
    )

    print(
        "BT VISION SHAPE WARMUP SEC =",
        f"{bt_vision_warmup_sec:.3f}",
    )

    print(
        "BT VISUAL CACHE BUILD SEC =",
        f"{bt_visual_cache_sec:.3f}",
    )

    print(
        "BT LANGUAGE SHAPE WARMUP SEC =",
        f"{bt_language_warmup_sec:.3f}",
    )

    print(
        "BT CACHED LANGUAGE PAIRWISE SEC =",
        f"{cached_pairwise_sec:.3f}",
    )

    print(
        "PAIRWISE SEC =",
        f"{pairwise_sec:.3f}",
    )

    print(
        "BT FIT SEC =",
        f"{bt_fit_sec:.3f}",
    )

    print()

    print(
        "RAW FIRST-IMAGE PIPELINE SEC =",
        f"{raw_pipeline_sec:.3f}",
    )

    print(
        "PERSISTENT PIPELINE SEC =",
        f"{pipeline_sec:.3f}",
    )

    print(
        "COLD MODEL+WARMUP+PIPELINE SEC =",
        f"{model_load_sec + warmup_sec + raw_pipeline_sec:.3f}",
    )

    print()

    print(
        "RESULT =",
        result_path,
    )

    print(
        "TOP12 =",
        top12_contact,
    )

    print()

    if (
        reference_iou is None
        or reference_iou >= 0.70
    ):
        print(
            "FAST_TEACHER_PERSISTENT_V6_CACHED=PASS"
        )
    else:
        print(
            "FAST_TEACHER_PERSISTENT_V6_CACHED=FAIL"
        )


if __name__ == "__main__":
    main()
