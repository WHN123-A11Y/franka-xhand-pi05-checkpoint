from pathlib import Path
import argparse
import math
import sys

import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

sys.path.insert(
    0,
    "experiments/delta_grounding",
)

import test_generic_hierarchical_semantic_screen as screen


PAIR_OFFSETS = (
    1,
    2,
    4,
    7,
)

BT_STEPS = 2500
BT_LR = 0.05


def build_balanced_pairs(
    candidate_count,
):
    pairs = set()

    for i in range(
        candidate_count
    ):
        for offset in PAIR_OFFSETS:
            j = (
                i + offset
            ) % candidate_count

            a = min(
                i,
                j,
            )

            b = max(
                i,
                j,
            )

            if a != b:
                pairs.add(
                    (
                        a,
                        b,
                    )
                )

    pairs = sorted(
        pairs
    )

    degree = [
        0
        for _ in range(
            candidate_count
        )
    ]

    for i, j in pairs:
        degree[i] += 1
        degree[j] += 1

    return (
        pairs,
        degree,
    )


def label_probability(
    result,
    wanted_label,
):
    logits = torch.tensor(
        [
            result[
                "label_scores"
            ][
                "A"
            ],
            result[
                "label_scores"
            ][
                "B"
            ],
        ],
        dtype=torch.float64,
    )

    probs = torch.softmax(
        logits,
        dim=0,
    )

    if wanted_label == "A":
        return float(
            probs[
                0
            ].item()
        )

    if wanted_label == "B":
        return float(
            probs[
                1
            ].item()
        )

    raise ValueError(
        wanted_label
    )


@torch.inference_mode()
def score_pair(
    model,
    processor,
    i_index,
    j_index,
    candidate_views,
):
    # --------------------------------------------------------
    # Assignment 1:
    # i -> A
    # j -> B
    # --------------------------------------------------------

    result_ab = screen.choose_one(
        model,
        processor,
        [
            i_index,
            j_index,
        ],
        candidate_views,
    )

    p_i_ab = label_probability(
        result_ab,
        "A",
    )

    # --------------------------------------------------------
    # Assignment 2:
    # j -> A
    # i -> B
    #
    # Explicit label permutation removes A/B preference.
    # --------------------------------------------------------

    result_ba = screen.choose_one(
        model,
        processor,
        [
            j_index,
            i_index,
        ],
        candidate_views,
    )

    p_i_ba = label_probability(
        result_ba,
        "B",
    )

    p_i = 0.5 * (
        p_i_ab
        +
        p_i_ba
    )

    return {
        "p_i":
            float(
                p_i
            ),
        "p_i_ab":
            float(
                p_i_ab
            ),
        "p_i_ba":
            float(
                p_i_ba
            ),
        "scores_ab":
            result_ab[
                "label_scores"
            ],
        "scores_ba":
            result_ba[
                "label_scores"
            ],
    }


def fit_bradley_terry(
    candidate_count,
    matches,
):
    theta = torch.nn.Parameter(
        torch.zeros(
            candidate_count,
            dtype=torch.float64,
        )
    )

    optimizer = torch.optim.Adam(
        [
            theta
        ],
        lr=BT_LR,
    )

    i_indices = torch.tensor(
        [
            match[
                "i_pos"
            ]
            for match in matches
        ],
        dtype=torch.long,
    )

    j_indices = torch.tensor(
        [
            match[
                "j_pos"
            ]
            for match in matches
        ],
        dtype=torch.long,
    )

    targets = torch.tensor(
        [
            match[
                "p_i"
            ]
            for match in matches
        ],
        dtype=torch.float64,
    )

    report_steps = {
        1,
        100,
        500,
        1000,
        BT_STEPS,
    }

    for step in range(
        1,
        BT_STEPS + 1,
    ):
        optimizer.zero_grad()

        logits = (
            theta[
                i_indices
            ]
            -
            theta[
                j_indices
            ]
        )

        loss = (
            F.binary_cross_entropy_with_logits(
                logits,
                targets,
            )
        )

        loss.backward()

        optimizer.step()

        with torch.no_grad():
            theta -= theta.mean()

        if step in report_steps:
            print(
                "BT STEP:",
                step,
                "LOSS:",
                float(
                    loss.item()
                ),
            )

    return (
        theta
        .detach()
        .float()
        .cpu()
    )


def save_contact_sheet(
    ranking,
    candidate_views,
    output_path,
    top_k=12,
):
    selected = ranking[
        :min(
            top_k,
            len(ranking),
        )
    ]

    thumb_width = 320
    thumb_height = 240
    header = 44
    columns = 3

    rows = math.ceil(
        len(selected)
        /
        columns
    )

    canvas = Image.new(
        "RGB",
        (
            columns
            *
            thumb_width,
            rows
            *
            (
                thumb_height
                +
                header
            ),
        ),
        "white",
    )

    draw = ImageDraw.Draw(
        canvas
    )

    for position, item in enumerate(
        selected
    ):
        row = (
            position
            //
            columns
        )

        column = (
            position
            %
            columns
        )

        x = (
            column
            *
            thumb_width
        )

        y = (
            row
            *
            (
                thumb_height
                +
                header
            )
        )

        view = candidate_views[
            item[
                "index"
            ]
        ].copy()

        view.thumbnail(
            (
                thumb_width,
                thumb_height,
            )
        )

        canvas.paste(
            view,
            (
                x,
                y
                +
                header,
            ),
        )

        draw.text(
            (
                x + 8,
                y + 6,
            ),
            (
                f"rank={position + 1} "
                f"idx={item['index']} "
                f"BT={item['bt_score']:.3f}"
            ),
            fill="black",
        )

    canvas.save(
        output_path
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--stage1",
        required=True,
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

    args = parser.parse_args()

    stage1_path = Path(
        args.stage1
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

    print("=" * 100)
    print(
        "GENERIC STAGE-2 BALANCED BRADLEY-TERRY"
    )
    print("=" * 100)

    print(
        "TARGET TEXT:",
        target_text,
    )

    screen.TARGET_TEXT = (
        target_text
    )

    stage1 = torch.load(
        stage1_path,
        map_location="cpu",
        weights_only=False,
    )

    if (
        "final_candidates"
        not in stage1
    ):
        print(
            "STAGE1 KEYS:",
            list(
                stage1.keys()
            ),
        )

        raise RuntimeError(
            "final_candidates not found"
        )

    candidates = [
        int(
            index
        )
        for index in stage1[
            "final_candidates"
        ]
    ]

    print(
        "STAGE1 FINAL COUNT:",
        len(
            candidates
        ),
    )

    print(
        "STAGE1 FINAL CANDIDATES:",
        candidates,
    )

    image = Image.open(
        image_path
    ).convert(
        "RGB"
    )

    # --------------------------------------------------------
    # Recreate exactly the same 1071 compact proposal bank.
    # No GT information is used.
    # --------------------------------------------------------

    pooler = (
        screen
        .CompactCandidatePooler()
    )

    dummy_tokens = torch.zeros(
        1,
        15 * 20,
        1,
    )

    proposals = pooler(
        dummy_tokens,
        grid_h=15,
        grid_w=20,
    )

    candidate_boxes = (
        proposals
        .candidate_boxes
        .float()
        .cpu()
    )

    print(
        "COMPACT CANDIDATE COUNT:",
        int(
            candidate_boxes.shape[
                0
            ]
        ),
    )

    # ============================================================
    # Phase-aware Candidate Bank:
    #
    # Stage2 MUST inherit the exact candidate geometry produced
    # and saved by Stage1.
    #
    # Do NOT regenerate the old 1071-candidate bank here.
    # Candidate IDs must remain identical across Stage1 / Stage2.
    # ============================================================

    if "candidate_boxes" not in stage1:
        raise RuntimeError(
            "Stage1 result contains no candidate_boxes."
        )

    candidate_boxes = (
        stage1[
            "candidate_boxes"
        ]
        .detach()
        .float()
        .cpu()
    )

    candidate_count = int(
        candidate_boxes.shape[0]
    )

    print(
        "INHERITED STAGE1 CANDIDATE COUNT:",
        candidate_count,
    )

    if candidate_count <= 0:
        raise RuntimeError(
            "Inherited candidate bank is empty."
        )

    if len(candidates) == 0:
        raise RuntimeError(
            "Stage1 final candidate list is empty."
        )

    max_candidate_id = max(
        candidates
    )

    print(
        "MAX STAGE1 FINAL CANDIDATE ID:",
        max_candidate_id,
    )

    if max_candidate_id >= candidate_count:
        raise RuntimeError(
            "Stage1 candidate ID exceeds inherited "
            "candidate-box bank."
        )

    print(
        "PHASEAWARE_STAGE1_STAGE2_ID_ALIGNMENT=PASS"
    )

    candidate_views = {
        index:
            screen.make_candidate_view(
                image,
                candidate_boxes[
                    index
                ],
            )
        for index in candidates
    }

    pairs, degree = build_balanced_pairs(
        len(
            candidates
        )
    )

    print(
        "PAIR OFFSETS:",
        PAIR_OFFSETS,
    )

    print(
        "PAIR COUNT:",
        len(
            pairs
        ),
    )

    print(
        "PAIR DEGREE VALUES:",
        sorted(
            set(
                degree
            )
        ),
    )

    print(
        "PAIR DEGREES:",
        degree,
    )

    if sorted(
        set(
            degree
        )
    ) != [
        8
    ]:
        raise RuntimeError(
            f"Unbalanced pair graph: {degree}"
        )

    processor = (
        AutoProcessor
        .from_pretrained(
            screen.MODEL_PATH,
            local_files_only=True,
        )
    )

    model = (
        Qwen3_5ForConditionalGeneration
        .from_pretrained(
            screen.MODEL_PATH,
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
            local_files_only=True,
        )
        .to(
            "cuda"
        )
        .eval()
    )

    print(
        "MODEL_LOAD=PASS"
    )

    matches = []

    pair_probs = {
        position: []
        for position in range(
            len(
                candidates
            )
        )
    }

    hard_wins = {
        position: []
        for position in range(
            len(
                candidates
            )
        )
    }

    for match_number, (
        i_pos,
        j_pos,
    ) in enumerate(
        pairs,
        start=1,
    ):
        i_index = candidates[
            i_pos
        ]

        j_index = candidates[
            j_pos
        ]

        result = score_pair(
            model,
            processor,
            i_index,
            j_index,
            candidate_views,
        )

        p_i = result[
            "p_i"
        ]

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
                "i_index":
                    i_index,
                "j_index":
                    j_index,
                "p_i":
                    p_i,
                "p_i_ab":
                    result[
                        "p_i_ab"
                    ],
                "p_i_ba":
                    result[
                        "p_i_ba"
                    ],
                "scores_ab":
                    result[
                        "scores_ab"
                    ],
                "scores_ba":
                    result[
                        "scores_ba"
                    ],
            }
        )

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
            p_i,
            "P_I_AB:",
            result[
                "p_i_ab"
            ],
            "P_I_BA:",
            result[
                "p_i_ba"
            ],
        )

    print(
        "PAIRWISE_SCORING=PASS"
    )

    theta = fit_bradley_terry(
        len(
            candidates
        ),
        matches,
    )

    ranking = []

    for position, index in enumerate(
        candidates
    ):
        probabilities = (
            pair_probs[
                position
            ]
        )

        wins = hard_wins[
            position
        ]

        ranking.append(
            {
                "position":
                    position,
                "index":
                    index,
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
                    candidate_boxes[
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

    print()
    print("=" * 100)
    print(
        "BRADLEY-TERRY GLOBAL RANKING"
    )
    print("=" * 100)

    for rank, item in enumerate(
        ranking,
        start=1,
    ):
        print(
            "BT RANK:",
            rank,
            "INDEX:",
            item[
                "index"
            ],
            "BT SCORE:",
            item[
                "bt_score"
            ],
            "PAIR MEAN:",
            item[
                "pair_mean"
            ],
            "HARD WIN RATE:",
            item[
                "hard_win_rate"
            ],
            "BOX:",
            item[
                "box"
            ],
        )

    result_path = (
        output_dir
        /
        "stage2_bt_results.pt"
    )

    torch.save(
        {
            "target_text":
                target_text,
            "stage1_path":
                str(
                    stage1_path
                ),
            "candidate_boxes":
                candidate_boxes,
            "candidates":
                candidates,
            "pair_offsets":
                PAIR_OFFSETS,
            "pair_degree":
                degree,
            "matches":
                matches,
            "bt_scores":
                theta,
            "ranking":
                ranking,
        },
        result_path,
    )

    contact_path = (
        output_dir
        /
        "top12_stage2_bt.png"
    )

    save_contact_sheet(
        ranking,
        candidate_views,
        contact_path,
        top_k=12,
    )

    print()
    print(
        "TOP1 INDEX:",
        ranking[
            0
        ][
            "index"
        ],
    )

    print(
        "TOP3 INDICES:",
        [
            item[
                "index"
            ]
            for item in ranking[
                :3
            ]
        ],
    )

    print(
        "RESULT:",
        result_path,
    )

    print(
        "TOP12 IMAGE:",
        contact_path,
    )

    print(
        "GENERIC_STAGE2_BT=EXECUTION_PASS"
    )


if __name__ == "__main__":
    main()
