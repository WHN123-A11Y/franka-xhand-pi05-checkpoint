from pathlib import Path
import argparse
import math
import random

import torch
from PIL import Image, ImageDraw
from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

from test_qwen_auto_proposal_teacher import (
    MODEL_PATH,
    find_answer_token,
)
from starVLA.model.modules.delta_grounding.compact_candidate_pooler import (
    CompactCandidatePooler,
)


IMAGE_PATH = None
OUTPUT_DIR = None
TARGET_TEXT = None

GROUP_SIZE = 6

# 两个完全固定、可复现的赛程。
# 不依赖 GT（真实标注）。
SCHEDULE_SEEDS = (
    1301,
    2903,
)

# 1071 -> ~350 -> ~120 -> ~40
FINAL_MAX = 48
MAX_ROUNDS = 4

LABELS = (
    "A",
    "B",
    "C",
    "D",
    "E",
    "F",
)


def normalized_to_pixel_box(
    box,
    width,
    height,
):
    return (
        int(round(float(box[0]) * width)),
        int(round(float(box[1]) * height)),
        int(round(float(box[2]) * width)),
        int(round(float(box[3]) * height)),
    )


def make_candidate_view(
    image,
    box,
):
    output = image.copy()

    draw = ImageDraw.Draw(
        output
    )

    pixel_box = normalized_to_pixel_box(
        box,
        image.width,
        image.height,
    )

    draw.rectangle(
        pixel_box,
        outline="red",
        width=6,
    )

    return output


def build_content(
    group,
    candidate_views,
):
    content = [
        {
            "type": "text",
            "text": (
                "The following candidate views come from the SAME scene. "
                "Each image contains exactly one automatically generated "
                "proposal outlined in red. "
                "Candidate labels are arbitrary."
            ),
        }
    ]

    for position, index in enumerate(
        group
    ):
        label = LABELS[
            position
        ]

        content.append(
            {
                "type": "text",
                "text": (
                    f"Candidate {label}:"
                ),
            }
        )

        content.append(
            {
                "type": "image",
                "image": candidate_views[
                    index
                ],
            }
        )

    valid_labels = ", ".join(
        LABELS[
            :len(group)
        ]
    )

    content.append(
        {
            "type": "text",
            "text": (
                "Target description: "
                f"{TARGET_TEXT}. "
                "Which candidate BEST corresponds to the target description? "
                "Judge the actual region inside the red rectangle. "
                "Prefer a candidate that tightly covers the complete target "
                "object rather than background, an unrelated object, or only "
                "a small fragment of the target. "
                f"Answer only one label from: {valid_labels}."
            ),
        }
    )

    return content


@torch.inference_mode()
def choose_one(
    model,
    processor,
    group,
    candidate_views,
):
    content = build_content(
        group,
        candidate_views,
    )

    valid_labels = LABELS[
        :len(group)
    ]

    # --------------------------------------------------------
    # Build the user-only prompt.
    # It is used only to locate the assistant answer token.
    # --------------------------------------------------------

    user_messages = [[
        {
            "role": "user",
            "content": content,
        }
    ]]

    prompt_inputs = processor.apply_chat_template(
        user_messages,
        tokenize=True,
        padding=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )

    prompt_ids = (
        prompt_inputs[
            "input_ids"
        ][0]
        .tolist()
    )

    # --------------------------------------------------------
    # Construct teacher-forced templates for every label.
    # All labels MUST share exactly the same causal prefix.
    # --------------------------------------------------------

    positions = {}
    token_ids = {}
    prefixes = {}
    teacher_inputs_by_label = {}

    for label in valid_labels:
        teacher_messages = [[
            {
                "role": "user",
                "content": content,
            },
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": label,
                    }
                ],
            },
        ]]

        teacher_inputs = processor.apply_chat_template(
            teacher_messages,
            tokenize=True,
            padding=True,
            add_generation_prompt=False,
            return_dict=True,
            return_tensors="pt",
        )

        teacher_ids = (
            teacher_inputs[
                "input_ids"
            ][0]
            .tolist()
        )

        target_position, target_token_id = find_answer_token(
            processor.tokenizer,
            prompt_ids,
            teacher_ids,
            label,
        )

        positions[
            label
        ] = int(
            target_position
        )

        token_ids[
            label
        ] = int(
            target_token_id
        )

        prefixes[
            label
        ] = teacher_ids[
            :target_position
        ]

        teacher_inputs_by_label[
            label
        ] = teacher_inputs

    reference_label = valid_labels[0]

    reference_position = positions[
        reference_label
    ]

    reference_prefix = prefixes[
        reference_label
    ]

    for label in valid_labels:
        if positions[label] != reference_position:
            raise RuntimeError(
                f"Teacher target-position mismatch: {positions}"
            )

        if prefixes[label] != reference_prefix:
            raise RuntimeError(
                "Teacher causal-prefix mismatch."
            )

    # --------------------------------------------------------
    # Exact fast path:
    #
    # all labels share the same causal prefix, therefore one
    # forward produces the exact teacher-forced probability
    # for every candidate label.
    # --------------------------------------------------------

    reference_inputs = {
        key:
            value.to("cuda")
            if torch.is_tensor(value)
            else value
        for key, value in (
            teacher_inputs_by_label[
                reference_label
            ].items()
        )
    }

    outputs = model(
        **reference_inputs,
        use_cache=False,
        return_dict=True,
    )

    logits = (
        outputs.logits[
            0,
            reference_position - 1,
        ].float()
    )

    log_probs = torch.log_softmax(
        logits,
        dim=-1,
    )

    label_scores = {}

    for label in valid_labels:
        label_scores[
            label
        ] = float(
            log_probs[
                token_ids[
                    label
                ]
            ].item()
        )

    winner_label = max(
        valid_labels,
        key=lambda label:
            label_scores[
                label
            ],
    )

    winner_position = LABELS.index(
        winner_label
    )

    winner = int(
        group[
            winner_position
        ]
    )

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



def make_schedule(
    candidates,
    seed,
):
    schedule = list(
        candidates
    )

    rng = random.Random(
        seed
    )

    rng.shuffle(
        schedule
    )

    return schedule


def run_round(
    model,
    processor,
    candidates,
    candidate_views,
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

    for schedule_index, base_seed in enumerate(
        SCHEDULE_SEEDS,
        start=1,
    ):
        seed = (
            base_seed
            + round_index * 10000
        )

        schedule = make_schedule(
            candidates,
            seed,
        )

        schedule_survivors = set()

        groups = [
            schedule[
                start:
                start + GROUP_SIZE
            ]
            for start in range(
                0,
                len(schedule),
                GROUP_SIZE,
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

        for group_index, group in enumerate(
            groups,
            start=1,
        ):
            # A one-element final group needs no VLM call.
            if len(
                group
            ) == 1:
                result = {
                    "status":
                        "SINGLE",
                    "answer":
                        "",
                    "winner":
                        int(
                            group[0]
                        ),
                }

                schedule_survivors.add(
                    int(
                        group[0]
                    )
                )

            else:
                result = choose_one(
                    model,
                    processor,
                    group,
                    candidate_views,
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
                    # Parsing failed:
                    # retain all candidates from this group.
                    for index in group:
                        schedule_survivors.add(
                            int(
                                index
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
                group_index == 1
                or group_index % 10 == 0
                or group_index == len(
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


def save_contact_sheet(
    candidates,
    candidate_views,
    output_path,
):
    thumb_width = 320
    thumb_height = 240
    header = 30
    columns = 4

    rows = math.ceil(
        len(candidates)
        / columns
    )

    canvas = Image.new(
        "RGB",
        (
            columns
            * thumb_width,
            rows
            * (
                thumb_height
                + header
            ),
        ),
        "white",
    )

    draw = ImageDraw.Draw(
        canvas
    )

    for position, index in enumerate(
        candidates
    ):
        row = (
            position
            // columns
        )

        column = (
            position
            % columns
        )

        x = (
            column
            * thumb_width
        )

        y = (
            row
            * (
                thumb_height
                + header
            )
        )

        view = candidate_views[
            index
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
                y + header,
            ),
        )

        draw.text(
            (
                x + 8,
                y + 8,
            ),
            (
                f"candidate={index}"
            ),
            fill="black",
        )

    canvas.save(
        output_path
    )


def main():
    global IMAGE_PATH
    global OUTPUT_DIR
    global TARGET_TEXT

    parser = argparse.ArgumentParser()

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

    IMAGE_PATH = Path(
        args.image
    )

    OUTPUT_DIR = Path(
        args.output_dir
    )

    TARGET_TEXT = str(
        args.target
    )

    print("=" * 100)
    print(
        "NO-GT GENERIC HIERARCHICAL SEMANTIC SCREEN"
    )
    print("=" * 100)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "TARGET TEXT:",
        TARGET_TEXT,
    )

    print(
        "IMAGE:",
        IMAGE_PATH,
    )

    image = Image.open(
        IMAGE_PATH
    ).convert(
        "RGB"
    )

    print(
        "IMAGE SIZE:",
        image.size,
    )

    # ============================================================
    # Generic Compact Candidate Bank
    #
    # No GT.
    # No white/black reference boxes.
    # No semantic-pair cache.
    # ============================================================

    grid_h = 15
    grid_w = 20

    pooler = CompactCandidatePooler()

    dummy = torch.zeros(
        1,
        grid_h * grid_w,
        1,
    )

    proposals = pooler(
        dummy,
        grid_h=grid_h,
        grid_w=grid_w,
    )

    candidate_boxes = (
        proposals
        .candidate_boxes
        .float()
        .cpu()
    )

    candidate_count = int(
        candidate_boxes.shape[0]
    )

    print(
        "COMPACT CANDIDATE COUNT:",
        candidate_count,
    )

    if (
        candidate_count
        != 1071
    ):
        raise RuntimeError(
            f"Expected 1071 candidates, got {candidate_count}"
        )

    all_candidates = list(
        range(
            candidate_count
        )
    )

    candidate_views = {}

    for index in all_candidates:
        candidate_views[
            index
        ] = make_candidate_view(
            image,
            candidate_boxes[
                index
            ],
        )

    processor = (
        AutoProcessor
        .from_pretrained(
            MODEL_PATH,
            local_files_only=True,
        )
    )

    model = (
        Qwen3_5ForConditionalGeneration
        .from_pretrained(
            MODEL_PATH,
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

    current = all_candidates
    history = []

    for round_index in range(
        1,
        MAX_ROUNDS + 1,
    ):
        if len(
            current
        ) <= FINAL_MAX:
            break

        current, record = run_round(
            model,
            processor,
            current,
            candidate_views,
            round_index,
        )

        history.append(
            record
        )

    print()
    print("=" * 100)
    print(
        "FINAL NO-GT SCREENING RESULT"
    )
    print("=" * 100)

    print(
        "FINAL CANDIDATE COUNT:",
        len(
            current
        ),
    )

    print(
        "FINAL CANDIDATES:",
        current,
    )

    result_path = (
        OUTPUT_DIR
        / "hierarchical_screen_results.pt"
    )

    torch.save(
        {
            "target_text":
                TARGET_TEXT,
            "image_path":
                str(
                    IMAGE_PATH
                ),
            "candidate_boxes":
                candidate_boxes,
            "initial_candidate_count":
                candidate_count,
            "final_candidates":
                current,
            "history":
                history,
            "group_size":
                GROUP_SIZE,
            "schedule_seeds":
                SCHEDULE_SEEDS,
        },
        result_path,
    )

    contact_path = (
        OUTPUT_DIR
        / "final_candidates_contact_sheet.png"
    )

    save_contact_sheet(
        current,
        candidate_views,
        contact_path,
    )

    for index in current:
        candidate_views[
            index
        ].save(
            OUTPUT_DIR
            / f"candidate_{index}.png"
        )

    print(
        "RESULT:",
        result_path,
    )

    print(
        "CONTACT SHEET:",
        contact_path,
    )

    print(
        "NO_GT_GENERIC_HIERARCHICAL_SCREEN=EXECUTION_PASS"
    )


if __name__ == "__main__":
    main()
