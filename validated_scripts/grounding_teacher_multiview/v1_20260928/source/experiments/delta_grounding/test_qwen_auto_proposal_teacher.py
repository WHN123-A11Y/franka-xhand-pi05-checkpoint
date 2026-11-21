from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

from starVLA.model.modules.delta_grounding.sliding_candidate_pooler import (
    SlidingCandidatePooler,
)


MODEL_PATH = Path(
    "/home/user/FineVLA-Data/Pretrained_models/Qwen3.5-4B"
)

IMAGE_PATH = Path(
    "/tmp/drawer_clean_frame.png"
)

OUTPUT_DIR = Path(
    "experiments/delta_grounding/"
    "qwen_auto_proposal_teacher"
)

GRID_H = 15
GRID_W = 20


# ================================================================
# Diagnostic-only proposal indices.
#
# These are NOT part of the method.
#
# They came from the full-image coverage audit:
#
# TOP:
#     candidate 612
#
# MIDDLE:
#     candidate 646
#
# BOTTOM:
#     candidate 663
#
# Their only purpose is to answer:
#
# "If automatic proposals contain the correct regions,
#  can Frozen Qwen select the correct one?"
# ================================================================

TARGET_PROPOSALS = {
    "TOP": 612,
    "MIDDLE": 646,
    "BOTTOM": 663,
}


PERMUTATIONS = {
    "PERM1": {
        "TOP": "C",
        "MIDDLE": "A",
        "BOTTOM": "B",
    },
    "PERM2": {
        "TOP": "B",
        "MIDDLE": "C",
        "BOTTOM": "A",
    },
}


TESTS = (
    ("SECOND", "second"),
    ("THIRD", "third"),
)


def get_font():
    paths = (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    )

    for path in paths:
        if Path(path).exists():
            return ImageFont.truetype(
                path,
                24,
            )

    return ImageFont.load_default()


def generate_proposals():
    pooler = SlidingCandidatePooler(
        scale_fractions=(
            0.10,
            0.15,
            0.20,
            0.25,
            1.0 / 3.0,
            0.50,
            2.0 / 3.0,
            0.75,
            0.90,
            1.00,
        ),
        stride_fraction=0.33,
    )

    dummy = torch.zeros(
        1,
        GRID_H * GRID_W,
        1,
    )

    output = pooler(
        dummy,
        grid_h=GRID_H,
        grid_w=GRID_W,
    )

    return (
        output.candidate_boxes,
        output.candidate_grid_boxes,
    )


def normalized_to_pixel_box(
    box,
    width,
    height,
):
    x1, y1, x2, y2 = [
        float(x)
        for x in box.tolist()
    ]

    px1 = int(
        round(x1 * width)
    )

    py1 = int(
        round(y1 * height)
    )

    px2 = int(
        round(x2 * width)
    )

    py2 = int(
        round(y2 * height)
    )

    return (
        px1,
        py1,
        px2,
        py2,
    )


def draw_candidates(
    image,
    candidate_boxes,
    assignment,
):
    out = image.copy()

    draw = ImageDraw.Draw(
        out
    )

    font = get_font()

    width, height = (
        out.size
    )

    pixel_boxes = {}

    for position, proposal_index in (
        TARGET_PROPOSALS.items()
    ):
        label = assignment[
            position
        ]

        normalized_box = (
            candidate_boxes[
                proposal_index
            ]
        )

        pixel_box = normalized_to_pixel_box(
            normalized_box,
            width,
            height,
        )

        pixel_boxes[
            position
        ] = pixel_box

        draw.rectangle(
            pixel_box,
            outline="red",
            width=4,
        )

        x1, y1, _, _ = (
            pixel_box
        )

        text_bbox = draw.textbbox(
            (0, 0),
            label,
            font=font,
        )

        text_w = (
            text_bbox[2]
            - text_bbox[0]
        )

        text_h = (
            text_bbox[3]
            - text_bbox[1]
        )

        tx = x1 + 4
        ty = y1 + 3

        draw.rectangle(
            (
                tx - 3,
                ty - 3,
                tx + text_w + 4,
                ty + text_h + 4,
            ),
            fill="white",
        )

        draw.text(
            (tx, ty),
            label,
            fill="red",
            font=font,
        )

    return (
        out,
        pixel_boxes,
    )


def question_for(
    ordinal,
):
    return (
        "Look carefully at the complete image. "
        "Three red rectangular candidate regions are labeled A, B, and C. "
        "The labels are arbitrary and their alphabetical order does NOT "
        "indicate top-to-bottom order. "
        "Each candidate was generated automatically by a visual proposal "
        "algorithm. "
        f"Which candidate corresponds to the {ordinal} drawer from the top? "
        "Judge the physical drawer represented by each red region. "
        "Choose only A, B, or C."
    )


def make_content(
    image,
    question,
):
    return [
        {
            "type": "image",
            "image": image,
        },
        {
            "type": "text",
            "text": question,
        },
    ]


def find_answer_token(
    tokenizer,
    prompt_ids,
    teacher_ids,
    letter,
):
    common = 0

    max_common = min(
        len(prompt_ids),
        len(teacher_ids),
    )

    while (
        common < max_common
        and prompt_ids[common]
        == teacher_ids[common]
    ):
        common += 1

    matches = []

    for position in range(
        max(
            1,
            common - 3,
        ),
        len(teacher_ids),
    ):
        token_id = int(
            teacher_ids[
                position
            ]
        )

        token_text = (
            tokenizer
            .decode(
                [token_id],
                skip_special_tokens=False,
            )
            .strip()
            .upper()
        )

        if token_text == letter:
            matches.append(
                (
                    position,
                    token_id,
                )
            )

    if not matches:
        raise RuntimeError(
            f"Cannot locate teacher token {letter}."
        )

    return matches[-1]


@torch.inference_mode()
def score_letter(
    model,
    processor,
    image,
    question,
    letter,
):
    user_messages = [[{
        "role": "user",
        "content": make_content(
            image,
            question,
        ),
    }]]

    teacher_messages = [[
        {
            "role": "user",
            "content": make_content(
                image,
                question,
            ),
        },
        {
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": letter,
                }
            ],
        },
    ]]

    prompt_inputs = (
        processor
        .apply_chat_template(
            user_messages,
            tokenize=True,
            padding=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
    )

    teacher_inputs = (
        processor
        .apply_chat_template(
            teacher_messages,
            tokenize=True,
            padding=True,
            add_generation_prompt=False,
            return_dict=True,
            return_tensors="pt",
        )
        .to("cuda")
    )

    prompt_ids = (
        prompt_inputs[
            "input_ids"
        ][0]
        .tolist()
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
    ) = find_answer_token(
        processor.tokenizer,
        prompt_ids,
        teacher_ids,
        letter,
    )

    outputs = model(
        **teacher_inputs,
        use_cache=False,
        return_dict=True,
    )

    logits = (
        outputs.logits[
            0,
            target_position - 1,
        ]
        .float()
    )

    log_probs = torch.log_softmax(
        logits,
        dim=-1,
    )

    score = float(
        log_probs[
            target_token_id
        ].item()
    )

    return score


def expected_letter(
    assignment,
    ordinal,
):
    if ordinal == "second":
        return assignment[
            "MIDDLE"
        ]

    if ordinal == "third":
        return assignment[
            "BOTTOM"
        ]

    raise ValueError(
        ordinal
    )


def main():
    print("=" * 100)
    print("FROZEN QWEN AUTOMATIC PROPOSAL TEACHER TEST")
    print("=" * 100)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
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

    (
        candidate_boxes,
        candidate_grid_boxes,
    ) = generate_proposals()

    print(
        "TOTAL AUTOMATIC PROPOSALS:",
        candidate_boxes.shape[0],
    )

    for position, index in (
        TARGET_PROPOSALS.items()
    ):
        print(
            position,
            "PROPOSAL INDEX:",
            index,
            "GRID BOX:",
            candidate_grid_boxes[
                index
            ].tolist(),
            "NORMALIZED BOX:",
            [
                round(
                    float(x),
                    4,
                )
                for x in candidate_boxes[
                    index
                ].tolist()
            ],
        )

    processor = AutoProcessor.from_pretrained(
        MODEL_PATH,
        local_files_only=True,
    )

    model = (
        Qwen3_5ForConditionalGeneration
        .from_pretrained(
            MODEL_PATH,
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
            local_files_only=True,
        )
        .to("cuda")
        .eval()
    )

    print(
        "MODEL_LOAD=PASS"
    )

    total = 0
    correct = 0

    for perm_name, assignment in (
        PERMUTATIONS.items()
    ):
        print()
        print("=" * 100)
        print(
            "PERMUTATION:",
            perm_name,
        )

        print(
            "ASSIGNMENT:",
            assignment,
        )
        print("=" * 100)

        (
            annotated,
            pixel_boxes,
        ) = draw_candidates(
            image,
            candidate_boxes,
            assignment,
        )

        output_path = (
            OUTPUT_DIR
            / f"{perm_name.lower()}_auto_candidates.png"
        )

        annotated.save(
            output_path
        )

        print(
            "IMAGE:",
            output_path,
        )

        print(
            "PIXEL BOXES:",
            pixel_boxes,
        )

        for test_name, ordinal in (
            TESTS
        ):
            print()
            print("-" * 100)

            expected = expected_letter(
                assignment,
                ordinal,
            )

            question = question_for(
                ordinal
            )

            print(
                "TEST:",
                test_name,
            )

            print(
                "ORDINAL:",
                ordinal,
            )

            print(
                "EXPECTED:",
                expected,
            )

            scores = {}

            for letter in (
                "A",
                "B",
                "C",
            ):
                score = score_letter(
                    model,
                    processor,
                    annotated,
                    question,
                    letter,
                )

                scores[
                    letter
                ] = score

                print(
                    f"SCORE {letter}:",
                    score,
                )

            ranking = sorted(
                scores.items(),
                key=lambda item: item[1],
                reverse=True,
            )

            predicted = (
                ranking[0][0]
            )

            margin = (
                ranking[0][1]
                - ranking[1][1]
            )

            is_correct = (
                predicted
                == expected
            )

            total += 1

            correct += int(
                is_correct
            )

            print(
                "PREDICTED:",
                predicted,
            )

            print(
                "EXPECTED:",
                expected,
            )

            print(
                "TOP1-TOP2 LOGPROB MARGIN:",
                margin,
            )

            print(
                "CORRECT:",
                is_correct,
            )

    accuracy = (
        correct / total
    )

    print()
    print("=" * 100)
    print("FINAL")
    print("=" * 100)

    print(
        "CORRECT:",
        correct,
    )

    print(
        "TOTAL:",
        total,
    )

    print(
        "ACCURACY:",
        accuracy,
    )

    if correct == total:
        print(
            "AUTO_PROPOSAL_TEACHER=PASS"
        )
    else:
        print(
            "AUTO_PROPOSAL_TEACHER=FAIL"
        )


if __name__ == "__main__":
    main()
