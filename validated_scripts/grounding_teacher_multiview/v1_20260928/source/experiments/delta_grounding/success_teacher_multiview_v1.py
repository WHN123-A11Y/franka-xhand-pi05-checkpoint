from pathlib import Path
import argparse
import csv
import importlib.util
import sys
import time

import torch


PROJECT = Path(
    "/root/autodl-tmp/FineVLA_bundle_20260827_v1/"
    "FineVLA_ablation_improved_eager/FineVLA-Policy"
)

LOCK_DIR = (
    PROJECT
    / "experiments/delta_grounding/"
      "LOCKED_MULTIVIEW_TEACHER_BT_V1_20260903"
)

PER_VIEW_CORE = (
    LOCK_DIR
    / "teacher_per_view.py"
)

CROSSVIEW_MODULE = (
    PROJECT
    / "experiments/delta_grounding/"
      "teacher_crossview_bt.py"
)

VIEW_ORDER = (
    "cam_high",
    "cam_left_wrist",
    "cam_right_wrist",
)


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


def load_manifest(
    path,
):
    with path.open(
        "r",
        newline="",
        encoding="utf-8",
    ) as f:

        return list(
            csv.DictReader(
                f,
                delimiter="\t",
            )
        )


def run_per_view_teacher(
    core,
    row,
    sample_output_dir,
):

    image_path = str(
        row[
            "image_path"
        ]
    ).strip()

    original_target_text = str(
        row[
            "target_text"
        ]
    ).strip()

    grounding_target_text = str(
        row.get(
            "grounding_target_text",
            ""
        )
    ).strip()

    # Anchor-Fine Grounding:
    # only the per-view visual Teacher receives Fine Target.
    # The original instruction remains untouched in the manifest
    # and in the multi-view task metadata.
    target_text = (
        grounding_target_text
        if grounding_target_text
        else original_target_text
    )

    argv = [
        str(
            PER_VIEW_CORE
        ),
        "--image",
        image_path,
        "--target",
        target_text,
        "--output-dir",
        str(
            sample_output_dir
        ),
    ]

    reference_stage2 = str(
        row.get(
            "reference_stage2",
            "",
        )
        or ""
    ).strip()

    if reference_stage2:
        argv.extend(
            [
                "--reference-stage2",
                reference_stage2,
            ]
        )

    old_argv = sys.argv

    try:

        sys.argv = argv

        core.main()

    finally:

        sys.argv = (
            old_argv
        )


def make_multiview_pool(
    group_id,
    group_rows,
    group_dir,
    topk_per_view,
):

    by_view = {
        str(
            row[
                "view_id"
            ]
        ).strip():
            row

        for row in group_rows
    }

    missing = [
        view_id
        for view_id in VIEW_ORDER
        if view_id
        not in by_view
    ]

    if missing:
        raise RuntimeError(
            f"{group_id}: missing views {missing}"
        )


    original_targets = {
        str(
            row[
                "target_text"
            ]
        ).strip()

        for row in group_rows
    }

    if len(original_targets) != 1:
        raise RuntimeError(
            f"{group_id}: target_text mismatch."
        )

    original_target_text = next(
        iter(
            original_targets
        )
    )


    grounding_targets = {
        str(
            row.get(
                "grounding_target_text",
                ""
            )
        ).strip()

        for row in group_rows
    }

    grounding_targets.discard("")

    if grounding_targets:

        if len(grounding_targets) != 1:
            raise RuntimeError(
                f"{group_id}: grounding_target_text mismatch."
            )

        target_text = next(
            iter(
                grounding_targets
            )
        )

    else:

        target_text = original_target_text


    episode_values = {
        str(
            row[
                "episode_index"
            ]
        ).strip()

        for row in group_rows
    }

    frame_values = {
        str(
            row[
                "frame_index"
            ]
        ).strip()

        for row in group_rows
    }

    if len(episode_values) != 1:
        raise RuntimeError(
            f"{group_id}: episode mismatch."
        )

    if len(frame_values) != 1:
        raise RuntimeError(
            f"{group_id}: frame mismatch."
        )


    views_output = {}

    candidate_pool = []


    for view_id in VIEW_ORDER:

        row = by_view[
            view_id
        ]

        sample_id = str(
            row[
                "sample_id"
            ]
        ).strip()

        result_path = (
            group_dir
            / f"sample_{sample_id}"
            / "fast_teacher_results.pt"
        )

        if not result_path.exists():
            raise RuntimeError(
                f"Missing per-view result: "
                f"{result_path}"
            )

        result = torch.load(
            result_path,
            map_location="cpu",
            weights_only=False,
        )

        ranking = result.get(
            "ranking",
            [],
        )

        if len(ranking) < topk_per_view:
            raise RuntimeError(
                f"{view_id}: ranking count "
                f"{len(ranking)} < {topk_per_view}"
            )

        evidences = []

        for rank_position, item in enumerate(
            ranking[
                :topk_per_view
            ],
            start=1,
        ):

            candidate_index = int(
                item[
                    "index"
                ]
            )

            box = item[
                "box"
            ]

            if torch.is_tensor(
                box
            ):
                box = (
                    box.detach()
                    .cpu()
                    .tolist()
                )

            box = [
                float(v)
                for v in box
            ]

            evidence = {
                "group_id":
                    group_id,

                "view_id":
                    view_id,

                "rank":
                    rank_position,

                "candidate_index":
                    candidate_index,

                "box":
                    box,

                "bt_score":
                    float(
                        item[
                            "bt_score"
                        ]
                    ),

                "pair_mean":
                    float(
                        item[
                            "pair_mean"
                        ]
                    ),

                "hard_win_rate":
                    float(
                        item[
                            "hard_win_rate"
                        ]
                    ),

                "visibility":
                    None,

                "reliability":
                    None,

                "raw_ranking_item":
                    item,
            }

            evidences.append(
                evidence
            )

            candidate_pool.append(
                evidence.copy()
            )


        views_output[
            view_id
        ] = {
            "sample_id":
                sample_id,

            "image_path":
                str(
                    row[
                        "image_path"
                    ]
                ),

            "result_path":
                str(
                    result_path
                ),

            "visibility":
                None,

            "reliability":
                None,

            "topk":
                evidences,
        }


    return {
        "version":
            "multiview_teacher_bt_v1",

        "group_id":
            group_id,

        "episode_index":
            next(
                iter(
                    episode_values
                )
            ),

        "frame_index":
            next(
                iter(
                    frame_values
                )
            ),

        "original_target_text":
            original_target_text,

        "target_text":
            target_text,

        "view_order":
            list(
                VIEW_ORDER
            ),

        "topk_per_view":
            int(
                topk_per_view
            ),

        "views":
            views_output,

        "candidate_pool":
            candidate_pool,

        "candidate_pool_count":
            len(
                candidate_pool
            ),

        "cross_view_reliability":
            None,

        "final_multiview_ranking":
            None,
    }


def main():

    parser = argparse.ArgumentParser(
        description=(
            "Production Multi-view Teacher: "
            "per-view Teacher + unified Cross-view BT."
        )
    )

    parser.add_argument(
        "--manifest",
        required=True,
    )

    parser.add_argument(
        "--output-root",
        required=True,
    )

    parser.add_argument(
        "--topk-per-view",
        type=int,
        default=6,
    )

    parser.add_argument(
        "--crossview-topk",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--resume",
        action="store_true",
    )

    parser.add_argument(
        "--limit-groups",
        type=int,
        default=0,
        help=(
            "0 = all groups."
        ),
    )

    args = (
        parser.parse_args()
    )


    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA required."
        )


    manifest = Path(
        args.manifest
    )

    output_root = Path(
        args.output_root
    )


    if not manifest.exists():
        raise RuntimeError(
            f"Manifest missing: {manifest}"
        )

    if not PER_VIEW_CORE.exists():
        raise RuntimeError(
            f"Locked per-view Teacher missing: "
            f"{PER_VIEW_CORE}"
        )

    if not CROSSVIEW_MODULE.exists():
        raise RuntimeError(
            f"Cross-view module missing: "
            f"{CROSSVIEW_MODULE}"
        )


    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )


    rows = load_manifest(
        manifest
    )

    if not rows:
        raise RuntimeError(
            "Manifest is empty."
        )


    required_fields = {
        "sample_id",
        "group_id",
        "view_id",
        "episode_index",
        "frame_index",
        "image_path",
        "target_text",
    }

    missing_fields = (
        required_fields
        -
        set(
            rows[
                0
            ].keys()
        )
    )

    if missing_fields:
        raise RuntimeError(
            "Manifest missing fields: "
            f"{sorted(missing_fields)}"
        )


    # ========================================================
    # Group rows by same timestamp / same group_id.
    # ========================================================

    groups = {}

    for row in rows:

        group_id = str(
            row[
                "group_id"
            ]
        ).strip()

        groups.setdefault(
            group_id,
            [],
        ).append(
            row
        )


    group_items = list(
        groups.items()
    )

    if (
        args.limit_groups
        >
        0
    ):
        group_items = (
            group_items[
                :args.limit_groups
            ]
        )


    print("=" * 120)
    print("PRODUCTION MULTI-VIEW TEACHER")
    print("=" * 120)

    print(
        "MANIFEST =",
        manifest,
    )

    print(
        "OUTPUT ROOT =",
        output_root,
    )

    print(
        "LOCKED PER-VIEW CORE =",
        PER_VIEW_CORE,
    )

    print(
        "GROUP COUNT =",
        len(
            group_items
        ),
    )

    print(
        "TOPK / VIEW =",
        args.topk_per_view,
    )

    print(
        "CROSSVIEW TOPK / VIEW =",
        args.crossview_topk,
    )

    print(
        "RESUME =",
        args.resume,
    )


    # ========================================================
    # IMPORTANT:
    # Load per-view core EXACTLY ONCE.
    #
    # Its:
    #   _SHARED_PROCESSOR
    #   _SHARED_MODEL
    #
    # remain alive for all cameras / groups.
    # ========================================================

    per_view_core = load_module(
        "locked_production_teacher_per_view",
        PER_VIEW_CORE,
    )

    crossview = load_module(
        "production_teacher_crossview_bt",
        CROSSVIEW_MODULE,
    )


    pass_count = 0
    skip_count = 0
    fail_count = 0

    total_start = (
        time.perf_counter()
    )


    for group_number, (
        group_id,
        group_rows,
    ) in enumerate(
        group_items,
        start=1,
    ):

        print()
        print("=" * 120)
        print(
            f"GROUP {group_number}/{len(group_items)} "
            f"= {group_id}"
        )
        print("=" * 120)


        by_view = {
            str(
                row[
                    "view_id"
                ]
            ).strip():
                row

            for row in group_rows
        }


        missing_views = [
            view_id
            for view_id in VIEW_ORDER
            if view_id
            not in by_view
        ]

        if missing_views:

            print(
                "GROUP FAIL: missing views",
                missing_views,
            )

            fail_count += 1
            continue


        group_dir = (
            output_root
            / group_id
        )

        group_dir.mkdir(
            parents=True,
            exist_ok=True,
        )


        final_result_path = (
            group_dir
            / "multiview_results_bt_v1.pt"
        )


        if (
            args.resume
            and
            final_result_path.exists()
        ):

            print(
                "GROUP SKIP_DONE =",
                final_result_path,
            )

            skip_count += 1
            continue


        group_start = (
            time.perf_counter()
        )


        try:

            # ================================================
            # PHASE 1:
            # Three independent per-view Teachers.
            # ================================================

            for view_id in VIEW_ORDER:

                row = by_view[
                    view_id
                ]

                sample_id = str(
                    row[
                        "sample_id"
                    ]
                ).strip()

                sample_output_dir = (
                    group_dir
                    / f"sample_{sample_id}"
                )

                sample_result_path = (
                    sample_output_dir
                    / "fast_teacher_results.pt"
                )


                if (
                    args.resume
                    and
                    sample_result_path.exists()
                ):

                    print(
                        view_id,
                        "PER_VIEW=SKIP_DONE",
                    )

                    continue


                print()
                print(
                    "PER-VIEW TEACHER:",
                    view_id,
                )


                run_per_view_teacher(
                    per_view_core,
                    row,
                    sample_output_dir,
                )


                if not sample_result_path.exists():
                    raise RuntimeError(
                        "Per-view Teacher did not "
                        f"produce result: "
                        f"{sample_result_path}"
                    )


            # ================================================
            # Shared model MUST exist now.
            # ================================================

            model = (
                per_view_core
                ._SHARED_MODEL
            )

            processor = (
                per_view_core
                ._SHARED_PROCESSOR
            )


            if model is None:
                raise RuntimeError(
                    "_SHARED_MODEL is None."
                )

            if processor is None:
                raise RuntimeError(
                    "_SHARED_PROCESSOR is None."
                )


            print()
            print(
                "SHARED_QWEN_REUSE_FOR_CROSSVIEW=PASS"
            )


            # ================================================
            # PHASE 2:
            # Build 18-evidence Multi-view pool.
            # ================================================

            multiview_pool = (
                make_multiview_pool(
                    group_id=
                        group_id,

                    group_rows=
                        group_rows,

                    group_dir=
                        group_dir,

                    topk_per_view=
                        args.topk_per_view,
                )
            )


            pool_path = (
                group_dir
                / "multiview_pool_before_crossview.pt"
            )

            torch.save(
                multiview_pool,
                pool_path,
            )


            print(
                "MULTIVIEW POOL COUNT =",
                multiview_pool[
                    "candidate_pool_count"
                ],
            )


            # ================================================
            # PHASE 3:
            # Unified Cross-view Bradley-Terry.
            #
            # Uses SAME already-loaded Qwen.
            # ================================================

            (
                final_result,
                saved_path,
            ) = (
                crossview.run_crossview_bt(
                    multiview_result=
                        multiview_pool,

                    model=
                        model,

                    processor=
                        processor,

                    output_dir=
                        group_dir,

                    topk=
                        args.crossview_topk,

                    temperature=
                        args.temperature,
                )
            )


            if not saved_path.exists():
                raise RuntimeError(
                    "Cross-view result missing."
                )


            group_sec = (
                time.perf_counter()
                -
                group_start
            )


            print()
            print(
                "GROUP BEST VIEW =",
                final_result[
                    "best_view"
                ],
            )

            print(
                "GROUP WALL SEC =",
                f"{group_sec:.3f}",
            )

            print(
                "GROUP=PASS"
            )


            pass_count += 1


        except Exception as exc:

            fail_count += 1

            print()
            print(
                "GROUP=FAIL"
            )

            print(
                "ERROR =",
                repr(
                    exc
                ),
            )


    total_sec = (
        time.perf_counter()
        -
        total_start
    )


    # ========================================================
    # Final production summary.
    # ========================================================

    print()
    print("=" * 120)
    print("MULTI-VIEW TEACHER FINAL SUMMARY")
    print("=" * 120)

    print(
        "GROUP TOTAL =",
        len(
            group_items
        ),
    )

    print(
        "PASS =",
        pass_count,
    )

    print(
        "SKIP_DONE =",
        skip_count,
    )

    print(
        "FAIL =",
        fail_count,
    )

    print(
        "TOTAL WALL SEC =",
        f"{total_sec:.3f}",
    )


    if fail_count == 0:

        print(
            "RUN_MULTIVIEW_TEACHER=PASS"
        )

    else:

        print(
            "RUN_MULTIVIEW_TEACHER=PARTIAL"
        )


if __name__ == "__main__":
    main()
