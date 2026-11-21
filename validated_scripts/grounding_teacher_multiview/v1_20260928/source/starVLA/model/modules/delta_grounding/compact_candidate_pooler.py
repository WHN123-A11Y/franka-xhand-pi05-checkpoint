from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class CompactCandidateOutput:
    candidate_embeddings: torch.Tensor
    candidate_boxes: torch.Tensor
    candidate_grid_boxes: torch.Tensor
    candidate_shapes: torch.Tensor


class CompactCandidatePooler(nn.Module):
    """
    Generic compact multi-scale region candidate pooler.

    Unlike the dense SlidingCandidatePooler, this module directly
    builds a compact region bank from generic spatial scales and
    aspect ratios.

    No task-specific concepts are encoded:
        no drawer
        no second / third
        no color
        no left / right
        no object category

    Candidate shapes are generated from:
        height scales
        ×
        normalized aspect ratios

    Region features are obtained by integral-image average pooling.
    """

    def __init__(
        self,
        height_fractions: Sequence[float] = (
            0.10,
            0.20,
            1.0 / 3.0,
            0.50,
            0.75,
            1.00,
        ),
        aspect_ratios: Sequence[float] = (
            0.50,
            1.00,
            1.50,
            2.00,
            3.00,
        ),
        stride_fraction: float = 0.66,
    ) -> None:
        super().__init__()

        if len(height_fractions) == 0:
            raise ValueError(
                "height_fractions must not be empty."
            )

        if len(aspect_ratios) == 0:
            raise ValueError(
                "aspect_ratios must not be empty."
            )

        self.height_fractions = tuple(
            float(x)
            for x in height_fractions
        )

        self.aspect_ratios = tuple(
            float(x)
            for x in aspect_ratios
        )

        self.stride_fraction = float(
            stride_fraction
        )

        for value in self.height_fractions:
            if not 0.0 < value <= 1.0:
                raise ValueError(
                    "height fraction must be in (0,1]."
                )

        for value in self.aspect_ratios:
            if value <= 0.0:
                raise ValueError(
                    "aspect ratio must be positive."
                )

        if not 0.0 < self.stride_fraction <= 1.0:
            raise ValueError(
                "stride_fraction must be in (0,1]."
            )

    def _build_shapes(
        self,
        grid_h: int,
        grid_w: int,
    ) -> tuple[tuple[int, int], ...]:
        """
        Aspect ratio is defined in normalized image coordinates:

            ratio =
                normalized_width
                /
                normalized_height

        Therefore:

            w / grid_w
            ---------------- = ratio
            h / grid_h

        =>
            w = ratio * h * grid_w / grid_h
        """

        shapes = set()

        for height_fraction in self.height_fractions:
            window_h = int(
                round(
                    grid_h
                    * height_fraction
                )
            )

            window_h = max(
                1,
                min(
                    grid_h,
                    window_h,
                ),
            )

            for aspect_ratio in self.aspect_ratios:
                window_w = int(
                    round(
                        aspect_ratio
                        * window_h
                        * grid_w
                        / grid_h
                    )
                )

                window_w = max(
                    1,
                    min(
                        grid_w,
                        window_w,
                    ),
                )

                shapes.add(
                    (
                        window_h,
                        window_w,
                    )
                )

        # Always retain whole-image context.
        shapes.add(
            (
                grid_h,
                grid_w,
            )
        )

        return tuple(
            sorted(
                shapes
            )
        )

    def _axis_starts(
        self,
        total: int,
        window: int,
    ) -> tuple[int, ...]:
        max_start = (
            total
            - window
        )

        if max_start == 0:
            return (
                0,
            )

        stride = max(
            1,
            int(
                round(
                    window
                    * self.stride_fraction
                )
            ),
        )

        starts = list(
            range(
                0,
                max_start + 1,
                stride,
            )
        )

        if starts[-1] != max_start:
            starts.append(
                max_start
            )

        return tuple(
            starts
        )

    def _build_grid_boxes(
        self,
        grid_h: int,
        grid_w: int,
        device: torch.device,
    ):
        shapes = self._build_shapes(
            grid_h,
            grid_w,
        )

        boxes = []
        shape_rows = []

        for window_h, window_w in shapes:
            y_starts = self._axis_starts(
                grid_h,
                window_h,
            )

            x_starts = self._axis_starts(
                grid_w,
                window_w,
            )

            for y1 in y_starts:
                y2 = (
                    y1
                    + window_h
                )

                for x1 in x_starts:
                    x2 = (
                        x1
                        + window_w
                    )

                    boxes.append(
                        (
                            x1,
                            y1,
                            x2,
                            y2,
                        )
                    )

                    shape_rows.append(
                        (
                            window_h,
                            window_w,
                        )
                    )

        grid_boxes = torch.tensor(
            boxes,
            dtype=torch.long,
            device=device,
        )

        candidate_shapes = torch.tensor(
            shape_rows,
            dtype=torch.long,
            device=device,
        )

        return (
            grid_boxes,
            candidate_shapes,
            shapes,
        )

    @staticmethod
    def _normalize_boxes(
        grid_boxes: torch.Tensor,
        grid_h: int,
        grid_w: int,
    ) -> torch.Tensor:
        boxes = (
            grid_boxes
            .float()
        )

        return torch.stack(
            (
                boxes[:, 0]
                / float(grid_w),

                boxes[:, 1]
                / float(grid_h),

                boxes[:, 2]
                / float(grid_w),

                boxes[:, 3]
                / float(grid_h),
            ),
            dim=-1,
        )

    @staticmethod
    def _region_pool(
        visual_map: torch.Tensor,
        grid_boxes: torch.Tensor,
    ) -> torch.Tensor:
        original_dtype = (
            visual_map.dtype
        )

        work = (
            visual_map
            .float()
        )

        integral = (
            work
            .cumsum(
                dim=1
            )
            .cumsum(
                dim=2
            )
        )

        integral = F.pad(
            integral,
            (
                0,
                0,
                1,
                0,
                1,
                0,
            ),
        )

        x1 = grid_boxes[:, 0]
        y1 = grid_boxes[:, 1]
        x2 = grid_boxes[:, 2]
        y2 = grid_boxes[:, 3]

        region_sums = (
            integral[:, y2, x2, :]
            - integral[:, y1, x2, :]
            - integral[:, y2, x1, :]
            + integral[:, y1, x1, :]
        )

        area = (
            (
                x2 - x1
            )
            *
            (
                y2 - y1
            )
        ).to(
            dtype=region_sums.dtype
        ).view(
            1,
            -1,
            1,
        )

        return (
            region_sums
            / area
        ).to(
            dtype=original_dtype
        )

    def forward(
        self,
        visual_tokens: torch.Tensor,
        *,
        grid_h: int,
        grid_w: int,
    ) -> CompactCandidateOutput:
        if visual_tokens.ndim != 3:
            raise ValueError(
                "visual_tokens must be [B,N,H]."
            )

        (
            batch_size,
            token_count,
            hidden_dim,
        ) = visual_tokens.shape

        grid_h = int(
            grid_h
        )

        grid_w = int(
            grid_w
        )

        expected = (
            grid_h
            * grid_w
        )

        if token_count != expected:
            raise ValueError(
                "visual token count mismatch: "
                f"{token_count} != "
                f"{grid_h}x{grid_w}"
            )

        visual_map = (
            visual_tokens
            .reshape(
                batch_size,
                grid_h,
                grid_w,
                hidden_dim,
            )
        )

        (
            grid_boxes,
            candidate_shapes,
            _,
        ) = self._build_grid_boxes(
            grid_h,
            grid_w,
            visual_tokens.device,
        )

        candidate_embeddings = (
            self._region_pool(
                visual_map,
                grid_boxes,
            )
        )

        candidate_boxes = (
            self._normalize_boxes(
                grid_boxes,
                grid_h,
                grid_w,
            )
        )

        return CompactCandidateOutput(
            candidate_embeddings=
            candidate_embeddings,
            candidate_boxes=
            candidate_boxes,
            candidate_grid_boxes=
            grid_boxes,
            candidate_shapes=
            candidate_shapes,
        )
