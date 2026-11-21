from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class SlidingCandidateOutput:
    candidate_embeddings: torch.Tensor
    candidate_boxes: torch.Tensor
    candidate_grid_boxes: torch.Tensor
    candidate_shapes: torch.Tensor


class SlidingCandidatePooler(nn.Module):
    """
    Generic multi-scale sliding-region candidate proposal.

    Input
    -----
    visual_tokens:
        [B, N, H]

    grid_h, grid_w:
        N = grid_h * grid_w

    Output
    ------
    candidate_embeddings:
        [B, K, H]

    candidate_boxes:
        [K, 4]
        normalized [x1, y1, x2, y2]

    candidate_grid_boxes:
        [K, 4]
        integer grid coordinates

    candidate_shapes:
        [K, 2]
        [window_h, window_w]

    Notes
    -----
    This module is completely task-agnostic.

    It does NOT contain:
        drawer
        second / third
        color
        left / right
        object category
        fixed number of instances

    Height and width use the SAME generic scale family.

    Region embeddings are produced with integral-image pooling,
    avoiding a Python loop over candidate feature tensors.
    """

    def __init__(
        self,
        scale_fractions: Sequence[float] = (
            0.20,
            0.25,
            1.0 / 3.0,
            0.50,
            2.0 / 3.0,
            0.75,
            0.90,
            1.00,
        ),
        stride_fraction: float = 0.50,
    ) -> None:
        super().__init__()

        if len(scale_fractions) == 0:
            raise ValueError(
                "scale_fractions must not be empty."
            )

        normalized = []

        for value in scale_fractions:
            value = float(value)

            if not (
                0.0 < value <= 1.0
            ):
                raise ValueError(
                    "Each scale fraction must satisfy "
                    "0 < scale <= 1."
                )

            normalized.append(
                value
            )

        if not (
            0.0 < stride_fraction <= 1.0
        ):
            raise ValueError(
                "stride_fraction must satisfy "
                "0 < stride_fraction <= 1."
            )

        self.scale_fractions = tuple(
            normalized
        )

        self.stride_fraction = float(
            stride_fraction
        )

    @staticmethod
    def _sizes_from_fractions(
        total: int,
        fractions: Sequence[float],
    ) -> tuple[int, ...]:
        sizes = {
            max(
                1,
                min(
                    total,
                    int(
                        round(
                            total * fraction
                        )
                    ),
                ),
            )
            for fraction in fractions
        }

        return tuple(
            sorted(
                sizes
            )
        )

    def _axis_starts(
        self,
        total: int,
        window: int,
    ) -> tuple[int, ...]:
        max_start = (
            total - window
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

        # Always include the far boundary.
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
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
    ]:
        heights = self._sizes_from_fractions(
            grid_h,
            self.scale_fractions,
        )

        widths = self._sizes_from_fractions(
            grid_w,
            self.scale_fractions,
        )

        boxes = []
        shapes = []

        for window_h in heights:
            y_starts = self._axis_starts(
                grid_h,
                window_h,
            )

            for window_w in widths:
                x_starts = self._axis_starts(
                    grid_w,
                    window_w,
                )

                for y1 in y_starts:
                    y2 = (
                        y1 + window_h
                    )

                    for x1 in x_starts:
                        x2 = (
                            x1 + window_w
                        )

                        boxes.append(
                            (
                                x1,
                                y1,
                                x2,
                                y2,
                            )
                        )

                        shapes.append(
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
            shapes,
            dtype=torch.long,
            device=device,
        )

        return (
            grid_boxes,
            candidate_shapes,
        )

    @staticmethod
    def _normalized_boxes(
        grid_boxes: torch.Tensor,
        grid_h: int,
        grid_w: int,
    ) -> torch.Tensor:
        grid_boxes_float = (
            grid_boxes.float()
        )

        x1 = (
            grid_boxes_float[:, 0]
            / float(grid_w)
        )

        y1 = (
            grid_boxes_float[:, 1]
            / float(grid_h)
        )

        x2 = (
            grid_boxes_float[:, 2]
            / float(grid_w)
        )

        y2 = (
            grid_boxes_float[:, 3]
            / float(grid_h)
        )

        return torch.stack(
            (
                x1,
                y1,
                x2,
                y2,
            ),
            dim=-1,
        )

    @staticmethod
    def _region_pool(
        visual_map: torch.Tensor,
        grid_boxes: torch.Tensor,
    ) -> torch.Tensor:
        """
        Integral-image average pooling.

        visual_map:
            [B, Gh, Gw, H]

        grid_boxes:
            [K,4]

        returns:
            [B,K,H]
        """

        original_dtype = (
            visual_map.dtype
        )

        # Do cumulative sums in fp32 for stability.
        work = (
            visual_map.float()
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

        # Add zero top/left border:
        #
        # [B,Gh,Gw,H]
        #      ↓
        # [B,Gh+1,Gw+1,H]
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
            integral[
                :,
                y2,
                x2,
                :,
            ]
            -
            integral[
                :,
                y1,
                x2,
                :,
            ]
            -
            integral[
                :,
                y2,
                x1,
                :,
            ]
            +
            integral[
                :,
                y1,
                x1,
                :,
            ]
        )

        areas = (
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

        pooled = (
            region_sums
            / areas
        )

        return pooled.to(
            dtype=original_dtype
        )

    def forward(
        self,
        visual_tokens: torch.Tensor,
        *,
        grid_h: int,
        grid_w: int,
    ) -> SlidingCandidateOutput:
        if visual_tokens.ndim != 3:
            raise ValueError(
                "visual_tokens must be [B,N,H], "
                f"got {tuple(visual_tokens.shape)}."
            )

        batch_size, token_count, hidden_dim = (
            visual_tokens.shape
        )

        grid_h = int(
            grid_h
        )

        grid_w = int(
            grid_w
        )

        if (
            grid_h <= 0
            or grid_w <= 0
        ):
            raise ValueError(
                f"Invalid grid "
                f"{grid_h}x{grid_w}."
            )

        expected_tokens = (
            grid_h
            * grid_w
        )

        if (
            token_count
            != expected_tokens
        ):
            raise ValueError(
                "Visual token count mismatch: "
                f"N={token_count}, "
                f"grid={grid_h}x{grid_w}="
                f"{expected_tokens}."
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
            self._normalized_boxes(
                grid_boxes,
                grid_h,
                grid_w,
            )
        )

        return SlidingCandidateOutput(
            candidate_embeddings=
            candidate_embeddings,
            candidate_boxes=
            candidate_boxes,
            candidate_grid_boxes=
            grid_boxes,
            candidate_shapes=
            candidate_shapes,
        )
