"""Reusable segmentation output schemas."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

MaskArray: TypeAlias = NDArray[np.bool_]
Point: TypeAlias = tuple[int, int]
Contour: TypeAlias = list[Point]


@dataclass(frozen=True)
class BoundingBox:
    """Axis-aligned bounding box in pixel coordinates."""

    x_min: int
    y_min: int
    x_max: int
    y_max: int

    @property
    def width(self) -> int:
        """Return bounding box width in pixels."""

        return max(0, self.x_max - self.x_min)

    @property
    def height(self) -> int:
        """Return bounding box height in pixels."""

        return max(0, self.y_max - self.y_min)

    def as_xyxy(self) -> tuple[int, int, int, int]:
        """Return the bounding box as `(x_min, y_min, x_max, y_max)`."""

        return self.x_min, self.y_min, self.x_max, self.y_max


@dataclass(frozen=True)
class SegmentedObject:
    """Processed mask and geometry for one segmented object."""

    id: int
    mask: MaskArray
    bbox: BoundingBox
    contour: Contour
    area: int

    @property
    def has_geometry(self) -> bool:
        """Return whether the object has non-empty mask geometry."""

        return self.area > 0 and bool(self.contour)
