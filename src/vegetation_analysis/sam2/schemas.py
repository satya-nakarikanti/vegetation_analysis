"""Output schemas for SAM 2 segmentation results."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from numpy.typing import NDArray

from vegetation_analysis.grounding.schemas import DetectionBox


@dataclass(frozen=True)
class MaskObject:
    """A single object segmented by SAM 2.

    Attributes:
        label: Text label assigned by the upstream detector.
        confidence: Detector confidence score, if available.
        box: The bounding box that was used to prompt SAM 2.
        mask: Boolean mask array of shape (H, W).
        mask_area_pixels: Total number of True pixels in the mask.
        contours: List of boundary contours for the mask (if computed),
            where each contour is an (N, 1, 2) integer array.
    """

    label: str
    confidence: float
    box: DetectionBox
    mask: NDArray[Any]  # bool array
    mask_area_pixels: int
    contours: tuple[NDArray[Any], ...] | None = None


@dataclass(frozen=True)
class SegmentationResult:
    """Aggregated output of a SAM 2 inference pass on an image.

    Attributes:
        objects: Tuple of segmented MaskObjects.
        image_width: Width of the source image in pixels.
        image_height: Height of the source image in pixels.
    """

    objects: tuple[MaskObject, ...]
    image_width: int
    image_height: int

    @property
    def is_empty(self) -> bool:
        """Return whether the result contains no segmented objects."""
        return len(self.objects) == 0

    def filter_by_label(self, label: str) -> tuple[MaskObject, ...]:
        """Return only the objects whose label matches the given string exactly.

        Args:
            label: Label string to filter on.

        Returns:
            A tuple of matching :class:`MaskObject` instances.
        """
        return tuple(obj for obj in self.objects if obj.label == label)
