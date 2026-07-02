"""Output schemas for Grounding DINO detection results."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

# Axis-aligned bounding box expressed as (x_min, y_min, x_max, y_max) in pixels.
BoxXYXY: TypeAlias = tuple[int, int, int, int]


@dataclass(frozen=True)
class DetectionBox:
    """Axis-aligned bounding box for a single detected object.

    Coordinates are expressed in pixel space, origin at the top-left corner of
    the image.

    Attributes:
        x_min: Left edge of the bounding box in pixels.
        y_min: Top edge of the bounding box in pixels.
        x_max: Right edge of the bounding box in pixels (exclusive).
        y_max: Bottom edge of the bounding box in pixels (exclusive).
        label: Text label assigned by the detection prompt.
        confidence: Detector confidence score in the range [0.0, 1.0].
    """

    x_min: int
    y_min: int
    x_max: int
    y_max: int
    label: str
    confidence: float

    @property
    def width(self) -> int:
        """Return bounding box width in pixels."""

        return max(0, self.x_max - self.x_min)

    @property
    def height(self) -> int:
        """Return bounding box height in pixels."""

        return max(0, self.y_max - self.y_min)

    @property
    def area(self) -> int:
        """Return bounding box area in pixels squared."""

        return self.width * self.height

    def as_xyxy(self) -> BoxXYXY:
        """Return the bounding box as ``(x_min, y_min, x_max, y_max)``."""

        return self.x_min, self.y_min, self.x_max, self.y_max


@dataclass(frozen=True)
class DetectionResult:
    """Aggregated output of a single Grounding DINO inference pass.

    Attributes:
        boxes: Detected bounding boxes with labels and confidence scores.
        image_width: Width of the source image in pixels.
        image_height: Height of the source image in pixels.
        prompt: The text prompt used to produce this result.
    """

    boxes: tuple[DetectionBox, ...]
    image_width: int
    image_height: int
    prompt: str

    @property
    def is_empty(self) -> bool:
        """Return whether the result contains no detected objects."""

        return len(self.boxes) == 0

    def filter_by_label(self, label: str) -> tuple[DetectionBox, ...]:
        """Return only the boxes whose label matches the given string exactly.

        Args:
            label: Label string to filter on.

        Returns:
            A tuple of matching :class:`DetectionBox` instances, possibly empty.
        """

        return tuple(box for box in self.boxes if box.label == label)
