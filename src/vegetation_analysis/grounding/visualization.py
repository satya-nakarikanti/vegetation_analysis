"""Visualization utilities for Grounding DINO detection results."""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray
from PIL import Image

from vegetation_analysis.grounding.schemas import DetectionBox, DetectionResult

logger = logging.getLogger(__name__)

# Default colour palette for bounding boxes, cycled by detection index.
_BOX_COLORS: tuple[tuple[int, int, int], ...] = (
    (230, 25, 75),
    (60, 180, 75),
    (0, 130, 200),
    (245, 130, 48),
    (145, 30, 180),
    (70, 240, 240),
    (240, 50, 230),
    (210, 245, 60),
)

# Font scale and thickness used when drawing label text on visualizations.
_LABEL_FONT_SCALE: float = 0.55
_LABEL_THICKNESS: int = 1
_BOX_THICKNESS: int = 2


class DetectionVisualizer:
    """Create and save bounding-box visualizations for detection results.

    Each detected object is drawn with a colour-coded axis-aligned bounding
    box and a label that includes the object category and confidence score.

    Args:
        font: OpenCV font identifier used for label text.  Defaults to
            ``cv2.FONT_HERSHEY_SIMPLEX``.
    """

    def __init__(self, font: int = cv2.FONT_HERSHEY_SIMPLEX) -> None:
        self._font = font

    def draw_boxes(
        self,
        image: NDArray[np.uint8] | Image.Image,
        result: DetectionResult,
    ) -> NDArray[np.uint8]:
        """Draw bounding boxes and labels on an RGB image.

        Args:
            image: Source image as a NumPy array or PIL Image.
            result: Detection result containing bounding boxes.

        Returns:
            A copy of the source image as an ``(H, W, 3)`` uint8 NumPy array
            with bounding boxes and labels drawn.
        """

        image_array = self._to_rgb_array(image)
        canvas = image_array.copy()

        for index, box in enumerate(result.boxes):
            color = _color_for_index(index)
            _draw_box(canvas, box, color, self._font)

        return canvas

    def save_visualization(
        self,
        image: NDArray[np.uint8] | Image.Image,
        result: DetectionResult,
        output_path: Path,
    ) -> Path:
        """Save a bounding-box visualization to disk.

        Args:
            image: Source image as a NumPy array or PIL Image.
            result: Detection result containing bounding boxes.
            output_path: Destination file path.  Parent directories are
                created automatically.

        Returns:
            The resolved output path.
        """

        output_path.parent.mkdir(parents=True, exist_ok=True)
        canvas = self.draw_boxes(image=image, result=result)
        Image.fromarray(canvas).save(output_path)
        logger.info("Saved detection visualization to %s.", output_path)
        return output_path

    @staticmethod
    def _to_rgb_array(image: NDArray[np.uint8] | Image.Image) -> NDArray[np.uint8]:
        if isinstance(image, Image.Image):
            return np.asarray(image.convert("RGB"), dtype=np.uint8)

        image_array = np.asarray(image, dtype=np.uint8)
        if image_array.ndim == 2:
            return np.stack([image_array] * 3, axis=-1)
        if image_array.ndim == 3 and image_array.shape[2] == 1:
            return np.repeat(image_array, repeats=3, axis=2)
        if image_array.ndim == 3 and image_array.shape[2] == 3:
            return image_array
        if image_array.ndim == 3 and image_array.shape[2] == 4:
            return image_array[:, :, :3]

        raise ValueError("Image must be grayscale, RGB, or RGBA.")


def _color_for_index(index: int) -> tuple[int, int, int]:
    """Return a deterministic colour for a given detection index."""

    return _BOX_COLORS[index % len(_BOX_COLORS)]


def _draw_box(
    canvas: NDArray[np.uint8],
    box: DetectionBox,
    color: tuple[int, int, int],
    font: int,
) -> None:
    """Draw a single bounding box and its label on *canvas* in place."""

    cv2.rectangle(
        canvas,
        (box.x_min, box.y_min),
        (box.x_max, box.y_max),
        color,
        _BOX_THICKNESS,
    )
    label_text = f"{box.label} {box.confidence:.2f}"
    cv2.putText(
        canvas,
        label_text,
        (box.x_min, max(box.y_min - 6, 0)),
        font,
        _LABEL_FONT_SCALE,
        color,
        _LABEL_THICKNESS,
        cv2.LINE_AA,
    )
