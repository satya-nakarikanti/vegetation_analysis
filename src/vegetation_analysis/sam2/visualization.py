"""Visualization utilities for SAM 2 segmentation results."""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray
from PIL import Image

from vegetation_analysis.sam2.schemas import SegmentationResult

logger = logging.getLogger(__name__)

# Default colour palette for masks and bounding boxes.
_COLORS: tuple[tuple[int, int, int], ...] = (
    (230, 25, 75),
    (60, 180, 75),
    (0, 130, 200),
    (245, 130, 48),
    (145, 30, 180),
    (70, 240, 240),
    (240, 50, 230),
    (210, 245, 60),
)


def _color_for_index(index: int) -> tuple[int, int, int]:
    """Return a deterministic colour for a given detection index."""
    return _COLORS[index % len(_COLORS)]


_LABEL_FONT_SCALE: float = 0.55
_LABEL_THICKNESS: int = 1
_BOX_THICKNESS: int = 2
_MASK_ALPHA: float = 0.5


class SegmentationVisualizer:
    """Create and save visualizations for SAM 2 segmentation results.

    Args:
        font: OpenCV font identifier used for label text.
    """

    def __init__(self, font: int = cv2.FONT_HERSHEY_SIMPLEX) -> None:
        self._font = font

    def draw_masks_and_boxes(
        self,
        image: NDArray[np.uint8] | Image.Image,
        result: SegmentationResult,
    ) -> NDArray[np.uint8]:
        """Draw masks, bounding boxes, and labels on an RGB image.

        Args:
            image: Source image.
            result: Segmentation result containing masks.

        Returns:
            An (H, W, 3) uint8 NumPy array with overlays.
        """
        image_array = self._to_rgb_array(image)
        canvas = image_array.copy()

        for index, obj in enumerate(result.objects):
            color = _color_for_index(index)
            # Draw mask overlay
            self._draw_mask(canvas, obj.mask, color)
            # Draw contours
            if obj.contours:
                cv2.drawContours(canvas, obj.contours, -1, color, 1)

            # Draw bounding box and label
            box = obj.box
            cv2.rectangle(
                canvas,
                (box.x_min, box.y_min),
                (box.x_max, box.y_max),
                color,
                _BOX_THICKNESS,
            )

            label_text = f"{obj.label} {obj.confidence:.2f}"
            cv2.putText(
                canvas,
                label_text,
                (box.x_min, max(box.y_min - 6, 0)),
                self._font,
                _LABEL_FONT_SCALE,
                color,
                _LABEL_THICKNESS,
                cv2.LINE_AA,
            )

        return canvas

    def save_visualization(
        self,
        image: NDArray[np.uint8] | Image.Image,
        result: SegmentationResult,
        output_path: Path,
    ) -> Path:
        """Save a visualization to disk."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        canvas = self.draw_masks_and_boxes(image=image, result=result)
        Image.fromarray(canvas).save(output_path)
        logger.info("Saved segmentation visualization to %s.", output_path)
        return output_path

    @staticmethod
    def _draw_mask(
        canvas: NDArray[np.uint8],
        mask: NDArray[np.bool_],
        color: tuple[int, int, int],
    ) -> None:
        """Blend a single boolean mask into the canvas."""
        overlay = canvas.copy()
        overlay[mask] = color
        cv2.addWeighted(overlay, _MASK_ALPHA, canvas, 1.0 - _MASK_ALPHA, 0, canvas)

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
