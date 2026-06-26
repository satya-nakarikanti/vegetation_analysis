"""Visualization utilities for segmentation masks."""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray
from PIL import Image

from vegetation_analysis.segmentation.schemas import SegmentedObject

logger = logging.getLogger(__name__)


class MaskVisualizer:
    """Create and save mask overlay visualizations."""

    def __init__(self, alpha: float = 0.45) -> None:
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("Alpha must be between 0.0 and 1.0.")
        self._alpha = alpha

    def overlay_masks(
        self,
        image: NDArray[np.uint8] | Image.Image,
        objects: list[SegmentedObject],
        draw_contours: bool = True,
    ) -> NDArray[np.uint8]:
        """Overlay masks and optional contours on an RGB image."""

        image_array = self._to_rgb_array(image)
        overlay = image_array.copy()

        for segmented_object in objects:
            color = _color_for_id(segmented_object.id)
            overlay[segmented_object.mask] = (
                (1.0 - self._alpha) * overlay[segmented_object.mask]
                + self._alpha * np.array(color, dtype=np.uint8)
            ).astype(np.uint8)

            if draw_contours and segmented_object.contour:
                contour = np.array(segmented_object.contour, dtype=np.int32).reshape(
                    (-1, 1, 2)
                )
                cv2.drawContours(overlay, [contour], -1, color, thickness=2)

        return overlay

    def save_overlay(
        self,
        image: NDArray[np.uint8] | Image.Image,
        objects: list[SegmentedObject],
        output_path: Path,
    ) -> Path:
        """Save a mask overlay visualization to disk."""

        output_path.parent.mkdir(parents=True, exist_ok=True)
        overlay = self.overlay_masks(image=image, objects=objects)
        Image.fromarray(overlay).save(output_path)
        logger.info("Saved segmentation visualization to %s.", output_path)
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


def _color_for_id(object_id: int) -> tuple[int, int, int]:
    colors = (
        (230, 25, 75),
        (60, 180, 75),
        (0, 130, 200),
        (245, 130, 48),
        (145, 30, 180),
        (70, 240, 240),
        (240, 50, 230),
        (210, 245, 60),
    )
    return colors[object_id % len(colors)]
