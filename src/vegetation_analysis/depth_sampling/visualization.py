"""Visualization utilities for depth sampling results."""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray
from PIL import Image

from vegetation_analysis.depth.schemas import DepthMapResult
from vegetation_analysis.depth.visualization import DepthVisualizer
from vegetation_analysis.depth_sampling.constants import (
    CENTROID_COLOR,
    CENTROID_RADIUS,
)
from vegetation_analysis.depth_sampling.schemas import DepthSamplingResult

logger = logging.getLogger(__name__)


class DepthSamplingVisualizer:
    """Create and save visualizations for depth sampling results.

    The visualizer overlays object centroids and key depth statistics onto
    a colorized depth heatmap.

    Args:
        font: OpenCV font identifier used for label text. Defaults to
            ``cv2.FONT_HERSHEY_SIMPLEX``.
        font_scale: Scale of the text labels.
        font_thickness: Thickness of the text labels.
    """

    def __init__(
        self,
        font: int = cv2.FONT_HERSHEY_SIMPLEX,
        font_scale: float = 0.55,
        font_thickness: int = 1,
    ) -> None:
        self._depth_visualizer = DepthVisualizer()
        self._font = font
        self._font_scale = font_scale
        self._font_thickness = font_thickness

    def draw_overlays(
        self,
        depth_map: DepthMapResult,
        sampling_result: DepthSamplingResult,
    ) -> NDArray[np.uint8]:
        """Draw centroids and stats on a colorized depth heatmap.

        Args:
            depth_map: Original depth map result.
            sampling_result: Depth sampling result with object statistics.

        Returns:
            A copy of the base heatmap as an ``(H, W, 3)`` uint8 NumPy array
            in RGB format with overlays drawn.
        """

        # Generate the base RGB heatmap using the depth visualization module
        canvas = self._depth_visualizer.create_heatmap(depth_map)

        for obj in sampling_result.objects:
            cx, cy = int(round(obj.centroid_x)), int(round(obj.centroid_y))

            # Draw centroid marker
            cv2.circle(
                canvas,
                (cx, cy),
                CENTROID_RADIUS,
                CENTROID_COLOR,
                -1,
                cv2.LINE_AA,
            )

            # Draw text label (e.g. "tree: 4.25 (median)")
            label_text = f"{obj.label}: {obj.median_depth:.2f}"

            # Add a slight drop shadow / outline effect for text readability
            text_size, _ = cv2.getTextSize(
                label_text,
                self._font,
                self._font_scale,
                self._font_thickness,
            )

            text_x = max(0, cx - text_size[0] // 2)
            text_y = max(text_size[1], cy - 10)

            # Outline
            cv2.putText(
                canvas,
                label_text,
                (text_x, text_y),
                self._font,
                self._font_scale,
                (0, 0, 0),
                self._font_thickness + 1,
                cv2.LINE_AA,
            )

            # Main text
            cv2.putText(
                canvas,
                label_text,
                (text_x, text_y),
                self._font,
                self._font_scale,
                CENTROID_COLOR,
                self._font_thickness,
                cv2.LINE_AA,
            )

        return canvas

    def save_visualization(
        self,
        depth_map: DepthMapResult,
        sampling_result: DepthSamplingResult,
        output_path: Path,
    ) -> Path:
        """Save a sampling visualization to disk.

        Args:
            depth_map: Original depth map result.
            sampling_result: Depth sampling result with object statistics.
            output_path: Destination file path. Parent directories are
                created automatically.

        Returns:
            The resolved output path.
        """

        output_path.parent.mkdir(parents=True, exist_ok=True)
        canvas = self.draw_overlays(
            depth_map=depth_map,
            sampling_result=sampling_result,
        )
        Image.fromarray(canvas).save(output_path)
        logger.info("Saved depth sampling visualization to %s.", output_path)
        return output_path
