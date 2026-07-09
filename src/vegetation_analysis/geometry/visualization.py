"""Visualization utilities for Relative Geometry Engine results."""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray
from PIL import Image

from vegetation_analysis.depth.schemas import DepthMapResult
from vegetation_analysis.depth.visualization import DepthVisualizer
from vegetation_analysis.geometry.constants import (
    CENTROID_MARKER_COLOR,
    CENTROID_RADIUS,
    COORD_COLOR,
    LABEL_COLOR,
    OVERLAY_FONT_SCALE,
    OVERLAY_FONT_THICKNESS,
    OVERLAY_LINE_SPACING,
)
from vegetation_analysis.geometry.schemas import GeometryResult

logger = logging.getLogger(__name__)

_FONT = cv2.FONT_HERSHEY_SIMPLEX


class GeometryVisualizer:
    """Render camera-relative geometry overlays on a depth heatmap.

    The visualizer draws the object centroid marker, label, and
    relative X/Y/Z coordinates for every :class:`GeometricObject`
    in a :class:`GeometryResult`.

    Args:
        font_scale: Scale of text labels on the canvas.
        font_thickness: Stroke thickness of text labels.
        line_spacing: Pixel gap between successive text lines per object.
    """

    def __init__(
        self,
        font_scale: float = OVERLAY_FONT_SCALE,
        font_thickness: int = OVERLAY_FONT_THICKNESS,
        line_spacing: int = OVERLAY_LINE_SPACING,
    ) -> None:
        self._depth_visualizer = DepthVisualizer()
        self._font_scale = font_scale
        self._font_thickness = font_thickness
        self._line_spacing = line_spacing

    def draw_overlays(
        self,
        depth_map: DepthMapResult,
        geometry_result: GeometryResult,
    ) -> NDArray[np.uint8]:
        """Draw geometry overlays on a colorized depth heatmap.

        Args:
            depth_map: Raw depth map used to produce the background heatmap.
            geometry_result: Relative geometry result with per-object data.

        Returns:
            An ``(H, W, 3)`` uint8 NumPy array in RGB format with geometry
            overlays rendered on top of the depth heatmap.
        """

        canvas = self._depth_visualizer.create_heatmap(depth_map)

        for obj in geometry_result.objects:
            cx = int(round(obj.centroid_x))
            cy = int(round(obj.centroid_y))

            # Centroid marker
            cv2.circle(
                canvas,
                (cx, cy),
                CENTROID_RADIUS,
                CENTROID_MARKER_COLOR,
                -1,
                cv2.LINE_AA,
            )

            # Build text lines
            if obj.species is None:
                label_text = obj.label
            else:
                label_text = f"{obj.label} ({obj.species})"
            x_text = f"rx={obj.relative_x:+.3f}"
            y_text = f"ry={obj.relative_y:+.3f}"
            z_text = f"rz={obj.relative_z:+.3f}"

            lines: list[tuple[str, tuple[int, int, int]]] = [
                (label_text, LABEL_COLOR),
                (x_text, COORD_COLOR),
                (y_text, COORD_COLOR),
                (z_text, COORD_COLOR),
            ]

            # Position text block so it doesn't exceed image bounds
            text_x = cx + CENTROID_RADIUS + 4
            half_lines = len(lines) // 2
            text_y_start = max(
                cy - self._line_spacing * half_lines,
                self._line_spacing,
            )

            for i, (text, color) in enumerate(lines):
                ty = text_y_start + i * self._line_spacing

                # Outline for readability
                cv2.putText(
                    canvas,
                    text,
                    (text_x, ty),
                    _FONT,
                    self._font_scale,
                    (0, 0, 0),
                    self._font_thickness + 1,
                    cv2.LINE_AA,
                )
                # Main text
                cv2.putText(
                    canvas,
                    text,
                    (text_x, ty),
                    _FONT,
                    self._font_scale,
                    color,
                    self._font_thickness,
                    cv2.LINE_AA,
                )

        return canvas

    def save_visualization(
        self,
        depth_map: DepthMapResult,
        geometry_result: GeometryResult,
        output_path: Path,
    ) -> Path:
        """Save a geometry visualization to disk.

        Args:
            depth_map: Raw depth map used to produce the background heatmap.
            geometry_result: Relative geometry result with per-object data.
            output_path: Destination file path. Parent directories are
                created automatically.

        Returns:
            The resolved output path.
        """

        output_path.parent.mkdir(parents=True, exist_ok=True)
        canvas = self.draw_overlays(
            depth_map=depth_map,
            geometry_result=geometry_result,
        )
        Image.fromarray(canvas).save(output_path)
        logger.info("Saved geometry visualization to %s.", output_path)
        return output_path
