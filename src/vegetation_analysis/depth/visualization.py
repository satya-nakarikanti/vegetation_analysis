"""Visualization utilities for Depth Anything V2 results."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import cast

import cv2
import numpy as np
from numpy.typing import NDArray
from PIL import Image

from vegetation_analysis.depth.constants import DEFAULT_COLORMAP
from vegetation_analysis.depth.schemas import DepthMapResult

logger = logging.getLogger(__name__)


class DepthVisualizer:
    """Create and save visualizations for depth estimation results.

    The visualizer normalizes the raw floating-point depth map and applies
    a colour map to create a human-readable heatmap representation.

    Args:
        colormap: OpenCV colormap identifier used for the heatmap. Defaults to
            ``cv2.COLORMAP_INFERNO``.
    """

    def __init__(self, colormap: int = DEFAULT_COLORMAP) -> None:
        self._colormap = colormap

    def create_heatmap(
        self, result: DepthMapResult, invert: bool = False
    ) -> NDArray[np.uint8]:
        """Convert a raw depth map into a colorized heatmap.

        Args:
            result: Depth estimation result containing the raw depth array.
            invert: If True, inverts the normalization so larger depth values
                become smaller normalized values (useful for metric depth to
                match relative depth visualization).

        Returns:
            An ``(H, W, 3)`` uint8 NumPy array in RGB format representing the
            colorized depth map.
        """

        depth_array = result.depth_map

        depth_min = np.min(depth_array)
        depth_max = np.max(depth_array)

        if depth_max - depth_min > 1e-6:
            if invert:
                normalized_depth = (depth_max - depth_array) / (depth_max - depth_min)
            else:
                normalized_depth = (depth_array - depth_min) / (depth_max - depth_min)
        else:
            normalized_depth = np.zeros_like(depth_array)

        # Scale to 0-255 and convert to uint8
        depth_uint8 = (normalized_depth * 255.0).astype(np.uint8)

        # Apply colormap (returns BGR)
        heatmap_bgr = cv2.applyColorMap(depth_uint8, self._colormap)

        # Convert BGR to RGB for consistent PIL/Matplotlib handling
        heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)

        return cast(NDArray[np.uint8], heatmap_rgb)

    def save_visualization(
        self,
        result: DepthMapResult,
        output_path: Path,
        invert: bool = False,
    ) -> Path:
        """Save a depth map visualization to disk.

        Args:
            result: Depth estimation result containing the raw depth array.
            output_path: Destination file path. Parent directories are
                created automatically.
            invert: If True, inverts the depth normalization.

        Returns:
            The resolved output path.
        """

        output_path.parent.mkdir(parents=True, exist_ok=True)
        heatmap = self.create_heatmap(result=result, invert=invert)
        Image.fromarray(heatmap).save(output_path)
        logger.info("Saved depth visualization to %s.", output_path)
        return output_path

    def create_grayscale(
        self, result: DepthMapResult, invert: bool = False
    ) -> NDArray[np.uint8]:
        """Create a normalised grayscale depth image before any colormap.

        Useful for inspection and as an intermediate for further processing.
        The output is a single-channel (H, W) uint8 array where 0 = minimum
        depth and 255 = maximum depth in the map.

        Args:
            result: Depth estimation result containing the raw depth array.
            invert: If True, inverts the normalization.

        Returns:
            An ``(H, W)`` uint8 NumPy array representing normalised depth.
        """

        depth_array = result.depth_map
        depth_min = np.min(depth_array)
        depth_max = np.max(depth_array)

        if depth_max - depth_min > 1e-6:
            if invert:
                normalized = (depth_max - depth_array) / (depth_max - depth_min)
            else:
                normalized = (depth_array - depth_min) / (depth_max - depth_min)
        else:
            normalized = np.zeros_like(depth_array)

        return (normalized * 255.0).astype(np.uint8)

    def save_grayscale(
        self,
        result: DepthMapResult,
        output_path: Path,
        invert: bool = False,
    ) -> Path:
        """Save a normalised grayscale depth image to disk.

        Args:
            result: Depth estimation result containing the raw depth array.
            output_path: Destination file path. Parent directories are
                created automatically.
            invert: If True, inverts the normalization.

        Returns:
            The resolved output path.
        """

        output_path.parent.mkdir(parents=True, exist_ok=True)
        grayscale = self.create_grayscale(result=result, invert=invert)
        Image.fromarray(grayscale, mode="L").save(output_path)
        logger.info("Saved grayscale depth image to %s.", output_path)
        return output_path
