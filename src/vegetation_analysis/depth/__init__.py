"""Depth Anything V2 estimation module.

This module provides a standalone interface for generating relative depth maps
from single images using the Depth Anything V2 architecture.

The public API is restricted to the components exported below.
"""

from __future__ import annotations

from vegetation_analysis.depth.constants import (
    DEFAULT_COLORMAP,
    DEFAULT_DEVICE_PREFERENCE,
    DEFAULT_MODEL_ID,
)
from vegetation_analysis.depth.depth_loader import (
    DepthAnythingLoader,
    DepthAnythingModelConfig,
    LoadedDepthAnything,
)
from vegetation_analysis.depth.estimator import DepthEstimator
from vegetation_analysis.depth.schemas import DepthMapResult
from vegetation_analysis.depth.visualization import DepthVisualizer

__all__ = [
    "DEFAULT_COLORMAP",
    "DEFAULT_DEVICE_PREFERENCE",
    "DEFAULT_MODEL_ID",
    "DepthAnythingLoader",
    "DepthAnythingModelConfig",
    "DepthEstimator",
    "DepthMapResult",
    "DepthVisualizer",
    "LoadedDepthAnything",
]
