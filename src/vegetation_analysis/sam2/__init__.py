"""SAM 2 foundation and infrastructure module.

This module provides a Segmenter that consumes DetectionResults
from Grounding DINO and runs them through SAM 2 to generate SegmentationResults.
"""

from vegetation_analysis.sam2.constants import (
    DEFAULT_CHECKPOINT,
    DEFAULT_DEVICE_PREFERENCE,
    DEFAULT_MASK_THRESHOLD,
    DEFAULT_MODEL_CFG,
    DevicePreference,
)
from vegetation_analysis.sam2.sam2_loader import LoadedSAM2, SAM2Loader, SAM2ModelConfig
from vegetation_analysis.sam2.schemas import MaskObject, SegmentationResult
from vegetation_analysis.sam2.segmenter import SAM2Segmenter
from vegetation_analysis.sam2.visualization import SegmentationVisualizer

__all__ = [
    "DEFAULT_CHECKPOINT",
    "DEFAULT_DEVICE_PREFERENCE",
    "DEFAULT_MASK_THRESHOLD",
    "DEFAULT_MODEL_CFG",
    "DevicePreference",
    "LoadedSAM2",
    "MaskObject",
    "SAM2Loader",
    "SAM2ModelConfig",
    "SAM2Segmenter",
    "SegmentationResult",
    "SegmentationVisualizer",
]
