"""FastSAM segmentation package."""

from vegetation_analysis.segmentation.fastsam_loader import (
    FastSAMLoader,
    FastSAMModelConfig,
    LoadedFastSAM,
)
from vegetation_analysis.segmentation.mask_utils import extract_segmented_objects
from vegetation_analysis.segmentation.schemas import BoundingBox, SegmentedObject
from vegetation_analysis.segmentation.segmenter import FastSAMSegmenter
from vegetation_analysis.segmentation.visualization import MaskVisualizer

__all__ = [
    "BoundingBox",
    "FastSAMLoader",
    "FastSAMModelConfig",
    "FastSAMSegmenter",
    "LoadedFastSAM",
    "MaskVisualizer",
    "SegmentedObject",
    "extract_segmented_objects",
]
