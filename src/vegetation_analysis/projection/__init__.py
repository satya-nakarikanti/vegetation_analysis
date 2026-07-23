"""Camera Projection module.

Converts depth-sampled objects from image-space observations into camera-space
3D coordinates using standard pinhole projection.
"""

from vegetation_analysis.projection.projection_engine import ProjectionEngine
from vegetation_analysis.projection.schemas import (
    CameraIntrinsics,
    CoordinateSystem,
    ProjectedObject,
    ProjectionResult,
)

__all__ = [
    "CameraIntrinsics",
    "CoordinateSystem",
    "ProjectedObject",
    "ProjectionEngine",
    "ProjectionResult",
]
