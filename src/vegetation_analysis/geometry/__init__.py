"""Relative Geometry Engine.

Computes camera-relative normalised coordinates (relative_x, relative_y,
relative_z) for every segmented object, using Phase 5.2 depth-sampling
output as input.

The camera is treated as the coordinate origin.  The image centre is used
as the assumed principal point — no camera calibration is required.

Public API
----------
- :class:`RelativeGeometryEngine` — main computation class.
- :class:`GeometryResult` — aggregated output schema.
- :class:`GeometricObject` — per-object output schema.
- :class:`GeometryImageMetadata` — image-level metadata schema.
- :class:`GeometryVisualizer` — visualization utility.
"""

from __future__ import annotations

from vegetation_analysis.geometry.constants import (
    CENTROID_MARKER_COLOR,
    CENTROID_RADIUS,
    COORD_COLOR,
    LABEL_COLOR,
    OVERLAY_FONT_SCALE,
    OVERLAY_FONT_THICKNESS,
    OVERLAY_LINE_SPACING,
)
from vegetation_analysis.geometry.geometry_engine import GeometryEngine
from vegetation_analysis.geometry.schemas import (
    GeometricObject,
    GeometryImageMetadata,
    GeometryResult,
)
from vegetation_analysis.geometry.visualization import GeometryVisualizer

__all__ = [
    "CENTROID_MARKER_COLOR",
    "CENTROID_RADIUS",
    "COORD_COLOR",
    "LABEL_COLOR",
    "OVERLAY_FONT_SCALE",
    "OVERLAY_FONT_THICKNESS",
    "OVERLAY_LINE_SPACING",
    "GeometricObject",
    "GeometryEngine",
    "GeometryImageMetadata",
    "GeometryResult",
    "GeometryVisualizer",
]
