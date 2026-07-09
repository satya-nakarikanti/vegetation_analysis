"""Relative Geometry Engine.

Converts Phase 5.2 depth-sampling results into camera-relative normalised
coordinates for every detected object.

The camera is treated as the coordinate origin.  The image centre is used as
the principal point so no camera calibration is required.

Coordinate conventions
----------------------
* ``relative_x`` — normalised horizontal offset from the image centre.
  Range: -0.5 (far left) to +0.5 (far right).
* ``relative_y`` — normalised vertical offset from the image centre.
  Range: -0.5 (top) to +0.5 (bottom).
* ``relative_z`` — the representative depth value (median) from Phase 5.2.
  Same unit as Depth Anything V2 output (relative, dimensionless).
"""

from __future__ import annotations

import logging

from vegetation_analysis.depth_sampling.schemas import DepthSamplingResult
from vegetation_analysis.geometry.schemas import (
    GeometricObject,
    GeometryImageMetadata,
    GeometryResult,
)

logger = logging.getLogger(__name__)


class RelativeGeometryEngine:
    """Compute camera-relative normalised coordinates for sampled objects.

    Uses the image centre as the assumed camera principal point.  No focal
    length or metric calibration is performed — all coordinates are
    dimensionless and internally consistent.

    Usage::

        engine = RelativeGeometryEngine()
        result = engine.compute(sampling_result)
    """

    def compute(self, sampling_result: DepthSamplingResult) -> GeometryResult:
        """Compute relative geometry for every object in a sampling result.

        Args:
            sampling_result: Phase 5.2 depth sampling result containing
                per-object depth statistics and centroid coordinates.

        Returns:
            A :class:`GeometryResult` with normalised relative coordinates
            for every object and image-level metadata.
        """

        image_width = sampling_result.metadata.width
        image_height = sampling_result.metadata.height

        # Principal point: image centre.
        principal_x = image_width / 2.0
        principal_y = image_height / 2.0

        geometric_objects: list[GeometricObject] = []
        for obj in sampling_result.objects:
            geo_obj = self._compute_object(
                obj=obj,
                principal_x=principal_x,
                principal_y=principal_y,
                image_width=image_width,
                image_height=image_height,
            )
            geometric_objects.append(geo_obj)

        metadata = GeometryImageMetadata(
            width=image_width,
            height=image_height,
            principal_x=principal_x,
            principal_y=principal_y,
            depth_model=sampling_result.metadata.depth_model,
        )

        logger.info(
            "Relative geometry computed for %d object(s).",
            len(geometric_objects),
        )

        return GeometryResult(
            objects=tuple(geometric_objects),
            metadata=metadata,
        )

    @staticmethod
    def _compute_object(
        obj: object,
        principal_x: float,
        principal_y: float,
        image_width: int,
        image_height: int,
    ) -> GeometricObject:
        """Compute relative coordinates for a single sampled object.

        Normalisation divides the pixel offset from the principal point by
        the image half-width and half-height respectively, so values are
        always in [-0.5, 0.5] for objects whose centroid lies within the
        image boundaries.

        Args:
            obj: A :class:`~vegetation_analysis.depth_sampling.schemas.SampledObject`.
            principal_x: X-coordinate of the camera principal point.
            principal_y: Y-coordinate of the camera principal point.
            image_width: Full width of the image in pixels.
            image_height: Full height of the image in pixels.

        Returns:
            A fully populated :class:`GeometricObject`.
        """

        # Import inline to avoid circular dependency at module level.
        from vegetation_analysis.depth_sampling.schemas import SampledObject

        assert isinstance(obj, SampledObject)  # noqa: S101 (dev assertion)

        # Normalise centroid offset against half-image dimensions so the
        # coordinate range is [-0.5, 0.5] independent of image resolution.
        half_w = image_width / 2.0
        half_h = image_height / 2.0

        relative_x = (obj.centroid_x - principal_x) / half_w if half_w else 0.0
        relative_y = (obj.centroid_y - principal_y) / half_h if half_h else 0.0
        relative_z = obj.median_depth  # Representative depth from Phase 5.2.

        return GeometricObject(
            label=obj.label,
            species=None,
            species_confidence=None,
            confidence=obj.confidence,
            pixel_count=obj.pixel_count,
            centroid_x=obj.centroid_x,
            centroid_y=obj.centroid_y,
            relative_x=relative_x,
            relative_y=relative_y,
            relative_z=relative_z,
        )
