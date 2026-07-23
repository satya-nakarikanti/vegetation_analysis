"""Projection Engine.

Converts depth-sampled image observations into camera-space 3D coordinates
using a pinhole camera model.
"""

from __future__ import annotations

import logging

from vegetation_analysis.depth_sampling.schemas import DepthSamplingResult
from vegetation_analysis.projection.schemas import (
    CameraIntrinsics,
    CoordinateSystem,
    ProjectedObject,
    ProjectionResult,
)

logger = logging.getLogger(__name__)


class ProjectionEngine:
    """Project image observations into camera-space 3D coordinates.

    Applies the standard pinhole camera projection:
    X = (u - cx) * Z / fx
    Y = (v - cy) * Z / fy
    Z = depth

    This engine is purely a coordinate transformer and does not compute
    semantic distances or bounding geometry.

    Args:
        coordinate_system: The coordinate system of the provided depth
            values (RELATIVE or METRIC).
    """

    def __init__(self, coordinate_system: CoordinateSystem) -> None:
        self._coordinate_system = coordinate_system

    def project(
        self,
        sampling_result: DepthSamplingResult,
        intrinsics: CameraIntrinsics,
    ) -> ProjectionResult:
        """Project all sampled objects into camera space.

        Args:
            sampling_result: Depth sampling result containing 2D objects.
            intrinsics: Camera parameters used for projection.

        Returns:
            A :class:`ProjectionResult` containing the 3D projected objects.
        """

        projected_objects: list[ProjectedObject] = []

        for obj in sampling_result.objects:
            u = obj.centroid_x
            v = obj.centroid_y
            z = obj.median_depth

            if intrinsics.fx != 0 and intrinsics.fy != 0:
                x = (u - intrinsics.cx) * z / intrinsics.fx
                y = (v - intrinsics.cy) * z / intrinsics.fy
            else:
                x, y = 0.0, 0.0

            proj_obj = ProjectedObject(
                label=obj.label,
                confidence=obj.confidence,
                pixel_count=obj.pixel_count,
                centroid_x=obj.centroid_x,
                centroid_y=obj.centroid_y,
                median_depth=obj.median_depth,
                mean_depth=obj.mean_depth,
                std_depth=obj.std_depth,
                min_depth=obj.min_depth,
                max_depth=obj.max_depth,
                bounding_box=obj.bounding_box,
                original_mask=obj.original_mask,
                sampling_mask=obj.sampling_mask,
                contours=obj.contours,
                mask_area_pixels=obj.mask_area_pixels,
                camera_x=x,
                camera_y=y,
                camera_z=z,
            )
            projected_objects.append(proj_obj)

        logger.info(
            "Projected %d object(s) into %s camera space.",
            len(projected_objects),
            self._coordinate_system.value,
        )

        return ProjectionResult(
            objects=tuple(projected_objects),
            intrinsics=intrinsics,
            coordinate_system=self._coordinate_system,
            depth_model=sampling_result.metadata.depth_model,
        )
