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
import math

from vegetation_analysis.geometry.schemas import (
    GeometricObject,
    GeometryImageMetadata,
    GeometryResult,
    ObjectRelationship,
)
from vegetation_analysis.projection.schemas import ProjectionResult

logger = logging.getLogger(__name__)


class GeometryEngine:
    """Verify and forward projected camera-space coordinates.

    The engine receives ProjectedObject instances from the ProjectionEngine
    and forwards them. Future phases will introduce distances, angles, and
    clearance computations here.

    Usage::

        engine = GeometryEngine()
        result = engine.compute(projection_result)
    """

    def compute(self, projection_result: ProjectionResult) -> GeometryResult:
        """Process projection result into geometric objects.

        Args:
            projection_result: Result from the projection engine.

        Returns:
            A :class:`GeometryResult` with verified geometric objects.
        """

        image_width = projection_result.intrinsics.image_width
        image_height = projection_result.intrinsics.image_height

        principal_x = projection_result.intrinsics.cx
        principal_y = projection_result.intrinsics.cy

        geometric_objects: list[GeometricObject] = []
        for obj in projection_result.objects:
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
            depth_model=projection_result.depth_model,
            coordinate_system=projection_result.coordinate_system,
        )

        import itertools
        relationships: list[ObjectRelationship] = []
        
        for obj_a, obj_b in itertools.combinations(geometric_objects, 2):
            dot_product = (
                obj_a.camera_x * obj_b.camera_x +
                obj_a.camera_y * obj_b.camera_y +
                obj_a.camera_z * obj_b.camera_z
            )
            
            denominator = obj_a.camera_distance * obj_b.camera_distance
            if denominator > 0.0:
                cos_theta = dot_product / denominator
                clamped_cos_theta = max(min(cos_theta, 1.0), -1.0)
                if cos_theta > 1.0 or cos_theta < -1.0:
                    logger.debug("Clamped cos_theta from %s to %s", cos_theta, clamped_cos_theta)
                
                theta_rad = math.acos(clamped_cos_theta)
                theta_deg = math.degrees(theta_rad)
            else:
                theta_rad = 0.0
                theta_deg = 0.0
                clamped_cos_theta = 1.0
                
            # Method 1 - Law of Cosines
            a = obj_a.camera_distance
            b = obj_b.camera_distance
            dist_sq = a**2 + b**2 - 2 * a * b * clamped_cos_theta
            # dist_sq might be slightly negative due to precision if a == b and theta == 0
            centroid_distance_law_of_cosines = math.sqrt(max(dist_sq, 0.0))
            
            # Method 2 - Direct 3D Euclidean Distance
            dx = obj_a.camera_x - obj_b.camera_x
            dy = obj_a.camera_y - obj_b.camera_y
            dz = obj_a.camera_z - obj_b.camera_z
            centroid_distance_euclidean = math.sqrt(dx**2 + dy**2 + dz**2)
            
            # Validation
            distance_difference = abs(centroid_distance_law_of_cosines - centroid_distance_euclidean)
            validation_passed = distance_difference < 1e-6
            
            if not validation_passed:
                raise ValueError(
                    f"Centroid distance validation failed for '{obj_a.label}' <-> '{obj_b.label}':\n"
                    f"  Law of Cosines: {centroid_distance_law_of_cosines}\n"
                    f"  Euclidean     : {centroid_distance_euclidean}\n"
                    f"  Difference    : {distance_difference}"
                )
                
            relationships.append(ObjectRelationship(
                object_a_label=obj_a.label,
                object_b_label=obj_b.label,
                angle_radians=theta_rad,
                angle_degrees=theta_deg,
                dot_product=dot_product,
                centroid_distance_law_of_cosines=centroid_distance_law_of_cosines,
                centroid_distance_euclidean=centroid_distance_euclidean,
                distance_difference=distance_difference,
                validation_passed=validation_passed
            ))

        logger.info(
            "Geometry verified and forwarded for %d object(s).",
            len(geometric_objects),
        )

        return GeometryResult(
            objects=tuple(geometric_objects),
            metadata=metadata,
            relationships=tuple(relationships),
        )

    @staticmethod
    def _compute_object(
        obj: object,
        principal_x: float,
        principal_y: float,
        image_width: int,
        image_height: int,
    ) -> GeometricObject:
        """Convert a ProjectedObject to a GeometricObject.

        Preserves the camera-space coordinates and computes deprecated
        relative coordinates for backward compatibility.
        """

        from vegetation_analysis.projection.schemas import ProjectedObject

        assert isinstance(obj, ProjectedObject)  # noqa: S101

        half_w = image_width / 2.0
        half_h = image_height / 2.0

        relative_x = (obj.centroid_x - principal_x) / half_w if half_w else 0.0
        relative_y = (obj.centroid_y - principal_y) / half_h if half_h else 0.0
        relative_z = obj.median_depth
        
        camera_distance = math.sqrt(
            obj.camera_x**2 +
            obj.camera_y**2 +
            obj.camera_z**2
        )
        
        if camera_distance < obj.camera_z:
            raise ValueError(f"Validation failed: camera_distance ({camera_distance}) cannot be less than camera_z ({obj.camera_z}).")

        return GeometricObject(
            label=obj.label,
            species=None,
            species_confidence=None,
            confidence=obj.confidence,
            pixel_count=obj.pixel_count,
            centroid_x=obj.centroid_x,
            centroid_y=obj.centroid_y,
            camera_x=obj.camera_x,
            camera_y=obj.camera_y,
            camera_z=obj.camera_z,
            camera_distance=camera_distance,
            relative_x=relative_x,
            relative_y=relative_y,
            relative_z=relative_z,
        )
