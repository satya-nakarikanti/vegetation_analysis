"""Output schemas for the Relative Geometry Engine."""

from __future__ import annotations

from dataclasses import dataclass

from vegetation_analysis.projection.schemas import CoordinateSystem


@dataclass(frozen=True)
class GeometricObject:
    """A single object with computed geometric coordinates.

    Attributes:
        label: Text label assigned by the upstream detector.
        species: Tree species name, or ``None`` if unavailable.
        species_confidence: Species classification confidence, or ``None``.
        confidence: Detector confidence score.
        pixel_count: Number of pixels used for depth sampling.
        centroid_x: X-coordinate of the mask centroid in image pixels.
        centroid_y: Y-coordinate of the mask centroid in image pixels.
        camera_x: X-coordinate in camera space.
        camera_y: Y-coordinate in camera space.
        camera_z: Z-coordinate in camera space (depth).
        camera_distance: True Euclidean distance from the camera.
        relative_x: (Deprecated) Normalised horizontal position.
        relative_y: (Deprecated) Normalised vertical position.
        relative_z: (Deprecated) Representative depth.
    """

    label: str
    species: str | None
    species_confidence: float | None
    confidence: float
    pixel_count: int
    centroid_x: float
    centroid_y: float
    camera_x: float
    camera_y: float
    camera_z: float
    camera_distance: float
    relative_x: float | None
    relative_y: float | None
    relative_z: float | None


@dataclass(frozen=True)
class GeometryImageMetadata:
    """Image-level metadata for geometry results.

    Attributes:
        width: Width of the source image in pixels.
        height: Height of the source image in pixels.
        principal_x: X-coordinate of the assumed principal point (image centre).
        principal_y: Y-coordinate of the assumed principal point (image centre).
        depth_model: Name of the depth model used.
        coordinate_system: The coordinate system (RELATIVE or METRIC).
    """

    width: int
    height: int
    principal_x: float
    principal_y: float
    depth_model: str
    coordinate_system: CoordinateSystem


@dataclass(frozen=True)
class ObjectRelationship:
    """Pairwise angular relationship between two detected objects.

    Attributes:
        object_a_label: Label of the first object.
        object_b_label: Label of the second object.
        angle_radians: The angle between the two objects' camera vectors in radians.
        angle_degrees: The angle between the two objects' camera vectors in degrees.
        dot_product: The dot product of the two objects' camera vectors.
        centroid_distance_law_of_cosines: Distance between centroids using the Law of Cosines.
        centroid_distance_euclidean: Direct 3D Euclidean distance between the centroids.
        distance_difference: Absolute difference between the two methods.
        validation_passed: True if the difference is within floating-point tolerance (< 1e-6).
    """
    object_a_label: str
    object_b_label: str
    angle_radians: float
    angle_degrees: float
    dot_product: float
    centroid_distance_law_of_cosines: float
    centroid_distance_euclidean: float
    distance_difference: float
    validation_passed: bool


@dataclass(frozen=True)
class GeometryResult:
    """Aggregated output of the Relative Geometry Engine.

    Attributes:
        objects: Tuple of geometric objects with relative coordinates.
        metadata: Image-level metadata describing the geometry coordinate space.
        relationships: Pairwise angular relationships between the objects.
    """

    objects: tuple[GeometricObject, ...]
    metadata: GeometryImageMetadata
    relationships: tuple[ObjectRelationship, ...] = ()
