"""Output schemas for the Relative Geometry Engine."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GeometricObject:
    """A single object with camera-relative geometric coordinates.

    The camera is treated as the coordinate origin. ``relative_x`` and
    ``relative_y`` are normalised image-plane coordinates centred on the
    principal point (image centre).  ``relative_z`` is the representative
    depth value from Phase 5.2 and carries the same unit as the Depth
    Anything V2 output (relative, dimensionless).

    The optional ``species`` / ``species_confidence`` fields are reserved for
    future tree-species classification (Phase 4) and default to ``None`` when
    species information is unavailable.

    Attributes:
        label: Text label assigned by the upstream detector.
        species: Tree species name, or ``None`` if unavailable.
        species_confidence: Species classification confidence, or ``None``.
        confidence: Detector confidence score.
        pixel_count: Number of pixels used for depth sampling.
        centroid_x: X-coordinate of the mask centroid in image pixels.
        centroid_y: Y-coordinate of the mask centroid in image pixels.
        relative_x: Normalised horizontal position relative to image centre.
            Negative values are left of centre, positive values are right.
        relative_y: Normalised vertical position relative to image centre.
            Negative values are above centre, positive values are below.
        relative_z: Representative depth (median) from Phase 5.2.
    """

    label: str
    species: str | None
    species_confidence: float | None
    confidence: float
    pixel_count: int
    centroid_x: float
    centroid_y: float
    relative_x: float
    relative_y: float
    relative_z: float


@dataclass(frozen=True)
class GeometryImageMetadata:
    """Image-level metadata for geometry results.

    Attributes:
        width: Width of the source image in pixels.
        height: Height of the source image in pixels.
        principal_x: X-coordinate of the assumed principal point (image centre).
        principal_y: Y-coordinate of the assumed principal point (image centre).
        depth_model: Name of the depth model used.
    """

    width: int
    height: int
    principal_x: float
    principal_y: float
    depth_model: str


@dataclass(frozen=True)
class GeometryResult:
    """Aggregated output of the Relative Geometry Engine.

    Attributes:
        objects: Tuple of geometric objects with relative coordinates.
        metadata: Image-level metadata describing the geometry coordinate space.
    """

    objects: tuple[GeometricObject, ...]
    metadata: GeometryImageMetadata
