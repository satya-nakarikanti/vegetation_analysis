"""Schemas for the Projection Engine."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from vegetation_analysis.depth_sampling.schemas import SampledObject


class CoordinateSystem(Enum):
    """Coordinate system representations."""

    RELATIVE = "relative"
    METRIC = "metric"


@dataclass(frozen=True)
class CameraIntrinsics:
    """Camera intrinsic parameters for pinhole projection.

    Attributes:
        fx: Focal length in x.
        fy: Focal length in y.
        cx: Principal point x-coordinate.
        cy: Principal point y-coordinate.
        image_width: Width of the sensor image.
        image_height: Height of the sensor image.
    """

    fx: float
    fy: float
    cx: float
    cy: float
    image_width: int
    image_height: int


@dataclass(frozen=True)
class ProjectedObject(SampledObject):
    """An object with camera-space coordinates computed via projection.

    Inherits all properties of the upstream :class:`SampledObject`.

    Attributes:
        camera_x: X-coordinate in camera space.
        camera_y: Y-coordinate in camera space.
        camera_z: Z-coordinate in camera space (depth).
    """

    camera_x: float
    camera_y: float
    camera_z: float


@dataclass(frozen=True)
class ProjectionResult:
    """Aggregated output of the Projection Engine.

    Attributes:
        objects: Tuple of projected objects.
        intrinsics: Camera intrinsic parameters used for projection.
        coordinate_system: The coordinate system of the projected coordinates.
        depth_model: Name of the depth model that provided the original depth.
    """

    objects: tuple[ProjectedObject, ...]
    intrinsics: CameraIntrinsics
    coordinate_system: CoordinateSystem
    depth_model: str
