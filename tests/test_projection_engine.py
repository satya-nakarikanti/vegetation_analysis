"""Unit tests for the Projection Engine."""

import numpy as np
import pytest

from vegetation_analysis.depth_sampling.schemas import (
    DepthSamplingResult,
    ImageMetadata,
    SampledObject,
)
from vegetation_analysis.grounding.schemas import DetectionBox
from vegetation_analysis.projection import ProjectionEngine
from vegetation_analysis.projection.schemas import (
    CameraIntrinsics,
    CoordinateSystem,
)


def _make_sampled_object(
    centroid_x: float = 320.0,
    centroid_y: float = 240.0,
    median_depth: float = 10.0,
) -> SampledObject:
    return SampledObject(
        label="tree",
        confidence=0.9,
        pixel_count=100,
        centroid_x=centroid_x,
        centroid_y=centroid_y,
        median_depth=median_depth,
        mean_depth=median_depth,
        std_depth=0.0,
        min_depth=median_depth,
        max_depth=median_depth,
        bounding_box=DetectionBox(0, 0, 10, 10, "tree", 0.9),
        original_mask=np.zeros((10, 10), dtype=bool),
        sampling_mask=np.zeros((10, 10), dtype=bool),
        contours=(),
        mask_area_pixels=100,
    )


def test_projection_pinhole_math() -> None:
    engine = ProjectionEngine(coordinate_system=CoordinateSystem.METRIC)
    obj = _make_sampled_object(centroid_x=420.0, centroid_y=340.0, median_depth=5.0)
    sampling_result = DepthSamplingResult(
        objects=(obj,),
        metadata=ImageMetadata(width=640, height=480, depth_model="test"),
    )
    intrinsics = CameraIntrinsics(
        fx=1000.0, fy=1000.0, cx=320.0, cy=240.0, image_width=640, image_height=480
    )

    result = engine.project(sampling_result, intrinsics)
    assert len(result.objects) == 1
    proj_obj = result.objects[0]

    # Z = depth
    assert proj_obj.camera_z == 5.0
    # X = (u - cx) * Z / fx = (420 - 320) * 5.0 / 1000.0 = 100 * 5 / 1000 = 0.5
    assert proj_obj.camera_x == pytest.approx(0.5)
    # Y = (v - cy) * Z / fy = (340 - 240) * 5.0 / 1000.0 = 100 * 5 / 1000 = 0.5
    assert proj_obj.camera_y == pytest.approx(0.5)

    assert result.coordinate_system == CoordinateSystem.METRIC
    assert result.depth_model == "test"
