"""Unit tests for GeometryVisualizer and DepthVisualizer.save_grayscale."""

from __future__ import annotations

import numpy as np

from vegetation_analysis.depth.schemas import DepthMapResult
from vegetation_analysis.depth.visualization import DepthVisualizer
from vegetation_analysis.geometry import (
    GeometryEngine,
    GeometryResult,
    GeometryVisualizer,
)
from vegetation_analysis.geometry.schemas import GeometryImageMetadata
from vegetation_analysis.grounding.schemas import DetectionBox
from vegetation_analysis.projection.schemas import (
    CameraIntrinsics,
    CoordinateSystem,
    ProjectedObject,
    ProjectionResult,
)


def _make_depth_result(height: int = 20, width: int = 30) -> DepthMapResult:
    n = height * width
    depth_map = np.linspace(0.0, 5.0, n, dtype=np.float32).reshape(height, width)
    return DepthMapResult(
        depth_map=depth_map,
        image_width=width,
        image_height=height,
        model_name="test-model",
    )


def _make_geometry_result(width: int = 30, height: int = 20) -> GeometryResult:
    obj = ProjectedObject(
        label="tree",
        confidence=0.9,
        pixel_count=100,
        centroid_x=15.0,
        centroid_y=10.0,
        median_depth=2.0,
        mean_depth=2.0,
        std_depth=0.1,
        min_depth=1.8,
        max_depth=2.2,
        bounding_box=DetectionBox(0, 0, 10, 10, "tree", 0.9),
        original_mask=np.zeros((10, 10), dtype=bool),
        sampling_mask=np.zeros((10, 10), dtype=bool),
        contours=(),
        mask_area_pixels=100,
        camera_x=0.0,
        camera_y=0.0,
        camera_z=2.0,
    )
    intrinsics = CameraIntrinsics(
        fx=1000.0,
        fy=1000.0,
        cx=width / 2.0,
        cy=height / 2.0,
        image_width=width,
        image_height=height,
    )
    projection = ProjectionResult(
        objects=(obj,),
        intrinsics=intrinsics,
        coordinate_system=CoordinateSystem.RELATIVE,
        depth_model="test",
    )
    return GeometryEngine().compute(projection)


def test_geometry_visualizer_output_shape() -> None:
    depth = _make_depth_result(height=20, width=30)
    geo = _make_geometry_result(width=30, height=20)

    visualizer = GeometryVisualizer()
    canvas = visualizer.draw_overlays(depth_map=depth, geometry_result=geo)

    assert canvas.shape == (20, 30, 3)
    assert canvas.dtype == np.uint8


def test_geometry_visualizer_empty_result() -> None:
    depth = _make_depth_result(height=20, width=30)
    empty_geo = GeometryResult(
        objects=(),
        metadata=GeometryImageMetadata(
            width=30,
            height=20,
            principal_x=15.0,
            principal_y=10.0,
            depth_model="test",
            coordinate_system=CoordinateSystem.RELATIVE,
        ),
    )

    canvas = GeometryVisualizer().draw_overlays(
        depth_map=depth, geometry_result=empty_geo
    )

    assert canvas.shape == (20, 30, 3)


def test_depth_visualizer_grayscale_shape_and_type() -> None:
    depth = _make_depth_result(height=20, width=30)
    grayscale = DepthVisualizer().create_grayscale(depth)

    assert grayscale.shape == (20, 30)
    assert grayscale.dtype == np.uint8


def test_depth_visualizer_grayscale_range() -> None:
    """Grayscale output should span [0, 255] for a non-uniform depth map."""
    depth = _make_depth_result(height=20, width=30)
    grayscale = DepthVisualizer().create_grayscale(depth)

    assert int(grayscale.min()) == 0
    assert int(grayscale.max()) == 255


def test_depth_visualizer_grayscale_uniform_depth() -> None:
    """A uniform depth map should produce an all-zero grayscale image."""
    depth_map = np.ones((20, 30), dtype=np.float32) * 3.0
    result = DepthMapResult(
        depth_map=depth_map,
        image_width=30,
        image_height=20,
        model_name="test",
    )
    grayscale = DepthVisualizer().create_grayscale(result)

    assert np.all(grayscale == 0)
