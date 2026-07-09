"""Unit tests for the Relative Geometry Engine."""

from __future__ import annotations

import pytest

from vegetation_analysis.depth_sampling.schemas import (
    DepthSamplingResult,
    ImageMetadata,
    SampledObject,
)
from vegetation_analysis.geometry import GeometryResult, RelativeGeometryEngine


def _make_sampled_object(
    label: str = "tree",
    centroid_x: float = 320.0,
    centroid_y: float = 240.0,
    median_depth: float = 1.5,
    confidence: float = 0.9,
    pixel_count: int = 500,
) -> SampledObject:
    return SampledObject(
        label=label,
        confidence=confidence,
        pixel_count=pixel_count,
        centroid_x=centroid_x,
        centroid_y=centroid_y,
        median_depth=median_depth,
        mean_depth=median_depth,
        std_depth=0.1,
        min_depth=median_depth - 0.2,
        max_depth=median_depth + 0.2,
    )


def _make_sampling_result(
    objects: tuple[SampledObject, ...],
    width: int = 640,
    height: int = 480,
) -> DepthSamplingResult:
    return DepthSamplingResult(
        objects=objects,
        metadata=ImageMetadata(
            width=width,
            height=height,
            depth_model="test-model",
        ),
    )


def test_engine_produces_geometry_result() -> None:
    obj = _make_sampled_object(centroid_x=320.0, centroid_y=240.0)
    sampling = _make_sampling_result(objects=(obj,), width=640, height=480)

    engine = RelativeGeometryEngine()
    result = engine.compute(sampling)

    assert isinstance(result, GeometryResult)
    assert len(result.objects) == 1


def test_centred_object_has_zero_relative_xy() -> None:
    """An object at the image centre should have relative_x = relative_y = 0."""
    obj = _make_sampled_object(centroid_x=320.0, centroid_y=240.0)
    sampling = _make_sampling_result(objects=(obj,), width=640, height=480)

    result = RelativeGeometryEngine().compute(sampling)
    geo = result.objects[0]

    assert geo.relative_x == pytest.approx(0.0, abs=1e-6)
    assert geo.relative_y == pytest.approx(0.0, abs=1e-6)


def test_relative_x_range_for_edge_objects() -> None:
    """Objects at the left/right edges should have relative_x near -0.5 / +0.5."""
    left = _make_sampled_object(label="left", centroid_x=0.0, centroid_y=240.0)
    right = _make_sampled_object(label="right", centroid_x=640.0, centroid_y=240.0)
    sampling = _make_sampling_result(objects=(left, right), width=640, height=480)

    result = RelativeGeometryEngine().compute(sampling)
    geo_left, geo_right = result.objects

    assert geo_left.relative_x == pytest.approx(-1.0, abs=1e-6)
    assert geo_right.relative_x == pytest.approx(1.0, abs=1e-6)


def test_relative_z_equals_median_depth() -> None:
    """relative_z must equal the median depth from Phase 5.2."""
    obj = _make_sampled_object(median_depth=3.14)
    sampling = _make_sampling_result(objects=(obj,))

    result = RelativeGeometryEngine().compute(sampling)

    assert result.objects[0].relative_z == pytest.approx(3.14)


def test_species_fields_are_none_by_default() -> None:
    obj = _make_sampled_object()
    sampling = _make_sampling_result(objects=(obj,))

    result = RelativeGeometryEngine().compute(sampling)
    geo = result.objects[0]

    assert geo.species is None
    assert geo.species_confidence is None


def test_metadata_contains_image_dimensions_and_principal_point() -> None:
    sampling = _make_sampling_result(objects=(), width=800, height=600)
    result = RelativeGeometryEngine().compute(sampling)

    assert result.metadata.width == 800
    assert result.metadata.height == 600
    assert result.metadata.principal_x == pytest.approx(400.0)
    assert result.metadata.principal_y == pytest.approx(300.0)


def test_empty_sampling_result_produces_empty_geometry() -> None:
    sampling = _make_sampling_result(objects=())
    result = RelativeGeometryEngine().compute(sampling)

    assert result.objects == ()


def test_multiple_objects_are_all_processed() -> None:
    objects = tuple(
        _make_sampled_object(
            label=f"obj{i}", centroid_x=float(i * 100), centroid_y=240.0
        )
        for i in range(5)
    )
    sampling = _make_sampling_result(objects=objects, width=640, height=480)
    result = RelativeGeometryEngine().compute(sampling)

    assert len(result.objects) == 5
