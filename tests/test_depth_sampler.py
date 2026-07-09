from __future__ import annotations

import numpy as np
import pytest

from vegetation_analysis.depth.schemas import DepthMapResult
from vegetation_analysis.depth_sampling import DepthSampler
from vegetation_analysis.grounding.schemas import DetectionBox
from vegetation_analysis.sam2.schemas import MaskObject, SegmentationResult


def _create_mock_data(
    mask_size: int = 10, fill_depth: float = 2.0
) -> tuple[SegmentationResult, DepthMapResult]:
    # 20x20 image
    image_width, image_height = 20, 20

    # Create a depth map filled with `fill_depth`
    depth_map = np.full((image_height, image_width), fill_depth, dtype=np.float32)

    # Create a mask that is True in the top-left mask_size x mask_size square
    mask = np.zeros((image_height, image_width), dtype=bool)
    mask[0:mask_size, 0:mask_size] = True

    obj = MaskObject(
        label="tree",
        confidence=0.9,
        box=DetectionBox(0, 0, mask_size, mask_size, "tree", 0.9),
        mask=mask,
        mask_area_pixels=mask_size * mask_size,
    )

    seg_result = SegmentationResult(
        objects=(obj,),
        image_width=image_width,
        image_height=image_height,
    )

    depth_result = DepthMapResult(
        depth_map=depth_map,
        image_width=image_width,
        image_height=image_height,
        model_name="test-model",
    )

    return seg_result, depth_result


def test_sampler_computes_correct_stats() -> None:
    seg_result, depth_result = _create_mock_data(mask_size=10, fill_depth=3.5)

    # Use small kernel so erosion doesn't completely erase the 10x10 mask
    sampler = DepthSampler(
        erosion_kernel_size=3, erosion_iterations=1, min_eroded_pixels=10
    )
    result = sampler.sample(seg_result, depth_result)

    assert len(result.objects) == 1
    obj = result.objects[0]

    assert obj.label == "tree"
    assert obj.confidence == 0.9
    assert obj.median_depth == 3.5
    assert obj.mean_depth == 3.5
    assert obj.min_depth == 3.5
    assert obj.max_depth == 3.5
    assert obj.std_depth == 0.0

    # Centroid of 0..9 is 4.5
    assert obj.centroid_x == pytest.approx(4.5)
    assert obj.centroid_y == pytest.approx(4.5)


def test_sampler_falls_back_when_eroded_mask_too_small() -> None:
    seg_result, depth_result = _create_mock_data(mask_size=4, fill_depth=2.0)

    # 4x4 mask eroded with 5x5 kernel will be completely empty (0 pixels).
    # It should fallback to the original mask which has 16 pixels.
    sampler = DepthSampler(
        erosion_kernel_size=5, erosion_iterations=1, min_eroded_pixels=10
    )
    result = sampler.sample(seg_result, depth_result)

    assert len(result.objects) == 1
    obj = result.objects[0]

    assert obj.pixel_count == 16
    assert obj.median_depth == 2.0


def test_sampler_raises_on_dimension_mismatch() -> None:
    seg_result, _ = _create_mock_data()
    _, depth_result = _create_mock_data()

    # Force a mismatch
    seg_result = SegmentationResult(
        objects=seg_result.objects,
        image_width=100,
        image_height=100,
    )

    sampler = DepthSampler()
    with pytest.raises(ValueError, match="Dimension mismatch"):
        sampler.sample(seg_result, depth_result)
