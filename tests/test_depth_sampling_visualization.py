from __future__ import annotations

import numpy as np

from vegetation_analysis.depth.schemas import DepthMapResult
from vegetation_analysis.depth_sampling.schemas import (
    DepthSamplingResult,
    ImageMetadata,
    SampledObject,
)
from vegetation_analysis.depth_sampling.visualization import DepthSamplingVisualizer


def test_sampling_visualizer_creates_overlay() -> None:
    # 20x20 image
    image_width, image_height = 20, 20

    depth_map = np.zeros((image_height, image_width), dtype=np.float32)
    depth_result = DepthMapResult(
        depth_map=depth_map,
        image_width=image_width,
        image_height=image_height,
        model_name="test-model",
    )

    sampled_obj = SampledObject(
        label="tree",
        confidence=0.9,
        pixel_count=100,
        centroid_x=10.0,
        centroid_y=10.0,
        median_depth=5.0,
        mean_depth=5.0,
        std_depth=0.1,
        min_depth=4.9,
        max_depth=5.1,
    )

    sampling_result = DepthSamplingResult(
        objects=(sampled_obj,),
        metadata=ImageMetadata(
            width=image_width,
            height=image_height,
            depth_model="test-model",
        ),
    )

    visualizer = DepthSamplingVisualizer()
    canvas = visualizer.draw_overlays(
        depth_map=depth_result, sampling_result=sampling_result
    )

    assert canvas.shape == (20, 20, 3)
    assert canvas.dtype == np.uint8
