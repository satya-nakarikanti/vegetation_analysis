from __future__ import annotations

import numpy as np

from vegetation_analysis.depth import DepthMapResult, DepthVisualizer


def test_visualizer_creates_heatmap_with_correct_shape() -> None:
    # Create a dummy depth map
    depth_array = np.random.rand(48, 64).astype(np.float32)
    result = DepthMapResult(
        depth_map=depth_array,
        image_width=64,
        image_height=48,
        model_name="test-model",
    )

    visualizer = DepthVisualizer()
    heatmap = visualizer.create_heatmap(result)

    assert heatmap.shape == (48, 64, 3)
    assert heatmap.dtype == np.uint8


def test_visualizer_handles_uniform_depth() -> None:
    # Create a uniform depth map
    depth_array = np.ones((48, 64), dtype=np.float32) * 5.0
    result = DepthMapResult(
        depth_map=depth_array,
        image_width=64,
        image_height=48,
        model_name="test-model",
    )

    visualizer = DepthVisualizer()
    heatmap = visualizer.create_heatmap(result)

    assert heatmap.shape == (48, 64, 3)
    assert heatmap.dtype == np.uint8
