"""Tests for SAM 2 visualization module."""

import tempfile
from pathlib import Path

import numpy as np

from vegetation_analysis.grounding.schemas import DetectionBox
from vegetation_analysis.sam2.schemas import MaskObject, SegmentationResult
from vegetation_analysis.sam2.visualization import SegmentationVisualizer


def test_draw_masks_and_boxes_empty():
    """Test drawing on an empty result returns unmodified image copy."""
    visualizer = SegmentationVisualizer()
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    result = SegmentationResult(objects=(), image_width=100, image_height=100)

    canvas = visualizer.draw_masks_and_boxes(image, result)
    assert canvas.shape == (100, 100, 3)
    assert np.array_equal(canvas, image)
    assert canvas is not image  # Ensure it's a copy


def test_draw_masks_and_boxes_with_objects():
    """Test drawing returns correctly shaped output."""
    visualizer = SegmentationVisualizer()
    image = np.zeros((100, 100, 3), dtype=np.uint8)

    box = DetectionBox(10, 10, 50, 50, "tree", 0.9)
    mask = np.zeros((100, 100), dtype=bool)
    mask[20:40, 20:40] = True

    obj = MaskObject(
        label="tree",
        confidence=0.9,
        box=box,
        mask=mask,
        mask_area_pixels=400,
        contours=None,
    )

    result = SegmentationResult(objects=(obj,), image_width=100, image_height=100)

    canvas = visualizer.draw_masks_and_boxes(image, result)
    assert canvas.shape == (100, 100, 3)
    # The canvas shouldn't be entirely black anymore
    assert np.any(canvas > 0)


def test_save_visualization():
    """Test saving a visualization to disk."""
    visualizer = SegmentationVisualizer()
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    result = SegmentationResult(objects=(), image_width=100, image_height=100)

    with tempfile.TemporaryDirectory() as temp_dir:
        output_path = Path(temp_dir) / "test.png"
        resolved_path = visualizer.save_visualization(image, result, output_path)

        assert resolved_path == output_path
        assert output_path.exists()
