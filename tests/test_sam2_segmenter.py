"""Tests for SAM 2 segmenter module."""

import numpy as np
from PIL import Image

from vegetation_analysis.grounding.schemas import DetectionBox, DetectionResult
from vegetation_analysis.sam2.sam2_loader import LoadedSAM2, SAM2ModelConfig
from vegetation_analysis.sam2.segmenter import SAM2Segmenter


class MockPredictor:
    def set_image(self, image):
        self.image = image

    def predict(self, point_coords, point_labels, box, multimask_output):
        N = len(box)
        H, W = self.image.shape[:2]
        # Return dummy masks (N, 1, H, W) and scores (N, 1)
        masks = np.zeros((N, 1, H, W), dtype=bool)
        scores = np.ones((N, 1), dtype=float) * 0.9
        logits = np.zeros((N, 1, H, W), dtype=float)

        # Add some true pixels to the masks to give non-zero area
        masks[:, 0, 10:20, 10:20] = True

        return masks, scores, logits


def test_segmenter_empty_result():
    """Test segmentation correctly skips on empty detection results."""
    config = SAM2ModelConfig()
    mock_model = LoadedSAM2(predictor=MockPredictor(), device="cpu", config=config)
    segmenter = SAM2Segmenter(mock_model)

    detection_result = DetectionResult(
        boxes=(),
        image_width=640,
        image_height=480,
        prompt="test",
    )

    image = np.zeros((480, 640, 3), dtype=np.uint8)
    result = segmenter.segment(image, detection_result)

    assert result.is_empty
    assert result.image_width == 640
    assert result.image_height == 480
    assert len(result.objects) == 0


def test_segmenter_valid_boxes():
    """Test segmentation returns valid MaskObjects."""
    config = SAM2ModelConfig()
    mock_model = LoadedSAM2(predictor=MockPredictor(), device="cpu", config=config)
    segmenter = SAM2Segmenter(mock_model)

    box = DetectionBox(
        x_min=10, y_min=10, x_max=50, y_max=50, label="tree", confidence=0.8
    )
    detection_result = DetectionResult(
        boxes=(box,),
        image_width=640,
        image_height=480,
        prompt="tree",
    )

    image = Image.new("RGB", (640, 480))
    result = segmenter.segment(image, detection_result)

    assert not result.is_empty
    assert len(result.objects) == 1

    obj = result.objects[0]
    assert obj.label == "tree"
    assert obj.box == box
    assert obj.mask.shape == (480, 640)
    assert obj.mask_area_pixels == 100  # 10x10 block we mocked
    assert obj.confidence == 0.9  # Returned by MockPredictor
