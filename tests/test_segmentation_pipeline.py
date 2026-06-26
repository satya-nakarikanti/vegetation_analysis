from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray
from PIL import Image

from vegetation_analysis.segmentation import (
    MaskVisualizer,
    extract_segmented_objects,
)
from vegetation_analysis.segmentation.segmenter import (
    FastSAMInferenceConfig,
    FastSAMSegmenter,
)


@dataclass
class FakeMasks:
    data: NDArray[np.uint8]


@dataclass
class FakeResult:
    masks: FakeMasks | None


class FakeFastSAMModel:
    def __init__(self, masks: NDArray[Any]) -> None:
        self.masks = masks
        self.calls: list[dict[str, Any]] = []

    def __call__(self, image: Any, **kwargs: Any) -> list[FakeResult]:
        self.calls.append({"image": image, **kwargs})
        return [FakeResult(masks=FakeMasks(data=self.masks))]


def test_tree_like_image_generates_structured_mask_objects() -> None:
    image = _tree_like_image()
    masks = _masks_for_objects(_height_width(image), boxes=[(30, 15, 72, 90)])
    model = FakeFastSAMModel(masks=masks)
    segmenter = FastSAMSegmenter(model=model, device="cpu")

    raw_results = segmenter.segment(image)
    objects = extract_segmented_objects(raw_results)

    assert len(objects) == 1
    assert objects[0].area == 3150
    assert objects[0].bbox.as_xyxy() == (30, 15, 72, 90)
    assert objects[0].has_geometry
    assert model.calls[0]["device"] == "cpu"
    assert model.calls[0]["retina_masks"] is True


def test_pole_like_image_generates_tall_narrow_bbox() -> None:
    image = _pole_like_image()
    masks = _masks_for_objects(_height_width(image), boxes=[(52, 5, 60, 110)])
    model = FakeFastSAMModel(masks=masks)

    raw_results = FastSAMSegmenter(model=model, device="cpu").segment(image)
    objects = extract_segmented_objects(raw_results)

    assert len(objects) == 1
    assert objects[0].bbox.width == 8
    assert objects[0].bbox.height == 105
    assert objects[0].area == 840


def test_multiple_objects_generate_multiple_structured_masks() -> None:
    image = _multiple_object_image()
    masks = _masks_for_objects(
        _height_width(image),
        boxes=[(5, 5, 30, 40), (50, 10, 80, 60), (20, 65, 90, 90)],
    )
    model = FakeFastSAMModel(masks=masks)

    raw_results = FastSAMSegmenter(model=model, device="cpu").segment(image)
    objects = extract_segmented_objects(raw_results)

    assert [item.id for item in objects] == [0, 1, 2]
    assert [item.area for item in objects] == [875, 1500, 1750]


@pytest.mark.parametrize(
    ("image_name", "box"),
    [
        ("small", (2, 2, 8, 8)),
        ("large", (200, 120, 700, 500)),
    ],
)
def test_small_and_large_images_are_supported(
    image_name: str,
    box: tuple[int, int, int, int],
) -> None:
    image = _small_image() if image_name == "small" else _large_image()
    masks = _masks_for_objects(_height_width(image), boxes=[box])
    model = FakeFastSAMModel(masks=masks)

    raw_results = FastSAMSegmenter(model=model, device="cpu").segment(image)
    objects = extract_segmented_objects(raw_results)

    assert len(objects) == 1
    assert objects[0].bbox.as_xyxy() == box


def test_invalid_image_path_is_rejected(tmp_path: Path) -> None:
    invalid_image = tmp_path / "invalid.txt"
    invalid_image.write_text("not an image", encoding="utf-8")

    segmenter = FastSAMSegmenter(
        model=FakeFastSAMModel(masks=np.zeros((1, 10, 10))),
        device="cpu",
    )

    with pytest.raises(ValueError, match="Invalid image file"):
        segmenter.segment(invalid_image)


def test_empty_image_array_is_rejected() -> None:
    segmenter = FastSAMSegmenter(
        model=FakeFastSAMModel(masks=np.zeros((1, 10, 10))),
        device="cpu",
    )

    with pytest.raises(ValueError, match="empty"):
        segmenter.segment(np.array([], dtype=np.uint8))


def test_visualization_overlay_is_saved(tmp_path: Path) -> None:
    image = _multiple_object_image()
    masks = _masks_for_objects(_height_width(image), boxes=[(5, 5, 30, 40)])
    objects = extract_segmented_objects([FakeResult(masks=FakeMasks(data=masks))])
    output_path = tmp_path / "segmentation_overlay.png"

    saved_path = MaskVisualizer().save_overlay(
        image=image,
        objects=objects,
        output_path=output_path,
    )

    assert saved_path == output_path
    assert output_path.exists()
    with Image.open(output_path) as saved_image:
        assert saved_image.size == (image.shape[1], image.shape[0])


def test_inference_configuration_is_passed_to_model() -> None:
    image = _small_image()
    model = FakeFastSAMModel(
        masks=_masks_for_objects(_height_width(image), [(1, 1, 5, 5)])
    )
    config = FastSAMInferenceConfig(
        image_size=640,
        confidence=0.25,
        iou=0.75,
        retina_masks=False,
    )

    FastSAMSegmenter(model=model, device="cpu", config=config).segment(image)

    call = model.calls[0]
    assert call["imgsz"] == 640
    assert call["conf"] == 0.25
    assert call["iou"] == 0.75
    assert call["retina_masks"] is False


def _tree_like_image() -> NDArray[np.uint8]:
    image = np.zeros((120, 100, 3), dtype=np.uint8)
    image[15:90, 30:72] = (30, 140, 45)
    image[80:115, 46:56] = (110, 75, 45)
    return image


def _pole_like_image() -> NDArray[np.uint8]:
    image = np.zeros((120, 100, 3), dtype=np.uint8)
    image[5:110, 52:60] = (180, 180, 180)
    return image


def _multiple_object_image() -> NDArray[np.uint8]:
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    image[5:40, 5:30] = (20, 130, 20)
    image[10:60, 50:80] = (170, 170, 170)
    image[65:90, 20:90] = (220, 90, 30)
    return image


def _small_image() -> NDArray[np.uint8]:
    return np.zeros((10, 10, 3), dtype=np.uint8)


def _large_image() -> NDArray[np.uint8]:
    return np.zeros((720, 960, 3), dtype=np.uint8)


def _masks_for_objects(
    shape: tuple[int, int],
    boxes: list[tuple[int, int, int, int]],
) -> NDArray[np.uint8]:
    masks = np.zeros((len(boxes), shape[0], shape[1]), dtype=np.uint8)
    for index, (x_min, y_min, x_max, y_max) in enumerate(boxes):
        masks[index, y_min:y_max, x_min:x_max] = 1
    return masks


def _height_width(image: NDArray[np.uint8]) -> tuple[int, int]:
    return int(image.shape[0]), int(image.shape[1])
