from __future__ import annotations

from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
import pytest
from numpy.typing import NDArray
from PIL import Image

from vegetation_analysis.grounding import (
    DEFAULT_VEGETATION_PROMPT,
    GroundingDINODetector,
    GroundingDINOModelConfig,
    LoadedGroundingDINO,
)


class FakeInputs(dict[str, Any]):
    def __init__(self) -> None:
        super().__init__({"input_ids": [[1, 2, 3]]})
        self.device: str | None = None

    @property
    def input_ids(self) -> list[list[int]]:
        return self["input_ids"]

    def to(self, device: str) -> FakeInputs:
        self.device = device
        return self


class FakeProcessor:
    def __init__(self) -> None:
        self.inputs = FakeInputs()
        self.calls: list[dict[str, Any]] = []
        self.postprocess_calls: list[dict[str, Any]] = []

    def __call__(
        self,
        *,
        images: Image.Image,
        text: str,
        return_tensors: str,
    ) -> FakeInputs:
        self.calls.append(
            {"images": images, "text": text, "return_tensors": return_tensors}
        )
        return self.inputs

    # def post_process_grounded_object_detection(
    #     self,
    #     outputs: Any,
    #     input_ids: Any,
    #     *,
    #     threshold: float,
    #     text_threshold: float,
    #     target_sizes: list[tuple[int, int]],
    # ) -> list[dict[str, Any]]:
    #     self.postprocess_calls.append(
    #         {
    #             "outputs": outputs,
    #             "input_ids": input_ids,
    #             "threshold": threshold,
    #             "text_threshold": text_threshold,
    #             "target_sizes": target_sizes,
    #         }
    #     )
    #     return [
    #         {
    #             "boxes": [[10.2, 5.7, 40.6, 50.1], [0.0, 0.0, 3.0, 4.0]],
    #             "scores": [0.91, 0.12],
    #             "labels": ["tree", "utility pole"],
    #         }
    #     ]

    def post_process_grounded_object_detection(
        self,
        outputs: Any,
        input_ids: Any,
        *,
        threshold: float | None = None,
        box_threshold: float | None = None,
        text_threshold: float,
        target_sizes: list[tuple[int, int]],
        text_labels: Any = None,
    ) -> list[dict[str, Any]]:
        threshold = threshold if threshold is not None else box_threshold

        self.postprocess_calls.append(
            {
                "outputs": outputs,
                "input_ids": input_ids,
                "threshold": threshold,
                "text_threshold": text_threshold,
                "target_sizes": target_sizes,
            }
        )

        return [
            {
                "boxes": [[10.2, 5.7, 40.6, 50.1], [0.0, 0.0, 3.0, 4.0]],
                "scores": [0.91, 0.12],
                "labels": ["tree", "utility pole"],
                "text_labels": ["tree", "utility pole"],
            }
        ]


class FakeModel:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def __call__(self, **kwargs: Any) -> dict[str, str]:
        self.calls.append(kwargs)
        return {"raw": "outputs"}


def test_detector_accepts_numpy_image_and_returns_structured_boxes() -> None:
    processor = FakeProcessor()
    model = FakeModel()
    detector = GroundingDINODetector(
        loaded_model=_loaded_model(model=model, processor=processor),
    )

    result = detector.detect(_sample_array(), prompt=DEFAULT_VEGETATION_PROMPT)

    assert result.image_width == 64
    assert result.image_height == 48
    assert result.prompt == DEFAULT_VEGETATION_PROMPT
    assert len(result.boxes) == 1
    assert result.boxes[0].label == "tree"
    assert result.boxes[0].confidence == pytest.approx(0.91)
    assert result.boxes[0].as_xyxy() == (10, 6, 41, 48)
    assert processor.inputs.device == "cpu"
    assert model.calls == [{"input_ids": [[1, 2, 3]]}]
    assert processor.postprocess_calls[0]["target_sizes"] == [(48, 64)]


def test_detector_accepts_pil_image() -> None:
    detector = GroundingDINODetector(
        loaded_model=_loaded_model(model=FakeModel(), processor=FakeProcessor()),
    )
    image = Image.fromarray(_sample_array(), mode="RGB")

    result = detector.detect(image, prompt=DEFAULT_VEGETATION_PROMPT)

    assert result.image_width == 64
    assert result.image_height == 48


def test_detector_accepts_image_path() -> None:
    image_path = _test_output_dir() / "sample.png"
    Image.fromarray(_sample_array(), mode="RGB").save(image_path)
    detector = GroundingDINODetector(
        loaded_model=_loaded_model(model=FakeModel(), processor=FakeProcessor()),
    )

    result = detector.detect(image_path, prompt=DEFAULT_VEGETATION_PROMPT)

    assert result.image_width == 64
    assert result.image_height == 48


def test_detector_rejects_invalid_prompt() -> None:
    detector = GroundingDINODetector(
        loaded_model=_loaded_model(model=FakeModel(), processor=FakeProcessor()),
    )

    with pytest.raises(ValueError, match="Prompt"):
        detector.detect(_sample_array(), prompt="tree")


def test_detector_rejects_empty_array() -> None:
    detector = GroundingDINODetector(
        loaded_model=_loaded_model(model=FakeModel(), processor=FakeProcessor()),
    )

    with pytest.raises(ValueError, match="empty"):
        detector.detect(np.array([], dtype=np.uint8), prompt=DEFAULT_VEGETATION_PROMPT)


def _loaded_model(model: FakeModel, processor: FakeProcessor) -> LoadedGroundingDINO:
    return LoadedGroundingDINO(
        model=model,
        processor=processor,
        device="cpu",
        config=GroundingDINOModelConfig(device_preference="cpu"),
    )


def _sample_array() -> NDArray[np.uint8]:
    image = np.zeros((48, 64, 3), dtype=np.uint8)
    image[6:40, 10:45] = (40, 140, 55)
    return image


def _test_output_dir() -> Path:
    path = Path("outputs") / "test_tmp" / f"grounding_detector_{uuid4().hex}"
    path.mkdir(parents=True, exist_ok=False)
    return path
