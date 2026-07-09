from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import numpy as np
import pytest
from numpy.typing import NDArray
from PIL import Image

from vegetation_analysis.depth import DepthEstimator
from vegetation_analysis.depth.depth_loader import (
    DepthAnythingModelConfig,
    LoadedDepthAnything,
)


class FakeModel:
    def __init__(self) -> None:
        pass


class FakeProcessor:
    def __init__(self) -> None:
        pass


def _loaded_model() -> LoadedDepthAnything:
    return LoadedDepthAnything(
        model=FakeModel(),
        processor=FakeProcessor(),
        device="cpu",
        config=DepthAnythingModelConfig(device_preference="cpu"),
    )


def _sample_array() -> NDArray[np.uint8]:
    image = np.zeros((48, 64, 3), dtype=np.uint8)
    image[6:40, 10:45] = (40, 140, 55)
    return image


def _test_output_dir() -> Path:
    path = Path("outputs") / "test_tmp" / f"depth_estimator_{uuid4().hex}"
    path.mkdir(parents=True, exist_ok=False)
    return path


def test_estimator_prepare_image_accepts_numpy_array() -> None:
    estimator = DepthEstimator(loaded_model=_loaded_model())
    image = estimator._prepare_image(_sample_array())

    assert image.size == (64, 48)
    assert image.mode == "RGB"


def test_estimator_prepare_image_accepts_pil_image() -> None:
    estimator = DepthEstimator(loaded_model=_loaded_model())
    pil_image = Image.fromarray(_sample_array(), mode="RGB")
    image = estimator._prepare_image(pil_image)

    assert image.size == (64, 48)
    assert image.mode == "RGB"


def test_estimator_prepare_image_accepts_path() -> None:
    image_path = _test_output_dir() / "sample.png"
    Image.fromarray(_sample_array(), mode="RGB").save(image_path)

    estimator = DepthEstimator(loaded_model=_loaded_model())
    image = estimator._prepare_image(image_path)

    assert image.size == (64, 48)
    assert image.mode == "RGB"


def test_estimator_prepare_image_rejects_empty_array() -> None:
    estimator = DepthEstimator(loaded_model=_loaded_model())
    with pytest.raises(ValueError, match="empty"):
        estimator._prepare_image(np.array([], dtype=np.uint8))
