from __future__ import annotations

import pytest

from vegetation_analysis.depth import (
    DEFAULT_MODEL_ID,
    DepthAnythingLoader,
    DepthAnythingModelConfig,
)


def test_model_config_uses_shared_defaults() -> None:
    config = DepthAnythingModelConfig()

    assert config.model_id == DEFAULT_MODEL_ID
    assert config.device_preference == "auto"


def test_model_config_rejects_empty_model_id() -> None:
    with pytest.raises(ValueError, match="model_id"):
        DepthAnythingModelConfig(model_id=" ")


def test_auto_device_uses_cuda_when_available(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        DepthAnythingLoader,
        "_cuda_is_available",
        staticmethod(lambda: True),
    )

    assert DepthAnythingLoader.select_device("auto") == "cuda"


def test_cuda_request_fails_when_cuda_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        DepthAnythingLoader,
        "_cuda_is_available",
        staticmethod(lambda: False),
    )

    with pytest.raises(RuntimeError, match="CUDA was requested"):
        DepthAnythingLoader.select_device("cuda")
