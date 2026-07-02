from __future__ import annotations

import pytest

from vegetation_analysis.grounding import (
    DEFAULT_BOX_THRESHOLD,
    DEFAULT_MODEL_ID,
    DEFAULT_TEXT_THRESHOLD,
    GroundingDINOLoader,
    GroundingDINOModelConfig,
)


def test_model_config_uses_shared_defaults() -> None:
    config = GroundingDINOModelConfig()

    assert config.model_id == DEFAULT_MODEL_ID
    assert config.device_preference == "auto"
    assert config.box_threshold == DEFAULT_BOX_THRESHOLD
    assert config.text_threshold == DEFAULT_TEXT_THRESHOLD


@pytest.mark.parametrize("field", ["box_threshold", "text_threshold"])
def test_model_config_rejects_invalid_threshold(field: str) -> None:
    with pytest.raises(ValueError, match=field):
        GroundingDINOModelConfig(**{field: 1.5})


def test_model_config_rejects_empty_model_id() -> None:
    with pytest.raises(ValueError, match="model_id"):
        GroundingDINOModelConfig(model_id=" ")


def test_auto_device_uses_cuda_when_available(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        GroundingDINOLoader,
        "_cuda_is_available",
        staticmethod(lambda: True),
    )

    assert GroundingDINOLoader.select_device("auto") == "cuda"


def test_cuda_request_fails_when_cuda_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        GroundingDINOLoader,
        "_cuda_is_available",
        staticmethod(lambda: False),
    )

    with pytest.raises(RuntimeError, match="CUDA was requested"):
        GroundingDINOLoader.select_device("cuda")
