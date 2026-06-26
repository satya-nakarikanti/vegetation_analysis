from __future__ import annotations

import pytest

from vegetation_analysis.segmentation import FastSAMLoader, FastSAMModelConfig


class FakeModel:
    pass


def test_loader_initializes_model_with_selected_cpu_device() -> None:
    created_with: list[str] = []

    def factory(model_name: str) -> FakeModel:
        created_with.append(model_name)
        return FakeModel()

    loader = FastSAMLoader(
        config=FastSAMModelConfig(model_name="FastSAM-s.pt", device_preference="cpu"),
        model_factory=factory,
    )

    loaded = loader.load()

    assert isinstance(loaded.model, FakeModel)
    assert loaded.device == "cpu"
    assert created_with == ["FastSAM-s.pt"]


def test_auto_device_uses_cuda_when_available(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(FastSAMLoader, "_cuda_is_available", staticmethod(lambda: True))

    assert FastSAMLoader.select_device("auto") == "cuda"


def test_cuda_request_fails_when_cuda_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        FastSAMLoader,
        "_cuda_is_available",
        staticmethod(lambda: False),
    )

    with pytest.raises(RuntimeError, match="CUDA was requested"):
        FastSAMLoader.select_device("cuda")
