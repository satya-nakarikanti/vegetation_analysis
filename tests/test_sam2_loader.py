"""Tests for SAM 2 loader module."""

import pytest

from vegetation_analysis.sam2.sam2_loader import SAM2Loader, SAM2ModelConfig


def test_config_validation():
    """Test SAM 2 config validation logic."""
    with pytest.raises(ValueError, match="model_cfg must not be empty"):
        SAM2ModelConfig(model_cfg="")

    with pytest.raises(ValueError, match="checkpoint must not be empty"):
        SAM2ModelConfig(checkpoint="  ")

    valid_config = SAM2ModelConfig()
    assert valid_config.model_cfg == "configs/sam2.1/sam2.1_hiera_t.yaml"
    assert valid_config.checkpoint == "checkpoints/sam2.1_hiera_tiny.pt"


def test_select_device_cpu():
    """Test device selection explicitly requesting CPU."""
    assert SAM2Loader.select_device("cpu") == "cpu"


def test_select_device_cuda_without_gpu(monkeypatch):
    """Test device selection explicitly requesting CUDA when unavailable."""

    def mock_cuda_is_available():
        return False

    monkeypatch.setattr(SAM2Loader, "_cuda_is_available", mock_cuda_is_available)

    with pytest.raises(RuntimeError, match="no CUDA device is available"):
        SAM2Loader.select_device("cuda")


def test_select_device_auto_fallback(monkeypatch):
    """Test automatic device selection falling back to CPU."""

    def mock_cuda_is_available():
        return False

    monkeypatch.setattr(SAM2Loader, "_cuda_is_available", mock_cuda_is_available)
    assert SAM2Loader.select_device("auto") == "cpu"


def test_load_predictor_failure(monkeypatch):
    """Test load failure when sam2 is not installed or checkpoints are missing."""

    def mock_import_module(name):
        raise ImportError("No module named sam2")

    monkeypatch.setattr("importlib.import_module", mock_import_module)
    loader = SAM2Loader(SAM2ModelConfig())

    with pytest.raises(RuntimeError, match="The 'sam2' package is required"):
        loader.load()
