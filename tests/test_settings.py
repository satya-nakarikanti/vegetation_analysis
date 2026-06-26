from pathlib import Path

from vegetation_analysis.config import AppSettings


def test_default_settings() -> None:
    settings = AppSettings()

    assert settings.environment == "development"
    assert settings.log_level == "INFO"
    assert settings.model_dir == Path("models")
    assert settings.output_dir == Path("outputs")
    assert settings.fastsam_model_name == "FastSAM-s.pt"
    assert settings.fastsam_image_size == 1024
    assert settings.fastsam_confidence == 0.4
    assert settings.fastsam_iou == 0.9
