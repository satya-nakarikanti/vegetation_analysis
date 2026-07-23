from pathlib import Path

from vegetation_analysis.config import AppSettings


def test_default_settings() -> None:
    settings = AppSettings()

    assert settings.environment == "development"
    assert settings.log_level == "INFO"
    assert settings.model_dir == Path("models")
    assert settings.output_dir == Path("outputs")
