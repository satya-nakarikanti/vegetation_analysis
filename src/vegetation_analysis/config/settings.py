"""Application settings loaded from environment variables."""

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class AppSettings(BaseSettings):
    """Runtime settings for the vegetation analysis module."""

    environment: str = Field(default="development", alias="VEGETATION_ENV")
    log_level: str = Field(default="INFO", alias="VEGETATION_LOG_LEVEL")
    model_dir: Path = Field(default=Path("models"), alias="VEGETATION_MODEL_DIR")
    output_dir: Path = Field(default=Path("outputs"), alias="VEGETATION_OUTPUT_DIR")
    fastsam_model_name: str = Field(
        default="FastSAM-s.pt",
        alias="VEGETATION_FASTSAM_MODEL",
    )
    fastsam_image_size: int = Field(default=1024, alias="VEGETATION_FASTSAM_IMGSZ")
    fastsam_confidence: float = Field(default=0.4, alias="VEGETATION_FASTSAM_CONF")
    fastsam_iou: float = Field(default=0.9, alias="VEGETATION_FASTSAM_IOU")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )
