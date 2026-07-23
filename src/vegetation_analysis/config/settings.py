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


    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )
