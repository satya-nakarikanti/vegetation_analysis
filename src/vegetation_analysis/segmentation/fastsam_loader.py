"""FastSAM model loading and device selection."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from importlib import import_module
from typing import Any, Literal, cast

DevicePreference = Literal["auto", "cpu", "cuda"]

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FastSAMModelConfig:
    """Configuration required to initialize a FastSAM model."""

    model_name: str = "FastSAM-s.pt"
    device_preference: DevicePreference = "auto"


@dataclass(frozen=True)
class LoadedFastSAM:
    """Loaded FastSAM model and the device selected for inference."""

    model: Any
    device: str


class FastSAMLoader:
    """Load FastSAM and select the best available inference device."""

    def __init__(
        self,
        config: FastSAMModelConfig | None = None,
        model_factory: Callable[[str], Any] | None = None,
    ) -> None:
        self._config = config or FastSAMModelConfig()
        self._model_factory = model_factory

    def load(self) -> LoadedFastSAM:
        """Initialize FastSAM and return the model with its selected device."""

        device = self.select_device(self._config.device_preference)
        model_factory = self._model_factory or self._load_ultralytics_factory()

        try:
            model = model_factory(self._config.model_name)
        except Exception as exc:
            message = f"Failed to load FastSAM model '{self._config.model_name}'."
            logger.exception(message)
            raise RuntimeError(message) from exc

        logger.info(
            "Loaded FastSAM model '%s' on device '%s'.",
            self._config.model_name,
            device,
        )
        return LoadedFastSAM(model=model, device=device)

    @staticmethod
    def select_device(preference: DevicePreference = "auto") -> str:
        """Select CPU or CUDA according to availability and preference."""

        if preference == "cpu":
            return "cpu"

        cuda_available = FastSAMLoader._cuda_is_available()

        if preference == "cuda":
            if not cuda_available:
                raise RuntimeError(
                    "CUDA was requested, but no CUDA device is available."
                )
            return "cuda"

        return "cuda" if cuda_available else "cpu"

    @staticmethod
    def _cuda_is_available() -> bool:
        try:
            import torch
        except ImportError:
            return False

        return bool(torch.cuda.is_available())

    @staticmethod
    def _load_ultralytics_factory() -> Callable[[str], Any]:
        try:
            ultralytics_module = import_module("ultralytics")
        except ImportError as exc:
            message = (
                "The 'ultralytics' package is required for FastSAM integration. "
                "Install dependencies with 'pip install -r requirements-dev.txt'."
            )
            raise RuntimeError(message) from exc

        fastsam_factory = getattr(ultralytics_module, "FastSAM", None)
        if fastsam_factory is None:
            raise RuntimeError(
                "The installed 'ultralytics' package has no FastSAM API."
            )

        return cast(Callable[[str], Any], fastsam_factory)
