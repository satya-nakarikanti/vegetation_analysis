"""SAM 2 model loading and device selection.

This module is responsible for:
- Validating the provided configuration.
- Selecting the appropriate inference device (CPU or CUDA).
- Loading the SAM 2 model using Meta's official API.
- Returning a LoadedSAM2 container.

No inference logic is performed here.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from importlib import import_module
from typing import Any

from vegetation_analysis.sam2.constants import (
    DEFAULT_CHECKPOINT,
    DEFAULT_DEVICE_PREFERENCE,
    DEFAULT_MODEL_CFG,
    DevicePreference,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SAM2ModelConfig:
    """Configuration required to initialise a SAM 2 model.

    Attributes:
        model_cfg: The SAM 2 configuration file name (e.g. 'sam2_hiera_t.yaml').
        checkpoint: The path to the model checkpoint.
        device_preference: Inference device selection strategy.
            - "auto" — selects CUDA when available, falls back to CPU.
            - "cpu"  — forces CPU.
            - "cuda" — forces CUDA; raises RuntimeError when no GPU is detected.
    """

    model_cfg: str = DEFAULT_MODEL_CFG
    checkpoint: str = DEFAULT_CHECKPOINT
    device_preference: DevicePreference = DEFAULT_DEVICE_PREFERENCE

    def __post_init__(self) -> None:
        if not self.model_cfg.strip():
            raise ValueError("model_cfg must not be empty.")
        if not self.checkpoint.strip():
            raise ValueError("checkpoint must not be empty.")


@dataclass(frozen=True)
class LoadedSAM2:
    """Container for a successfully loaded SAM 2 predictor.

    Attributes:
        predictor: The SAM2ImagePredictor instance.
        device: The device string on which the model resides ("cpu" or "cuda").
        config: The configuration used.
    """

    predictor: Any
    device: str
    config: SAM2ModelConfig


class SAM2Loader:
    """Load SAM 2 and select the best available inference device.

    Args:
        config: Model configuration. Defaults to SAM2ModelConfig().
    """

    def __init__(self, config: SAM2ModelConfig | None = None) -> None:
        self._config = config or SAM2ModelConfig()

    def load(self) -> LoadedSAM2:
        """Initialise SAM 2 and return a container with the predictor.

        Returns:
            LoadedSAM2 holding the predictor, device, and config.

        Raises:
            RuntimeError: If sam2 or torch is not installed, or if the
                model cannot be loaded.
        """
        device = self.select_device(self._config.device_preference)

        logger.info(
            "Loading SAM 2 model with config '%s' from '%s' on device '%s'.",
            self._config.model_cfg,
            self._config.checkpoint,
            device,
        )

        predictor = self._load_predictor(
            model_cfg=self._config.model_cfg,
            checkpoint=self._config.checkpoint,
            device=device,
        )

        logger.info("SAM 2 model loaded successfully.")

        return LoadedSAM2(
            predictor=predictor,
            device=device,
            config=self._config,
        )

    @staticmethod
    def select_device(preference: DevicePreference = "auto") -> str:
        """Select CPU or CUDA according to availability and caller preference."""
        if preference == "cpu":
            return "cpu"

        cuda_available = SAM2Loader._cuda_is_available()

        if preference == "cuda":
            if not cuda_available:
                raise RuntimeError(
                    "CUDA was requested, but no CUDA device is available."
                )
            return "cuda"

        selected = "cuda" if cuda_available else "cpu"
        logger.debug("Auto device selection resolved to '%s'.", selected)
        return selected

    @staticmethod
    def _load_predictor(model_cfg: str, checkpoint: str, device: str) -> Any:
        try:
            build_sam2_module = import_module("sam2.build_sam")
            predictor_module = import_module("sam2.sam2_image_predictor")
        except ImportError as exc:
            raise RuntimeError(
                "The 'sam2' package is required. "
                "Install it with: pip install git+https://github.com/facebookresearch/sam2.git"
            ) from exc

        build_sam2 = getattr(build_sam2_module, "build_sam2", None)
        SAM2ImagePredictor = getattr(predictor_module, "SAM2ImagePredictor", None)

        if build_sam2 is None or SAM2ImagePredictor is None:
            raise RuntimeError(
                "The installed 'sam2' package does not expose the required APIs."
            )

        try:
            model = build_sam2(model_cfg, checkpoint, device=device)
            predictor = SAM2ImagePredictor(model)
        except Exception as exc:
            message = (
                f"Failed to load SAM 2 from cfg '{model_cfg}' and checkpoint "
                f"'{checkpoint}'. Verify the paths and files exist."
            )
            logger.exception(message)
            raise RuntimeError(message) from exc

        return predictor

    @staticmethod
    def _cuda_is_available() -> bool:
        try:
            import torch
        except ImportError:
            return False
        return bool(torch.cuda.is_available())
