"""Depth Anything V2 model loading and device selection.

This module is the sole entry point for initialising a Depth Anything V2 model.
It is responsible for:

- Validating the provided configuration.
- Selecting the appropriate inference device (CPU or CUDA).
- Loading the official Metric Depth Anything V2 model from the local repository.
- Returning a :class:`LoadedDepthAnything` container that other modules can
  use without re-loading the model.

The loader intentionally performs no inference. Detection logic lives
exclusively in :mod:`vegetation_analysis.depth.estimator`.
"""

from __future__ import annotations

import importlib
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from vegetation_analysis.depth.constants import (
    DEFAULT_DEVICE_PREFERENCE,
    DEFAULT_MODEL_ID,
)

DevicePreference = Literal["auto", "cpu", "cuda"]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DepthAnythingModelConfig:
    """Configuration required to initialise a Depth Anything V2 model.

    Attributes:
        model_id: Absolute or relative path to the local checkpoint file (.pth).
        device_preference: Inference device selection strategy.
            - ``"auto"`` — selects CUDA when available, falls back to CPU.
            - ``"cpu"``  — forces CPU regardless of available hardware.
            - ``"cuda"`` — forces CUDA; raises :class:`RuntimeError` when no
              GPU is detected.
        encoder: The encoder size used by the model checkpoint.
            Supported values are 'vits', 'vitb', 'vitl', 'vitg'.
        max_depth: The maximum metric depth scale configured during training (in meters).
    """

    model_id: str = DEFAULT_MODEL_ID
    device_preference: DevicePreference = DEFAULT_DEVICE_PREFERENCE
    encoder: str = "vits"
    max_depth: float = 20.0

    def __post_init__(self) -> None:
        """Validate configuration values after dataclass initialisation."""
        if not self.model_id.strip():
            raise ValueError("model_id must not be empty or whitespace-only.")
        if self.encoder not in {"vits", "vitb", "vitl", "vitg"}:
            raise ValueError(f"Unsupported encoder: {self.encoder}")
        if self.max_depth <= 0.0:
            raise ValueError("max_depth must be positive.")


# ---------------------------------------------------------------------------
# Loaded-model container
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LoadedDepthAnything:
    """Container for a successfully loaded Depth Anything V2 model.

    This dataclass is the immutable result of :meth:`DepthAnythingLoader.load`.
    It carries everything the estimator needs to run inference without
    reloading the model.

    Attributes:
        model: The underlying metric depth estimation model object.
        processor: Left as None (preserved for interface compatibility).
        device: The device string on which the model resides (``"cpu"`` or
            ``"cuda"``).
        config: The :class:`DepthAnythingModelConfig` used to produce this
            container, preserved for auditability.
    """

    model: Any
    processor: Any | None
    device: str
    config: DepthAnythingModelConfig


# ---------------------------------------------------------------------------
# Loader class
# ---------------------------------------------------------------------------


class DepthAnythingLoader:
    """Load Depth Anything V2 and select the best available inference device.

    The loader is intentionally decoupled from estimation so that model
    initialisation can be exercised and tested independently of inference.

    A single loader instance represents a single configuration. Call
    :meth:`load` to obtain a :class:`LoadedDepthAnything` container. The
    container can then be passed to
    :class:`~vegetation_analysis.depth.estimator.DepthEstimator`.

    Args:
        config: Model configuration. Defaults to
            :class:`DepthAnythingModelConfig` with factory defaults from
            :mod:`vegetation_analysis.depth.constants`.
    """

    def __init__(self, config: DepthAnythingModelConfig | None = None) -> None:
        self._config = config or DepthAnythingModelConfig()

        self._model_configs = {
            "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
            "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
            "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
            "vitg": {"encoder": "vitg", "features": 384, "out_channels": [1536, 1536, 1536, 1536]},
        }

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def load(self) -> LoadedDepthAnything:
        """Initialise Depth Anything V2 and return an immutable model container.

        The method performs the following steps in order:

        1. Resolve the target device using :meth:`select_device`.
        2. Load the official DepthAnythingV2 model class from the third_party repository.
        3. Instantiate the model with encoder specifications and max_depth.
        4. Load the weights from the .pth file.
        5. Move it to the selected device and set it to evaluation mode.
        6. Return a :class:`LoadedDepthAnything` container.

        Returns:
            A :class:`LoadedDepthAnything` holding the model,
            device string, and the configuration used.

        Raises:
            RuntimeError: If the model checkpoint cannot be fetched or loaded.
            RuntimeError: If ``device_preference="cuda"`` but no CUDA device
                is available.
        """

        device = self.select_device(self._config.device_preference)

        logger.info(
            "Loading Metric Depth Anything V2 model '%s' (encoder=%s) on device '%s'.",
            self._config.model_id,
            self._config.encoder,
            device,
        )

        model = self._load_model(
            model_id=self._config.model_id,
            encoder=self._config.encoder,
            max_depth=self._config.max_depth,
            device=device,
        )

        logger.info(
            "Metric Depth Anything V2 model '%s' loaded successfully.",
            self._config.model_id,
        )

        return LoadedDepthAnything(
            model=model,
            processor=None,
            device=device,
            config=self._config,
        )

    @staticmethod
    def select_device(preference: DevicePreference = "auto") -> str:
        """Select CPU or CUDA according to availability and caller preference."""
        if preference == "cpu":
            return "cpu"

        cuda_available = DepthAnythingLoader._cuda_is_available()

        if preference == "cuda":
            if not cuda_available:
                raise RuntimeError(
                    "CUDA was requested, but no CUDA device is available."
                )
            return "cuda"

        selected = "cuda" if cuda_available else "cpu"
        logger.debug("Auto device selection resolved to '%s'.", selected)
        return selected

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_model(
        self,
        model_id: str,
        encoder: str,
        max_depth: float,
        device: str,
    ) -> Any:
        """Load the official Metric Depth Anything V2 model.

        Args:
            model_id: Local path to the .pth checkpoint file.
            encoder: The encoder architecture string.
            max_depth: The metric maximum depth value.
            device: Target device string (``"cpu"`` or ``"cuda"``).

        Returns:
            The initialised and loaded model in eval mode.
        """
        import torch
        import os

        # Isolate the sys.path modification required to import the module
        repo_root = Path(__file__).resolve().parents[3]
        metric_depth_path = repo_root / "third_party" / "Depth-Anything-V2" / "metric_depth"
        
        path_added = False
        if str(metric_depth_path) not in sys.path:
            sys.path.insert(0, str(metric_depth_path))
            path_added = True
            
        try:
            dpt_module = importlib.import_module("depth_anything_v2.dpt")
            DepthAnythingV2 = getattr(dpt_module, "DepthAnythingV2")
        except ImportError as exc:
            raise RuntimeError(
                "Failed to import the official DepthAnythingV2 module from third_party."
            ) from exc
        finally:
            # We avoid polluting sys.path globally.
            if path_added and str(metric_depth_path) in sys.path:
                sys.path.remove(str(metric_depth_path))

        model_kwargs = self._model_configs[encoder].copy()
        model_kwargs["max_depth"] = max_depth
        
        # Resolve model_id relative to repo_root if it's a relative path
        model_path = Path(model_id)
        if not model_path.is_absolute():
            model_path = repo_root / model_path

        try:
            model = DepthAnythingV2(**model_kwargs)
            state_dict = torch.load(str(model_path), map_location="cpu", weights_only=True)
            model.load_state_dict(state_dict)
            model = model.to(device)
            model.eval()
        except Exception as exc:
            message = (
                f"Failed to load model from '{model_id}'. "
                "Verify that the checkpoint file exists and matches the encoder architecture."
            )
            logger.exception(message)
            raise RuntimeError(message) from exc

        return model

    @staticmethod
    def _cuda_is_available() -> bool:
        """Return whether a CUDA-capable device is present and accessible."""
        try:
            import torch
        except ImportError:
            return False

        return bool(torch.cuda.is_available())
