"""Depth Anything V2 model loading and device selection.

This module is the sole entry point for initialising a Depth Anything V2 model.
It is responsible for:

- Validating the provided configuration.
- Selecting the appropriate inference device (CPU or CUDA).
- Loading the model and processor from a Hugging Face checkpoint or a local
  path.
- Returning a :class:`LoadedDepthAnything` container that other modules can
  use without re-loading the model.

The loader intentionally performs no inference. Detection logic lives
exclusively in :mod:`vegetation_analysis.depth.estimator`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from importlib import import_module
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
        model_id: Hugging Face model identifier or absolute path to a local
            checkpoint directory. The recommended public checkpoint is
            ``depth-anything/Depth-Anything-V2-Small-hf``.
        device_preference: Inference device selection strategy.

            - ``"auto"`` — selects CUDA when available, falls back to CPU.
            - ``"cpu"``  — forces CPU regardless of available hardware.
            - ``"cuda"`` — forces CUDA; raises :class:`RuntimeError` when no
              GPU is detected.
    """

    model_id: str = DEFAULT_MODEL_ID
    device_preference: DevicePreference = DEFAULT_DEVICE_PREFERENCE

    def __post_init__(self) -> None:
        """Validate threshold values after dataclass initialisation."""

        if not self.model_id.strip():
            raise ValueError("model_id must not be empty or whitespace-only.")


# ---------------------------------------------------------------------------
# Loaded-model container
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LoadedDepthAnything:
    """Container for a successfully loaded Depth Anything V2 model and processor.

    This dataclass is the immutable result of :meth:`DepthAnythingLoader.load`.
    It carries everything the estimator needs to run inference without
    reloading the model.

    Attributes:
        model: The underlying depth estimation model object.
        processor: The processor instance paired with the model.
        device: The device string on which the model resides (``"cpu"`` or
            ``"cuda"``).
        config: The :class:`DepthAnythingModelConfig` used to produce this
            container, preserved for auditability.
    """

    model: Any
    processor: Any
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

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def load(self) -> LoadedDepthAnything:
        """Initialise Depth Anything V2 and return an immutable model container.

        The method performs the following steps in order:

        1. Resolve the target device using :meth:`select_device`.
        2. Import ``transformers`` (raises :class:`RuntimeError` with an
           actionable message if the package is not installed).
        3. Load the processor with ``AutoImageProcessor.from_pretrained``.
        4. Load the model with ``AutoModelForDepthEstimation.from_pretrained``
           and move it to the selected device.
        5. Set the model to evaluation mode.
        6. Return a :class:`LoadedDepthAnything` container.

        Returns:
            A :class:`LoadedDepthAnything` holding the model, processor,
            device string, and the configuration used.

        Raises:
            RuntimeError: If ``transformers`` is not installed or if the
                model checkpoint cannot be fetched or loaded.
            RuntimeError: If ``device_preference="cuda"`` but no CUDA device
                is available.
        """

        device = self.select_device(self._config.device_preference)

        logger.info(
            "Loading Depth Anything V2 model '%s' on device '%s'.",
            self._config.model_id,
            device,
        )

        processor, model = self._load_from_pretrained(
            model_id=self._config.model_id,
            device=device,
        )

        logger.info(
            "Depth Anything V2 model '%s' loaded successfully.",
            self._config.model_id,
        )

        return LoadedDepthAnything(
            model=model,
            processor=processor,
            device=device,
            config=self._config,
        )

    @staticmethod
    def select_device(preference: DevicePreference = "auto") -> str:
        """Select CPU or CUDA according to availability and caller preference.

        Args:
            preference: Device selection strategy — ``"auto"``, ``"cpu"``, or
                ``"cuda"``.

        Returns:
            The resolved device string: ``"cpu"`` or ``"cuda"``.

        Raises:
            RuntimeError: If ``preference="cuda"`` but no CUDA device is
                available in the current environment.
        """

        if preference == "cpu":
            return "cpu"

        cuda_available = DepthAnythingLoader._cuda_is_available()

        if preference == "cuda":
            if not cuda_available:
                raise RuntimeError(
                    "CUDA was requested, but no CUDA device is available."
                )
            return "cuda"

        # preference == "auto"
        selected = "cuda" if cuda_available else "cpu"
        logger.debug("Auto device selection resolved to '%s'.", selected)
        return selected

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load_from_pretrained(
        model_id: str,
        device: str,
    ) -> tuple[Any, Any]:
        """Import transformers and load the processor and model.

        Args:
            model_id: Hugging Face model identifier or local path.
            device: Target device string (``"cpu"`` or ``"cuda"``).

        Returns:
            A ``(processor, model)`` tuple.

        Raises:
            RuntimeError: If ``transformers`` is not installed.
            RuntimeError: If the model cannot be loaded from the given
                ``model_id``.
        """

        try:
            transformers_module = import_module("transformers")
        except ImportError as exc:
            raise RuntimeError(
                "The 'transformers' package is required for Depth Anything V2. "
                "Install it with: pip install transformers torch"
            ) from exc

        auto_processor = getattr(transformers_module, "AutoImageProcessor", None)
        model_factory = getattr(
            transformers_module,
            "AutoModelForDepthEstimation",
            None,
        )
        if auto_processor is None or model_factory is None:
            raise RuntimeError(
                "The installed 'transformers' package does not expose the "
                "APIs required by this project."
            )

        try:
            processor = auto_processor.from_pretrained(model_id)
        except Exception as exc:
            message = (
                f"Failed to load processor from '{model_id}'. "
                "Verify that the model identifier is correct and that a "
                "network connection is available for the initial download."
            )
            logger.exception(message)
            raise RuntimeError(message) from exc

        try:
            model = model_factory.from_pretrained(model_id)
            model = model.to(device)
            model.eval()
        except Exception as exc:
            message = (
                f"Failed to load model from '{model_id}'. "
                "Verify that the model identifier is correct and that "
                "sufficient disk space and memory are available."
            )
            logger.exception(message)
            raise RuntimeError(message) from exc

        return processor, model

    @staticmethod
    def _cuda_is_available() -> bool:
        """Return whether a CUDA-capable device is present and accessible."""

        try:
            import torch
        except ImportError:
            return False

        return bool(torch.cuda.is_available())
