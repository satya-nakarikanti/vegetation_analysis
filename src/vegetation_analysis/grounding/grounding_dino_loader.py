"""Grounding DINO model loading and device selection.

This module is the sole entry point for initialising a Grounding DINO model.
It is responsible for:

- Validating the provided configuration.
- Selecting the appropriate inference device (CPU or CUDA).
- Loading the model and processor from a Hugging Face checkpoint or a local
  path.
- Returning a :class:`LoadedGroundingDINO` container that other modules can
  use without re-loading the model.

The loader intentionally performs no inference.  Detection logic lives
exclusively in :mod:`vegetation_analysis.grounding.detector`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from importlib import import_module
from typing import Any, Literal

from vegetation_analysis.grounding.constants import (
    DEFAULT_BOX_THRESHOLD,
    DEFAULT_DEVICE_PREFERENCE,
    DEFAULT_MODEL_ID,
    DEFAULT_TEXT_THRESHOLD,
)

DevicePreference = Literal["auto", "cpu", "cuda"]

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Public constants re-exported for convenience
# ---------------------------------------------------------------------------

#: Box-score threshold exported from this module so callers that import only
#: the loader can still access the default without a separate constants import.
BOX_THRESHOLD_DEFAULT: float = DEFAULT_BOX_THRESHOLD
TEXT_THRESHOLD_DEFAULT: float = DEFAULT_TEXT_THRESHOLD


# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GroundingDINOModelConfig:
    """Configuration required to initialise a Grounding DINO model.

    All defaults are pulled from :mod:`vegetation_analysis.grounding.constants`
    so that no value is ever duplicated across the codebase.

    Attributes:
        model_id: Hugging Face model identifier or absolute path to a local
            checkpoint directory.  The recommended public checkpoints are
            ``IDEA-Research/grounding-dino-tiny`` (approx. 172 MB) and
            ``IDEA-Research/grounding-dino-base`` (approx. 360 MB).
        device_preference: Inference device selection strategy.

            - ``"auto"`` — selects CUDA when available, falls back to CPU.
            - ``"cpu"``  — forces CPU regardless of available hardware.
            - ``"cuda"`` — forces CUDA; raises :class:`RuntimeError` when no
              GPU is detected.

        box_threshold: Minimum objectness score for a predicted box to be
            retained.  Must be in the range ``(0.0, 1.0)``.
        text_threshold: Minimum token-level alignment score for label
            assignment.  Must be in the range ``(0.0, 1.0)``.
    """

    model_id: str = DEFAULT_MODEL_ID
    device_preference: DevicePreference = DEFAULT_DEVICE_PREFERENCE
    box_threshold: float = DEFAULT_BOX_THRESHOLD
    text_threshold: float = DEFAULT_TEXT_THRESHOLD

    def __post_init__(self) -> None:
        """Validate threshold values after dataclass initialisation."""

        if not (0.0 < self.box_threshold < 1.0):
            raise ValueError(
                f"box_threshold must be in the open interval (0.0, 1.0), "
                f"received {self.box_threshold!r}."
            )
        if not (0.0 < self.text_threshold < 1.0):
            raise ValueError(
                f"text_threshold must be in the open interval (0.0, 1.0), "
                f"received {self.text_threshold!r}."
            )
        if not self.model_id.strip():
            raise ValueError("model_id must not be empty or whitespace-only.")


# ---------------------------------------------------------------------------
# Loaded-model container
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LoadedGroundingDINO:
    """Container for a successfully loaded Grounding DINO model and processor.

    This dataclass is the immutable result of :meth:`GroundingDINOLoader.load`.
    It carries everything the detector needs to run inference without
    reloading the model.

    Attributes:
        model: The underlying ``GroundingDinoForObjectDetection`` model object.
        processor: The ``AutoProcessor`` instance paired with the model.
        device: The device string on which the model resides (``"cpu"`` or
            ``"cuda"``).
        config: The :class:`GroundingDINOModelConfig` used to produce this
            container, preserved for auditability.
    """

    model: Any
    processor: Any
    device: str
    config: GroundingDINOModelConfig


# ---------------------------------------------------------------------------
# Loader class
# ---------------------------------------------------------------------------


class GroundingDINOLoader:
    """Load Grounding DINO and select the best available inference device.

    The loader is intentionally decoupled from detection so that model
    initialisation can be exercised and tested independently of inference.

    A single loader instance represents a single configuration.  Call
    :meth:`load` to obtain a :class:`LoadedGroundingDINO` container.  The
    container can then be passed to
    :class:`~vegetation_analysis.grounding.detector.GroundingDINODetector`.

    Args:
        config: Model configuration.  Defaults to
            :class:`GroundingDINOModelConfig` with factory defaults from
            :mod:`vegetation_analysis.grounding.constants`.

    Example::

        from vegetation_analysis.grounding.grounding_dino_loader import (
            GroundingDINOLoader,
            GroundingDINOModelConfig,
        )

        loader = GroundingDINOLoader(
            config=GroundingDINOModelConfig(device_preference="cpu")
        )
        loaded = loader.load()
    """

    def __init__(self, config: GroundingDINOModelConfig | None = None) -> None:
        self._config = config or GroundingDINOModelConfig()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def load(self) -> LoadedGroundingDINO:
        """Initialise Grounding DINO and return an immutable model container.

        The method performs the following steps in order:

        1. Resolve the target device using :meth:`select_device`.
        2. Import ``transformers`` (raises :class:`RuntimeError` with an
           actionable message if the package is not installed).
        3. Load the processor with ``AutoProcessor.from_pretrained``.
        4. Load the model with ``GroundingDinoForObjectDetection.from_pretrained``
           and move it to the selected device.
        5. Set the model to evaluation mode.
        6. Return a :class:`LoadedGroundingDINO` container.

        Returns:
            A :class:`LoadedGroundingDINO` holding the model, processor,
            device string, and the configuration used.

        Raises:
            RuntimeError: If ``transformers`` is not installed or if the
                model checkpoint cannot be fetched or loaded.
            RuntimeError: If ``device_preference="cuda"`` but no CUDA device
                is available.
        """

        device = self.select_device(self._config.device_preference)

        logger.info(
            "Loading Grounding DINO model '%s' on device '%s'.",
            self._config.model_id,
            device,
        )

        processor, model = self._load_from_pretrained(
            model_id=self._config.model_id,
            device=device,
        )

        logger.info(
            "Grounding DINO model '%s' loaded successfully.",
            self._config.model_id,
        )

        return LoadedGroundingDINO(
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

        cuda_available = GroundingDINOLoader._cuda_is_available()

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
                "The 'transformers' package is required for Grounding DINO. "
                "Install it with: pip install transformers torch"
            ) from exc

        auto_processor = getattr(transformers_module, "AutoProcessor", None)
        model_factory = getattr(
            transformers_module,
            "GroundingDinoForObjectDetection",
            None,
        )
        if auto_processor is None or model_factory is None:
            raise RuntimeError(
                "The installed 'transformers' package does not expose the "
                "Grounding DINO APIs required by this project."
            )

        try:
            processor = auto_processor.from_pretrained(model_id)
        except Exception as exc:
            message = (
                f"Failed to load Grounding DINO processor from '{model_id}'. "
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
                f"Failed to load Grounding DINO model from '{model_id}'. "
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
