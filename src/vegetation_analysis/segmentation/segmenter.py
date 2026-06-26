"""FastSAM inference runner."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeAlias

import numpy as np
from numpy.typing import NDArray
from PIL import Image, UnidentifiedImageError

ImageInput: TypeAlias = str | Path | NDArray[np.uint8] | Image.Image

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FastSAMInferenceConfig:
    """Inference parameters for generating all visible object masks."""

    image_size: int = 1024
    confidence: float = 0.4
    iou: float = 0.9
    retina_masks: bool = True


class FastSAMSegmenter:
    """Run FastSAM inference and return raw model results."""

    def __init__(
        self,
        model: Any,
        device: str,
        config: FastSAMInferenceConfig | None = None,
    ) -> None:
        self._model = model
        self._device = device
        self._config = config or FastSAMInferenceConfig()

    def segment(self, image: ImageInput) -> Any:
        """Run segmentation for every detected object in the image."""

        prepared_image = self._prepare_image(image)

        try:
            return self._model(
                prepared_image,
                device=self._device,
                retina_masks=self._config.retina_masks,
                imgsz=self._config.image_size,
                conf=self._config.confidence,
                iou=self._config.iou,
            )
        except Exception as exc:
            logger.exception("FastSAM inference failed.")
            raise RuntimeError("FastSAM inference failed.") from exc

    @staticmethod
    def _prepare_image(image: ImageInput) -> str | NDArray[np.uint8]:
        if isinstance(image, str | Path):
            return FastSAMSegmenter._validate_image_path(Path(image))

        if isinstance(image, Image.Image):
            image_array = np.asarray(image.convert("RGB"), dtype=np.uint8)
            return FastSAMSegmenter._validate_image_array(image_array)

        return FastSAMSegmenter._validate_image_array(image)

    @staticmethod
    def _validate_image_path(image_path: Path) -> str:
        if not image_path.exists():
            raise FileNotFoundError(f"Image does not exist: {image_path}")
        if not image_path.is_file():
            raise ValueError(f"Image path is not a file: {image_path}")

        try:
            with Image.open(image_path) as image:
                image.verify()
        except (UnidentifiedImageError, OSError) as exc:
            raise ValueError(f"Invalid image file: {image_path}") from exc

        return str(image_path)

    @staticmethod
    def _validate_image_array(image: NDArray[np.uint8]) -> NDArray[np.uint8]:
        image_array = np.asarray(image)
        if image_array.size == 0:
            raise ValueError("Image array is empty.")
        if image_array.ndim not in {2, 3}:
            raise ValueError(
                "Image array must have shape (height, width) or "
                "(height, width, channels)."
            )
        if image_array.shape[0] == 0 or image_array.shape[1] == 0:
            raise ValueError("Image array height and width must be greater than zero.")
        if image_array.ndim == 3 and image_array.shape[2] not in {1, 3, 4}:
            raise ValueError("Image array must have 1, 3, or 4 channels.")

        return image_array.astype(np.uint8, copy=False)
