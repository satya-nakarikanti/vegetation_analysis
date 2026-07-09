"""Depth Anything V2 depth estimation runner."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, TypeAlias

import numpy as np
from numpy.typing import NDArray
from PIL import Image, UnidentifiedImageError

from vegetation_analysis.depth.depth_loader import LoadedDepthAnything
from vegetation_analysis.depth.schemas import DepthMapResult

ImageInput: TypeAlias = str | Path | NDArray[np.uint8] | Image.Image

logger = logging.getLogger(__name__)


class DepthEstimator:
    """Run Depth Anything V2 inference and return structured depth results.

    The estimator is constructed with a pre-loaded model container. Estimation
    is initiated by calling :meth:`estimate` with an image.

    This class is responsible only for running inference and converting raw
    model output into :class:`~vegetation_analysis.depth.schemas.DepthMapResult`
    instances. Model loading is handled by
    :class:`~vegetation_analysis.depth.depth_loader.DepthAnythingLoader`.

    Args:
        loaded_model: A :class:`LoadedDepthAnything` container produced by
            :class:`DepthAnythingLoader`.
    """

    def __init__(self, loaded_model: LoadedDepthAnything) -> None:
        self._loaded_model = loaded_model

    def estimate(self, image: ImageInput) -> DepthMapResult:
        """Run Depth Anything V2 estimation for the given image.

        Args:
            image: Source image as a file path, NumPy array, or PIL Image.

        Returns:
            A :class:`~vegetation_analysis.depth.schemas.DepthMapResult`
            containing the raw depth map array and metadata.

        Raises:
            ValueError: If the image is invalid.
            RuntimeError: If model inference or post-processing fails.
        """

        prepared_image = self._prepare_image(image)
        image_width, image_height = prepared_image.size

        logger.info(
            "Running Depth Anything V2 estimation on %dx%d image.",
            image_width,
            image_height,
        )

        inputs = self._preprocess(prepared_image)
        outputs = self._run_inference(inputs)
        depth_map = self._postprocess(
            outputs=outputs,
            image_size=(image_width, image_height),
        )

        logger.info("Depth estimation completed successfully.")
        return DepthMapResult(
            depth_map=depth_map,
            image_width=image_width,
            image_height=image_height,
            model_name=self._loaded_model.config.model_id,
        )

    @staticmethod
    def _prepare_image(image: ImageInput) -> Image.Image:
        """Convert the input image to a PIL RGB image.

        Args:
            image: Source image in any supported format.

        Returns:
            A PIL :class:`~PIL.Image.Image` in RGB mode.

        Raises:
            FileNotFoundError: If a path is given but the file does not exist.
            ValueError: If the provided array has an unsupported shape.
        """

        if isinstance(image, Image.Image):
            return image.convert("RGB")

        if isinstance(image, str | Path):
            image_path = Path(image)
            if not image_path.exists():
                raise FileNotFoundError(f"Image does not exist: {image_path}")
            if not image_path.is_file():
                raise ValueError(f"Image path is not a file: {image_path}")
            try:
                return Image.open(image_path).convert("RGB")
            except (UnidentifiedImageError, OSError) as exc:
                raise ValueError(f"Invalid image file: {image_path}") from exc

        array = np.asarray(image, dtype=np.uint8)
        if array.size == 0:
            raise ValueError("Image array is empty.")
        if array.ndim not in {2, 3}:
            raise ValueError(
                "Image array must have shape (height, width) or "
                "(height, width, channels)."
            )
        if array.shape[0] == 0 or array.shape[1] == 0:
            raise ValueError("Image array height and width must be greater than zero.")
        if array.ndim == 3 and array.shape[2] not in {1, 3, 4}:
            raise ValueError("Image array must have 1, 3, or 4 channels.")
        return Image.fromarray(array).convert("RGB")

    def _preprocess(self, image: Image.Image) -> Any:
        """Convert a PIL image into model-ready tensors."""

        try:
            inputs = self._loaded_model.processor(
                images=image,
                return_tensors="pt",
            )
            to_device = getattr(inputs, "to", None)
            if callable(to_device):
                inputs = to_device(self._loaded_model.device)
        except Exception as exc:
            logger.exception("Depth Anything V2 preprocessing failed.")
            raise RuntimeError("Depth Anything V2 preprocessing failed.") from exc

        return inputs

    def _run_inference(self, inputs: Any) -> Any:
        """Run model inference without tracking gradients."""

        try:
            import torch
        except ImportError as exc:
            raise RuntimeError(
                "The 'torch' package is required for inference."
            ) from exc

        try:
            with torch.no_grad():
                return self._loaded_model.model(**inputs)
        except Exception as exc:
            logger.exception("Depth Anything V2 inference failed.")
            raise RuntimeError("Depth Anything V2 inference failed.") from exc

    def _postprocess(
        self,
        outputs: Any,
        image_size: tuple[int, int],
    ) -> NDArray[np.float32]:
        """Convert raw model output into resized NumPy array."""

        try:
            import torch.nn.functional as F
        except ImportError as exc:
            raise RuntimeError(
                "The 'torch' package is required for postprocessing."
            ) from exc

        try:
            image_width, image_height = image_size
            predicted_depth = outputs.predicted_depth

            # Resize to original image size
            prediction = F.interpolate(
                predicted_depth.unsqueeze(1),
                size=(image_height, image_width),
                mode="bicubic",
                align_corners=False,
            )

            depth_array = prediction.squeeze().cpu().numpy()
            return depth_array.astype(np.float32)

        except Exception as exc:
            logger.exception("Depth Anything V2 post-processing failed.")
            raise RuntimeError("Depth Anything V2 post-processing failed.") from exc
