"""SAM 2 segmenter implementation."""

from __future__ import annotations

import logging
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray
from PIL import Image

from vegetation_analysis.grounding.schemas import DetectionResult
from vegetation_analysis.sam2.sam2_loader import LoadedSAM2
from vegetation_analysis.sam2.schemas import MaskObject, SegmentationResult

logger = logging.getLogger(__name__)


class SAM2Segmenter:
    """Segment objects using SAM 2 based on bounding box prompts.

    Args:
        model: A LoadedSAM2 instance containing the predictor.
    """

    def __init__(self, model: LoadedSAM2) -> None:
        self._predictor = model.predictor

    def segment(
        self,
        image: NDArray[np.uint8] | Image.Image,
        detection_result: DetectionResult,
    ) -> SegmentationResult:
        """Run SAM 2 segmentation using bounding boxes from detection_result.

        Args:
            image: Source image as a NumPy array or PIL Image.
            detection_result: DetectionResult from Grounding DINO.

        Returns:
            A SegmentationResult containing the predicted masks.
        """
        image_array = self._to_rgb_array(image)
        height, width = image_array.shape[:2]

        if detection_result.is_empty:
            logger.info("Empty detection result received. Skipping segmentation.")
            return SegmentationResult(
                objects=(),
                image_width=width,
                image_height=height,
            )

        # Set image in the predictor
        self._predictor.set_image(image_array)

        # Prepare bounding boxes: shape (N, 4)
        boxes_array = np.array([box.as_xyxy() for box in detection_result.boxes])

        # Run prediction
        # SAM 2 returns masks of shape (N, 1, H, W) when multimask_output=False
        masks, scores, _ = self._predictor.predict(
            point_coords=None,
            point_labels=None,
            box=boxes_array,
            multimask_output=False,
        )

        # Squeeze the multimask dimension: (N, H, W)
        if masks.ndim == 4 and masks.shape[1] == 1:
            masks = masks[:, 0, :, :]
        if scores.ndim == 2 and scores.shape[1] == 1:
            scores = scores[:, 0]

        objects = []
        for i, box in enumerate(detection_result.boxes):
            mask = masks[i].astype(bool)
            score = float(scores[i])
            area = int(mask.sum())
            contours = self._extract_contours(mask)
            obj = MaskObject(
                label=box.label,
                confidence=score,
                box=box,
                mask=mask,
                mask_area_pixels=area,
                contours=contours,
            )
            objects.append(obj)

        logger.info("Segmented %d objects using SAM 2.", len(objects))

        return SegmentationResult(
            objects=tuple(objects),
            image_width=width,
            image_height=height,
        )

    @staticmethod
    def _extract_contours(mask: NDArray[Any]) -> tuple[NDArray[Any], ...]:
        """Extract boundaries from a boolean mask as a tuple of arrays."""
        mask_uint8 = mask.astype(np.uint8) * 255
        contours, _ = cv2.findContours(
            mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        return tuple(contours)

    @staticmethod
    def _to_rgb_array(image: NDArray[np.uint8] | Image.Image) -> NDArray[np.uint8]:
        if isinstance(image, Image.Image):
            return np.asarray(image.convert("RGB"), dtype=np.uint8)

        image_array = np.asarray(image, dtype=np.uint8)
        if image_array.ndim == 2:
            return np.stack([image_array] * 3, axis=-1)
        if image_array.ndim == 3 and image_array.shape[2] == 1:
            return np.repeat(image_array, repeats=3, axis=2)
        if image_array.ndim == 3 and image_array.shape[2] == 3:
            return image_array
        if image_array.ndim == 3 and image_array.shape[2] == 4:
            return image_array[:, :, :3]

        raise ValueError("Image must be grayscale, RGB, or RGBA.")
