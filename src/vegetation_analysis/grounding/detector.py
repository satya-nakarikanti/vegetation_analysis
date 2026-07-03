"""Grounding DINO object detection runner."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, TypeAlias

import numpy as np
from numpy.typing import NDArray
from PIL import Image, UnidentifiedImageError

from vegetation_analysis.grounding.constants import (
    DEFAULT_BOX_THRESHOLD,
    DEFAULT_TEXT_THRESHOLD,
)
from vegetation_analysis.grounding.grounding_dino_loader import LoadedGroundingDINO
from vegetation_analysis.grounding.prompts import PromptBuilder
from vegetation_analysis.grounding.schemas import DetectionBox, DetectionResult

ImageInput: TypeAlias = str | Path | NDArray[np.uint8] | Image.Image

logger = logging.getLogger(__name__)


class GroundingDINODetector:
    """Run Grounding DINO inference and return structured detection results.

    The detector is constructed with a pre-loaded model container and a set of
    inference parameters.  Detection is initiated by calling :meth:`detect`
    with an image and a text prompt.

    This class is responsible only for running inference and converting raw
    model output into :class:`~vegetation_analysis.grounding.schemas.DetectionResult`
    instances.  Model loading is handled by
    :class:`~vegetation_analysis.grounding.grounding_dino_loader.GroundingDINOLoader`.

    Args:
        loaded_model: A :class:`LoadedGroundingDINO` container produced by
            :class:`GroundingDINOLoader`.
        box_threshold: Minimum objectness confidence for a box to be retained.
        text_threshold: Minimum token-level confidence for label assignment.
    """

    def __init__(
        self,
        loaded_model: LoadedGroundingDINO,
        box_threshold: float = DEFAULT_BOX_THRESHOLD,
        text_threshold: float = DEFAULT_TEXT_THRESHOLD,
    ) -> None:
        _validate_threshold("box_threshold", box_threshold)
        _validate_threshold("text_threshold", text_threshold)
        self._loaded_model = loaded_model
        self._box_threshold = box_threshold
        self._text_threshold = text_threshold

    def detect(self, image: ImageInput, prompt: str) -> DetectionResult:
        """Run Grounding DINO detection for the given image and text prompt.

        Args:
            image: Source image as a file path, NumPy array, or PIL Image.
            prompt: Grounding DINO text prompt in the format
                ``"label_a . label_b ."``.  Use
                :class:`~vegetation_analysis.grounding.prompts.PromptBuilder`
                to construct valid prompts.

        Returns:
            A :class:`~vegetation_analysis.grounding.schemas.DetectionResult`
            containing all retained bounding boxes with labels and confidence
            scores.

        Raises:
            ValueError: If the image or prompt is invalid.
            RuntimeError: If model inference or post-processing fails.
        """

        if not PromptBuilder.validate(prompt):
            raise ValueError(
                "Prompt must be non-empty and formatted as dot-separated labels."
            )

        prepared_image = self._prepare_image(image)
        image_width, image_height = prepared_image.size

        logger.info(
            "Running Grounding DINO detection with prompt '%s' on %dx%d image.",
            prompt,
            image_width,
            image_height,
        )

        inputs = self._preprocess(prepared_image, prompt)
        outputs = self._run_inference(inputs)
        boxes = self._postprocess(
            outputs=outputs,
            inputs=inputs,
            image_size=(image_width, image_height),
        )

        logger.info("Grounding DINO retained %d detection(s).", len(boxes))
        return DetectionResult(
            boxes=boxes,
            image_width=image_width,
            image_height=image_height,
            prompt=prompt,
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

    def _preprocess(self, image: Image.Image, prompt: str) -> Any:
        """Convert a PIL image and prompt into model-ready tensors."""

        try:
            inputs = self._loaded_model.processor(
                images=image,
                text=prompt,
                return_tensors="pt",
            )
            to_device = getattr(inputs, "to", None)
            if callable(to_device):
                inputs = to_device(self._loaded_model.device)
        except Exception as exc:
            logger.exception("Grounding DINO preprocessing failed.")
            raise RuntimeError("Grounding DINO preprocessing failed.") from exc

        return inputs

    def _run_inference(self, inputs: Any) -> Any:
        """Run model inference without tracking gradients."""

        try:
            import torch
        except ImportError as exc:
            raise RuntimeError(
                "The 'torch' package is required for Grounding DINO inference."
            ) from exc

        try:
            with torch.no_grad():
                return self._loaded_model.model(**inputs)
        except Exception as exc:
            logger.exception("Grounding DINO inference failed.")
            raise RuntimeError("Grounding DINO inference failed.") from exc

    def _postprocess(
        self,
        outputs: Any,
        inputs: Any,
        image_size: tuple[int, int],
    ) -> tuple[DetectionBox, ...]:
        """Convert raw model output into validated detection boxes."""

        image_width, image_height = image_size
        input_ids = _extract_input_ids(inputs)

        # try:

        #     processed = postprocess(
        #         outputs,
        #         input_ids=input_ids,
        #         threshold=self._box_threshold,
        #         text_threshold=self._text_threshold,
        #         target_sizes=[(image_height, image_width)],
        #     )
        # except Exception as exc:
        #     logger.exception("Grounding DINO post-processing failed.")
        #     raise RuntimeError("Grounding DINO post-processing failed.") from exc

        try:
            postprocess = (
                self._loaded_model.processor.post_process_grounded_object_detection
            )

            try:
                processed = postprocess(
                    outputs,
                    input_ids=input_ids,
                    threshold=self._box_threshold,
                    text_threshold=self._text_threshold,
                    target_sizes=[(image_height, image_width)],
                )
            except TypeError:
                processed = postprocess(
                    outputs,
                    input_ids=input_ids,
                    box_threshold=self._box_threshold,
                    text_threshold=self._text_threshold,
                    target_sizes=[(image_height, image_width)],
                )

        except Exception as exc:
            logger.exception("Grounding DINO post-processing failed.")
            raise RuntimeError("Grounding DINO post-processing failed.") from exc

        if not processed:
            return ()

        first_result = processed[0]
        raw_boxes = _as_list(first_result.get("boxes", []))
        raw_scores = _as_list(first_result.get("scores", []))
        raw_labels = list(
            first_result.get(
                "text_labels",
                first_result.get("labels", []),
            )
        )

        detection_boxes: list[DetectionBox] = []
        for raw_box, raw_score, raw_label in zip(
            raw_boxes,
            raw_scores,
            raw_labels,
            strict=False,
        ):
            confidence = float(_to_python_scalar(raw_score))
            if confidence < self._box_threshold:
                continue

            x_min, y_min, x_max, y_max = _normalize_box(
                raw_box=raw_box,
                image_width=image_width,
                image_height=image_height,
            )
            if x_max <= x_min or y_max <= y_min:
                continue

            detection_boxes.append(
                DetectionBox(
                    x_min=x_min,
                    y_min=y_min,
                    x_max=x_max,
                    y_max=y_max,
                    label=str(raw_label).strip(),
                    confidence=confidence,
                )
            )
        before = len(detection_boxes)
        detection_boxes = self._filter_duplicate_poles(detection_boxes)

        logger.info(
            "Filtered %d duplicate pole detection(s).",
            before - len(detection_boxes),
        )

        return tuple(detection_boxes)

    def _filter_duplicate_poles(
        self,
        detections: list[DetectionBox],
    ) -> list[DetectionBox]:
        """Remove duplicate pole detections while keeping the larger box."""

        def area(box: DetectionBox) -> float:
            return max(0.0, box.x_max - box.x_min) * max(0.0, box.y_max - box.y_min)

        def intersection(box1: DetectionBox, box2: DetectionBox) -> float:
            x1 = max(box1.x_min, box2.x_min)
            y1 = max(box1.y_min, box2.y_min)
            x2 = min(box1.x_max, box2.x_max)
            y2 = min(box1.y_max, box2.y_max)

            if x2 <= x1 or y2 <= y1:
                return 0.0

            return (x2 - x1) * (y2 - y1)

        def iou(box1: DetectionBox, box2: DetectionBox) -> float:
            inter = intersection(box1, box2)
            union = area(box1) + area(box2) - inter

            if union <= 0:
                return 0.0

            return inter / union

        def containment(box1: DetectionBox, box2: DetectionBox) -> float:
            inter = intersection(box1, box2)
            smaller = min(area(box1), area(box2))

            if smaller <= 0:
                return 0.0

            return inter / smaller

        poles = []
        others = []

        for det in detections:
            label = det.label.lower()

            if "pole" in label:
                poles.append(det)
            else:
                others.append(det)

        keep = [True] * len(poles)

        for i in range(len(poles)):
            if not keep[i]:
                continue

            for j in range(i + 1, len(poles)):
                if not keep[j]:
                    continue

                duplicate = (
                    iou(poles[i], poles[j]) > 0.70
                    or containment(poles[i], poles[j]) > 0.90
                )

                if not duplicate:
                    continue

                if area(poles[i]) >= area(poles[j]):
                    keep[j] = False
                else:
                    keep[i] = False
                    break

        filtered_poles = [pole for pole, use in zip(poles, keep, strict=True) if use]

        return others + filtered_poles


def _validate_threshold(name: str, value: float) -> None:
    if not (0.0 < value < 1.0):
        raise ValueError(f"{name} must be in the open interval (0.0, 1.0).")


def _extract_input_ids(inputs: Any) -> Any:
    if hasattr(inputs, "input_ids"):
        return inputs.input_ids
    if isinstance(inputs, dict) and "input_ids" in inputs:
        return inputs["input_ids"]
    raise RuntimeError("Grounding DINO processor output does not include input_ids.")


def _as_list(value: Any) -> list[Any]:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "tolist"):
        return list(value.tolist())
    return list(value)


def _to_python_scalar(value: Any) -> float:
    if hasattr(value, "item"):
        return float(value.item())
    return float(value)


def _normalize_box(
    raw_box: Any,
    image_width: int,
    image_height: int,
) -> tuple[int, int, int, int]:
    values = _as_list(raw_box)
    if len(values) != 4:
        raise RuntimeError(f"Expected a four-value box, received {values!r}.")

    x_min, y_min, x_max, y_max = (int(round(float(value))) for value in values)
    return (
        max(0, min(image_width, x_min)),
        max(0, min(image_height, y_min)),
        max(0, min(image_width, x_max)),
        max(0, min(image_height, y_max)),
    )
