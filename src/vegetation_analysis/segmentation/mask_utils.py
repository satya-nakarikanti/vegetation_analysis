"""Mask extraction and geometry utilities."""

from __future__ import annotations

import logging
from typing import Any, cast

import cv2
import numpy as np
from numpy.typing import NDArray

from vegetation_analysis.segmentation.schemas import (
    BoundingBox,
    Contour,
    MaskArray,
    SegmentedObject,
)

logger = logging.getLogger(__name__)


def extract_segmented_objects(raw_results: Any) -> list[SegmentedObject]:
    """Extract structured segmented objects from raw FastSAM results."""

    masks = extract_masks(raw_results)
    objects: list[SegmentedObject] = []

    for object_id, mask in enumerate(masks):
        normalized_mask = normalize_mask(mask)
        if not normalized_mask.any():
            logger.debug("Skipping empty mask with id %s.", object_id)
            continue

        objects.append(
            SegmentedObject(
                id=object_id,
                mask=normalized_mask,
                bbox=mask_to_bbox(normalized_mask),
                contour=mask_to_largest_contour(normalized_mask),
                area=mask_area(normalized_mask),
            )
        )

    return objects


def extract_masks(raw_results: Any) -> list[NDArray[Any]]:
    """Extract mask arrays from Ultralytics-style FastSAM results."""

    result_items = _as_result_items(raw_results)
    extracted_masks: list[NDArray[Any]] = []

    for result in result_items:
        masks_container = getattr(result, "masks", None)
        if masks_container is None:
            continue

        mask_data = getattr(masks_container, "data", None)
        if mask_data is None:
            continue

        mask_array = _to_numpy(mask_data)
        if mask_array.ndim == 2:
            extracted_masks.append(mask_array)
        elif mask_array.ndim == 3:
            extracted_masks.extend(
                mask_array[index] for index in range(mask_array.shape[0])
            )
        else:
            logger.warning(
                "Ignoring unsupported mask array shape: %s.",
                mask_array.shape,
            )

    return extracted_masks


def normalize_mask(mask: NDArray[Any]) -> MaskArray:
    """Convert a numeric mask into a two-dimensional boolean mask."""

    mask_array = np.asarray(mask)
    if mask_array.ndim != 2:
        raise ValueError(f"Expected a 2D mask, received shape {mask_array.shape}.")

    return mask_array.astype(bool)


def mask_area(mask: MaskArray) -> int:
    """Return the number of foreground pixels in a mask."""

    return int(np.count_nonzero(mask))


def mask_to_bbox(mask: MaskArray) -> BoundingBox:
    """Compute the tight bounding box for a boolean mask."""

    if not mask.any():
        return BoundingBox(0, 0, 0, 0)

    y_indices, x_indices = np.where(mask)
    return BoundingBox(
        x_min=int(x_indices.min()),
        y_min=int(y_indices.min()),
        x_max=int(x_indices.max()) + 1,
        y_max=int(y_indices.max()) + 1,
    )


def mask_to_largest_contour(mask: MaskArray) -> Contour:
    """Extract the largest external contour from a boolean mask."""

    mask_uint8 = mask.astype(np.uint8) * 255
    contours, _ = cv2.findContours(
        mask_uint8,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    if not contours:
        return []

    largest_contour = max(contours, key=cv2.contourArea)
    contour_points = cast(NDArray[np.integer[Any]], largest_contour.reshape(-1, 2))
    return [(int(point[0]), int(point[1])) for point in contour_points]


def _as_result_items(raw_results: Any) -> list[Any]:
    if isinstance(raw_results, list):
        return raw_results
    if isinstance(raw_results, tuple):
        return list(raw_results)
    return [raw_results]


def _to_numpy(value: Any) -> NDArray[Any]:
    if isinstance(value, np.ndarray):
        return value

    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        return np.asarray(value.numpy())

    return np.asarray(value)
