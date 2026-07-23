"""Depth sampling runner."""

from __future__ import annotations

import logging
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray

from vegetation_analysis.depth.schemas import DepthMapResult
from vegetation_analysis.depth_sampling.constants import (
    DEFAULT_EROSION_ITERATIONS,
    DEFAULT_EROSION_KERNEL_SIZE,
    MIN_ERODED_PIXELS,
)
from vegetation_analysis.depth_sampling.schemas import (
    DepthSamplingResult,
    ImageMetadata,
    SampledObject,
)
from vegetation_analysis.sam2.schemas import MaskObject, SegmentationResult

logger = logging.getLogger(__name__)


class DepthSampler:
    """Run depth sampling by projecting SAM2 masks onto Depth Anything V2 depth maps.

    The sampler computes robust statistical metrics (median, mean, min, max, std)
    for each object segmented in the image, applying morphological erosion where
    possible to avoid sampling background depth at object boundaries.

    Args:
        erosion_kernel_size: Size of the square morphological erosion kernel.
        erosion_iterations: Number of times to apply the erosion operation.
        min_eroded_pixels: The minimum number of pixels an eroded mask must have
            to be used. If it falls below this threshold, the sampler falls back
            to the original uneroded mask.
    """

    def __init__(
        self,
        erosion_kernel_size: int = DEFAULT_EROSION_KERNEL_SIZE,
        erosion_iterations: int = DEFAULT_EROSION_ITERATIONS,
        min_eroded_pixels: int = MIN_ERODED_PIXELS,
    ) -> None:
        self._erosion_kernel_size = erosion_kernel_size
        self._erosion_iterations = erosion_iterations
        self._min_eroded_pixels = min_eroded_pixels
        self._kernel = np.ones(
            (self._erosion_kernel_size, self._erosion_kernel_size),
            dtype=np.uint8,
        )

    def sample(
        self,
        segmentation: SegmentationResult,
        depth: DepthMapResult,
    ) -> DepthSamplingResult:
        """Sample depth values for each object in the segmentation result.

        Args:
            segmentation: Aggregated SAM2 segmentation result containing masks.
            depth: Aggregated Depth Anything V2 result containing the relative
                depth array.

        Returns:
            A :class:`DepthSamplingResult` containing object-wise statistics
            and image metadata.

        Raises:
            ValueError: If the segmentation and depth maps have mismatched dimensions.
        """

        if (
            segmentation.image_width != depth.image_width
            or segmentation.image_height != depth.image_height
        ):
            raise ValueError(
                f"Dimension mismatch: segmentation is "
                f"{segmentation.image_width}x{segmentation.image_height}, "
                f"but depth map is {depth.image_width}x{depth.image_height}."
            )

        sampled_objects: list[SampledObject] = []

        for obj in segmentation.objects:
            sampled_obj = self._sample_object(obj, depth.depth_map)
            if sampled_obj is not None:
                sampled_objects.append(sampled_obj)

        metadata = ImageMetadata(
            width=depth.image_width,
            height=depth.image_height,
            depth_model=depth.model_name,
        )

        logger.info(
            "Successfully sampled depth for %d object(s).",
            len(sampled_objects),
        )

        return DepthSamplingResult(
            objects=tuple(sampled_objects),
            metadata=metadata,
        )

    def _sample_object(
        self,
        obj: MaskObject,
        depth_map: NDArray[np.float32],
    ) -> SampledObject | None:
        """Sample depth for a single mask object."""

        # Determine the mask to use (try eroding first to avoid bleeding)
        sampling_mask = self._get_sampling_mask(obj.mask)

        # Calculate centroid based on the original mask
        # (more representative of position)
        centroid_x, centroid_y = self._calculate_centroid(obj.mask)

        # Extract depth values inside the chosen mask
        depth_values = depth_map[sampling_mask]

        if depth_values.size == 0:
            logger.warning(
                "Skipping object '%s' with empty depth sampling mask.",
                obj.label,
            )
            return None

        return SampledObject(
            label=obj.label,
            confidence=obj.confidence,
            pixel_count=depth_values.size,
            centroid_x=centroid_x,
            centroid_y=centroid_y,
            median_depth=float(np.median(depth_values)),
            mean_depth=float(np.mean(depth_values)),
            std_depth=float(np.std(depth_values)),
            min_depth=float(np.min(depth_values)),
            max_depth=float(np.max(depth_values)),
            bounding_box=obj.box,
            original_mask=obj.mask,
            sampling_mask=sampling_mask,
            contours=obj.contours,
            mask_area_pixels=obj.mask_area_pixels,
        )

    def _get_sampling_mask(self, original_mask: NDArray[Any]) -> NDArray[np.bool_]:
        """Apply morphological erosion to avoid boundary bleeding.

        Falls back to original mask if eroded mask is too small.
        """

        mask_uint8 = original_mask.astype(np.uint8)

        eroded = cv2.erode(
            mask_uint8,
            self._kernel,
            iterations=self._erosion_iterations,
        )

        eroded_bool = eroded > 0
        eroded_pixels = int(np.sum(eroded_bool))

        if eroded_pixels >= self._min_eroded_pixels:
            return eroded_bool

        logger.debug(
            "Eroded mask too small (%d pixels). Falling back to original mask.",
            eroded_pixels,
        )
        return original_mask.astype(np.bool_)

    @staticmethod
    def _calculate_centroid(mask: NDArray[Any]) -> tuple[float, float]:
        """Calculate the (x, y) centroid of a boolean mask."""

        y_coords, x_coords = np.where(mask)

        if len(x_coords) == 0:
            return 0.0, 0.0

        return float(np.mean(x_coords)), float(np.mean(y_coords))
