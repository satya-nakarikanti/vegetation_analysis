"""Output schemas for the Depth Sampling Engine."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SampledObject:
    """A single object with sampled depth statistics.

    Attributes:
        label: Text label assigned by the upstream detector.
        confidence: Detector confidence score.
        pixel_count: Total number of pixels used to calculate the depth statistics.
        centroid_x: X-coordinate of the object's mask centroid.
        centroid_y: Y-coordinate of the object's mask centroid.
        median_depth: The median depth value within the mask (representative depth).
        mean_depth: The mean depth value within the mask.
        std_depth: The standard deviation of depth values within the mask.
        min_depth: The minimum depth value within the mask.
        max_depth: The maximum depth value within the mask.
    """

    label: str
    confidence: float
    pixel_count: int
    centroid_x: float
    centroid_y: float
    median_depth: float
    mean_depth: float
    std_depth: float
    min_depth: float
    max_depth: float


@dataclass(frozen=True)
class ImageMetadata:
    """Image-level metadata for depth sampling results.

    Attributes:
        width: Width of the source image in pixels.
        height: Height of the source image in pixels.
        depth_model: Name of the model used to produce the depth map.
    """

    width: int
    height: int
    depth_model: str


@dataclass(frozen=True)
class DepthSamplingResult:
    """Aggregated output of the depth sampling engine.

    Attributes:
        objects: Tuple of sampled objects with their depth statistics.
        metadata: Image-level metadata and configuration info.
    """

    objects: tuple[SampledObject, ...]
    metadata: ImageMetadata
