"""Output schemas for Depth Anything V2 results."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class DepthMapResult:
    """Aggregated output of a single Depth Anything V2 inference pass.

    Attributes:
        depth_map: Raw floating-point depth array of shape (image_height, image_width).
            Values represent relative depth (larger means closer or further depending on
            the model convention, typically larger is closer in Depth Anything).
        image_width: Width of the source image in pixels.
        image_height: Height of the source image in pixels.
        model_name: The identifier of the model used to produce this result.
    """

    depth_map: NDArray[np.float32]
    image_width: int
    image_height: int
    model_name: str

    @property
    def min_depth(self) -> float:
        """Return the minimum depth value in the map."""
        return float(np.min(self.depth_map))

    @property
    def max_depth(self) -> float:
        """Return the maximum depth value in the map."""
        return float(np.max(self.depth_map))

    @property
    def mean_depth(self) -> float:
        """Return the mean depth value in the map."""
        return float(np.mean(self.depth_map))

    @property
    def std_depth(self) -> float:
        """Return the standard deviation of depth values in the map."""
        return float(np.std(self.depth_map))
