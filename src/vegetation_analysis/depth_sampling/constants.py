"""Shared constants for the Depth Sampling package."""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Morphological Operations (Erosion)
# ---------------------------------------------------------------------------

#: Size of the square kernel used for mask erosion (e.g., 5 means 5x5).
DEFAULT_EROSION_KERNEL_SIZE: int = 5

#: Number of iterations to apply the erosion operation.
DEFAULT_EROSION_ITERATIONS: int = 2

#: Minimum number of pixels required in an eroded mask to use it for depth
#: sampling. If the eroded mask has fewer pixels than this, the sampler falls
#: back to the original uneroded mask.
MIN_ERODED_PIXELS: int = 25

# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

#: Color of the centroid marker in visualizations (RGB).
CENTROID_COLOR: tuple[int, int, int] = (255, 255, 255)  # White

#: Radius of the centroid marker in pixels.
CENTROID_RADIUS: int = 4
