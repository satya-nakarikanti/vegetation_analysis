"""Shared constants for the Depth Anything V2 package.

All default values, model identifiers, and prompt strings used across the depth package
are defined here. No other module in the package should hardcode these values; they
must always be imported from this module.
"""

from __future__ import annotations

from typing import Literal

# ---------------------------------------------------------------------------
# Model identifiers
# ---------------------------------------------------------------------------

#: Default path to the Metric Depth Anything V2 weights.
DEFAULT_MODEL_ID: str = "third_party/Depth-Anything-V2/checkpoints/depth_anything_v2_metric_vkitti_vits.pth"

#: Alias for the metric model (now the only supported model).
METRIC_MODEL_ID: str = DEFAULT_MODEL_ID
# ---------------------------------------------------------------------------
# Device selection
# ---------------------------------------------------------------------------

#: Device preference used when none is specified by the caller.
#: ``"auto"`` selects CUDA when available and falls back to CPU.
DEFAULT_DEVICE_PREFERENCE: Literal["auto", "cpu", "cuda"] = "auto"

# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

#: Default colormap used for visualizing depth.
DEFAULT_COLORMAP: int = 2  # cv2.COLORMAP_INFERNO
