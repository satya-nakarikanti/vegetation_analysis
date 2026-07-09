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

#: Default Hugging Face model identifier for Depth Anything V2 (small variant).
DEFAULT_MODEL_ID: str = "depth-anything/Depth-Anything-V2-Small-hf"

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
