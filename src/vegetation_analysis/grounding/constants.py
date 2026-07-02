"""Shared constants for the Grounding DINO detection package.

All default values, model identifiers, threshold parameters, and prompt
strings used across the grounding package are defined here.  No other module
in the package should hardcode these values; they must always be imported from
this module.
"""

from __future__ import annotations

from typing import Literal

# ---------------------------------------------------------------------------
# Model identifiers
# ---------------------------------------------------------------------------

#: Default Hugging Face model identifier for Grounding DINO (small variant).
#: Approx. 172 MB — suitable for evaluation and CPU inference.
DEFAULT_MODEL_ID: str = "IDEA-Research/grounding-dino-tiny"

#: Hugging Face model identifier for the larger Grounding DINO checkpoint.
#: Approx. 360 MB — better recall, higher compute requirements.
LARGE_MODEL_ID: str = "IDEA-Research/grounding-dino-base"

# ---------------------------------------------------------------------------
# Device selection
# ---------------------------------------------------------------------------

#: Device preference used when none is specified by the caller.
#: ``"auto"`` selects CUDA when available and falls back to CPU.
DEFAULT_DEVICE_PREFERENCE: Literal["auto", "cpu", "cuda"] = "auto"

# ---------------------------------------------------------------------------
# Detection thresholds
# ---------------------------------------------------------------------------

#: Minimum objectness score required for a predicted box to be retained.
#: Boxes with a score below this value are discarded before label assignment.
DEFAULT_BOX_THRESHOLD: float = 0.35

#: Minimum token-level alignment score required for a label to be assigned.
#: Used in conjunction with :data:`DEFAULT_BOX_THRESHOLD`.
DEFAULT_TEXT_THRESHOLD: float = 0.25

# ---------------------------------------------------------------------------
# Object category labels
# ---------------------------------------------------------------------------

#: Label string for tree detections.
LABEL_TREE: str = "tree"

#: Label string for utility pole detections.
LABEL_POLE: str = "utility pole"

#: Prompt variants that can improve utility-pole recall across datasets.
POLE_PROMPT_VARIANTS: tuple[str, ...] = (
    "utility pole",
    "electric pole",
    "power pole",
)

#: Prompt variants that can improve tree recall across canopy types.
TREE_PROMPT_VARIANTS: tuple[str, ...] = (
    "tree",
    "large tree",
    "roadside tree",
)

#: All recognised object category labels for vegetation analysis.
VEGETATION_LABELS: tuple[str, ...] = (LABEL_TREE, LABEL_POLE)

# ---------------------------------------------------------------------------
# Default text prompts
# ---------------------------------------------------------------------------

#: Default Grounding DINO text prompt for detecting trees and utility poles.
#: Grounding DINO expects labels separated by " . " and terminated with " .".
DEFAULT_VEGETATION_PROMPT: str = "tree . utility pole ."

#: Alternative prompt that uses "electric pole" instead of "utility pole".
#: May improve recall on images where the pole type differs from utility poles.
ALT_VEGETATION_PROMPT: str = "tree . electric pole ."

#: Extended prompt that also targets wires/conductors near poles.
EXTENDED_VEGETATION_PROMPT: str = "tree . utility pole . wire ."
