"""Grounding DINO detection package.

This package provides the Grounding DINO object detection pipeline for the
vegetation analysis module.  Grounding DINO replaces the previously planned
CLIP classification stage as the primary mechanism for identifying tree and
utility pole regions in an image.

Architecture
------------
The package follows the same layered architecture as the Phase 2 segmentation
package:

1. **Constants** — :mod:`~vegetation_analysis.grounding.constants` holds all
   shared default values and configuration strings.
2. **Model loading** — :mod:`~vegetation_analysis.grounding.grounding_dino_loader`
   handles checkpoint loading and device selection.
3. **Detection** — :mod:`~vegetation_analysis.grounding.detector` runs inference
   and converts raw model output into structured schemas.
4. **Schemas** — :mod:`~vegetation_analysis.grounding.schemas` defines
   :class:`~vegetation_analysis.grounding.schemas.DetectionBox` and
   :class:`~vegetation_analysis.grounding.schemas.DetectionResult`.
5. **Prompts** — :mod:`~vegetation_analysis.grounding.prompts` constructs and
   validates the dot-separated text prompts required by Grounding DINO.
6. **Visualization** — :mod:`~vegetation_analysis.grounding.visualization`
   renders bounding-box overlays for inspection and debugging.

Status
------
Phase 3.1 (loader and constants) is complete.  Detection inference will be
added in the approved Phase 3.2 implementation.
"""

from vegetation_analysis.grounding.constants import (
    ALT_VEGETATION_PROMPT,
    DEFAULT_BOX_THRESHOLD,
    DEFAULT_DEVICE_PREFERENCE,
    DEFAULT_MODEL_ID,
    DEFAULT_TEXT_THRESHOLD,
    DEFAULT_VEGETATION_PROMPT,
    EXTENDED_VEGETATION_PROMPT,
    LABEL_POLE,
    LABEL_TREE,
    LARGE_MODEL_ID,
    POLE_PROMPT_VARIANTS,
    TREE_PROMPT_VARIANTS,
    VEGETATION_LABELS,
)
from vegetation_analysis.grounding.detector import GroundingDINODetector
from vegetation_analysis.grounding.grounding_dino_loader import (
    GroundingDINOLoader,
    GroundingDINOModelConfig,
    LoadedGroundingDINO,
)
from vegetation_analysis.grounding.prompts import PromptBuilder, build_vegetation_prompt
from vegetation_analysis.grounding.schemas import DetectionBox, DetectionResult
from vegetation_analysis.grounding.visualization import DetectionVisualizer

__all__ = [
    # Constants
    "ALT_VEGETATION_PROMPT",
    "DEFAULT_BOX_THRESHOLD",
    "DEFAULT_DEVICE_PREFERENCE",
    "DEFAULT_MODEL_ID",
    "DEFAULT_TEXT_THRESHOLD",
    "DEFAULT_VEGETATION_PROMPT",
    "EXTENDED_VEGETATION_PROMPT",
    "LABEL_POLE",
    "LABEL_TREE",
    "LARGE_MODEL_ID",
    "POLE_PROMPT_VARIANTS",
    "TREE_PROMPT_VARIANTS",
    "VEGETATION_LABELS",
    # Loader
    "GroundingDINOLoader",
    "GroundingDINOModelConfig",
    "LoadedGroundingDINO",
    # Detector (stub)
    "GroundingDINODetector",
    # Schemas
    "DetectionBox",
    "DetectionResult",
    # Prompts
    "PromptBuilder",
    "build_vegetation_prompt",
    # Visualization
    "DetectionVisualizer",
]
