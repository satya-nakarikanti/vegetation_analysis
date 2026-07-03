"""Constants and defaults for SAM 2 integration."""

from typing import Literal

DevicePreference = Literal["auto", "cpu", "cuda"]

#: Default SAM 2 model identifier.
#: The official repo provides configs like 'sam2.1_hiera_t.yaml',
#: 'sam2.1_hiera_s.yaml', etc.
DEFAULT_MODEL_CFG = "configs/sam2.1/sam2.1_hiera_t.yaml"

#: Default SAM 2 checkpoint path or identifier.
#: This usually points to a downloaded .pt file like 'sam2.1_hiera_tiny.pt'.
DEFAULT_CHECKPOINT: str = "checkpoints/sam2.1_hiera_tiny.pt"

#: Default device preference.
DEFAULT_DEVICE_PREFERENCE: DevicePreference = "auto"

#: Default score threshold for masks.
DEFAULT_MASK_THRESHOLD: float = 0.0
