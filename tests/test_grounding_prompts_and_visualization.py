from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import numpy as np
from PIL import Image

from vegetation_analysis.grounding import (
    POLE_PROMPT_VARIANTS,
    TREE_PROMPT_VARIANTS,
    DetectionBox,
    DetectionResult,
    DetectionVisualizer,
    PromptBuilder,
    build_vegetation_prompt,
)


def test_prompt_builder_formats_dot_separated_prompt() -> None:
    prompt = PromptBuilder().add("tree").add("power pole").build()

    assert prompt == "tree . power pole ."
    assert PromptBuilder.validate(prompt)


def test_prompt_variants_include_required_labels() -> None:
    assert {"tree", "large tree", "roadside tree"}.issubset(TREE_PROMPT_VARIANTS)
    assert {"utility pole", "electric pole", "power pole"}.issubset(
        POLE_PROMPT_VARIANTS
    )
    assert build_vegetation_prompt("large tree", "electric pole") == (
        "large tree . electric pole ."
    )


def test_detection_visualization_is_saved() -> None:
    image = np.zeros((80, 100, 3), dtype=np.uint8)
    result = DetectionResult(
        boxes=(
            DetectionBox(
                x_min=10,
                y_min=12,
                x_max=60,
                y_max=70,
                label="tree",
                confidence=0.88,
            ),
        ),
        image_width=100,
        image_height=80,
        prompt="tree .",
    )
    output_path = _test_output_dir() / "grounding.png"

    saved_path = DetectionVisualizer().save_visualization(
        image=image,
        result=result,
        output_path=output_path,
    )

    assert saved_path == output_path
    assert output_path.exists()
    with Image.open(output_path) as saved_image:
        assert saved_image.size == (100, 80)


def _test_output_dir() -> Path:
    path = Path("outputs") / "test_tmp" / f"grounding_visualization_{uuid4().hex}"
    path.mkdir(parents=True, exist_ok=False)
    return path
