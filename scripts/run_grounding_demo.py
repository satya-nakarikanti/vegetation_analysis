"""Phase 3Ademonstration script for Grounding DINO detection.

This script orchestrates the production Grounding DINO modules without
duplicating model loading, inference, post-processing, or visualization logic.

Usage
-----
Run from the project root with the virtual environment active:

    python scripts/run_grounding_demo.py

An optional image path may be supplied as the first positional argument:

    python scripts/run_grounding_demo.py path/to/image.jpg

Outputs are written to ``outputs/demo/`` and include:

* ``grounding_dino_annotated.png`` - detection boxes and labels.
* ``grounding_dino_statistics.json`` - detection summary and per-box data.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

from vegetation_analysis.config.settings import AppSettings
from vegetation_analysis.grounding import (
    DEFAULT_VEGETATION_PROMPT,
    DetectionResult,
    DetectionVisualizer,
    GroundingDINODetector,
    GroundingDINOLoader,
    GroundingDINOModelConfig,
)
from vegetation_analysis.utils.logging import configure_logging

logger = logging.getLogger(__name__)

_DEFAULT_SAMPLE_PATH = Path("demo") / "sample.jpg"
_DEMO_OUTPUT_DIR = Path("outputs") / "demo"
_ANNOTATED_FILENAME = "grounding_dino_annotated.png"
_STATISTICS_FILENAME = "grounding_dino_statistics.json"
_SYNTHETIC_IMAGE_SIZE = (480, 640, 3)


def load_image(image_path: Path) -> Image.Image:
    """Load an RGB image from disk."""

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    try:
        return Image.open(image_path).convert("RGB")
    except Exception as exc:
        raise ValueError(f"Cannot read image at '{image_path}': {exc}") from exc


def create_synthetic_image() -> Image.Image:
    """Create a simple tree-and-pole image for demo smoke runs."""

    canvas = np.full(_SYNTHETIC_IMAGE_SIZE, fill_value=235, dtype=np.uint8)
    canvas[:180, :, :] = [120, 170, 220]
    canvas[380:, :, :] = [70, 120, 65]
    canvas[80:320, 170:360, :] = [35, 135, 55]
    canvas[245:380, 255:300, :] = [120, 80, 45]
    canvas[40:430, 485:515, :] = [165, 165, 160]
    canvas[120:140, 455:545, :] = [145, 145, 140]
    logger.info("No sample image found. Using synthetic image.")
    return Image.fromarray(canvas, mode="RGB")


def resolve_input_image(cli_args: list[str]) -> tuple[Image.Image, str]:
    """Resolve the demo image from CLI, sample path, or synthetic fallback."""

    if cli_args:
        image_path = Path(cli_args[0])
        return load_image(image_path), str(image_path)

    if _DEFAULT_SAMPLE_PATH.exists():
        return load_image(_DEFAULT_SAMPLE_PATH), str(_DEFAULT_SAMPLE_PATH)

    return create_synthetic_image(), "synthetic (generated)"


def save_statistics(
    result: DetectionResult,
    inference_duration_s: float,
    output_dir: Path,
) -> Path:
    """Serialize detection statistics to JSON."""

    payload = {
        "summary": {
            "image_width": result.image_width,
            "image_height": result.image_height,
            "prompt": result.prompt,
            "box_count": len(result.boxes),
            "inference_time_s": round(inference_duration_s, 4),
        },
        "boxes": [
            {
                "label": box.label,
                "confidence": round(box.confidence, 4),
                "bbox": {
                    "x_min": box.x_min,
                    "y_min": box.y_min,
                    "x_max": box.x_max,
                    "y_max": box.y_max,
                    "width_px": box.width,
                    "height_px": box.height,
                    "area_px": box.area,
                },
            }
            for box in result.boxes
        ],
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    statistics_path = output_dir / _STATISTICS_FILENAME
    statistics_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info("Saved Grounding DINO statistics to '%s'.", statistics_path)
    return statistics_path


def print_runtime_summary(
    image_source: str,
    result: DetectionResult,
    inference_duration_s: float,
    annotated_path: Path,
    statistics_path: Path,
) -> None:
    """Print a concise human-readable demo summary."""

    separator = "-" * 60
    print(separator)
    print("  Vegetation Analysis - Phase 3AGrounding DINO Demo")
    print(separator)
    print(f"  Image source    : {image_source}")
    print(f"  Image size      : {result.image_width} x {result.image_height} px")
    print(f"  Prompt          : {result.prompt}")
    print(f"  Inference time  : {inference_duration_s:.3f} s")
    print(f"  Boxes found     : {len(result.boxes)}")
    print()

    if result.boxes:
        print("  Detection summary:")
        for box in result.boxes:
            print(
                f"    {box.label:<16} | "
                f"conf={box.confidence:.3f} | "
                f"bbox=({box.x_min},{box.y_min})->({box.x_max},{box.y_max})"
            )
    else:
        print("  No boxes were detected.")

    print()
    print("  Output files:")
    print(f"    Annotated   : {annotated_path}")
    print(f"    Statistics  : {statistics_path}")
    print(separator)


def run_demo(cli_args: list[str]) -> int:
    """Run Grounding DINO detection and write demo outputs."""

    settings = AppSettings()
    configure_logging(level=settings.log_level)

    try:
        image, image_source = resolve_input_image(cli_args)
    except (FileNotFoundError, ValueError) as exc:
        logger.error("Input image error: %s", exc)
        return 1

    config = GroundingDINOModelConfig(device_preference="auto")
    try:
        loaded_model = GroundingDINOLoader(config=config).load()
    except RuntimeError as exc:
        logger.error("Grounding DINO model loading failed: %s", exc)
        return 1

    detector = GroundingDINODetector(
        loaded_model=loaded_model,
        box_threshold=config.box_threshold,
        text_threshold=config.text_threshold,
    )

    t_start = time.perf_counter()
    try:
        result = detector.detect(image=image, prompt=DEFAULT_VEGETATION_PROMPT)
    except (RuntimeError, ValueError, FileNotFoundError) as exc:
        logger.error("Grounding DINO detection failed: %s", exc)
        return 1
    inference_duration_s = time.perf_counter() - t_start

    output_dir = _DEMO_OUTPUT_DIR
    annotated_path = output_dir / _ANNOTATED_FILENAME
    try:
        DetectionVisualizer().save_visualization(
            image=image,
            result=result,
            output_path=annotated_path,
        )
    except Exception as exc:  # noqa: BLE001
        logger.error("Visualization failed: %s", exc)
        return 1

    statistics_path = save_statistics(
        result=result,
        inference_duration_s=inference_duration_s,
        output_dir=output_dir,
    )
    print_runtime_summary(
        image_source=image_source,
        result=result,
        inference_duration_s=inference_duration_s,
        annotated_path=annotated_path,
        statistics_path=statistics_path,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(run_demo(sys.argv[1:]))
