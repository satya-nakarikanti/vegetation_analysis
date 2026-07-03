"""Phase 3B.1 demonstration script for SAM 2 segmentation.

This script orchestrates the production Grounding DINO + SAM 2 modules.

Usage
-----
Run from the project root with the virtual environment active:

    python scripts/run_sam2_demo.py

An optional image path may be supplied as the first positional argument:

    python scripts/run_sam2_demo.py path/to/image.jpg

Outputs are written to ``outputs/demo/`` and include:

* ``sam2_annotated.png`` - detection boxes, masks, and labels.
* ``sam2_statistics.json`` - segmentation summary and per-object data.
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
    GroundingDINODetector,
    GroundingDINOLoader,
    GroundingDINOModelConfig,
)
from vegetation_analysis.sam2 import (
    SAM2Loader,
    SAM2ModelConfig,
    SAM2Segmenter,
    SegmentationResult,
    SegmentationVisualizer,
)
from vegetation_analysis.utils.logging import configure_logging

logger = logging.getLogger(__name__)

_DEFAULT_SAMPLE_PATH = Path("demo") / "sample.jpg"
_DEMO_OUTPUT_DIR = Path("outputs") / "demo"
_ANNOTATED_FILENAME = "sam2_annotated.png"
_STATISTICS_FILENAME = "sam2_statistics.json"
_SYNTHETIC_IMAGE_SIZE = (480, 640, 3)


def load_image(image_path: Path) -> Image.Image:
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    try:
        return Image.open(image_path).convert("RGB")
    except Exception as exc:
        raise ValueError(f"Cannot read image at '{image_path}': {exc}") from exc


def create_synthetic_image() -> Image.Image:
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
    if cli_args:
        image_path = Path(cli_args[0])
        return load_image(image_path), str(image_path)
    if _DEFAULT_SAMPLE_PATH.exists():
        return load_image(_DEFAULT_SAMPLE_PATH), str(_DEFAULT_SAMPLE_PATH)
    return create_synthetic_image(), "synthetic (generated)"


def save_statistics(
    result: SegmentationResult,
    dino_time_s: float,
    sam2_time_s: float,
    output_dir: Path,
) -> Path:
    payload = {
        "summary": {
            "image_width": result.image_width,
            "image_height": result.image_height,
            "object_count": len(result.objects),
            "dino_inference_time_s": round(dino_time_s, 4),
            "sam2_inference_time_s": round(sam2_time_s, 4),
            "total_pipeline_time_s": round(dino_time_s + sam2_time_s, 4),
        },
        "objects": [
            {
                "label": obj.label,
                "confidence": round(obj.confidence, 4),
                "bbox": {
                    "x_min": obj.box.x_min,
                    "y_min": obj.box.y_min,
                    "x_max": obj.box.x_max,
                    "y_max": obj.box.y_max,
                },
                "mask_area_px": obj.mask_area_pixels,
            }
            for obj in result.objects
        ],
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    statistics_path = output_dir / _STATISTICS_FILENAME
    statistics_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info("Saved SAM 2 statistics to '%s'.", statistics_path)
    return statistics_path


def print_runtime_summary(
    image_source: str,
    result: SegmentationResult,
    dino_time_s: float,
    sam2_time_s: float,
    annotated_path: Path,
    statistics_path: Path,
) -> None:
    separator = "-" * 60
    print(separator)
    print("  Vegetation Analysis - Phase 3B.2 SAM 2 Demo")
    print(separator)
    print(f"  Image source    : {image_source}")
    print(f"  Image size      : {result.image_width} x {result.image_height} px")
    print()
    print("  Pipeline:")
    print("    Grounding DINO")
    print("      ↓")
    print("    Duplicate Pole Filtering")
    print("      ↓")
    print("    SAM 2 Segmentation")
    print()
    print(f"  Prompt          : {DEFAULT_VEGETATION_PROMPT}")
    print()
    print(f"  DINO time       : {dino_time_s:.3f} s")
    print(f"  SAM 2 time      : {sam2_time_s:.3f} s")
    print(f"  Objects found   : {len(result.objects)}")
    print()

    if result.objects:
        print("  Segmentation summary:")
        for obj in result.objects:
            print(
                f"    {obj.label:<16} | "
                f"conf={obj.confidence:.3f} | "
                f"area={obj.mask_area_pixels} px"
            )
    else:
        print("  No objects were segmented.")

    print()
    print("  Output files:")
    print(f"    Annotated   : {annotated_path}")
    print(f"    Statistics  : {statistics_path}")
    print(separator)


def run_demo(cli_args: list[str]) -> int:
    settings = AppSettings()
    configure_logging(level=settings.log_level)

    try:
        image, image_source = resolve_input_image(cli_args)
    except (FileNotFoundError, ValueError) as exc:
        logger.error("Input image error: %s", exc)
        return 1

    # Load Models
    try:
        dino_config = GroundingDINOModelConfig(device_preference="auto")
        loaded_dino = GroundingDINOLoader(config=dino_config).load()
        detector = GroundingDINODetector(
            loaded_model=loaded_dino,
            box_threshold=dino_config.box_threshold,
            text_threshold=dino_config.text_threshold,
        )

        sam2_config = SAM2ModelConfig(device_preference="auto")
        loaded_sam2 = SAM2Loader(config=sam2_config).load()
        segmenter = SAM2Segmenter(model=loaded_sam2)
    except RuntimeError as exc:
        logger.error("Model loading failed: %s", exc)
        return 1

    # Run Grounding DINO
    t0 = time.perf_counter()
    try:
        dino_result = detector.detect(image=image, prompt=DEFAULT_VEGETATION_PROMPT)
    except Exception as exc:
        logger.error("Grounding DINO detection failed: %s", exc)
        return 1
    dino_time_s = time.perf_counter() - t0

    # Run SAM 2
    t1 = time.perf_counter()
    try:
        sam2_result = segmenter.segment(image=image, detection_result=dino_result)
    except Exception as exc:
        logger.error("SAM 2 segmentation failed: %s", exc)
        return 1
    sam2_time_s = time.perf_counter() - t1

    output_dir = _DEMO_OUTPUT_DIR
    annotated_path = output_dir / _ANNOTATED_FILENAME
    try:
        SegmentationVisualizer().save_visualization(
            image=image,
            result=sam2_result,
            output_path=annotated_path,
        )
    except Exception as exc:  # noqa: BLE001
        logger.error("Visualization failed: %s", exc)
        return 1

    statistics_path = save_statistics(
        result=sam2_result,
        dino_time_s=dino_time_s,
        sam2_time_s=sam2_time_s,
        output_dir=output_dir,
    )
    print_runtime_summary(
        image_source=image_source,
        result=sam2_result,
        dino_time_s=dino_time_s,
        sam2_time_s=sam2_time_s,
        annotated_path=annotated_path,
        statistics_path=statistics_path,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(run_demo(sys.argv[1:]))
