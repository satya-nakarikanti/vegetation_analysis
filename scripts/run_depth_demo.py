"""Phase 5.1 demonstration script for Depth Anything V2 estimation.

This script orchestrates the production Depth Anything V2 modules without
duplicating model loading, inference, post-processing, or visualization logic.

Usage
-----
Run from the project root with the virtual environment active:

    python scripts/run_depth_demo.py

An optional image path may be supplied as the first positional argument:

    python scripts/run_depth_demo.py path/to/image.jpg

Outputs are written to ``outputs/demo/`` and include:

* ``depth_map.png`` - colorized depth heatmap.
* ``depth.npy`` - raw floating-point depth array.
* ``depth_statistics.json`` - inference summary and statistics.
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
from vegetation_analysis.depth import (
    DepthAnythingLoader,
    DepthAnythingModelConfig,
    DepthEstimator,
    DepthMapResult,
    DepthVisualizer,
)
from vegetation_analysis.utils.logging import configure_logging

logger = logging.getLogger(__name__)

_DEFAULT_SAMPLE_PATH = Path("demo") / "sample.jpg"
_DEMO_OUTPUT_DIR = Path("outputs") / "demo"
_ANNOTATED_FILENAME = "depth_map.png"
_RAW_FILENAME = "depth.npy"
_STATISTICS_FILENAME = "depth_statistics.json"
_SYNTHETIC_IMAGE_SIZE = (1080, 1440, 3)


def load_image(image_path: Path) -> Image.Image:
    """Load an RGB image from disk."""

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    try:
        return Image.open(image_path).convert("RGB")
    except Exception as exc:
        raise ValueError(f"Cannot read image at '{image_path}': {exc}") from exc


def create_synthetic_image() -> Image.Image:
    """Create a simple synthetic image for demo smoke runs."""

    canvas = np.full(_SYNTHETIC_IMAGE_SIZE, fill_value=235, dtype=np.uint8)
    canvas[:360, :, :] = [120, 170, 220]
    canvas[720:, :, :] = [70, 120, 65]
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
    result: DepthMapResult,
    inference_duration_s: float,
    output_dir: Path,
    device: str,
) -> Path:
    """Serialize depth statistics to JSON."""

    # Using basic hardware detection for the JSON summary
    device_name = "CUDA" if device == "cuda" else "CPU"

    payload = {
        "image_size": [result.image_width, result.image_height],
        "model_name": result.model_name,
        "device": device,
        "gpu_cpu": device_name,
        "inference_time": round(inference_duration_s, 4),
        "depth_map_dimensions": list(result.depth_map.shape),
        "minimum_depth": round(result.min_depth, 4),
        "maximum_depth": round(result.max_depth, 4),
        "mean_depth": round(result.mean_depth, 4),
        "std_depth": round(result.std_depth, 4),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    statistics_path = output_dir / _STATISTICS_FILENAME
    statistics_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info("Saved Depth Anything V2 statistics to '%s'.", statistics_path)
    return statistics_path


def print_runtime_summary(
    image_source: str,
    result: DepthMapResult,
    inference_duration_s: float,
    annotated_path: Path,
    raw_path: Path,
    statistics_path: Path,
    device: str,
) -> None:
    """Print a concise human-readable demo summary."""

    device_name = "CUDA" if device == "cuda" else "CPU"
    separator = "-" * 60

    # As requested by the user:
    # Image source : demo/sample.jpg
    # Image size : 1080 x 1440
    # Model : Depth Anything V2 Small
    # Device : CUDA
    # Inference : 0.83 s
    # Depth range :
    # Minimum : ...
    # Maximum : ...
    # Mean : ...
    # Std : ...
    # Outputs
    # Depth Map : outputs/demo/depth_map.png
    # Raw Depth : outputs/demo/depth.npy
    # Statistics : outputs/demo/depth_statistics.json

    model_display_name = "Depth Anything V2 Small"
    if "base" in result.model_name.lower():
        model_display_name = "Depth Anything V2 Base"
    elif "large" in result.model_name.lower():
        model_display_name = "Depth Anything V2 Large"

    print(separator)
    print("  Vegetation Analysis - Phase 5.1 Depth Anything V2 Demo")
    print(separator)
    print(f"  Image source    : {image_source}")
    print(f"  Image size      : {result.image_width} x {result.image_height}")
    print(f"  Model           : {model_display_name}")
    print(f"  Device          : {device_name}")
    print(f"  Inference       : {inference_duration_s:.2f} s")
    print("  Depth range     :")
    print(f"    Minimum       : {result.min_depth:.4f}")
    print(f"    Maximum       : {result.max_depth:.4f}")
    print(f"    Mean          : {result.mean_depth:.4f}")
    print(f"    Std           : {result.std_depth:.4f}")
    print()
    print("  Outputs")
    print(f"  Depth Map       : {annotated_path.as_posix()}")
    print(f"  Raw Depth       : {raw_path.as_posix()}")
    print(f"  Statistics      : {statistics_path.as_posix()}")
    print(separator)


def run_demo(cli_args: list[str]) -> int:
    """Run Depth Anything V2 estimation and write demo outputs."""

    settings = AppSettings()
    configure_logging(level=settings.log_level)

    try:
        image, image_source = resolve_input_image(cli_args)
    except (FileNotFoundError, ValueError) as exc:
        logger.error("Input image error: %s", exc)
        return 1

    config = DepthAnythingModelConfig(device_preference="auto")
    try:
        loaded_model = DepthAnythingLoader(config=config).load()
    except RuntimeError as exc:
        logger.error("Depth Anything V2 model loading failed: %s", exc)
        return 1

    estimator = DepthEstimator(loaded_model=loaded_model)

    t_start = time.perf_counter()
    try:
        result = estimator.estimate(image=image)
    except (RuntimeError, ValueError, FileNotFoundError) as exc:
        logger.error("Depth Anything V2 estimation failed: %s", exc)
        return 1
    inference_duration_s = time.perf_counter() - t_start

    output_dir = _DEMO_OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    annotated_path = output_dir / _ANNOTATED_FILENAME
    raw_path = output_dir / _RAW_FILENAME

    try:
        # Save raw numpy array
        np.save(raw_path, result.depth_map)
        logger.info("Saved raw depth array to '%s'.", raw_path)

        # Save visualization
        DepthVisualizer().save_visualization(
            result=result,
            output_path=annotated_path,
        )
    except Exception as exc:  # noqa: BLE001
        logger.error("Output saving failed: %s", exc)
        return 1

    statistics_path = save_statistics(
        result=result,
        inference_duration_s=inference_duration_s,
        output_dir=output_dir,
        device=loaded_model.device,
    )

    print_runtime_summary(
        image_source=image_source,
        result=result,
        inference_duration_s=inference_duration_s,
        annotated_path=annotated_path,
        raw_path=raw_path,
        statistics_path=statistics_path,
        device=loaded_model.device,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(run_demo(sys.argv[1:]))
