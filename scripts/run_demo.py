"""Phase 2 demonstration script for the FastSAM segmentation pipeline.

This script orchestrates the existing production modules to demonstrate
end-to-end segmentation on a test image. It does not contain any
segmentation, mask extraction, or visualization logic; all such
responsibilities are delegated to the appropriate modules inside src/.

Usage
-----
Run from the project root with the virtual environment active:

    python scripts/run_demo.py

An optional image path may be supplied as the first positional argument:

    python scripts/run_demo.py path/to/image.jpg

If no path is given, the script looks for ``demo/sample.jpg`` relative to
the project root. When that file is also absent, a synthetic RGB image is
generated so the pipeline can always be demonstrated.

Outputs are written to ``outputs/demo/`` and include:

* ``overlay.png``       — segmentation mask overlay on the source image.
* ``statistics.json``   — per-object bounding boxes, areas, and summary.
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
from vegetation_analysis.segmentation import (
    FastSAMLoader,
    FastSAMModelConfig,
    FastSAMSegmenter,
    MaskVisualizer,
    SegmentedObject,
    extract_segmented_objects,
)
from vegetation_analysis.segmentation.segmenter import FastSAMInferenceConfig
from vegetation_analysis.utils.logging import configure_logging

# ---------------------------------------------------------------------------
# Module-level logger — configured by _setup() before first use.
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_DEFAULT_SAMPLE_PATH = Path("demo") / "sample.jpg"
_DEMO_OUTPUT_DIR = Path("outputs") / "demo"
_SYNTHETIC_IMAGE_SIZE = (480, 640, 3)  # (height, width, channels)


# ---------------------------------------------------------------------------
# Image loading
# ---------------------------------------------------------------------------


def load_image(image_path: Path) -> Image.Image:
    """Load an RGB image from *image_path* and return it as a PIL Image.

    Raises
    ------
    FileNotFoundError
        When *image_path* does not exist.
    ValueError
        When the file cannot be decoded as an image.
    """
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as exc:
        raise ValueError(f"Cannot read image at '{image_path}': {exc}") from exc

    logger.info("Loaded image from '%s' — size %s.", image_path, image.size)
    return image


def create_synthetic_image() -> Image.Image:
    """Create a minimal synthetic RGB image for pipeline demonstration.

    The image contains simple coloured rectangles that are visually distinct
    enough to produce at least one segmented object from FastSAM.
    """
    canvas = np.full(_SYNTHETIC_IMAGE_SIZE, fill_value=30, dtype=np.uint8)

    # Sky region (blue gradient approximation)
    canvas[:160, :, :] = [90, 140, 200]

    # Ground region
    canvas[380:, :, :] = [60, 100, 40]

    # Tree-like green blob
    canvas[80:340, 180:360, :] = [30, 130, 50]
    canvas[240:380, 240:310, :] = [100, 70, 35]  # trunk

    # Pole-like grey rectangle
    canvas[40:420, 480:510, :] = [160, 160, 160]
    canvas[110:130, 460:530, :] = [140, 140, 140]  # crossarm

    image = Image.fromarray(canvas, mode="RGB")
    logger.info(
        "No sample image found. Using synthetic image (%d × %d).",
        image.width,
        image.height,
    )
    return image


def resolve_input_image(cli_args: list[str]) -> tuple[Image.Image, str]:
    """Return the demo image and a human-readable source description.

    Resolution order:

    1. CLI argument (``sys.argv[1]``).
    2. ``demo/sample.jpg`` relative to the project root.
    3. Synthetic image generated in memory.

    Parameters
    ----------
    cli_args:
        ``sys.argv[1:]`` — may be empty.

    Returns
    -------
    tuple[Image.Image, str]
        The loaded or synthesised PIL image and a short source label used in
        runtime output.
    """
    if cli_args:
        image_path = Path(cli_args[0])
        return load_image(image_path), str(image_path)

    if _DEFAULT_SAMPLE_PATH.exists():
        return load_image(_DEFAULT_SAMPLE_PATH), str(_DEFAULT_SAMPLE_PATH)

    return create_synthetic_image(), "synthetic (generated)"


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def build_loader(settings: AppSettings) -> FastSAMLoader:
    """Construct a :class:`FastSAMLoader` from application settings."""
    config = FastSAMModelConfig(
        model_name=settings.fastsam_model_name,
        device_preference="auto",
    )
    return FastSAMLoader(config=config)


def build_segmenter(
    settings: AppSettings, model: object, device: str
) -> FastSAMSegmenter:
    """Construct a :class:`FastSAMSegmenter` from application settings."""
    inference_config = FastSAMInferenceConfig(
        image_size=settings.fastsam_image_size,
        confidence=settings.fastsam_confidence,
        iou=settings.fastsam_iou,
    )
    return FastSAMSegmenter(model=model, device=device, config=inference_config)


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------


def save_statistics(
    objects: list[SegmentedObject],
    image_size: tuple[int, int],
    inference_duration_s: float,
    output_dir: Path,
) -> Path:
    """Serialise per-object statistics and a summary to ``statistics.json``.

    Parameters
    ----------
    objects:
        Structured segmentation results produced by the pipeline.
    image_size:
        ``(width, height)`` of the source image in pixels.
    inference_duration_s:
        Wall-clock inference time in seconds.
    output_dir:
        Directory where ``statistics.json`` will be written.

    Returns
    -------
    Path
        Absolute path to the written file.
    """
    image_width, image_height = image_size
    total_pixels = image_width * image_height

    per_object = [
        {
            "id": obj.id,
            "area_px": obj.area,
            "area_pct": (
                round(obj.area / total_pixels * 100, 2) if total_pixels else 0.0
            ),
            "bbox": {
                "x_min": obj.bbox.x_min,
                "y_min": obj.bbox.y_min,
                "x_max": obj.bbox.x_max,
                "y_max": obj.bbox.y_max,
                "width_px": obj.bbox.width,
                "height_px": obj.bbox.height,
            },
            "has_geometry": obj.has_geometry,
        }
        for obj in objects
    ]

    payload = {
        "summary": {
            "image_width": image_width,
            "image_height": image_height,
            "total_pixels": total_pixels,
            "object_count": len(objects),
            "inference_time_s": round(inference_duration_s, 4),
        },
        "objects": per_object,
    }

    statistics_path = output_dir / "statistics.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    statistics_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info("Saved statistics to '%s'.", statistics_path)
    return statistics_path


# ---------------------------------------------------------------------------
# Runtime summary
# ---------------------------------------------------------------------------


def print_runtime_summary(
    image_source: str,
    image: Image.Image,
    objects: list[SegmentedObject],
    inference_duration_s: float,
    overlay_path: Path,
    statistics_path: Path,
) -> None:
    """Print a structured human-readable summary of the demo run.

    Parameters
    ----------
    image_source:
        Short label describing where the image came from.
    image:
        The source PIL image used for segmentation.
    objects:
        Structured segmentation results produced by the pipeline.
    inference_duration_s:
        Wall-clock inference time in seconds.
    overlay_path:
        Path of the saved overlay image.
    statistics_path:
        Path of the saved statistics file.
    """
    separator = "-" * 60
    print(separator)
    print("  Vegetation Analysis — Phase 2 Demo")
    print(separator)
    print(f"  Image source    : {image_source}")
    print(f"  Image size      : {image.width} × {image.height} px")
    print(f"  Inference time  : {inference_duration_s:.3f} s")
    print(f"  Objects found   : {len(objects)}")
    print()

    if objects:
        print("  Per-object summary:")
        for obj in objects:
            bbox = obj.bbox
            print(
                f"    Object {obj.id:>3d} | "
                f"area={obj.area:>7,d} px | "
                f"bbox=({bbox.x_min},{bbox.y_min})→({bbox.x_max},{bbox.y_max})"
            )
    else:
        print("  No objects were segmented.")

    print()
    print("  Output files:")
    print(f"    Overlay     : {overlay_path}")
    print(f"    Statistics  : {statistics_path}")
    print(separator)


# ---------------------------------------------------------------------------
# Pipeline orchestration
# ---------------------------------------------------------------------------


def run_demo(cli_args: list[str]) -> int:
    """Orchestrate the Phase 2 segmentation pipeline and save all outputs.

    Parameters
    ----------
    cli_args:
        Command-line arguments following the script name (``sys.argv[1:]``).

    Returns
    -------
    int
        Process exit code: ``0`` on success, ``1`` on error.
    """
    settings = AppSettings()
    configure_logging(level=settings.log_level)

    logger.info("Phase 2 demo starting.")

    # ------------------------------------------------------------------
    # 1. Resolve input image
    # ------------------------------------------------------------------
    try:
        image, image_source = resolve_input_image(cli_args)
    except (FileNotFoundError, ValueError) as exc:
        logger.error("Input image error: %s", exc)
        return 1

    # ------------------------------------------------------------------
    # 2. Load FastSAM model
    # ------------------------------------------------------------------
    logger.info("Loading FastSAM model '%s'.", settings.fastsam_model_name)
    try:
        loader = build_loader(settings)
        loaded_model = loader.load()
    except RuntimeError as exc:
        logger.error("Model loading failed: %s", exc)
        return 1

    logger.info("Model loaded on device '%s'.", loaded_model.device)

    # ------------------------------------------------------------------
    # 3. Run segmentation
    # ------------------------------------------------------------------
    segmenter = build_segmenter(
        settings=settings,
        model=loaded_model.model,
        device=loaded_model.device,
    )

    logger.info("Running segmentation.")
    t_start = time.perf_counter()
    try:
        raw_results = segmenter.segment(image)
    except (RuntimeError, ValueError, FileNotFoundError) as exc:
        logger.error("Segmentation failed: %s", exc)
        return 1
    inference_duration_s = time.perf_counter() - t_start
    logger.info("Segmentation complete in %.3f s.", inference_duration_s)

    # ------------------------------------------------------------------
    # 4. Convert raw results to structured objects
    # ------------------------------------------------------------------
    objects = extract_segmented_objects(raw_results)
    logger.info("Extracted %d segmented object(s).", len(objects))

    # ------------------------------------------------------------------
    # 5. Save overlay visualization
    # ------------------------------------------------------------------
    output_dir = _DEMO_OUTPUT_DIR
    overlay_path = output_dir / "overlay.png"
    visualizer = MaskVisualizer()
    try:
        visualizer.save_overlay(
            image=image,
            objects=objects,
            output_path=overlay_path,
        )
    except Exception as exc:  # noqa: BLE001
        logger.error("Visualization failed: %s", exc)
        return 1

    # ------------------------------------------------------------------
    # 6. Save statistics
    # ------------------------------------------------------------------
    statistics_path = save_statistics(
        objects=objects,
        image_size=image.size,
        inference_duration_s=inference_duration_s,
        output_dir=output_dir,
    )

    # ------------------------------------------------------------------
    # 7. Print runtime summary
    # ------------------------------------------------------------------
    print_runtime_summary(
        image_source=image_source,
        image=image,
        objects=objects,
        inference_duration_s=inference_duration_s,
        overlay_path=overlay_path,
        statistics_path=statistics_path,
    )

    logger.info("Phase 2 demo finished successfully.")
    return 0


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    raise SystemExit(run_demo(sys.argv[1:]))
