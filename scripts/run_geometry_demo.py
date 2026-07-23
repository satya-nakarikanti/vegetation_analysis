"""Phase 5.3 demonstration script for the Relative Geometry Engine.

This script orchestrates the full pipeline:

    Grounding DINO -> SAM2 -> Depth Anything V2 -> Depth Sampling
    -> Relative Geometry Engine

Usage
-----
Run from the project root with the virtual environment active:

    python scripts/run_geometry_demo.py

An optional image path may be supplied as the first positional argument:

    python scripts/run_geometry_demo.py path/to/image.jpg

Outputs are written to ``outputs/demo/`` and include:

* ``geometry_annotated.png``  — geometry overlay on the depth heatmap.
* ``geometry_statistics.json`` — per-object relative geometry data.
* ``depth_grayscale.png``     — normalised grayscale depth (no colormap).
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
)
from vegetation_analysis.depth.visualization import DepthVisualizer
from vegetation_analysis.depth_sampling import DepthSampler
from vegetation_analysis.geometry import (
    GeometryEngine,
    GeometryVisualizer,
)
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
)
from vegetation_analysis.utils.logging import configure_logging

logger = logging.getLogger(__name__)

_DEFAULT_SAMPLE_PATH = Path("demo") / "sample.jpg"
_DEMO_OUTPUT_DIR = Path("outputs") / "demo"
_ANNOTATED_FILENAME = "geometry_annotated.png"
_STATISTICS_FILENAME = "geometry_statistics.json"
_GRAYSCALE_FILENAME = "depth_grayscale.png"
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


def save_statistics(geometry_result: object, output_dir: Path) -> Path:
    from vegetation_analysis.geometry.schemas import GeometryResult

    assert isinstance(geometry_result, GeometryResult)  # noqa: S101

    payload = {
        "metadata": {
            "image_width": geometry_result.metadata.width,
            "image_height": geometry_result.metadata.height,
            "principal_x": round(geometry_result.metadata.principal_x, 2),
            "principal_y": round(geometry_result.metadata.principal_y, 2),
            "depth_model": geometry_result.metadata.depth_model,
            "coordinate_system": geometry_result.metadata.coordinate_system.value,
        },
        "objects": [
            {
                "label": obj.label,
                "species": obj.species,
                "species_confidence": obj.species_confidence,
                "confidence": round(obj.confidence, 4),
                "pixel_count": obj.pixel_count,
                "centroid_x": round(obj.centroid_x, 2),
                "centroid_y": round(obj.centroid_y, 2),
                "camera_x": round(obj.camera_x, 6),
                "camera_y": round(obj.camera_y, 6),
                "camera_z": round(obj.camera_z, 6),
                "camera_distance": round(obj.camera_distance, 6),
                "relative_x": round(obj.relative_x, 6)
                if obj.relative_x is not None
                else None,
                "relative_y": round(obj.relative_y, 6)
                if obj.relative_y is not None
                else None,
                "relative_z": round(obj.relative_z, 6)
                if obj.relative_z is not None
                else None,
            }
            for obj in geometry_result.objects
        ],
        "relationships": [
            {
                "object_a_label": rel.object_a_label,
                "object_b_label": rel.object_b_label,
                "angle_radians": round(rel.angle_radians, 6),
                "angle_degrees": round(rel.angle_degrees, 6),
                "dot_product": round(rel.dot_product, 6),
                "centroid_distance_law_of_cosines": round(rel.centroid_distance_law_of_cosines, 6),
                "centroid_distance_euclidean": round(rel.centroid_distance_euclidean, 6),
                "distance_difference": rel.distance_difference,
                "validation_passed": rel.validation_passed
            }
            for rel in geometry_result.relationships
        ]
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    statistics_path = output_dir / _STATISTICS_FILENAME
    statistics_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info("Saved geometry statistics to '%s'.", statistics_path)
    return statistics_path


def print_runtime_summary(
    image_source: str,
    geometry_result: object,
    dino_time_s: float,
    sam2_time_s: float,
    depth_time_s: float,
    sampling_time_s: float,
    projection_time_s: float,
    geometry_time_s: float,
    annotated_path: Path,
    statistics_path: Path,
    grayscale_path: Path,
) -> None:
    from vegetation_analysis.geometry.schemas import GeometryResult

    assert isinstance(geometry_result, GeometryResult)  # noqa: S101

    sep = "-" * 62
    print(sep)
    print("  Vegetation Analysis - Phase 5.3 / 6.2 Geometry Demo")
    print(sep)
    print(f"  Image source    : {image_source}")
    print(
        f"  Image size      : "
        f"{geometry_result.metadata.width} x "
        f"{geometry_result.metadata.height} px"
    )
    print(
        f"  Principal point : "
        f"({geometry_result.metadata.principal_x:.1f}, "
        f"{geometry_result.metadata.principal_y:.1f})"
    )
    print()
    print("  Pipeline:")
    print("    Grounding DINO")
    print("      |")
    print("      v")
    print("    SAM 2 Segmentation")
    print("      |")
    print("      v")
    print("    Depth Anything V2")
    print("      |")
    print("      v")
    print("    Depth Sampling")
    print("      |")
    print("      v")
    print("    Projection Engine")
    print("      |")
    print("      v")
    print("    Geometry Engine")
    print()
    print(f"  DINO time       : {dino_time_s:.3f} s")
    print(f"  SAM 2 time      : {sam2_time_s:.3f} s")
    print(f"  Depth time      : {depth_time_s:.3f} s")
    print(f"  Sampling time   : {sampling_time_s:.3f} s")
    print(f"  Proj time       : {projection_time_s:.3f} s")
    print(f"  Geometry time   : {geometry_time_s:.3f} s")
    print(f"  Objects found   : {len(geometry_result.objects)}")
    print()

    if geometry_result.objects:
        print("  Geometry summary:")
        for obj in geometry_result.objects:
            label = obj.label if obj.species is None else f"{obj.label}/{obj.species}"
            print(
                f"    {label:<18} | "
                f"cx={obj.camera_x:+.3f}  "
                f"cy={obj.camera_y:+.3f}  "
                f"cz={obj.camera_z:+.3f}  "
                f"dist={obj.camera_distance:+.3f}"
            )
    else:
        print("  No objects detected.")
        
    if geometry_result.relationships:
        print()
        print("  Camera Relationships:")
        
        # Build a lookup for distances so we can print them cleanly
        dist_lookup = {obj.label: obj.camera_distance for obj in geometry_result.objects}
        
        for rel in geometry_result.relationships:
            lbl_a = rel.object_a_label
            lbl_b = rel.object_b_label
            
            dist_a = dist_lookup.get(lbl_a, 0.0)
            dist_b = dist_lookup.get(lbl_b, 0.0)
            
            print(f"    {lbl_a} <-> {lbl_b}")
            print(f"    Camera->{lbl_a.capitalize()} : {dist_a:.3f} m")
            print(f"    Camera->{lbl_b.capitalize()} : {dist_b:.3f} m")
            print(f"    Angle        : {rel.angle_degrees:.2f}°")
            
        print()
        print("  Centroid Distance Validation")
        for rel in geometry_result.relationships:
            lbl_a = rel.object_a_label
            lbl_b = rel.object_b_label
            pass_str = "PASS" if rel.validation_passed else "FAIL"
            
            print(f"    {lbl_a} <-> {lbl_b}")
            print(f"    Law of Cosines : {rel.centroid_distance_law_of_cosines:.3f} m")
            print(f"    Euclidean      : {rel.centroid_distance_euclidean:.3f} m")
            print(f"    Difference     : {rel.distance_difference:.9f} m")
            print(f"    Validation     : {pass_str}")
            print()

    print("  Output files:")
    print(f"    Annotated       : {annotated_path}")
    print(f"    Statistics      : {statistics_path}")
    print(f"    Depth grayscale : {grayscale_path}")
    print(sep)


def run_demo(cli_args: list[str]) -> int:
    settings = AppSettings()
    configure_logging(level=settings.log_level)

    try:
        image, image_source = resolve_input_image(cli_args)
    except (FileNotFoundError, ValueError) as exc:
        logger.error("Input image error: %s", exc)
        return 1

    # ------------------------------------------------------------------ #
    # Load models
    # ------------------------------------------------------------------ #
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

        # Allow selecting Relative or Metric through configuration or a runtime flag
        # For this phase, we use the Metric model.
        from vegetation_analysis.depth.constants import METRIC_MODEL_ID
        depth_config = DepthAnythingModelConfig(
            model_id=METRIC_MODEL_ID, device_preference="auto"
        )
        loaded_depth = DepthAnythingLoader(config=depth_config).load()
        depth_estimator = DepthEstimator(loaded_model=loaded_depth)

        sampler = DepthSampler()
        
        from vegetation_analysis.projection import ProjectionEngine
        from vegetation_analysis.projection.schemas import CoordinateSystem
        projection_engine = ProjectionEngine(coordinate_system=CoordinateSystem.METRIC)
        
        geometry_engine = GeometryEngine()

    except RuntimeError as exc:
        logger.error("Model loading failed: %s", exc)
        return 1

    # ------------------------------------------------------------------ #
    # Run pipeline
    # ------------------------------------------------------------------ #
    t0 = time.perf_counter()
    try:
        dino_result = detector.detect(image=image, prompt=DEFAULT_VEGETATION_PROMPT)
    except Exception as exc:
        logger.error("Grounding DINO detection failed: %s", exc)
        return 1
    dino_time_s = time.perf_counter() - t0

    t1 = time.perf_counter()
    try:
        sam2_result = segmenter.segment(image=image, detection_result=dino_result)
    except Exception as exc:
        logger.error("SAM 2 segmentation failed: %s", exc)
        return 1
    sam2_time_s = time.perf_counter() - t1

    t2 = time.perf_counter()
    try:
        depth_result = depth_estimator.estimate(image=image)
    except Exception as exc:
        logger.error("Depth estimation failed: %s", exc)
        return 1
    depth_time_s = time.perf_counter() - t2

    t3 = time.perf_counter()
    try:
        sampling_result = sampler.sample(
            segmentation=sam2_result,
            depth=depth_result,
        )
    except Exception as exc:
        logger.error("Depth sampling failed: %s", exc)
        return 1
    sampling_time_s = time.perf_counter() - t3

    t4 = time.perf_counter()
    try:
        from vegetation_analysis.projection.schemas import CameraIntrinsics
        # For demo/testing only, use: fx = fy = 1000.0, cx = w/2, cy = h/2
        # These are temporary demonstration values and NOT production calibration.
        intrinsics = CameraIntrinsics(
            fx=1000.0,
            fy=1000.0,
            cx=image.width / 2.0,
            cy=image.height / 2.0,
            image_width=image.width,
            image_height=image.height,
        )
        projection_result = projection_engine.project(
            sampling_result=sampling_result,
            intrinsics=intrinsics,
        )
    except Exception as exc:
        logger.error("Projection failed: %s", exc)
        return 1
    projection_time_s = time.perf_counter() - t4

    t5 = time.perf_counter()
    try:
        geometry_result = geometry_engine.compute(projection_result=projection_result)
    except Exception as exc:
        logger.error("Geometry computation failed: %s", exc)
        return 1
    geometry_time_s = time.perf_counter() - t5

    # ------------------------------------------------------------------ #
    # Save outputs
    # ------------------------------------------------------------------ #
    output_dir = _DEMO_OUTPUT_DIR
    annotated_path = output_dir / _ANNOTATED_FILENAME
    grayscale_path = output_dir / _GRAYSCALE_FILENAME

    try:
        GeometryVisualizer().save_visualization(
            depth_map=depth_result,
            geometry_result=geometry_result,
            output_path=annotated_path,
        )
        DepthVisualizer().save_grayscale(
            result=depth_result,
            output_path=grayscale_path,
        )
    except Exception as exc:  # noqa: BLE001
        logger.error("Visualization failed: %s", exc)
        return 1

    statistics_path = save_statistics(
        geometry_result=geometry_result,
        output_dir=output_dir,
    )

    print_runtime_summary(
        image_source=image_source,
        geometry_result=geometry_result,
        dino_time_s=dino_time_s,
        sam2_time_s=sam2_time_s,
        depth_time_s=depth_time_s,
        sampling_time_s=sampling_time_s,
        projection_time_s=projection_time_s,
        geometry_time_s=geometry_time_s,
        annotated_path=annotated_path,
        statistics_path=statistics_path,
        grayscale_path=grayscale_path,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(run_demo(sys.argv[1:]))
