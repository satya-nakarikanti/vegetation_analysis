# Changelog

This file records meaningful project changes. Every future phase or meaningful
task should use the same structure so progress, decisions, and verification stay
traceable.

## 2026-07-23 - Phase 6.3 Centroid-Based Geometry Validation

### Date

2026-07-23

### Phase

Phase 6.3: Centroid-Based Geometry Validation.

### Status

Completed.

### Summary

Successfully implemented and mathematically validated the Centroid-Based Geometry computations, completing the centroid-based 3D tracking engine.

The Geometry module now computes the true camera-to-object Euclidean distance `camera_distance` separate from the depth `camera_z`. It also computes pairwise angular relationships between detected objects (dot product, angles).

Centroid-to-centroid true distance was computed using two independent approaches (Law of Cosines vs. 3D Euclidean distance), which validated precisely identical values within floating-point tolerance, thus verifying the mathematical integrity of the full metric projection process.

### Implemented

- Added `camera_distance` (Euclidean from camera) to `GeometricObject`.
- Implemented `ObjectRelationship` schema to track object-to-object interactions.
- Computed angle between camera-to-object vectors using dot product logic.
- Evaluated and mathematically validated distance matching between Cosine Law and 3D Euclidean space.
- Validated `camera_distance >= camera_z`.
- Updated schemas, JSON structures, and demonstration output formatting.

### Validation

- `pytest` passed.
- `ruff check` passed.
- `mypy` passed.
- All distances validated strictly equal (`diff < 1e-6`).

### Next Phase

Edge-Based Engineering Measurements: extracting SAM2 boundaries to find the minimum edge-to-edge true metric clearance.

## 2026-07-14 - Phase 6 Apple Depth Pro Evaluation (Retired)

### Date

2026-07-14

### Phase

Phase 6: Apple Depth Pro Evaluation.

### Status

Completed and Retired.

### Summary

Removed the experimental Apple Depth Pro module.
Retired the evaluation after completing Phase 6.
Removed all related scripts, datasets, tests, documentation, and generated outputs.

## 2026-07-08 - Phase 5.3 Relative Geometry Engine

### Date

2026-07-08

### Phase

Phase 5.3: Relative Geometry Engine.

### Status

Completed.

### Files Changed

- `src/vegetation_analysis/geometry/constants.py` — new
- `src/vegetation_analysis/geometry/__init__.py` — new
- `src/vegetation_analysis/geometry/geometry_engine.py` — new
- `src/vegetation_analysis/geometry/schemas.py` — new
- `src/vegetation_analysis/geometry/visualization.py` — new
- `scripts/run_geometry_demo.py` — new
- `tests/test_geometry_engine.py` — new
- `tests/test_geometry_visualization.py` — new
- `src/vegetation_analysis/depth/visualization.py`
- `README.md`
- `docs/PROJECT_STATUS.md`
- `docs/CHANGELOG.md`
- `docs/PROJECT_CONTEXT.md`
- `docs/RESEARCH_LOG.md`
- `docs/TODO.md`
- `demo/README.md`

### Summary

Completed the Relative Geometry Engine.

The perception pipeline now combines Grounding DINO, SAM 2, Depth Anything V2,
and the Depth Sampling Engine to compute camera-relative object coordinates.
Each detected object now contains normalized `(rx, ry, rz)` coordinates derived
from segmentation masks and relative depth estimation.

Added grayscale depth visualization alongside the existing colorized depth map
to simplify depth inspection and debugging.

### Implemented

- Relative Geometry module.
- Camera-relative coordinate computation.
- Geometry visualization.
- Geometry demo runner.
- Geometry schemas.
- Future-compatible species fields (`species`, `species_confidence`).
- Grayscale depth visualization.
- Automated unit tests.

### Validation

- `pytest` passed.
- `ruff check` passed.
- `ruff format --check` passed.
- `mypy` passed.
- Geometry demo successfully validated on CUDA.

### Architectural Decisions

- Camera coordinate system adopted as the intermediate reference frame.
- Camera-relative coordinates are stored instead of metric coordinates.
- Object centroids remain the current reference point.
- Metric calibration is intentionally deferred to a later phase.

### Next Phase

Phase 4: Tree Species Classification (parallel development).

Following species integration, the project will continue with metric calibration,
nearest-point extraction, and vegetation clearance estimation.


## 2026-07-08 - Phase 5.2 Depth Sampling Engine

### Date

2026-07-08

### Phase

Phase 5.2: Depth Sampling Engine.

### Status

Completed.

### Files Changed

- `src/vegetation_analysis/depth_sampling/constants.py` — new
- `src/vegetation_analysis/depth_sampling/__init__.py` — new
- `src/vegetation_analysis/depth_sampling/sampler.py` — new
- `src/vegetation_analysis/depth_sampling/schemas.py` — new
- `src/vegetation_analysis/depth_sampling/visualization.py` — new
- `scripts/run_depth_sampling_demo.py` — new
- `tests/test_depth_sampler.py` — new
- `tests/test_depth_sampling_visualization.py` — new

### Summary

Completed the Depth Sampling Engine.

The module combines segmentation masks from SAM 2 with dense depth predictions
from Depth Anything V2 to generate robust object-wise depth statistics.

### Implemented

- Depth Sampling module.
- Mask-based depth extraction.
- Median, mean, minimum, maximum and standard deviation depth computation.
- Depth sampling visualization.
- Demo runner.
- Automated tests.

### Validation

- `pytest` passed.
- `ruff` passed.
- `mypy` passed.
- Sampling demo validated successfully.

### Architectural Decisions

- Median depth selected as the representative object depth.
- Mask-based sampling replaces single-pixel sampling.
- Sampling remains completely independent from geometry computation.

### Next Phase

Phase 5.3: Relative Geometry Engine.


## 2026-07-08 - Phase 5.1 Depth Anything V2 Integration

### Date

2026-07-08

### Phase

Phase 5.1: Depth Anything V2 Integration.

### Status

Completed.

### Files Changed

- `src/vegetation_analysis/depth/constants.py` — new
- `src/vegetation_analysis/depth/__init__.py` — new
- `src/vegetation_analysis/depth/depth_loader.py` — new
- `src/vegetation_analysis/depth/estimator.py` — new
- `src/vegetation_analysis/depth/schemas.py` — new
- `src/vegetation_analysis/depth/visualization.py` — new
- `scripts/run_depth_demo.py` — new
- `tests/test_depth_loader.py` — new
- `tests/test_depth_estimator.py` — new
- `tests/test_depth_visualization.py` — new

### Summary

Integrated Depth Anything V2 as the production monocular depth estimation
module.

The depth estimation pipeline operates independently of object detection and
segmentation, producing dense relative depth maps for every input image.

### Implemented

- Depth Anything V2 loader.
- CUDA/CPU device selection.
- Relative depth estimation.
- Depth visualization.
- Raw depth export.
- Depth statistics export.
- Demo runner.
- Automated tests.

### Validation

- `pytest` passed.
- `ruff` passed.
- `mypy` passed.
- CUDA inference validated successfully.

### Architectural Decisions

- Hugging Face Transformers implementation selected.
- Dense depth estimation separated from segmentation.
- Relative depth retained without metric calibration.

### Next Phase

Phase 5.2: Depth Sampling Engine.

## 2026-07-03 - Phase 3B.2 SAM 2 Integration and Validation

### Date

2026-07-03

### Phase

Phase 3B.2: SAM 2 Integration and Validation.

### Status

Completed.

### Summary

Phase 3B.2 is functionally complete. The pipeline is finalized as:
Image → Grounding DINO → Duplicate Pole Filtering → SAM 2 → Segmentation Masks.

The end-to-end segmentation pipeline has been validated successfully on real images. FastSAM is officially retired in favor of the Grounding DINO + SAM 2 pipeline.

### Validation

- Automated tests pass (`pytest`).
- Linting and static analysis pass (`ruff`, `mypy`).
- Duplicate pole filtering verified.
- SAM 2 demo output verified.

### Next Phase

Next Phase: Depth Anything V2 integration for monocular depth estimation.(Phase for tree classification will be handled subsequently or in parallel).

## 2026-07-03 - Phase 3B.1 SAM 2 Foundation & Infrastructure

### Date

2026-07-03

### Phase

Phase 3B.1: SAM 2 Foundation & Infrastructure.

### Status

Completed.

### Files Changed

- `src/vegetation_analysis/sam2/constants.py` — new
- `src/vegetation_analysis/sam2/__init__.py` — new
- `src/vegetation_analysis/sam2/sam2_loader.py` — new
- `src/vegetation_analysis/sam2/segmenter.py` — new
- `src/vegetation_analysis/sam2/schemas.py` — new
- `src/vegetation_analysis/sam2/visualization.py` — new
- `scripts/run_sam2_demo.py` — new
- `tests/test_sam2_loader.py` — new
- `tests/test_sam2_segmenter.py` — new
- `tests/test_sam2_visualization.py` — new
- `pyproject.toml`
- `README.md`
- `docs/PROJECT_STATUS.md`
- `docs/CHANGELOG.md`
- `docs/TODO.md`

### Summary

Completed the foundation and infrastructure for SAM 2 integration. 

Implemented the loader, segmenter, schemas, and visualization module for SAM 2, mirroring the existing Grounding DINO architecture. Added a demo script that successfully chains Grounding DINO detection results into SAM 2 segmentation masks.

### Implemented

- SAM 2 model loader using Meta's official API (`sam2`).
- Segmenter capable of consuming Grounding DINO bounding boxes and generating boolean masks.
- Structured `MaskObject` and `SegmentationResult` schemas.
- Mask visualization overlaying masks and bounding boxes on the original image.
- Added SAM 2 dependency to `pyproject.toml`.
- SAM 2 demo runner.
- Automated unit tests with a mock predictor.

### Validation

- 10 automated tests pass.
- Ruff linting passes.
- mypy static typing passes.
- Demo runner tested to catch missing dependencies gracefully.

### Architectural Decisions

- Use Meta's official `sam2` repository directly rather than an Ultralytics or Hugging Face wrapper.
- `SAM2Segmenter` is purely responsible for converting `DetectionResult` boxes into SAM 2 masks; it performs no object detection itself.

### Next Phase

Phase 3B.2: SAM 2 mask generation and evaluation against the FastSAM baseline.

## 2026-07-02 - Phase 3A Grounding DINO Integration

### Date

2026-07-02

### Phase

Phase 3A: Grounding DINO Detection.

### Status

Completed.

### Files Changed

- `src/vegetation_analysis/grounding/constants.py`
- `src/vegetation_analysis/grounding/__init__.py`
- `src/vegetation_analysis/grounding/grounding_dino_loader.py`
- `src/vegetation_analysis/grounding/detector.py`
- `src/vegetation_analysis/grounding/schemas.py`
- `src/vegetation_analysis/grounding/prompts.py`
- `src/vegetation_analysis/grounding/visualization.py`
- `scripts/run_grounding_demo.py`
- `tests/test_grounding_loader.py`
- `tests/test_grounding_detector.py`
- `tests/test_grounding_prompts_and_visualization.py`
- `tests/test_segmentation_pipeline.py`
- `pyproject.toml`
- `README.md`
- `docs/PROJECT_STATUS.md`
- `docs/CHANGELOG.md`
- `docs/RESEARCH_LOG.md`
- `docs/TODO.md`
- `docs/PROJECT_CONTEXT.md`

### Summary

Completed the production Grounding DINO integration.

The repository now contains a fully working open-set object detection pipeline
capable of detecting trees and utility poles using natural-language prompts.
Grounding DINO replaces the originally planned FastSAM → CLIP architecture.

FastSAM remains archived as the completed Phase 2 baseline for comparison and
future benchmarking.

### Implemented

- Grounding DINO model loader.
- Hugging Face model loading.
- Automatic CPU/CUDA device selection.
- Prompt builder.
- Shared constants module.
- Detection pipeline.
- Structured `DetectionBox` and `DetectionResult` schemas.
- Detection visualization.
- Grounding DINO demo runner.
- Automated unit tests.
- Compatibility with multiple Transformers API versions (`threshold` and
  `box_threshold`, `text_labels` and `labels`).

### Validation

- 27 automated tests pass.
- Ruff linting passes.
- mypy passes.
- Grounding DINO successfully detects trees and utility poles on real images.
- Demo runner validated successfully.

### Architectural Decisions

- Grounding DINO is now the active object detector.
- FastSAM is retained as the archived Phase 2 baseline.
- SAM 2 will consume Grounding DINO bounding boxes during Phase 3B.
- Compatibility logic is included to support multiple Transformers releases.

### Observations

- Utility pole detection is reliable.
- Tree detection is significantly more reliable than the previous FastSAM
  segmentation approach.
- Detection quality depends on prompt wording and confidence thresholds.
- Bounding boxes are intentionally coarse and will later be refined by SAM 2.

### Next Phase

Phase 3B: SAM 2 mask generation.

## 2026-06-26 - Phase 2 Completion and Validation

### Date

2026-06-26

### Phase

Phase 2: FastSAM Integration — completion, validation, and documentation.

### Status

Completed.

### Files Changed

- `src/vegetation_analysis/segmentation/fastsam_loader.py` — new
- `src/vegetation_analysis/segmentation/segmenter.py` — new
- `src/vegetation_analysis/segmentation/mask_utils.py` — new
- `src/vegetation_analysis/segmentation/schemas.py` — new
- `src/vegetation_analysis/segmentation/visualization.py` — new
- `src/vegetation_analysis/segmentation/__init__.py` — new
- `scripts/run_demo.py` — new
- `demo/README.md` — new
- `tests/test_segmentation_loader.py` — new
- `tests/test_segmentation_pipeline.py` — new
- `docs/CHANGELOG.md`
- `docs/PROJECT_STATUS.md`
- `docs/RESEARCH_LOG.md`
- `docs/TODO.md`
- `README.md`

### Summary

Completed the Phase 2 FastSAM integration. Implemented a modular segmentation
pipeline covering model loading, inference, mask extraction, geometry
computation, and visualization. All components are isolated into focused modules
inside `src/vegetation_analysis/segmentation/`. Added a reusable demo script
(`scripts/run_demo.py`) that orchestrates the pipeline without duplicating
production logic. Validated the implementation against both a synthetic image
and real electric pole images. Recorded segmentation behaviour observations in
`RESEARCH_LOG.md`. Updated all documentation to reflect the current repository
state.

### Implemented

- FastSAM model loader with automatic CPU/CUDA device selection.
- Configurable inference parameters via `FastSAMInferenceConfig`.
- Mask extraction from Ultralytics-style raw results, including multi-mask
  batch and tensor-to-NumPy handling.
- Boolean mask normalization, area calculation, bounding box derivation, and
  largest-contour extraction.
- Structured `BoundingBox` and `SegmentedObject` output schemas.
- Colour-coded mask overlay visualization with per-object contour drawing.
- Demo runner with image resolution fallback (CLI argument → `demo/sample.jpg`
  → synthetic image).
- Statistics export to `outputs/demo/statistics.json` including per-object
  bounding boxes, pixel areas, area percentage, and inference summary.
- pytest configuration improvements (base temp directory set to `temp_pytest`).
- Documentation improvements across all five tracking files.

### Validation

- 13 automated tests pass (`pytest`).
- Ruff linting passes with no violations.
- mypy static typing passes across 19 source files.
- Real FastSAM inference verified on CPU using `FastSAM-s.pt`.
- Demo validated on both a synthetic RGB image and a real electric pole image.
- Overlay and statistics files confirmed in `outputs/demo/`.

### Architectural Decisions

- Segmentation modules are isolated inside `src/vegetation_analysis/segmentation/`
  so each module has a single, replaceable responsibility.
- The demo script lives in `scripts/` and never contains segmentation,
  extraction, or visualization logic. All such work is delegated to `src/`.
- `AppSettings` drives all configurable values in the demo script, preventing
  hardcoded paths or magic numbers.
- Raw FastSAM results are handled through an adapter (`extract_masks`) that
  converts Ultralytics-specific output into framework-independent NumPy arrays,
  keeping the rest of the pipeline model-agnostic.

### Observations

- Pole segmentation is reliable on real images.
- Sparse tree canopy segmentation is inconsistent and requires further
  evaluation before Phase 3 design is finalized.
- Phase 3 will not begin until the segmentation strategy is confirmed.

### Next Phase

Phase 3: Grounding DINO detection for tree and pole bounding boxes. Pending
explicit user approval. (Note: CLIP integration was the originally planned
Phase 3 approach. After FastSAM evaluation, Grounding DINO was selected as
the replacement strategy.)

## 2026-06-26 - Phase 2 Demo Script

### Date

2026-06-26

### Phase

Phase 2: FastSAM Integration — demonstration script.

### Status

Completed.

### Files Changed

- `scripts/run_demo.py` — new
- `demo/README.md` — new
- `docs/CHANGELOG.md`
- `docs/PROJECT_STATUS.md`

### Summary

Added `scripts/run_demo.py`, a lightweight orchestration script that
exercises the complete Phase 2 segmentation pipeline without duplicating
any logic from the production modules in `src/`. The script accepts an
optional image path, falls back to `demo/sample.jpg`, and synthesises a
representative image when neither is available. Outputs are written to
`outputs/demo/` and include a mask overlay (`overlay.png`) and a
per-object statistics file (`statistics.json`). Added `demo/README.md`
to document the purpose of the new sample-image directory.

### Architectural Decisions

- The demo script is kept entirely in `scripts/` and delegates all
  segmentation, mask extraction, and visualization work to existing
  modules in `src/`.
- `AppSettings` drives model name, inference parameters, and output
  directory so no paths or constants are hardcoded in the script.
- Inference timing is measured with `time.perf_counter` to provide
  accurate wall-clock duration without any additional dependency.
- `statistics.json` is written with `json.dumps` rather than a new
  helper so the script stays self-contained and lightweight.

### Verification

- Script executes without error using the synthetic image fallback.
- All pre-existing tests continue to pass (`pytest`).
- Ruff and mypy checks pass with no new violations.
- Overlay and statistics files are written to `outputs/demo/`.

### Reason for Change

A reusable demo script was needed to exercise and demonstrate the Phase 2
implementation without requiring a specific image or any modification to
the production codebase.

### Next Phase

Phase 3: Grounding DINO detection for tree and pole bounding boxes, pending
explicit user approval.

## 2026-06-26 - Documentation System Improvements

### Date

2026-06-26

### Phase

Documentation System Improvements before Phase 2.

### Status

Completed.

### Files Changed

- `CHANGELOG.md`
- `PROJECT_STATUS.md`
- `README.md`
- `RESEARCH_LOG.md`
- `TODO.md`

### Summary

Improved the documentation system so project progress, research notes,
architecture decisions, pending work, and verification history are easier to
maintain across future phases.

### Architectural Decisions

- Keep `CHANGELOG.md` focused on completed meaningful changes and verification.
- Use `PROJECT_STATUS.md` as the single source of truth for current progress.
- Use `TODO.md` for actionable work lists.
- Use `RESEARCH_LOG.md` for research notes, comparisons, rejected approaches,
  experiments, and architecture decisions.
- Synchronize all four documentation files after every meaningful task.

### Verification

- Documentation-only update.
- No source code, tests, dependencies, or AI functionality changed.

### Reason for Change

The project needs a stronger documentation workflow before Phase 2 so future
model integration work remains traceable, reviewable, and easy to resume.

### Next Phase

Phase 2 remains pending. Do not start FastSAM integration until the user reviews
and approves the documentation system and Phase 1 foundation.

## 2026-06-26 - Phase 1 Foundation

### Date

2026-06-26

### Phase

Phase 1: Environment Setup and Project Foundation.

### Status

Completed.

### Files Changed

- `.gitignore`
- `.env.example`
- `pyproject.toml`
- `requirements.txt`
- `requirements-dev.txt`
- `README.md`
- `PROJECT_STATUS.md`
- `CHANGELOG.md`
- `scripts/verify_environment.py`
- `src/vegetation_analysis/__init__.py`
- `src/vegetation_analysis/api/__init__.py`
- `src/vegetation_analysis/config/__init__.py`
- `src/vegetation_analysis/config/settings.py`
- `src/vegetation_analysis/core/__init__.py`
- `src/vegetation_analysis/utils/__init__.py`
- `src/vegetation_analysis/utils/logging.py`
- `tests/__init__.py`
- `tests/test_settings.py`

### Summary

Created the initial Python project foundation with a `src/` package layout,
configuration loading, logging helper, dependency declarations, environment
verification script, test scaffold, Git ignore rules, and project documentation.
Created a local virtual environment using the bundled Codex Python runtime,
installed Phase 1 development dependencies, and verified the foundation with the
environment checker, pytest, Ruff, and mypy.

### Architectural Decisions

- Use a `src/` layout to avoid accidental imports from the repository root.
- Separate future responsibilities into `api`, `config`, `core`, and `utils`.
- Keep Phase 1 dependencies minimal and avoid AI/model dependencies until the
  relevant implementation phase.
- Use environment-based settings to avoid hardcoded paths.
- Ignore generated outputs, caches, virtual environments, datasets, and model
  weights in Git.

### Verification

- `scripts/verify_environment.py`: passed.
- `pytest`: passed, 1 test.
- `ruff check .`: passed.
- `mypy`: passed.

### Reason for Change

Phase 1 requires a clean, scalable, maintainable foundation before any computer
vision or AI functionality is implemented.

### Next Phase

Phase 2: FastSAM integration, pending explicit user approval.

### 2026-07-23

Phase 6.2 completed. Metric Depth integration and repository cleanup executed. The official Depth Anything V2 Metric model from the third-party submodule replaces the Hugging Face relative depth integration. All obsolete FastSAM scripts, modules, tests, and dependencies (including ultralytics) have been permanently deleted from the repository to create a clean production codebase.
