# Project Status

This file is the single source of truth for project progress. It should be
updated whenever a meaningful task is completed.

## Current Phase

Phase 5: Relative Scene Geometry

Status: Phase 5.1, Phase 5.2, and Phase 5.3 are completed.

The perception pipeline now combines Grounding DINO, SAM 2, Depth Anything V2,
Depth Sampling, and the Relative Geometry Engine to generate normalized
camera-relative `(rx, ry, rz)` coordinates for every detected object.

Current parallel development:

- Phase 4: Tree Species Classification

## Overall Progress

- Phase 1 foundation is complete and verified.
- Documentation system is complete and synchronized.
- Phase 2 FastSAM integration is complete and archived as the baseline.
- FastSAM evaluation on real electric-pole images showed reliable pole
  segmentation but fragmented sparse tree canopies.
- Architecture decision: Grounding DINO replaced the planned FastSAM → CLIP
  pipeline.
- Phase 3A Grounding DINO implementation is complete.
- Grounding DINO successfully detects trees and utility poles using
  natural-language prompts.
- Phase 3B (SAM 2 mask generation) is complete. The pipeline: Image → Grounding DINO → Duplicate Pole Filtering → SAM 2 → Segmentation Masks.
- The end-to-end segmentation pipeline is validated on real images. FastSAM is officially retired.
- Phase 5.1 (Depth Anything V2 integration) is complete.
- Phase 5.2 (Depth Sampling Engine) is complete.
- Phase 5.3 (Relative Geometry Engine) is complete.
- The pipeline now produces normalized camera-relative object coordinates.
- Grayscale and colorized depth visualizations are available.

## Completed Phases

### Phase 1: Environment Setup and Project Foundation

Status: ✅ Completed.

Completed work:

- Initialized repository foundation.
- Created scalable `src/` package structure.
- Added configuration module using environment-based settings.
- Added logging configuration helper.
- Added dependency declarations.
- Added `.gitignore`.
- Added `.env.example`.
- Added environment verification script.
- Added initial test scaffold.
- Added README documentation.
- Added changelog.
- Created local virtual environment and installed Phase 1 development
  dependencies.
- Verified environment, tests, linting, and static typing.

Verification:

- `scripts/verify_environment.py`: passed.
- `pytest`: passed, 1 test.
- `ruff check .`: passed.
- `mypy`: passed.

---

### Documentation System Improvements

Status: ✅ Completed.

Completed work:

- Expanded `CHANGELOG.md` format.
- Created `RESEARCH_LOG.md`.
- Expanded `PROJECT_STATUS.md`.
- Created `TODO.md`.
- Expanded `README.md`.
- Added documentation synchronization rules.

---

### Phase 2: FastSAM Integration

Status: ✅ Completed. Archived as the Phase 2 baseline.

Completed work:

- FastSAM model loader with automatic CPU/CUDA device selection
  (`segmentation/fastsam_loader.py`).
- FastSAM segmenter with configurable inference parameters
  (`segmentation/segmenter.py`).
- Mask extraction and geometry utilities including contour and bounding box
  extraction (`segmentation/mask_utils.py`).
- Segmentation output schemas (`BoundingBox`, `SegmentedObject`)
  (`segmentation/schemas.py`).
- Mask overlay visualization with per-object colour coding and contour drawing
  (`segmentation/visualization.py`).
- Segmentation package with a clean public API (`segmentation/__init__.py`).
- Demo orchestration script that delegates to existing modules without
  duplicating logic (`scripts/run_demo.py`).
- Sample-image directory with documentation (`demo/README.md`).
- Automated tests for all Phase 2 modules.
- Validated against real images.

Evaluation findings:

- Pole segmentation is reliable. Utility poles are consistently isolated with
  clean bounding boxes and accurate masks.
- Sparse tree canopy segmentation is unreliable. Thin branches and fragmented
  foliage are split into multiple disconnected masks, making the FastSAM → CLIP
  pipeline unsuitable for tree identification.
- The architecture has therefore changed: Grounding DINO replaces CLIP.

Verification:

- `pytest`: 13 tests pass.
- `ruff check .`: no violations.
- `mypy`: no type errors in 19 source files.
- Demo script validated on representative real images.
- Demo script: executes successfully with both synthetic and real images.
- Overlay and statistics files verified in `outputs/demo/`.
- Real FastSAM inference confirmed on CPU using `FastSAM-s.pt`.

---

### Phase 3A: Grounding DINO Detection

Status: ✅ Completed.

Completed work:

- Grounding DINO model loader.
- Hugging Face model loading.
- Automatic CPU/CUDA device selection.
- Prompt builder and vegetation prompt constants.
- Grounding DINO inference pipeline.
- Structured `DetectionBox` and `DetectionResult` schemas.
- Detection visualization.
- Grounding DINO demo script (`scripts/run_grounding_demo.py`).
- Automated unit tests.
- Compatibility with multiple Transformers API versions.
- Real-image validation.

Evaluation findings:

- Utility poles are detected reliably.
- Trees are detected reliably using prompt-based zero-shot detection.
- Detection quality depends on prompt wording and confidence thresholds.
- Bounding boxes are intentionally coarse and will be refined by SAM 2.

Verification:

- pytest: 27 tests passed.
- ruff check .: passed.
- mypy: passed.
- Grounding DINO demo executed successfully.
---

### Phase 3B.1: SAM 2 Foundation & Infrastructure

Status: ✅ Completed.

Completed work:

- Meta SAM 2 dependency added.
- SAM 2 loader and configuration.
- SAM 2 segmenter consuming Grounding DINO boxes.
- `MaskObject` and `SegmentationResult` schemas.
- Mask visualization and contour generation.
- Demo script `run_sam2_demo.py`.
- Automated tests passing.

Verification:

- pytest: 10 tests passed.
- ruff check .: passed.
- mypy: passed.
- Demo script handles missing weights correctly.
---

### Phase 3B.2: SAM 2 Integration and Validation

Status: ✅ Completed.

Completed work:

- SAM 2 mask generation validated against real images.
- Duplicate pole filtering implemented.
- End-to-end segmentation pipeline finalized.
- FastSAM officially retired in favor of Grounding DINO + SAM 2.

Verification:

- pytest, ruff, and mypy passed.
- SAM 2 demo outputs verified.

---

### Phase 5.1: Depth Anything V2 Integration

Status: ✅ Completed.

Completed work:

- Depth Anything V2 loader and configuration.
- Depth Estimator.
- Output schemas (`DepthMapResult`).
- Depth visualization heatmap overlay.
- Demo script (`scripts/run_depth_demo.py`).
- Automated tests passing.

---

### Phase 5.2: Depth Sampling Engine

Status: ✅ Completed.

Completed work:

- `DepthSampler` module to extract depth metrics using SAM 2 masks and Depth Anything V2 maps.
- Morphological erosion of masks to prevent edge bleeding.
- Computation of `centroid_x`, `centroid_y`, and comprehensive depth statistics (median, mean, std, min, max, pixel count).
- Metadata extraction in the JSON response.
- Visualization module to overlay centroids and statistics on depth heatmaps.
- Complete demo orchestration script (`scripts/run_depth_sampling_demo.py`).
- Automated tests passing.

### Phase 5.3: Relative Geometry Engine

Status: ✅ Completed.

Completed work:

- Relative Geometry module.
- Camera-relative coordinate computation.
- Relative Geometry Engine.
- Geometry visualization.
- Grayscale depth visualization.
- Geometry demo (`scripts/run_geometry_demo.py`).
- Geometry schemas.
- Automated tests.

Outputs:

Each detected object now contains:

- relative_x
- relative_y
- relative_z

along with its sampled depth statistics.

Verification:

- pytest: passed.
- ruff check: passed.
- ruff format --check: passed.
- mypy: passed.
- Geometry demo validated successfully on CUDA.

---

## Current Tasks

- Integrate Tree Species Classification (Phase 4).
- Merge species metadata into the geometry pipeline.
- Begin research for metric distance estimation.

---

## Pending Tasks

- Merge Tree Species Classification into the main pipeline.
- Build a representative evaluation dataset.
- Implement nearest-point extraction.
- Design metric calibration strategy.
- Implement engineering distance estimation.
---

## Future Phases

- Phase 4: Tree Species Classification.
- Phase 6: Metric Distance Estimation.
- Production API integration.
- Pipeline parallelization and optimization.
- Production hardening.

---

## Repository Status

Completed:

- Phase 1
- Phase 2
- Phase 3A
- Phase 3B.1
- Phase 3B.2
- Phase 5.1
- Phase 5.2
- Phase 5.3

Current development:

Phase 4 (Tree Species Classification) is under parallel development.

The current production pipeline is:

Grounding DINO
↓

Duplicate Pole Filtering
↓

SAM 2

+

Depth Anything V2

↓

Depth Sampling

↓

Relative Geometry

---

## Known Issues

- Grounding DINO produces coarse bounding boxes by design.
- Detection quality depends on prompt wording and confidence thresholds.
- Depth Anything V2 provides relative depth only.
- Relative Geometry currently uses object centroids.
- Nearest-point extraction has not yet been implemented.
- Metric calibration remains future work.

---

## Risks

- SAM 2 integration quality depends on Grounding DINO bounding-box accuracy.
- Prompt engineering may significantly affect detection consistency.
- Metric distance estimation depends on accurate camera calibration.
- Nearest-point extraction will influence engineering accuracy.
- Species classification accuracy depends on available training data.
---

## Technical Debt

- No API framework selected.
- No model registry implemented.
- No benchmark dataset defined.
- No CI pipeline.
- Prompt evaluation framework not yet implemented.

---

## Research Topics

- Tree Species Classification.
- Camera calibration.
- Relative-to-metric depth conversion.
- Nearest-point extraction.
- Vegetation clearance estimation.
- Real-world engineering validation.

---

## Notes

- Phase 5.1, 5.2, and 5.3 are complete.
- The perception pipeline is functionally complete.
- Future work focuses on semantic understanding (species classification) and engineering measurement (metric distance estimation).
- Keep CHANGELOG.md, PROJECT_STATUS.md, TODO.md, and RESEARCH_LOG.md synchronized.

