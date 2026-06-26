# Project Status

This file is the single source of truth for project progress. It should be
updated whenever a meaningful task is completed.

## Current Phase

Phase 2: FastSAM Integration.

Status: ✅ Completed. Awaiting segmentation evaluation before Phase 3 begins.

## Overall Progress

- Phase 1 foundation is complete and verified.
- Documentation system has been expanded for long-term project tracking.
- Phase 2 FastSAM integration is complete and verified against real images.
- Phase 2 demo script (`scripts/run_demo.py`) is complete.
- Segmentation evaluation is in progress before Phase 3 starts.
- Phase 3 has not started.

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

Status: ✅ Completed.

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

Verification:

- `pytest`: 13 tests pass.
- `ruff check .`: no violations.
- `mypy`: no type errors in 19 source files.
- Demo script: executes successfully with both synthetic and real images.
- Overlay and statistics files verified in `outputs/demo/`.
- Real FastSAM inference confirmed on CPU using `FastSAM-s.pt`.

---

## Current Observations

The following observations were recorded during Phase 2 validation on real
images:

- Pole segmentation performs reliably. Utility poles are consistently isolated
  as distinct objects with clean bounding boxes and accurate masks.
- Sparse tree canopy segmentation remains inconsistent. Thin branches and
  fragmented foliage are frequently split into multiple small masks or missed
  entirely, producing less useful geometry than expected.
- Dense, solid objects produce significantly better masks than sparse
  vegetation. The segmentation model is well-suited for rigid structures.
- A more thorough evaluation across a broader and more representative image
  set is recommended before committing to the current segmentation strategy
  for Phase 3.

---

## Current Tasks

- Evaluate segmentation quality on a representative validation dataset.
- Compare segmentation approaches for sparse vegetation before finalizing the
  Phase 3 integration strategy.
- Wait for explicit user approval before starting Phase 3.

---

## Pending Tasks

- Select and organize a representative validation image dataset.
- Finalize segmentation strategy (current FastSAM approach versus alternatives)
  before Phase 3 design begins.
- Decide whether a persistent real sample image should be added to `demo/`.

---

## Future Phases

- Phase 3: CLIP integration for identifying tree and pole masks. Start pending
  segmentation evaluation and explicit user approval.
- Later phase: Tree species classification using EfficientNetV2.
- Later phase: Metric depth estimation using Depth Anything V2.
- Later phase: Distance estimation engine.
- Later phase: API integration.
- Later phase: Production hardening and integration testing.

---

## Repository Status

Stable development baseline.

All Phase 2 modules are complete, tested, and verified. The repository is
ready for segmentation evaluation before Phase 3 design begins.

---

## Known Issues

- Sparse tree canopy segmentation is inconsistent and requires further
  investigation before the segmentation strategy is finalized.

---

## Risks

- Future AI dependencies may be large and may require CUDA-specific
  installation steps.
- FastSAM, CLIP, depth estimation, and species classification may have
  different runtime requirements and model license considerations.
- Distance estimation from a single image may be approximate unless camera
  calibration, scale references, or reliable metric depth models are available.
- Sparse vegetation may require a different segmentation strategy or
  post-processing approach for reliable Phase 3 classification input.

---

## Technical Debt

- No API framework has been selected yet.
- No model registry or model-weight management strategy exists yet.
- No validated image dataset has been defined yet.
- No continuous integration workflow exists yet.

---

## Research Topics

- Segmentation quality evaluation across a representative dataset.
- Segmentation approaches for sparse vegetation (detector-assisted, alternative
  models, mask merging).
- CLIP prompt strategy for distinguishing pole and tree masks.
- Tree species classification dataset and model choice.
- Depth Anything V2 Metric setup, limitations, and calibration needs.
- Distance estimation strategy using segmentation masks and depth outputs.
- Edge extraction approaches for future distance estimation.

---

## Notes

- Do not start Phase 3 until segmentation evaluation is complete and the user
  explicitly approves moving forward.
- Do not implement vegetation growth prediction in the current roadmap.
- Keep `CHANGELOG.md`, `PROJECT_STATUS.md`, `TODO.md`, and `RESEARCH_LOG.md`
  synchronized after every meaningful task.
