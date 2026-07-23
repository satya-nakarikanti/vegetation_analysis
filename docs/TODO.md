# TODO

This file tracks actionable work. It should evolve throughout development and
stay synchronized with `PROJECT_STATUS.md`, `CHANGELOG.md`, and
`RESEARCH_LOG.md`.


## Immediate Tasks

- Merge the Tree Species Classification module (Phase 4).
- Integrate species predictions into the geometry pipeline.
- Validate species predictions on representative utility-pole images.
- Start Phase 6 - Edge-Based Engineering Measurements.

## Current Sprint — Phase 4: Tree Species Classification

- Complete the Tree Species Classification module.
- Validate species predictions.
- Integrate species metadata into geometry outputs.
- Update documentation after successful integration.

## Next Research Direction

Investigate Edge-Based Engineering Measurements (nearest boundary extraction from SAM2 masks) to compute minimum edge-to-edge true clearance.

## Future Improvements

- Execute Grounding DINO/SAM 2 and Depth Anything V2 in parallel after the
  perception pipeline is finalized.
- Add continuous integration after repository stabilization.
- Add benchmark datasets.
- Add API documentation.
- Add model configuration documentation.
- Benchmark CPU vs CUDA inference.

## Research TODOs

- Compare Grounding DINO Tiny vs Base.
- Improve vegetation prompt engineering.
- Research camera calibration strategies.
- Research relative-to-metric depth conversion.
- Research nearest-point extraction algorithms.
- Evaluate engineering distance estimation methods.

## Completed

- ✅ Complete Phase 5.1: Depth Anything V2 integration.
- ✅ Implement Depth Anything V2 loader.
- ✅ Implement relative depth estimation.
- ✅ Generate grayscale and Inferno depth visualizations.
- ✅ Export raw depth maps and statistics.
- ✅ Complete Phase 5.2: Depth Sampling Engine.
- ✅ Implement mask-based depth sampling.
- ✅ Compute median, mean, minimum, maximum, and standard deviation depth.
- ✅ Implement object-wise depth visualization.
- ✅ Complete Phase 5.3: Relative Geometry Engine.
- ✅ Generate camera-relative `(rx, ry, rz)` coordinates.
- ✅ Implement geometry visualization.
- ✅ Generate grayscale depth outputs.
- ✅ Validate CUDA execution.
- ✅ Pass pytest, Ruff, and mypy validation.
- ✅ Complete Phase 6.3: Centroid-Based Geometry Validation.
- ✅ Compute true camera-to-object Euclidean distance `camera_distance`.
- ✅ Compute pairwise angle and dot product.
- ✅ Validate centroid distances using Law of Cosines and 3D Euclidean geometry.
- ✅ Review the documentation system.
- ✅ Research FastSAM installation options and repository/package choice.
- ✅ Research FastSAM output format and mask visualization workflow.
- ✅ Implement FastSAM model loader.
- ✅ Implement FastSAM segmenter with configurable inference parameters.
- ✅ Implement mask extraction and geometry utilities.
- ✅ Implement segmentation output schemas (`BoundingBox`, `SegmentedObject`).
- ✅ Implement mask overlay visualization.
- ✅ Implement segmentation package public API.
- ✅ Write automated tests for all Phase 2 modules.
- ✅ Implement Phase 2 demo script (`scripts/run_demo.py`).
- ✅ Validate demo on synthetic and real images.
- ✅ Update all documentation to reflect Phase 2 completion.
- ✅ Confirm Python environment and virtual environment setup.
- ✅ Decide model weight strategy (model file placed in project root).
- ✅ Evaluate FastSAM on real electric-pole images and record findings.
- ✅ Record architecture decision: Grounding DINO replaces CLIP as Phase 3.
- ✅ Archive FastSAM segmentation as the completed Phase 2 baseline.
- ✅ Create `src/vegetation_analysis/grounding/` package skeleton.
- ✅ Implement `schemas.py` (`DetectionBox`, `DetectionResult`).
- ✅ Implement `grounding_dino_loader.py` (config, loader, loaded-model stubs).
- ✅ Implement `detector.py` (detector skeleton).
- ✅ Implement `prompts.py` (`PromptBuilder`, prompt constants).
- ✅ Implement `visualization.py` (`DetectionVisualizer`).
- ✅ Update all documentation to reflect architecture change.
- ✅ Implement Grounding DINO model loader.
- ✅ Implement Grounding DINO inference pipeline.
- ✅ Implement prompt builder and shared constants.
- ✅ Implement structured detection schemas.
- ✅ Implement detection visualization.
- ✅ Implement `scripts/run_grounding_demo.py`.
- ✅ Add Grounding DINO dependencies (`torch`, `transformers`).
- ✅ Write automated Grounding DINO unit tests.
- ✅ Validate Grounding DINO on representative electric-pole images.
- ✅ Achieve 27 passing automated tests.
- ✅ Complete Phase 3A Grounding DINO integration.
- ✅ Complete Phase 3B.1 SAM 2 Foundation.
- ✅ Implement duplicate pole filtering.
- ✅ Complete Phase 3B.2 SAM 2 mask generation and evaluation.
- ✅ Finalize end-to-end segmentation pipeline (Grounding DINO + SAM 2).
- ✅ Retire FastSAM.

## Nice to Have

- Parallel pipeline orchestration after all functionality is complete.
- Repository cleanup before final submission.
- Pipeline performance benchmarking.
- Architecture diagrams for every major pipeline stage.
- Documentation index.
- Engineering benchmark dataset.
