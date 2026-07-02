# TODO

This file tracks actionable work. It should evolve throughout development and
stay synchronized with `PROJECT_STATUS.md`, `CHANGELOG.md`, and
`RESEARCH_LOG.md`.


## Immediate Tasks

- Review Grounding DINO performance on a larger validation dataset.
- Evaluate multiple prompt strategies for vegetation detection.
- Begin Phase 3B: SAM 2 integration.

## Current Sprint — Phase 3B: SAM 2 Integration

- Integrate SAM 2 with Grounding DINO bounding boxes.
- Generate tree and utility pole masks.
- Compare SAM 2 masks against the archived FastSAM baseline.
- Measure segmentation quality on representative images.
- Validate mask quality before beginning species classification.
- Record evaluation findings in `RESEARCH_LOG.md`.

## Phase 3B Design Decisions

- Select the preferred SAM 2 checkpoint.
- Define acceptable mask quality thresholds.
- Determine the mask refinement strategy for overlapping vegetation.
- Decide whether post-processing is required after SAM 2 inference.

## Future Improvements

- Add continuous integration checks after the project structure stabilizes.
- Add sample data organization and a validation dataset registry.
- Add API documentation when the API layer is introduced.
- Add model configuration documentation when additional AI dependencies are
  introduced.

## Research TODOs

- Compare Grounding DINO Tiny vs. Base checkpoints.
- Evaluate prompt engineering for vegetation detection.
- Research SAM 2 mask quality using Grounding DINO detections.
- Research edge extraction for future distance estimation.
- Evaluate Depth Anything V2 for engineering-grade distance estimation.

## Completed

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

## Nice to Have

- Add architecture decision records if decisions become too large for
  `RESEARCH_LOG.md`.
- Add a documentation index if the number of docs grows.
- Add diagrams for each major pipeline stage as the architecture matures.
