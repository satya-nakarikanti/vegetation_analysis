# TODO

This file tracks actionable work. It should evolve throughout development and
stay synchronized with `PROJECT_STATUS.md`, `CHANGELOG.md`, and
`RESEARCH_LOG.md`.

## Immediate Tasks

- Present Phase 2 results and documentation to the user for review.
- Wait for explicit user approval before starting Phase 3 design.

## Current Sprint — Segmentation Evaluation

- Collect a representative set of real electric pole images covering varied
  lighting conditions, canopy densities, and pole configurations.
- Run the Phase 2 pipeline against the validation dataset.
- Record segmentation quality observations for poles and trees separately.
- Evaluate whether a single unified tree mask is reliably produced, or
  whether masks are consistently fragmented.
- Determine whether the current FastSAM approach is sufficient for Phase 3
  input, or whether an alternative strategy is needed.

## Phase 3 Pre-work

- Research CLIP prompt engineering for tree and pole mask identification.
- Investigate mask merging strategies for fragmented tree canopy masks.
- Define a minimum acceptable segmentation quality threshold for Phase 3
  input before CLIP integration begins.
- Evaluate whether a detector-assisted segmentation approach would produce
  more reliable tree masks than the current any-object strategy.
- Research alternative segmentation models that may handle sparse vegetation
  with greater consistency.
- Design the CLIP integration module after segmentation evaluation is
  complete.

## Future Improvements

- Add continuous integration checks after the project structure stabilizes.
- Add sample data organization and a validation dataset registry.
- Add API documentation when the API layer is introduced.
- Add model configuration documentation when additional AI dependencies are
  introduced.

## Research TODOs

- Evaluate segmentation consistency across a wider variety of images
  including different seasons, lighting, and canopy types.
- Compare segmentation quality across multiple approaches before committing
  to Phase 3.
- Research edge extraction approaches for future distance estimation between
  segmented tree and pole objects.
- Research depth estimation limitations for distance approximation.
- Research tree mask merging strategies to unify fragmented canopy detections.

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

## Nice to Have

- Add architecture decision records if decisions become too large for
  `RESEARCH_LOG.md`.
- Add a documentation index if the number of docs grows.
- Add diagrams for each major pipeline stage as the architecture matures.
