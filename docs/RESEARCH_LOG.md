# Research Log

This file is the engineering notebook for AI research, technical findings,
architecture decisions, experiments, comparisons, rejected approaches, and open
research questions. It should not contain implementation code.

## 2026-06-26 - Initial Architecture Baseline

### Date

2026-06-26

### Research Topic

Initial vegetation analysis architecture.

### Objective

Record the baseline architecture proposed for the vegetation analysis module
before Phase 2 begins.

### Findings

- The planned pipeline is image upload, FastSAM masks, CLIP mask
  identification, EfficientNetV2 species classification, Depth Anything V2
  Metric depth estimation, distance calculation, and API response.
- The architecture should remain flexible because better model choices or
  integration patterns may emerge during development.
- Phase 1 intentionally avoided AI dependencies so the project foundation stays
  small, testable, and easy to reason about.

### Decision

Use the proposed architecture as the initial research direction, but do not
treat it as permanent. Validate each model and integration point during its
dedicated phase before building later phases on top of it.

### Pending Questions

- Which FastSAM package or repository should be used for Phase 2?
- Where should model weights be stored locally?
- Should model downloads be automatic, manual, or configured through environment
  variables?
- What sample images should be used to evaluate segmentation quality?

### References

- Project master instructions provided by the user.

## 2026-06-26 - Documentation System

### Date

2026-06-26

### Research Topic

Project documentation workflow.

### Objective

Define how progress, decisions, research notes, and pending work should be
tracked before model integration begins.

### Findings

- A single changelog is not enough for this project because implementation
  progress, research decisions, and active TODOs evolve at different speeds.
- `PROJECT_STATUS.md` is best suited for current truth and overall progress.
- `CHANGELOG.md` is best suited for completed meaningful changes.
- `TODO.md` is best suited for actionable next work.
- `RESEARCH_LOG.md` is best suited for model comparisons, experiments,
  rejected approaches, and pending research questions.

### Decision

Maintain four synchronized documentation files:

- `CHANGELOG.md`
- `PROJECT_STATUS.md`
- `TODO.md`
- `RESEARCH_LOG.md`

Update all relevant documentation after every meaningful task. Update
`RESEARCH_LOG.md` only when research findings, experiments, comparisons, or
architecture decisions are involved.

### Pending Questions

- Should research entries include experiment metrics once model testing begins?
- Should future model evaluations use a fixed benchmark image set?

### References

- User request to improve the documentation system before Phase 2.

## 2026-06-26 - Phase 2 FastSAM Segmentation Observations

### Date

2026-06-26

### Research Topic

FastSAM segmentation quality on real electric pole images.

### Objective

Record observed segmentation behaviour after validating the Phase 2
implementation against real-world images of electric poles and nearby
vegetation. The goal is to identify strengths, limitations, and open
questions before Phase 3 design begins.

### Observations

- FastSAM consistently and reliably segments utility poles. In all tested
  images, the pole body was isolated as a distinct object with a clean,
  accurate bounding box and well-formed mask geometry.
- Sparse tree canopies are frequently fragmented or missed. Thin branches,
  scattered leaves, and open-canopy trees often produce multiple small
  disconnected masks rather than a single unified tree object. In some
  images, sparse foliage was not detected at all.
- Dense, solid objects produce significantly better masks than sparse
  vegetation. Objects with continuous, high-contrast surfaces are
  well-suited to FastSAM's general segmentation approach.
- The quality gap between rigid structure segmentation and sparse vegetation
  segmentation is wide enough to warrant evaluation before committing the
  current approach to Phase 3.
- Additional evaluation across a larger, more representative image set is
  recommended before selecting a final segmentation strategy.

### Pending Research Directions

- Evaluate segmentation consistency across a wider variety of electric pole
  and vegetation images, including different lighting conditions, seasons,
  and canopy densities.
- Investigate mask merging strategies for unifying fragmented tree masks
  before passing them to a classification model.
- Evaluate whether a detector-assisted segmentation approach (e.g., using a
  region proposal or detection stage to constrain segmentation) would
  produce more reliable tree masks.
- Research alternative segmentation models that may handle sparse vegetation
  with greater consistency.
- Define a minimum acceptable segmentation quality threshold for Phase 3
  input before CLIP integration begins.
- Research edge extraction approaches for future distance estimation between
  segmented tree and pole objects.

### Decision

No final segmentation strategy has been selected. Phase 3 design will not
begin until the segmentation quality has been evaluated on a representative
dataset and the development team has reviewed the results.

### References

- Phase 2 implementation: `src/vegetation_analysis/segmentation/`.
- Phase 2 demo script: `scripts/run_demo.py`.
- Validation outputs: `outputs/demo/`.
