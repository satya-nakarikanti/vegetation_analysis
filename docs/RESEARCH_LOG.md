# Research Log

This file is the engineering notebook for AI research, technical findings,
architecture decisions, experiments, comparisons, rejected approaches, and open
research questions. It should not contain implementation code.

## 2026-07-03 - SAM 2 Mask Generation and Duplicate Pole Filtering

### Date

2026-07-03

### Research Topic

Phase 3B.2: SAM 2 Evaluation, Duplicate Pole Filtering, and FastSAM Retirement.

### Objective

Evaluate the end-to-end segmentation pipeline (Grounding DINO + SAM 2) on real images, implement duplicate pole filtering to refine detections, and officially retire FastSAM.

### Findings

- **Duplicate Pole Filtering**: Implementing duplicate filtering successfully removes overlapping bounding boxes for poles, leading to cleaner inputs for SAM 2.
- **SAM 2 Segmentation**: SAM 2 efficiently produces accurate masks based on Grounding DINO's coarse bounding boxes. It handles both pole structures and tree canopies much better than FastSAM.
- **End-to-End Pipeline**: The combined pipeline successfully isolates utility poles and vegetation in complex backgrounds without the fragmentation issues seen in FastSAM.
- **Filter**: Prompt engineering alone could not reliably suppress duplicate pole detections. A lightweight post-processing filter based on overlap/containment heuristics proved significantly more robust while preserving the modular architecture.

### Engineering Decisions

- FastSAM is officially retired in favor of the Grounding DINO + SAM 2 pipeline.
- The next development focus (Phase 5) will be on metric depth estimation using Depth Anything V2.

## 2026-07-02 - Grounding DINO Evaluation and Phase 3A Completion

### Date

2026-07-02

### Research Topic

## Phase 3B.1: SAM 2 Foundation Architecture

- **Decision:** SAM 2 segmenter uses Grounding DINO `DetectionResult` natively.
- **Reason:** It decouples object detection completely from SAM 2 mask generation.
- **Implementation:** `sam2` Meta package is used over Hugging Face for optimal compatibility with the official checkpoints.

## Phase 3A: Grounding DINO vs. FastSAM

### Objective

Evaluate Grounding DINO as the replacement for the archived FastSAM → CLIP
pipeline and determine whether zero-shot object detection provides a more
reliable foundation for later segmentation using SAM 2.

### Evaluation Setup

Model:

- `IDEA-Research/grounding-dino-tiny`

Prompt:

- `"tree . utility pole ."`

Evaluation images:

- Representative real electric-pole images containing vegetation.

### Findings

**Utility pole detection — reliable.**

Grounding DINO consistently detects the main utility pole with a single,
well-localized bounding box across tested images.

**Tree detection — reliable.**

Unlike FastSAM, Grounding DINO detects sparse tree canopies as a single object
instead of fragmenting them into multiple disconnected regions. This makes it
better suited for vegetation localization.

**Bounding box quality.**

Bounding boxes intentionally include contextual pixels surrounding the object.
This is expected behaviour because Grounding DINO is an object detector rather
than a segmentation model.

**Inference speed.**

CPU inference is suitable for development and testing. GPU acceleration can be
used later without architectural changes.

### Engineering Decisions

- Grounding DINO becomes the primary object detection model.
- FastSAM is retained as the archived Phase 2 baseline for comparison.
- CLIP is permanently removed from the planned pipeline.
- SAM 2 will be responsible for converting Grounding DINO bounding boxes into
  accurate object masks.

### Remaining Research

- Compare `grounding-dino-tiny` with `grounding-dino-base`.
- Evaluate additional prompt variations:
  - `"tree . utility pole ."`
  - `"tree . electric pole ."`
  - `"vegetation . utility pole ."`
  - `"tree . power pole ."`
- Measure detection consistency across different lighting conditions,
  vegetation densities, camera distances, and pole configurations.
- Determine the optimal confidence thresholds for tree and utility pole
  detection.
- Evaluate SAM 2 mask quality using Grounding DINO detections as prompts.

### References

- Grounding DINO paper (Liu et al., 2023).
- Hugging Face model: `IDEA-Research/grounding-dino-tiny`.
- Phase 3A implementation: `src/vegetation_analysis/grounding/`.
- Demo script: `scripts/run_grounding_demo.py`.

## 2026-07-01 - FastSAM Evaluation and Architecture Decision: Grounding DINO

### Date

2026-07-01

### Research Topic

FastSAM pipeline evaluation and Phase 3 architecture selection.

### Objective

Record the findings from evaluating the Phase 2 FastSAM segmentation pipeline
on real electric-pole images, and document the resulting architecture decision
for Phase 3.

### Evaluation Findings

FastSAM was evaluated on multiple real images of electric poles with nearby
vegetation. The following patterns were consistently observed:

**Utility pole segmentation — reliable.**
In every tested image, the pole body was isolated as a distinct object with a
clean bounding box and well-formed mask geometry. The FastSAM-s model handles
rigid, high-contrast vertical structures well.

**Tree canopy segmentation — unreliable.**
Sparse and semi-open tree canopies were consistently fragmented into multiple
small disconnected masks. In the worst cases, large sections of foliage were
not detected at all. Dense closed-canopy trees produced better results but were
still split at branch boundaries.

**Root cause.**
FastSAM is a class-agnostic segmentation model trained to produce instance
masks for all visible objects. It does not understand semantic categories.
When a tree canopy is visually non-contiguous (sparse branches, sky gaps),
FastSAM segments individual contiguous regions rather than producing a single
unified tree object. CLIP classification applied to individual sub-masks would
receive partial canopy fragments rather than a representative tree region,
making confidence scores unreliable.

### Architecture Decision

**CLIP classification is removed from the Phase 3 plan.**

The FastSAM → CLIP pipeline was designed under the assumption that FastSAM
would reliably produce one contiguous mask per tree. Evaluation shows this
assumption does not hold for sparse vegetation.

**Grounding DINO replaces CLIP as the Phase 3 detection approach.**

Grounding DINO is an open-set object detection model that produces labelled
axis-aligned bounding boxes directly from text prompts. It does not depend on
segmentation masks as input.

Advantages for this project:

- Produces a single bounding box per detected object regardless of canopy
  fragmentation. A sparse tree canopy is covered by one box, not split into
  many.
- Text prompts (`"tree . utility pole ."`) map naturally to the project's
  detection needs.
- Well-established Hugging Face integration with accessible pre-trained
  checkpoints (`IDEA-Research/grounding-dino-tiny`,
  `IDEA-Research/grounding-dino-base`).
- Bounding boxes from Grounding DINO are the standard input format for SAM 2,
  which is the planned mask generation model for a later phase.

**FastSAM segmentation is retained as the archived Phase 2 baseline.**
No code is deleted. The `src/vegetation_analysis/segmentation/` package remains
intact for reference, benchmarking, and potential future use.

**The updated pipeline is:**

```
Image
  → Grounding DINO
  → Tree & Utility Pole Bounding Boxes
  → SAM 2 (future phase)
  → Tree/Pole Masks
  → Depth Anything V2
  → Distance Estimation
  → API Response
```

### Pending Research Directions

- Compare Grounding DINO Tiny and Base checkpoints on a larger validation dataset.
- Refine prompt engineering for improved tree detection.
- Evaluate SAM 2 mask quality using Grounding DINO detections.
- Research edge extraction for future distance estimation.

### References

- Phase 2 implementation: `src/vegetation_analysis/segmentation/`.
- Phase 3 scaffold: `src/vegetation_analysis/grounding/`.
- Grounding DINO paper: Liu et al., 2023 — "Grounding DINO: Marrying DINO with
  Grounded Pre-Training for Open-Set Object Detection."
- Hugging Face model hub: `IDEA-Research/grounding-dino-tiny`,
  `IDEA-Research/grounding-dino-base`.

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

Note: The subsequent evaluation (recorded in the 2026-07-01 entry above)
confirmed that the FastSAM → CLIP approach is unsuitable and led to the
architecture decision to adopt Grounding DINO.

### References

- Phase 2 implementation: `src/vegetation_analysis/segmentation/`.
- Phase 2 demo script: `scripts/run_demo.py`.
- Validation outputs: `outputs/demo/`.
