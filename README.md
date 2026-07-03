# Vegetation Analysis Module

## Project Overview

This project is a production-oriented computer vision module for vegetation
analysis near electric poles. It will later integrate with an existing electric
pole inspection application and run only when the user selects Tree Analysis.

The module is built phase by phase. Phase 2 (FastSAM segmentation) is complete
and archived as the baseline. After evaluating FastSAM on real electric-pole
images, sparse tree canopies were found to fragment into multiple disconnected
masks, making the originally planned FastSAM → CLIP pipeline unsuitable.

The active pipeline now uses Grounding DINO for open-set object detection.
Phase 3A is complete and provides production-ready tree and utility pole
detection using natural-language prompts.

Grounding DINO replaces the originally planned FastSAM → CLIP pipeline after
evaluation showed that FastSAM fragmented sparse tree canopies into multiple
disconnected masks.

Future phases will use Depth Anything V2 for metric depth estimation, followed by species
classification, distance estimation, and API integration.

## Objectives

The long-term objective is to process an uploaded image containing an electric
pole and nearby vegetation, then eventually:

1. Identify the pole and tree.
2. Generate segmentation masks.
3. Identify the tree species.
4. Estimate the approximate distance between the nearest visible tree edge and
   nearest visible pole edge.
5. Return results through an API for integration into the main project.

Vegetation growth prediction is a future idea and is not part of the current
scope.

## Folder Structure

```text
vegetation_analysis/
|-- demo/
|   `-- README.md
    `-- sample.jpg (optional)
|-- docs/
|   |-- CHANGELOG.md
|   |-- CONTRIBUTION_GUIDELINES.md
|   |-- PROJECT_CONTEXT.md
|   |-- PROJECT_STATUS.md
|   |-- RESEARCH_LOG.md
|   `-- TODO.md
|-- outputs/ (generated at runtime, not tracked by Git)
|   `-- demo/
|       |-- overlay.png
|       `-- statistics.json
|-- scripts/
|   |-- run_demo.py
|   |-- run_grounding_demo.py
|   `-- verify_environment.py
|-- src/
|   `-- vegetation_analysis/
|       |-- api/
|       |-- config/
|       |   `-- settings.py
|       |-- core/
|       |-- grounding/                  ← Phase 3A Grounding DINO detection modules
|       |   |-- __init__.py
|       |   |-- constants.py
|       |   |-- detector.py
|       |   |-- grounding_dino_loader.py
|       |   |-- prompts.py
|       |   |-- schemas.py
|       |   `-- visualization.py
|       |-- segmentation/               ← Phase 2 archived baseline
|       |   |-- fastsam_loader.py
|       |   |-- mask_utils.py
|       |   |-- schemas.py
|       |   |-- segmenter.py
|       |   `-- visualization.py
|       `-- utils/
|           `-- logging.py
|-- tests/
|   |-- test_grounding_detector.py
|   |-- test_grounding_loader.py
|   |-- test_grounding_prompts_and_visualization.py
|   |-- test_segmentation_loader.py
|   |-- test_segmentation_pipeline.py
|   `-- test_settings.py
|-- .env.example
|-- .gitignore
|-- pyproject.toml
|-- requirements-dev.txt
`-- requirements.txt
```

Folder responsibilities:

- `demo`: sample images for demonstrations. Place `sample.jpg` here to use
  it as the default input for `scripts/run_demo.py`.
- `docs`: project documentation files.
- `outputs/demo`: generated outputs from the demo script (overlays,
  statistics).
- `scripts`: developer utilities and the Phase 2 demo runner.
- `src/vegetation_analysis/api`: future application/API integration boundary.
- `src/vegetation_analysis/config`: runtime settings loaded from environment
  variables.
- `src/vegetation_analysis/core`: future computer vision and domain logic.
- `src/vegetation_analysis/grounding`: Production Grounding DINO implementation responsible for
  model loading, prompt generation, object detection,
  structured detection results, and visualization.
- `src/vegetation_analysis/segmentation`: Phase 2 FastSAM segmentation modules
  (archived baseline — do not delete).
- `src/vegetation_analysis/utils`: shared infrastructure helpers.
- `tests`: automated tests.

## Current Implementation

The repository currently contains two completed computer vision stages.

Phase 2 (archived baseline)

• FastSAM segmentation
• Object masks
• Bounding boxes
• Overlay visualization

Phase 3A

• Grounding DINO model loader
• Prompt builder
• Open-set object detection
• Structured DetectionResult output
• Detection visualization
• Demo script
• Automated tests

Grounding DINO is now the active object detection pipeline.

Phase 3B.2

• SAM 2 mask generation evaluated and validated
• Duplicate pole filtering implemented
• End-to-end segmentation pipeline finalized
• FastSAM officially retired

FastSAM remains in the repository as the archived Phase 2 baseline for
reference and benchmarking.

## Current Limitations

Current limitations:

- Tree species classification has not started.
- Depth estimation has not started.
- Distance estimation has not started.
- API integration has not started.

## Architecture Diagram

The planned full pipeline is shown below. Implemented stages are marked.

```text
Uploaded Image
      |
      v
Grounding DINO  ← IMPLEMENTED (Phase 3A)
      |
      v
Tree / Pole Bounding Boxes (Duplicate poles filtered)
      |
      v
SAM 2 Mask Generation  ← IMPLEMENTED (Phase 3B.1)
      |
      v
Tree & Pole Masks
      |
      v
Depth Anything V2 (Phase 5)
      |
      v
Distance Engine
      |
      v
API Response
```

## Development Workflow

Development is phase-based:

1. Explain what will be built and why.
2. Discuss alternatives, advantages, and disadvantages.
3. Implement only the approved phase or task.
4. Verify the implementation.
5. Update documentation.
6. Stop and wait for user approval before the next phase.

No future phase should be implemented early.

## Documentation Files

- `README.md`: project overview, setup, architecture, and workflow.
- `docs/CHANGELOG.md`: meaningful completed changes, decisions, verification,
  and next phase notes.
- `docs/PROJECT_STATUS.md`: single source of truth for current progress.
- `docs/TODO.md`: actionable immediate, sprint, future, and research tasks.
- `docs/RESEARCH_LOG.md`: research findings, comparisons, rejected approaches,
  experiments, and architecture decisions.
- `docs/CONTRIBUTION_GUIDELINES.md`: engineering standards and collaboration
  rules.
- `docs/PROJECT_CONTEXT.md`: detailed project context and long-term vision.

Documentation rule:

- Update `CHANGELOG.md` after every meaningful completed task.
- Update `PROJECT_STATUS.md` after every meaningful completed task.
- Update `TODO.md` when tasks are added, completed, removed, or reprioritized.
- Update `RESEARCH_LOG.md` when research findings, experiments, comparisons, or
  architecture decisions are involved.

## Coding Standards

- Python 3.11 or newer.
- PEP 8 style.
- Type hints for production code.
- Docstrings for modules, public classes, and important functions.
- Logging instead of print statements in application code.
- Clear separation of concerns.
- No hardcoded paths.
- No unnecessary dependencies.
- Small modules with focused responsibilities.

## Completed Phases

### Phase 1: Environment Setup

Status: ✅ Completed.

Foundation created with project structure, configuration, dependency files,
verification script, tests, and documentation.

### Documentation System Improvements

Status: ✅ Completed.

Documentation tracking was expanded before starting Phase 2.

### Phase 2: FastSAM Integration

Status: ✅ Completed. Archived as the Phase 2 baseline.

Completed work:

- FastSAM model loader with automatic device selection.
- FastSAM segmenter with configurable inference parameters.
- Mask extraction and geometry utilities.
- Segmentation output schemas (`BoundingBox`, `SegmentedObject`).
- Mask overlay visualization module.
- Segmentation package with a clean public API.
- Demo orchestration script (`scripts/run_demo.py`).
- Automated tests for loader, segmenter, mask utilities, and visualization.
- Validated on synthetic and representative real images.

Evaluation finding: FastSAM reliably segments utility poles but consistently
fragments sparse tree canopies into multiple disconnected masks. This makes the
FastSAM → CLIP pipeline unsuitable. Architecture has changed to Grounding DINO.

### Phase 3A: Grounding DINO Detection

Status: ✅ Completed.

Completed work:

- Grounding DINO loader
- Hugging Face model loading
- Prompt builder
- Detection pipeline
- Detection schemas
- Detection visualization
- Demo runner
- Automated unit tests
- CPU inference validation
- Real-image validation

Evaluation:

Grounding DINO successfully detects both utility poles and trees using
natural-language prompts and produces structured DetectionResult objects.

### Phase 3B: SAM 2 Integration

Status: ✅ Completed.

Completed work:

- SAM 2 mask generation infrastructure (Phase 3B.1).
- Duplicate pole filtering (Phase 3B.2).
- Mask evaluation and validation (Phase 3B.2).
- Finalized end-to-end pipeline: Image → Grounding DINO → Duplicate Pole Filtering → SAM 2 → Masks.

FastSAM remains archived as the Phase 2 baseline.

## Upcoming Phase

### Phase 5: Depth Anything V2

Status: Planned.

Planned work:

- Integrate Depth Anything V2.
- Generate metric depth maps for pole and tree regions.

## Installation

Python 3.11 or newer is required.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip

pip install -r requirements.txt
pip install -r requirements-dev.txt

pip install transformers
```

If `python` is not available on PATH, install Python 3.11+ and enable the
option to add Python to PATH during installation.

## First-Time Setup

After cloning the repository:

```powershell
python scripts\verify_environment.py
pytest
python scripts\run_demo.py
python scripts\run_grounding_demo.py
```

If all three commands succeed, the environment has been configured correctly.

## Model Weights

Phase 2

FastSAM-s.pt

Used only by the archived segmentation pipeline.

Phase 3A

Grounding DINO weights are downloaded automatically from Hugging Face on the
first execution and cached locally.

No manual download is required.

## Running the Project

### Environment Verification

```powershell
python scripts\verify_environment.py
```

### Automated Tests

```powershell
pytest
```

### Linting and Type Checking

```powershell
ruff check .
mypy
```

### Phase 2 Segmentation Demo

Run the demo using the synthetic image fallback (no sample image required):

```powershell
python scripts\run_demo.py
```

Run the demo using a specific image:

```powershell
python scripts\run_demo.py path\to\image.jpg
```

Place a representative image at `demo\sample.jpg` to use it as the persistent
default input.

### Phase 3A Grounding DINO Demo

Run the Grounding DINO pipeline:

python scripts\run_grounding_demo.py

Outputs:

outputs/demo/grounding_dino_annotated.png

outputs/demo/grounding_dino_statistics.json

### Expected Demo Outputs

All generated outputs are written to `outputs\demo\`:

| File | Description |
|---|---|
| `overlay.png` | FastSAM segmentation overlay showing colour-coded object masks (Phase 2 archived baseline). |
| `statistics.json` | FastSAM segmentation statistics including object count, bounding boxes, pixel areas, and inference summary. |
| `grounding_dino_annotated.png` | Grounding DINO detection visualization showing labelled bounding boxes, confidence scores, and detected objects. |
| `grounding_dino_statistics.json` | Grounding DINO detection statistics including detected labels, confidence scores, bounding boxes, inference time, model information, and prompt used. |

The Grounding DINO demo also prints a runtime summary to the console:

```text
------------------------------------------------------------
  Vegetation Analysis - Phase 3AGrounding DINO Demo
------------------------------------------------------------
  Image source    : demo\sample.jpg
  Image size      : 1080 x 1440 px
  Prompt          : tree . utility pole .
  Inference time  : 8.954 s
  Boxes found     : 3

  Detection summary:
    utility pole     | conf=0.607 | bbox=(271,478)->(767,1437)
    utility pole     | conf=0.452 | bbox=(507,558)->(766,1437)
    tree             | conf=0.664 | bbox=(505,3)->(1078,1149)

  Output files:
    Annotated   : outputs\demo\grounding_dino_annotated.png
    Statistics  : outputs\demo\grounding_dino_statistics.json
------------------------------------------------------------
```
## Repository Status

Current Stable Release

v0.3.0

Completed

✓ Phase 1 Foundation

✓ Phase 2 FastSAM

✓ Phase 3A Grounding DINO

✓ Phase 3B.1 SAM 2 Foundation
✓ Phase 3B.2 SAM 2 Mask Evaluation

Current Development

Phase 5

Depth Anything V2

## Future Scope

- Metric depth estimation using Depth Anything V2.
- Tree species classification.
- Distance calculation between visible pole and tree edges.
- API response format for main application integration.
- Integration tests with representative electric pole images.
- Production deployment planning.
