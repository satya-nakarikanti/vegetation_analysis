# Vegetation Analysis Module

## Project Overview

This project is a production-oriented computer vision module for vegetation
analysis near electric poles. It will later integrate with an existing electric
pole inspection application and run only when the user selects Tree Analysis.

The module is built phase by phase. Phase 2 (FastSAM segmentation) is now
complete. The current repository contains a working segmentation pipeline that
can detect and mask objects in an image. CLIP-based classification, species
identification, depth estimation, distance estimation, and API integration are
planned for future phases and have not been implemented yet.

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
|   `-- verify_environment.py
|-- src/
|   `-- vegetation_analysis/
|       |-- api/
|       |-- config/
|       |   `-- settings.py
|       |-- core/
|       |-- segmentation/
|       |   |-- fastsam_loader.py
|       |   |-- mask_utils.py
|       |   |-- schemas.py
|       |   |-- segmenter.py
|       |   `-- visualization.py
|       `-- utils/
|           `-- logging.py
|-- tests/
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
- `src/vegetation_analysis/segmentation`: Phase 2 FastSAM segmentation modules.
- `src/vegetation_analysis/utils`: shared infrastructure helpers.
- `tests`: automated tests.

## Current Implementation

Only segmentation has been implemented. The pipeline currently:

1. Accepts an image (file path, NumPy array, or PIL Image).
2. Loads FastSAM using the Ultralytics package.
3. Runs inference and extracts boolean masks and contours for each detected
   object.
4. Returns structured `SegmentedObject` instances containing a mask, bounding
   box, contour, and pixel area.
5. Generates a colour-coded mask overlay image.

No object classification, species identification, depth estimation, or distance
measurement has been implemented.

## Current Limitations

- Object detection produces all visible objects without label assignment.
  There is no way to distinguish a tree from a pole at this stage.
- Sparse or thin tree canopies are segmented inconsistently. Dense, solid
  objects produce more reliable masks.
- No CLIP, depth, or distance logic is present. Those capabilities belong to
  future phases.
- The module does not expose an API endpoint yet.

## Architecture Diagram

The planned full pipeline is shown below. Implemented stages are marked.

```text
Uploaded Image
    |
    v
FastSAM  ← IMPLEMENTED (Phase 2)
    |
    v
Object Masks  ← IMPLEMENTED (Phase 2)
    |
    v
CLIP Mask Identification  ← planned (Phase 3)
    |
    +---> Tree Mask
    |
    +---> Pole Mask
             |
             v
EfficientNetV2 Species Classification  ← planned
             |
             v
Depth Anything V2 Metric  ← planned
             |
             v
Distance Engine  ← planned
             |
             v
API Response  ← planned
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

Status: ✅ Completed.

Completed work:

- FastSAM model loader with automatic device selection.
- FastSAM segmenter with configurable inference parameters.
- Mask extraction and geometry utilities.
- Segmentation output schemas (`BoundingBox`, `SegmentedObject`).
- Mask overlay visualization module.
- Segmentation package with a clean public API.
- Demo orchestration script (`scripts/run_demo.py`).
- Automated tests for loader, segmenter, mask utilities, and visualization.
- Validated on synthetic and representative real images..

## Upcoming Phase

### Phase 3: CLIP Integration

Status: Pending — segmentation evaluation must be finalized first.

Planned work:

- Load CLIP.
- Pass masked objects to CLIP.
- Identify tree and pole masks.
- Return confidence scores.
- Explain prompt engineering approach.
- Verify classification results.

Future phases will cover species classification, depth estimation, distance
estimation, API integration, and production hardening.

## Installation

Python 3.11 or newer is required.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip

pip install -r requirements.txt
pip install -r requirements-dev.txt
```

If `python` is not available on PATH, install Python 3.11+ and enable the
option to add Python to PATH during installation.

## First-Time Setup

After cloning the repository:

```powershell
python scripts\verify_environment.py
pytest
python scripts\run_demo.py
```

If all three commands succeed, the environment has been configured correctly.

## Model Weights

The Phase 2 implementation depends on the FastSAM model weights.

The repository includes:

```
FastSAM-s.pt
```

No additional download is required.

Future phases may require additional model weights (CLIP, EfficientNetV2, Depth Anything V2). Those requirements will be documented when those phases begin.

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

### Expected Demo Outputs

All outputs are written to `outputs\demo\`:

| File | Description |
|---|---|
| `overlay.png` | Source image with colour-coded mask overlays and contours |
| `statistics.json` | Per-object bounding boxes and areas; inference summary |

The script also prints a runtime summary to the console:

```text
------------------------------------------------------------
  Vegetation Analysis — Phase 2 Demo
------------------------------------------------------------
  Image source    : path\to\image.jpg
  Image size      : 1920 × 1080 px
  Inference time  : 1.243 s
  Objects found   : 7

  Per-object summary:
    Object   0 | area=  3,200 px | bbox=(120,45)→(390,310)
    ...

  Output files:
    Overlay     : outputs\demo\overlay.png
    Statistics  : outputs\demo\statistics.json
------------------------------------------------------------
```
## Repository Status

Current Stable Release:

**Phase 2**

Implemented:

- Environment setup
- FastSAM segmentation
- Automated testing
- Demo runner
- Documentation

Current Research Focus:

Segmentation evaluation before beginning CLIP integration.

## Future Scope

- CLIP-based tree and pole mask identification.
- Tree species classification.
- Metric depth estimation.
- Distance calculation between visible pole and tree edges.
- API response format for main application integration.
- Integration tests with representative electric pole images.
- Production deployment planning.
