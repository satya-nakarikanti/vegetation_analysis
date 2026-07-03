# PROJECT_CONTEXT.md

# Project Context

## Project Name

Vegetation Analysis Module for Electric Pole Inspection

---

# Project Overview

This project is a modular Computer Vision system designed to identify vegetation near electric poles and estimate the approximate distance between vegetation and the nearest visible part of the electric pole.

The vegetation module is designed as an independent subsystem that can later be integrated into a larger electric pole inspection platform.

The long-term objective is to automatically assist utility inspections by identifying nearby vegetation and estimating vegetation clearance using only RGB images.

---

# Overall System Vision

The complete pipeline is planned as:

Image
↓
Grounding DINO (Completed)
↓
Duplicate Pole Filtering (Completed)
↓
DetectionResult
↓
SAM 2 (Completed)
↓
TreeMask
PoleMask
↓
Tree Species Classification
↓
Depth Anything V2
↓
Distance Estimation
↓
Results
↓
Main Pole Inspection Project

This repository currently develops only the vegetation analysis module.

Note: The original pipeline used FastSAM → CLIP. After evaluating FastSAM on
real electric-pole images, sparse tree canopies were found to fragment into
multiple disconnected masks, making the FastSAM → CLIP approach unsuitable.
FastSAM is retained as the archived Phase 2 baseline. Grounding DINO was
selected as the Phase 3 replacement because it produces a single labelled
bounding box per detected object regardless of canopy fragmentation.

---

# Integration with Main Project

The vegetation module is **not** responsible for:

* Pole component inspection
* OCR
* Defect detection
* Pole analysis
* Report generation

Those responsibilities belong to the existing electric pole inspection project.

The vegetation module will eventually expose a clean API so the main project can send an image and receive vegetation analysis results.

---

# Current Development Philosophy

The project follows a modular architecture.

Every major AI model is isolated inside its own module.

The objective is to make every model replaceable without affecting the remaining system.

Future models should integrate through clean interfaces instead of tightly coupling implementations.

---

# Planned AI Pipeline

## Phase 2

FastSAM

Responsibilities:

* Segment every visible object
* Generate object masks
* Generate contours
* Generate bounding boxes
* Produce structured segmentation objects

Output:

SegmentedObject instances

Status:

Completed

---

## Phase 3A

Grounding DINO

Responsibilities:

* Detect trees using natural-language prompts.
* Detect utility poles using natural-language prompts.
* Produce structured bounding boxes.
* Produce confidence scores.
* Provide bounding-box prompts for SAM 2.

Output:

DetectionResult

Status:

Completed.

Implementation:

* Hugging Face Grounding DINO integration.
* Automatic CPU/CUDA device selection.
* Prompt builder.
* Detection visualization.
* Demo runner.
* Automated tests.

Evaluation:

Validated successfully on representative electric-pole images.

---

## Phase 3B

SAM 2

Responsibilities:

* Generate segmentation masks using Grounding DINO bounding boxes.
* Produce one mask per detected object.
* Improve boundary accuracy over bounding boxes.
* Generate masks suitable for later distance estimation.

Output:

TreeMask

PoleMask

Status:

Completed.

---

## Phase 4

Tree Species Classification

Model:

EfficientNetV2

Input:

Tree mask / crop

Output:

* Tree species
* Confidence score

Current dataset:

* 31 Indian tree species
* Approximately 50 images per species

Status:

Planned

---

## Phase 5

Depth Estimation

Model:

Depth Anything V2

Responsibilities:

Generate a monocular relative depth map for the entire image.

Output:

Relative depth map.

Important:

Depth Anything does **not** provide metric distance directly.

Additional algorithms are required for metric estimation.

Status:

Planned

---

## Phase 6

Distance Estimation

Goal:

Estimate the approximate distance between:

Nearest visible tree edge

↓

Nearest visible pole edge

The nearest pole edge may belong to:

* Pole body
* Pole component
* Conductor
* Wire

depending on visibility.

Current target accuracy:

Approximate engineering estimate.

Example:

3.5 m ± 0.3 m

Exact survey-grade measurements are not required.

Status:

Research ongoing.

---

# Current Architecture

The project currently follows this layered architecture.

Configuration

↓

Model Loading

↓

Inference

↓

Result Adapters

↓

Project Schemas

↓

Visualization

↓

Future AI Modules

Each layer has a single responsibility.

---

# Folder Responsibilities

src/

Production source code.

Never place demo code here.

scripts/

Executable helper scripts.

Examples:

* environment verification
* demonstrations
* utilities

demo/

Sample images and videos for demonstrations.

outputs/

Generated outputs.

Examples:

* overlays
* masks
* debug images

docs/

Project documentation.

tests/

Automated tests.

---

# Engineering Principles

The project prioritizes:

* Modular design
* Separation of concerns
* Production-quality code
* Replaceable AI models
* Strong typing
* Maintainability
* Readability

The objective is long-term maintainability rather than rapid prototyping.

---

# Data Flow

Active pipeline (Phase 3A):

Image

↓

Grounding DINO

↓

Duplicate Pole Filtering (Completed)

↓

DetectionResult

↓

Phase 3B (Completed)

↓

SAM 2

↓

TreeMask
PoleMask

↓

Species Classification

↓

Depth Anything

↓

Distance Engine

↓

Final Result
---

# Current API Goal

The vegetation module should eventually expose a simple interface.

Example:

Input:

Image

Output:

{
"tree_detected": true,
"species": "Neem",
"species_confidence": 0.96,
"distance_to_pole": 3.4,
"distance_confidence": 0.88
}

The implementation details should remain hidden behind the API.

---

# Coding Standards

Every new feature should:

* Reuse existing modules
* Avoid duplicate logic
* Preserve modular architecture
* Remain independently testable
* Keep files focused on one responsibility

Production code should never depend directly on demonstration scripts.

---

# Project Status

Current Phase:

Phase 3B completed.

Grounding DINO + SAM 2 is now the active object detection and segmentation pipeline.

Completed:

Completed:

* Environment setup
* Project foundation
* Documentation system
* FastSAM integration (archived Phase 2 baseline)
* Segmentation pipeline
* Grounding DINO architecture migration
* Grounding DINO implementation
* SAM 2 integration
* Duplicate pole filtering
* Automated testing
* Real-image validation

Upcoming:

Phase 5

Metric depth estimation using Depth Anything V2.

---

# Long-Term Goal

Create a scalable vegetation analysis module that integrates seamlessly into the existing electric pole inspection platform.

Every phase should strengthen the architecture so future models can be integrated with minimal code changes.

The project should remain maintainable, testable, and production-ready throughout development.
