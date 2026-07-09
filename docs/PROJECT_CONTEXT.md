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

RGB Image
        │
        ├──────────────────────────────┐
        │                              │
        ▼                              ▼
Grounding DINO (Completed)      Depth Anything V2 (Completed)
        │                              │
Duplicate Pole Filtering              Relative Depth Map
(Completed)                            │
        │                              │
        ▼                              │
SAM 2 Segmentation (Completed)         │
        │                              │
Segmentation Masks ────────────────────┘
        │
        ▼
Depth Sampling Engine (Completed)
        │
        ▼
Relative Geometry Engine (Completed)
        │
        ▼
Camera-Relative Object Coordinates
        │
        ▼
Tree Species Classification (In Progress)
        │
        ▼
Nearest Point Extraction
        │
        ▼
Metric Distance Estimation
        │
        ▼
Vegetation Clearance Estimation
        │
        ▼
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

## Phase 5.1

Depth Anything V2

Model:

Depth Anything V2 Small

Responsibilities:

* Generate a dense relative depth map.
* Preserve original image resolution.
* Produce grayscale and colorized depth visualizations.
* Export raw depth arrays.

Output:

Relative depth map.

Status:

Completed.

---

## Phase 5.2

Depth Sampling Engine

Responsibilities:

* Combine SAM 2 segmentation masks with the dense depth map.
* Compute robust object-wise depth statistics.
* Generate representative depth values for every detected object.

Output:

Object-wise depth statistics.

Includes:

* median depth
* mean depth
* minimum depth
* maximum depth
* standard deviation

Status:

Completed.

---

## Phase 5.3

Relative Geometry Engine

Responsibilities:

* Convert object image coordinates into normalized camera-relative coordinates.
* Combine centroid location with sampled object depth.
* Produce relative 3D object coordinates.

Output:

Camera-relative object coordinates.

Includes:

* relative_x
* relative_y
* relative_z

Status:

Completed.

---
## Phase 6

Metric Distance Estimation

Goal:

Convert the relative camera-coordinate representation into an engineering-scale distance estimate.

Future work includes:

* Camera calibration
* Pole reference frame
* Nearest-point extraction
* Metric scaling
* Vegetation clearance estimation

Status:

Planned.

---

# Current Architecture

The project currently follows this layered architecture.

Configuration

↓

Model Loading

↓

Inference

↓

Structured Schemas

↓

Visualization

↓

Geometry

↓

Future Distance Engine

Each layer has a single responsibility.

---

# Folder Responsibilities

src/

Production source code.

Current modules:

* grounding/
* sam2/
* depth/
* depth_sampling/
* geometry/
* segmentation/ (Phase 2 archived baseline)

Each module is independently testable and loosely coupled.

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

Current production pipeline:

RGB Image

        │
        ├──────────────────────────────┐
        │                              │
        ▼                              ▼
Grounding DINO                  Depth Anything V2
        │                              │
Duplicate Pole Filtering              │
        │                              │
        ▼                              ▼
SAM 2 Segmentation            Relative Depth Map
        │                              │
        └──────────────┬───────────────┘
                       ▼
              Depth Sampling Engine
                       ▼
             Relative Geometry Engine
                       ▼
        Camera-Relative Object Coordinates

---

# Current API Goal

The vegetation module should eventually expose a simple interface.

Example:

Input:

Image

Output:

{
  "tree_detected": true,
  "species": null,
  "species_confidence": null,

  "relative_coordinates": {
    "rx": -0.782,
    "ry": 0.430,
    "rz": 0.357
  },

  "distance_to_pole": null
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

Phase 5.3 completed.

Completed:

* Environment setup
* Documentation system
* FastSAM baseline (archived)
* Grounding DINO integration
* Duplicate pole filtering
* SAM 2 segmentation
* Depth Anything V2 integration
* Depth Sampling Engine
* Relative Geometry Engine
* CUDA acceleration
* Automated testing
* Real-image validation

Upcoming:

Phase 4

Tree Species Classification

followed by

Phase 6

Metric Distance Estimation

Nearest-point geometry

Vegetation clearance estimation

---

# Long-Term Goal

The long-term objective is to build a production-ready vegetation analysis
module capable of estimating the real-world clearance between vegetation and
utility infrastructure from a single RGB image.

The architecture is intentionally modular so that future improvements, including
species classification, nearest-point extraction, metric calibration, and
distance estimation, can be integrated with minimal changes to the existing
pipeline.