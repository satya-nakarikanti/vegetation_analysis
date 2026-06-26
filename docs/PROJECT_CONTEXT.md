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
FastSAM
↓
CLIP Classification
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

## Phase 3

CLIP

Responsibilities:

Classify each segmented object.

Current target labels:

* Tree
* Pole

The objective is to identify which segmented objects should continue through the vegetation pipeline.

Status:

Planned

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

Current:

Image

↓

FastSAM

↓

SegmentedObject

Future:

SegmentedObject

↓

CLIP

↓

Tree Objects

↓

Species Classifier

↓

Tree Species

↓

Depth Anything

↓

Relative Depth

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

Phase 2 completed.

Completed:

* Environment setup
* Project foundation
* FastSAM integration
* Segmentation pipeline
* Structured segmentation objects
* Visualization
* Automated testing

Upcoming:

Phase 3

CLIP integration for Tree/Pole classification.

---

# Long-Term Goal

Create a scalable vegetation analysis module that integrates seamlessly into the existing electric pole inspection platform.

Every phase should strengthen the architecture so future models can be integrated with minimal code changes.

The project should remain maintainable, testable, and production-ready throughout development.
