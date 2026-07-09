# Vegetation Analysis for Utility Pole Inspection

An AI-powered computer vision pipeline for analysing vegetation around utility poles from a single RGB image.

The project combines modern object detection, image segmentation, monocular depth estimation, and geometric reasoning to generate structured information about trees and utility poles. The long-term objective is to estimate the real-world clearance between vegetation and electrical infrastructure for automated inspection workflows.

---

# Features

- Utility pole detection using Grounding DINO
- Tree and vegetation detection
- Duplicate pole filtering
- High-quality object segmentation using SAM 2
- Dense monocular depth estimation using Depth Anything V2
- Object-wise depth sampling
- Camera-relative 3D geometry generation
- GPU (CUDA) and CPU execution
- Modular architecture for future AI model integration
- Automated testing with Pytest, Ruff, and MyPy

---

# Current Pipeline

```text
                         RGB Image
                              │
          ┌───────────────────┴───────────────────┐
          │                                       │
          ▼                                       ▼
   Grounding DINO                     Depth Anything V2
          │                                       │
 Duplicate Pole Filtering                Relative Depth Map
          │                                       │
          ▼                                       │
   SAM 2 Segmentation                             │
          │                                       │
 Segmentation Masks ──────────────────────────────┘
                     │
                     ▼
            Depth Sampling Engine
                     │
                     ▼
          Relative Geometry Engine
                     │
                     ▼
       Camera-Relative Object Coordinates
```

Future pipeline additions include:

- Tree Species Classification
- Nearest Point Extraction
- Metric Distance Estimation
- Vegetation Clearance Analysis

---

# Repository Structure

```text
vegetation_analysis/
│
├── checkpoints/
│
├── configs/
│
├── demo/
│
├── docs/
│
├── outputs/
│
├── scripts/
│
├── src/
│   └── vegetation_analysis/
│       ├── config/
│       ├── grounding/
│       ├── sam2/
│       ├── depth/
│       ├── depth_sampling/
│       ├── geometry/
│       ├── segmentation/
│       └── utils/
│
├── tests/
│
├── pyproject.toml
└── README.md
```

---

# Technology Stack

## Programming

- Python 3.10.11

## AI Models

- Grounding DINO Tiny
- SAM 2.1 Tiny
- Depth Anything V2 Small

## Libraries

- PyTorch
- TorchVision
- Transformers
- Hugging Face Hub
- OpenCV
- NumPy
- Pillow

## Development Tools

- Pytest
- Ruff
- MyPy

---

# Installation

Clone the repository

```bash
git clone <repository-url>

cd vegetation_analysis
```

Create a virtual environment

```bash
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

Verify the environment

```bash
python scripts/verify_environment.py
```

Run the automated tests

```bash
pytest
```

Grounding DINO Demo

```bash
python scripts/run_grounding_demo.py
```

SAM 2 Demo

```bash
python scripts/run_sam2_demo.py
```

Depth Anything V2 Demo

```bash
python scripts/run_depth_demo.py
```

Depth Sampling Demo

```bash
python scripts/run_depth_sampling_demo.py
```

Relative Geometry Demo

```bash
python scripts/run_geometry_demo.py
```

---

# Generated Outputs

All demo outputs are written to

```text
outputs/demo/
```
<img width="100" height="200" alt="Screenshot 2026-07-09 123158" src="https://github.com/user-attachments/assets/d9f983cf-188c-4853-b886-dbb62c861a60" />
<p float="left">
  <img width="200" alt="Screenshot 2026-07-09 122724" src="https://github.com/user-attachments/assets/a54bffcb-e7a9-4fde-b99d-466d92f9890f" />
  <img width="200" alt="Screenshot 2026-07-09 122816" src="https://github.com/user-attachments/assets/e8a30a65-23d2-4c75-9660-9c57626e5b20" />
  <img width="200" alt="Screenshot 2026-07-09 122858" src="https://github.com/user-attachments/assets/2bd953d9-a99b-4fb8-98fb-bdcb3d55cb92" />
</p>
<p float="left">
  <img width="200" alt="Screenshot 2026-07-09 122939" src="https://github.com/user-attachments/assets/5080fea1-f80f-483a-89cd-ee5a7b4078a7" />
  <img width="200" alt="Screenshot 2026-07-09 123029" src="https://github.com/user-attachments/assets/ea5095e8-ac25-4452-8ddf-66a49aa76bc0" />
</p>
---

# Example Output

The current pipeline generates structured information for every detected object, including:

- Object label
- Detection confidence
- Segmentation mask
- Relative depth
- Object depth statistics
- Camera-relative coordinates (rx, ry, rz)



Example:

```json
{
  "label": "utility pole",
  "confidence": 0.97,
  "median_depth": 2.756,
  "relative_coordinates": {
    "rx": 0.332,
    "ry": 0.203,
    "rz": 2.756
  }
}
```

---

# Project Documentation

Detailed project documentation is available in the `docs/` directory.

| Document | Description |
|----------|-------------|
| PROJECT_CONTEXT.md | Overall architecture and design decisions |
| PROJECT_STATUS.md | Current project progress |
| CHANGELOG.md | Development history |
| RESEARCH_LOG.md | Research findings and engineering notes |
| TODO.md | Current and future development tasks |

---

# Future Work

- Tree Species Classification
- Camera Calibration
- Nearest Point Extraction
- Metric Distance Estimation
- Vegetation Clearance Estimation
- REST API Integration
- Parallel Pipeline Execution
- Production Deployment

---
