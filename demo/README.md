# demo/

This directory contains representative images used for demonstrations,
development, validation, and exploratory testing of the vegetation analysis
pipeline.

## Usage

Place a representative utility-pole image in this directory and name it
`sample.jpg`.

The image will automatically be used as the default input for the production demo script:

- `scripts/run_geometry_demo.py` (End-to-End Perception Pipeline)

If `sample.jpg` is not present, the demo script requires a valid input image.

---

## Recommended Image

Use a representative image that contains:

- One or more utility poles.
- Nearby trees or vegetation.
- Natural outdoor lighting.
- Minimal motion blur.
- Clearly visible pole structures.
- Clearly visible tree trunks and foliage.

Keeping a consistent sample image allows direct comparison across different
pipeline stages and model versions.

---

## Current Processing Pipeline

The current perception pipeline is:

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
   Segmentation Masks ────────────────────────────┘
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

---

## Generated Outputs

Running the demo scripts creates outputs inside `outputs/demo/`.

### Production Pipeline

- `geometry_annotated.png`
- `geometry_statistics.json`

### Relative Geometry (Phase 5.3)

- `geometry_annotated.png`
- `geometry_statistics.json`

---

## Notes

- Images in this directory are intended only for development and testing.
- Do not place confidential or production images in this folder.
- Production code must never import files directly from this directory.
- Grounding DINO is the production object detection model.
- Duplicate pole filtering removes overlapping pole detections before segmentation.
- SAM 2 is the production segmentation model.
- Depth Anything V2 generates dense relative depth maps.
- Depth Sampling computes robust object-wise depth statistics using segmentation masks.
- Relative Geometry converts sampled object depth into camera-relative `(rx, ry, rz)` coordinates.
- Tree species classification is being developed in parallel.
- Future phases will introduce nearest-point extraction, metric calibration, and real-world vegetation clearance estimation.