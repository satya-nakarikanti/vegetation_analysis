# demo/

This directory contains representative images used for demonstrations,
development, validation, and exploratory testing of the vegetation analysis
pipeline.

## Usage

Place a representative utility-pole image in this directory and name it
`sample.jpg`.

The image will automatically be used as the default input for all available
demo scripts:

- `scripts/run_demo.py` (Phase 2 FastSAM archived baseline)
- `scripts/run_grounding_demo.py` (Phase 3A Grounding DINO)
- `scripts/run_sam2_demo.py` (Phase 3B SAM 2 Segmentation)
- `scripts/run_depth_demo.py` (Phase 5.1 Depth Anything V2)
- `scripts/run_depth_sampling_demo.py` (Phase 5.2 Depth Sampling)
- `scripts/run_geometry_demo.py` (Phase 5.3 Relative Geometry)

If `sample.jpg` is not present:

- `run_demo.py` automatically generates a synthetic RGB image for FastSAM testing.
- All other demo scripts require a valid input image.

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

### FastSAM (Phase 2 Archived)

- `overlay.png`
- `statistics.json`

### Grounding DINO (Phase 3A)

- `grounding_dino_annotated.png`
- `grounding_dino_statistics.json`

### SAM 2 (Phase 3B)

- `sam2_annotated.png`
- `sam2_statistics.json`

### Depth Anything V2 (Phase 5.1)

- `depth_map.png`
- `depth_grayscale.png`
- `depth.npy`
- `depth_statistics.json`

### Depth Sampling (Phase 5.2)

- `depth_sampling_annotated.png`
- `depth_sampling_statistics.json`

### Relative Geometry (Phase 5.3)

- `geometry_annotated.png`
- `geometry_statistics.json`

---

## Notes

- Images in this directory are intended only for development and testing.
- Do not place confidential or production images in this folder.
- Production code must never import files directly from this directory.
- FastSAM is retained as the archived Phase 2 segmentation baseline.
- Grounding DINO is the production object detection model.
- Duplicate pole filtering removes overlapping pole detections before segmentation.
- SAM 2 is the production segmentation model.
- Depth Anything V2 generates dense relative depth maps.
- Depth Sampling computes robust object-wise depth statistics using segmentation masks.
- Relative Geometry converts sampled object depth into camera-relative `(rx, ry, rz)` coordinates.
- Tree species classification is being developed in parallel.
- Future phases will introduce nearest-point extraction, metric calibration, and real-world vegetation clearance estimation.