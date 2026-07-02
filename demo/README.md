# demo/

This directory contains representative images used for demonstrations,
development, validation, and exploratory testing of the vegetation analysis
project.

## Usage

Place a representative electric-pole image in this directory and name it
`sample.jpg`.

The image will automatically be used as the default input for both demo
scripts:

- `scripts/run_demo.py` (Phase 2 FastSAM archived baseline)
- `scripts/run_grounding_demo.py` (Phase 3A Grounding DINO)

If `sample.jpg` is not present:

- `run_demo.py` automatically generates a synthetic RGB image for FastSAM
  testing.
- `run_grounding_demo.py` expects a valid input image and is intended for
  real-image evaluation.

## Recommended Image

Use a representative image that contains:

- One or more utility poles.
- Nearby trees or vegetation.
- Natural outdoor lighting.
- Minimal motion blur.
- Clearly visible pole and tree structures.

Keeping a consistent sample image allows developers to compare outputs across
different phases and models.

## Generated Outputs

Running the demo scripts creates outputs inside `outputs/demo/`:

### FastSAM (Phase 2 Archived)

- `overlay.png`
- `statistics.json`

### Grounding DINO (Phase 3A)

- `grounding_dino_annotated.png`
- `grounding_dino_statistics.json`

## Notes

- Images in this directory are intended only for development and testing.
- Do not place confidential or production images in this folder.
- Production code must never import files directly from this directory.
- FastSAM is retained as the archived Phase 2 baseline for comparison.
- Grounding DINO is the active object detection pipeline.