# demo/

This directory holds sample images used for demonstrations and exploratory
testing of the vegetation analysis pipeline.

## Usage

Place any representative image here and name it `sample.jpg` to use it as
the default input for `scripts/run_demo.py`.

If no image is present, the demo script generates a synthetic RGB image
automatically so the pipeline can always be exercised.

## Notes

- Images in this directory are ignored by Git (see `.gitignore`).
- Do not place production data here.
- Do not import from this directory in production source code.
