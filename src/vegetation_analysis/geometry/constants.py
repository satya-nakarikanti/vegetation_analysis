"""Shared constants for the Relative Geometry Engine."""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

#: Font scale for overlay text labels.
OVERLAY_FONT_SCALE: float = 0.50

#: Font thickness for overlay text labels.
OVERLAY_FONT_THICKNESS: int = 1

#: Line spacing in pixels between overlay text lines.
OVERLAY_LINE_SPACING: int = 14

#: Color used for object label text in the visualization (RGB).
LABEL_COLOR: tuple[int, int, int] = (255, 255, 255)

#: Color used for coordinate text in the visualization (RGB).
COORD_COLOR: tuple[int, int, int] = (180, 230, 255)

#: Radius of the centroid circle marker drawn in the visualization.
CENTROID_RADIUS: int = 5

#: Color of the centroid circle marker (RGB).
CENTROID_MARKER_COLOR: tuple[int, int, int] = (255, 255, 100)
