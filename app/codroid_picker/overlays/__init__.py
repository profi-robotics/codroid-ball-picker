"""Overlay drawing functions for the calibration UI.

This package contains all the overlay drawing functions that are used to
visualize various aspects of the calibration and detection process on
the camera feed.
"""

from .base import _plate_center_px
from .drawing import (
    _draw_label,
    _draw_dotted_polyline,
    _workspace_corners_px,
    _apply_workspace_overlay,
)
from .calibration import (
    draw_charuco_overlay,
    draw_workspace_overlay,
    draw_plate_overlay,
    _refresh_plate_status,
)
from .detection import (
    draw_detection_overlay,
    draw_drop_marker_overlay,
    draw_monitor_overlay,
)
from .auto_plan import draw_auto_plan_overlay

__all__ = [
    # Base utilities
    "_plate_center_px",
    # Drawing helpers
    "_draw_label",
    "_draw_dotted_polyline",
    "_workspace_corners_px",
    "_apply_workspace_overlay",
    # Calibration overlays
    "draw_charuco_overlay",
    "draw_workspace_overlay",
    "draw_plate_overlay",
    "_refresh_plate_status",
    # Detection overlays
    "draw_detection_overlay",
    "draw_drop_marker_overlay",
    "draw_monitor_overlay",
    "draw_auto_plan_overlay",
]
