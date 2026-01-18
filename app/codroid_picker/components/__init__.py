"""
UI builder components for the Calibration UI.

This module re-exports all public build_* functions from submodules.
"""

from __future__ import annotations

# Re-export from layout module
from app.codroid_picker.components.layout import (
    build_header,
    build_footer,
)

# Re-export from calibration_tab module
from app.codroid_picker.components.calibration_tab import (
    build_calibration_tab,
    build_stepper_card,
    build_workspace_step_content,
    build_camera_panel,
    build_settings_card,
)

# Re-export from tabs module
from app.codroid_picker.components.tabs import (
    build_detection_tab,
    build_pick_place_tab,
    build_application_tab,
    build_monitoring_tab,
    build_tracking_mode_controls,
    set_builder_dependencies,
)

# Re-export common utilities
from app.codroid_picker.components.common import (
    placeholder_frame,
    _encode_frame,
    GRIPPER_FORCE_MIN,
    GRIPPER_FORCE_MAX,
    GRIPPER_FORCE_DEFAULT,
)

__all__ = [
    # Layout
    "build_header",
    "build_footer",
    # Calibration tab
    "build_calibration_tab",
    "build_stepper_card",
    "build_workspace_step_content",
    "build_camera_panel",
    "build_settings_card",
    # Tabs
    "build_detection_tab",
    "build_pick_place_tab",
    "build_application_tab",
    "build_monitoring_tab",
    "build_tracking_mode_controls",
    "set_builder_dependencies",
    # Common utilities
    "placeholder_frame",
    "_encode_frame",
    "GRIPPER_FORCE_MIN",
    "GRIPPER_FORCE_MAX",
    "GRIPPER_FORCE_DEFAULT",
]

