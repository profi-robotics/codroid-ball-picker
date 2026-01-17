#!/usr/bin/env python3
"""Extract UI builder functions from main.py to components/__init__.py"""
from pathlib import Path

source = Path("app/main.py")
target = Path("app/codroid_picker/components/__init__.py")

content = source.read_text()
lines = content.split('\n')

# Functions to extract and their line ranges (1-indexed inclusive start, exclusive end)
functions_to_extract = [
    ("build_tracking_mode_controls", 236, 282),
    ("build_header", 4051, 4102),
    ("build_footer", 4102, 4582),
    ("build_setup_drawer", 4582, 5019),
    ("build_robot_calibration_modal", 5019, 5120),
    ("build_emergency_stop_modal", 5120, 5168),
    ("build_overspeed_modal", 5168, 5235),
    ("build_joint_protection_modal", 5235, 5288),
    ("build_wrong_tool_modal", 5288, 5341),
    ("build_drag_not_allowed_modal", 5341, 5394),
    ("_calibration_stepper_stack", 5394, 5536),
    ("_calibration_loaded_body", 5536, 5562),
    ("build_stepper_card", 5562, 5605),
    ("build_workspace_step_content", 5605, 5635),
    ("build_camera_panel", 5635, 5702),
    ("build_settings_card", 5702, 5885),
    ("build_calibration_tab", 5885, 5889),
    ("build_detection_tab", 5889, 6006),
    ("build_pick_place_tab", 6006, 6240),
    ("build_application_tab", 6240, 6274),
    ("build_monitoring_tab", 6274, 6335),
    ("build_layout", 6335, 6555),
]

extracted_code = []
for name, start, end in functions_to_extract:
    # Convert to 0-indexed
    func_lines = lines[start-1:end-1]
    extracted_code.append('\n'.join(func_lines))
    print(f"Extracted {name}: lines {start}-{end-1} ({end - start} lines)")

# Build the module header with imports
header = '''"""
UI builder components for the Calibration UI.

This module contains all the build_* functions that construct the UI layout.
"""

from __future__ import annotations
import dash_mantine_components as dmc
from dash import dcc, html

from app.codroid_picker.state import (
    _operation_snapshot,
    _error_snapshot,
)
from app.codroid_picker.session import SESSION
import app.settings as app_settings

# Re-export settings constants used by builders
SETTINGS = app_settings.SETTINGS
DEFAULT_RESOLUTION = app_settings.DEFAULT_RESOLUTION
DEFAULT_FPS = app_settings.DEFAULT_FPS
DEFAULT_BUNDLE_PATH = app_settings.DEFAULT_BUNDLE_PATH
DEFAULT_WORKSPACE_PATH = app_settings.DEFAULT_WORKSPACE_PATH
DEFAULT_ROBOT_CALIBRATION_PATH = app_settings.DEFAULT_ROBOT_CALIBRATION_PATH
DEFAULT_ROBOT_URI = app_settings.DEFAULT_ROBOT_URI
DEFAULT_GRIPPER_IP = app_settings.DEFAULT_GRIPPER_IP
GRIPPER_PROFILE_OPTIONS = app_settings.GRIPPER_PROFILE_OPTIONS

# Gripper constants - refreshed dynamically
GRIPPER_PROFILE = SETTINGS.gripper_profile
GRIPPER_FORCE_MIN = GRIPPER_PROFILE.force_min
GRIPPER_FORCE_MAX = GRIPPER_PROFILE.force_max
GRIPPER_FORCE_DEFAULT = GRIPPER_PROFILE.force_default
GRIPPER_SPEED_MIN = GRIPPER_PROFILE.speed_min


def _placeholder_frame(message: str, size: tuple[int, int] = (720, 1280)):
    """Create a placeholder frame with a message (lazy import cv2/numpy)."""
    import numpy as np
    import cv2
    h, w = size
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    frame[:] = (11, 15, 20)
    cv2.putText(
        frame,
        message,
        (w // 2 - 150, h // 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (128, 128, 128),
        2,
    )
    return frame


def _encode_frame(frame) -> str:
    """Encode a frame as a base64 data URI."""
    import cv2
    import base64
    _, buffer = cv2.imencode(".jpg", frame)
    encoded = base64.b64encode(buffer).decode("utf-8")
    return f"data:image/jpeg;base64,{encoded}"


# These will be set by main.py after import
format_gpu_status = None
_model_options = None


def set_builder_dependencies(gpu_status_fn, model_options_fn):
    """Set external dependencies for builders that need them."""
    global format_gpu_status, _model_options
    format_gpu_status = gpu_status_fn
    _model_options = model_options_fn

'''

# Write the module
module_content = header + '\n\n' + '\n\n\n'.join(extracted_code)
target.parent.mkdir(parents=True, exist_ok=True)
target.write_text(module_content)
print(f"\nWrote {len(module_content)} bytes to {target}")
print(f"Total functions extracted: {len(functions_to_extract)}")
total_lines = sum(end - start for _, start, end in functions_to_extract)
print(f"Total lines extracted: {total_lines}")
