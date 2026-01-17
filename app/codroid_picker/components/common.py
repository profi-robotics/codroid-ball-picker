"""Common utilities and constants for UI components.

This module provides shared utilities, constants, and imports used across
all component modules.
"""

from __future__ import annotations

import base64
from pathlib import Path
from typing import Tuple

import cv2
import dash_mantine_components as dmc
import numpy as np
from dash import dcc, html

from app.settings import (
    SETTINGS,
    DEFAULT_BUNDLE_PATH,
    DEFAULT_WORKSPACE_PATH,
    DEFAULT_ROBOT_CALIBRATION_PATH,
    DEFAULT_RESOLUTION,
    DEFAULT_FPS,
    DEFAULT_ROBOT_URI,
    DEFAULT_GRIPPER_IP,
    GRIPPER_PROFILE_OPTIONS,
)
from app.codroid_picker.state import _operation_snapshot, get_error_snapshot
from app.codroid_picker.utils import placeholder_frame

# Re-export commonly used items
__all__ = [
    # Types
    "Path",
    "Tuple",
    # Libraries
    "cv2",
    "np",
    "dmc",
    "dcc",
    "html",
    # Settings
    "SETTINGS",
    "DEFAULT_BUNDLE_PATH",
    "DEFAULT_WORKSPACE_PATH",
    "DEFAULT_ROBOT_CALIBRATION_PATH",
    "DEFAULT_RESOLUTION",
    "DEFAULT_FPS",
    "DEFAULT_ROBOT_URI",
    "DEFAULT_GRIPPER_IP",
    "GRIPPER_PROFILE_OPTIONS",
    # State
    "_operation_snapshot",
    "get_error_snapshot",
    # Constants
    "GRIPPER_FORCE_MIN",
    "GRIPPER_FORCE_MAX",
    "GRIPPER_FORCE_DEFAULT",
    # Functions
    "placeholder_frame",
    "_placeholder_frame",
    "_encode_frame",
]

# Compute gripper constants from profile
GRIPPER_FORCE_MIN = SETTINGS.gripper_profile.force_min
GRIPPER_FORCE_MAX = SETTINGS.gripper_profile.force_max
GRIPPER_FORCE_DEFAULT = SETTINGS.gripper_profile.force_default

# Keep underscore-prefixed alias for backward compatibility
_placeholder_frame = placeholder_frame


def _encode_frame(frame: np.ndarray) -> str:
    """Encode a frame as a base64 data URL.

    Args:
        frame: OpenCV BGR frame to encode.

    Returns:
        Base64 encoded JPEG data URL.

    Raises:
        RuntimeError: If JPEG encoding fails.
    """
    success, buffer = cv2.imencode(".jpg", frame)
    if not success:
        raise RuntimeError("Failed to encode camera frame.")
    encoded = base64.b64encode(buffer).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"
