"""
Dash calibration UI for the Codroid bring-up workflow.
"""

from __future__ import annotations
from vision.stream import DepthAIFrameStream, StreamConfig
from vision.camera_check import list_oak_devices
from vision import aruco_board, calibration, markers, plate, workspace
from onrobot.twofg import TWOFG as TwoFG7Gripper
from onrobot.device import Device as GripperDevice
from onrobot.gripper_profiles import DEFAULT_GRIPPER_TYPE, gripper_profile_options
from codroid_api import (
    CodroidAPI,
    CodroidSettings,
    RobotSession,
    DEFAULT_FLANGE_BUTTON_PORT,
    DEFAULT_FLANGE_BUTTON_PORTS,
)
from codroid_api.robot_session import RobotPosture
import app.settings as app_settings
from app.settings import _commit_settings, _save_settings, reload_defaults, set_asset_paths
from app.ui.session import CALIBRATION_PROGRESS, SESSION, _refresh_session_auto_config_defaults
from app.calibration_ui import state

# Temporary compatibility globals during refactoring Phase 1
# These will be removed as we continue to replace references
PLATE_STATUS = state.get_plate_status_snapshot()
ROBOT_CALIBRATION_GUIDANCE = state.get_calibration_guidance_snapshot()

def _update_compatibility_globals():
    """Update compatibility globals to reflect current state - called periodically."""
    global PLATE_STATUS, ROBOT_CALIBRATION_GUIDANCE
    PLATE_STATUS = state.get_plate_status_snapshot()
    ROBOT_CALIBRATION_GUIDANCE = state.get_calibration_guidance_snapshot()

import asyncio
import base64
import io
import json
import logging
import os
import re
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable, Deque, Optional, Tuple

import cv2
import numpy as np
from dash import Dash, Input, Output, State, callback_context, dcc, html, no_update
import dash_mantine_components as dmc

LOGGER = logging.getLogger(__name__)


try:
    import pynvml
    NVML_AVAILABLE = True
except ImportError:
    NVML_AVAILABLE = False

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

DEFAULT_BOARD_PATH = app_settings.DEFAULT_BOARD_PATH
DEFAULT_PLATE_PATH = app_settings.DEFAULT_PLATE_PATH
DEFAULT_BUNDLE_PATH = app_settings.DEFAULT_BUNDLE_PATH
DEFAULT_WORKSPACE_PATH = app_settings.DEFAULT_WORKSPACE_PATH
DEFAULT_ROBOT_CALIBRATION_PATH = app_settings.DEFAULT_ROBOT_CALIBRATION_PATH
DEFAULT_CAPTURE_DIR = app_settings.DEFAULT_CAPTURE_DIR
DEFAULT_RESOLUTION = app_settings.DEFAULT_RESOLUTION
DEFAULT_FPS = app_settings.DEFAULT_FPS
DEFAULT_MODEL_DIR = app_settings.DEFAULT_MODEL_DIR
DEFAULT_DROP_MARKER_ID = app_settings.DEFAULT_DROP_MARKER_ID
DROP_TARGET_PATH = app_settings.DROP_TARGET_PATH
SETTINGS = app_settings.SETTINGS
CODROID_DEFAULTS = app_settings.CODROID_DEFAULTS
DEFAULT_ROBOT_URI = app_settings.DEFAULT_ROBOT_URI
GRIPPER_DEFAULTS = app_settings.GRIPPER_DEFAULTS
DEFAULT_GRIPPER_IP = app_settings.DEFAULT_GRIPPER_IP
GRIPPER_PROFILE_OPTIONS = app_settings.GRIPPER_PROFILE_OPTIONS
GRIPPER_PROFILE_KEYS = {option["value"] for option in GRIPPER_PROFILE_OPTIONS}
GRIPPER_PROFILE = SETTINGS.gripper_profile
GRIPPER_FORCE_MIN = GRIPPER_PROFILE.force_min
GRIPPER_FORCE_MAX = GRIPPER_PROFILE.force_max
GRIPPER_FORCE_DEFAULT = GRIPPER_PROFILE.force_default
GRIPPER_SPEED_DEFAULT = GRIPPER_PROFILE.speed_default
GRIPPER_SPEED_MIN = GRIPPER_PROFILE.speed_min
GRIPPER_CALIBRATION_WIDTH_MM = GRIPPER_PROFILE.calibration_width
GRIPPER_OPEN_WIDTH_DEFAULT = GRIPPER_PROFILE.open_width_default
GRIPPER_CLOSE_WIDTH_DEFAULT = GRIPPER_PROFILE.close_width_default
ROBOT_COORDINATE_ID = SETTINGS.robot_coordinate_id


def _refresh_gripper_constants() -> None:
    """Refresh cached gripper limits when settings change."""
    global GRIPPER_PROFILE
    global GRIPPER_FORCE_MIN, GRIPPER_FORCE_MAX, GRIPPER_FORCE_DEFAULT
    global GRIPPER_SPEED_MIN, GRIPPER_SPEED_DEFAULT
    global GRIPPER_CALIBRATION_WIDTH_MM, GRIPPER_OPEN_WIDTH_DEFAULT, GRIPPER_CLOSE_WIDTH_DEFAULT

    GRIPPER_PROFILE = SETTINGS.gripper_profile
    GRIPPER_FORCE_MIN = GRIPPER_PROFILE.force_min
    GRIPPER_FORCE_MAX = GRIPPER_PROFILE.force_max
    GRIPPER_FORCE_DEFAULT = GRIPPER_PROFILE.force_default
    GRIPPER_SPEED_MIN = GRIPPER_PROFILE.speed_min
    GRIPPER_SPEED_DEFAULT = GRIPPER_PROFILE.speed_default
    GRIPPER_CALIBRATION_WIDTH_MM = GRIPPER_PROFILE.calibration_width
    GRIPPER_OPEN_WIDTH_DEFAULT = GRIPPER_PROFILE.open_width_default
    GRIPPER_CLOSE_WIDTH_DEFAULT = GRIPPER_PROFILE.close_width_default


def _camera_calibration_path() -> Path:
    """Return the currently configured camera calibration bundle path."""
    return DEFAULT_BUNDLE_PATH


def _robot_calibration_path() -> Path:
    """Return the currently configured robot calibration data path."""
    return DEFAULT_ROBOT_CALIBRATION_PATH


def _calibration_metadata_entries() -> list[str]:
    """Produce human-readable metadata about existing calibration artifacts."""
    entries: list[str] = []
    bundle = SESSION.calibration_bundle
    if bundle is not None:
        pixel_scale = bundle.pixel_to_mm
        entries.append(f"Pixel scale: {pixel_scale:.3f} mm/px")
        cam = bundle.camera.camera_matrix
        entries.append(
            f"Camera focal: fx={cam[0,0]:.1f}, fy={cam[1,1]:.1f} px"
        )
        plane = bundle.workspace_plane
        entries.append(
            f"Workspace plane offset: {plane.offset:.1f} mm · normal="
            f"{tuple(float(v) for v in plane.normal.round(3))}"
        )
    if DEFAULT_BUNDLE_PATH.exists():
        entries.append(
            f"Camera bundle saved: "
            f"{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(DEFAULT_BUNDLE_PATH.stat().st_mtime))}"
        )
    if SESSION.robot_calibration_points:
        entries.append(
            f"Robot calibration points: {len(SESSION.robot_calibration_points)}"
        )
    if SESSION.robot_calibration_min_z is not None:
        entries.append(
            f"Robot min Z: {SESSION.robot_calibration_min_z:.1f} mm"
        )
    if DEFAULT_ROBOT_CALIBRATION_PATH.exists():
        entries.append(
            f"Robot calibration saved: "
            f"{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(DEFAULT_ROBOT_CALIBRATION_PATH.stat().st_mtime))}"
        )
    accuracy = state.get_calibration_guidance_snapshot().get("accuracy")
    if accuracy:
        max_err = accuracy.get("max_xy_error")
        mean_err = accuracy.get("mean_xy_error")
        if max_err is not None and mean_err is not None:
            entries.append(
                f"Calibration accuracy max/mean XY error: "
                f"{max_err:.2f}/{mean_err:.2f} mm"
            )
    return entries


def _calibration_data_complete() -> bool:
    """Check whether both camera and robot calibration artifacts exist."""
    return _camera_calibration_path().exists() and _robot_calibration_path().exists()


def _refresh_calibration_menu_visibility() -> bool:
    """Update the session flag that controls whether the calibration stepper is visible."""
    SESSION.calibration_menu_open = not _calibration_data_complete()
    return SESSION.calibration_menu_open


def _reset_calibration_assets() -> None:
    """Delete saved calibration files and clear cached session state."""
    for path in (_camera_calibration_path(), _robot_calibration_path()):
        try:
            path.unlink()
        except FileNotFoundError:
            pass
    SESSION.calibration_bundle = None
    SESSION.plate_calibration = None
    SESSION.robot_calibration_points = []
    SESSION.robot_calibration_min_z = None
    SESSION.plate_reference_homography = None
    SESSION.plate_to_robot_transform = None
    SESSION.plate_origin_mm = None
    SESSION.robot_calibration_accuracy = None
    SESSION.robot_calibration_gripper_width = None
    _refresh_calibration_menu_visibility()


_refresh_gripper_constants()


def build_tracking_mode_controls() -> dmc.Group:
    """Build shared tracking mode controls used in both plate test and dynamic pick & place."""
    return dmc.Group(
        [
            dmc.Select(
                id="shared-tracking-mode",
                label="Tracking mode",
                data=[
                    {"label": "Move target (default)", "value": "move"},
                    {"label": "Segmented tracking", "value": "segment"},
                ],
                value=SETTINGS.tracking_mode,
                clearable=False,
            ),
            dmc.NumberInput(
                id="shared-segment-min",
                label="Segment min (mm)",
                description="Minimum step size for segmented tracking.",
                value=SETTINGS.segment_min,
                min=1.0,
                max=200.0,
                step=1.0,
            ),
            dmc.NumberInput(
                id="shared-segment-max",
                label="Segment max (mm)",
                description="Maximum step size for segmented tracking.",
                value=SETTINGS.segment_max,
                min=5.0,
                max=500.0,
                step=5.0,
            ),
            dmc.NumberInput(
                id="shared-segment-fraction",
                label="Segment fraction",
                description="Fraction of remaining XY distance each step; higher = fewer, longer segments.",
                value=SETTINGS.segment_fraction,
                min=0.1,
                max=1.0,
                step=0.1,
            ),
        ],
        grow=True,
    )


def _clamp_z_for_coord5(z: float, *, use_base_coords: bool) -> float:
    """Clamp Z to safe maximum when using coordinate system 5.

    CRITICAL: Coord5 Z-axis is INVERTED:
    - Negative Z = UP (safe, away from table)
    - Positive Z = DOWN (dangerous, toward table)

    This prevents the robot from diving into the table by clamping
    Z values that are TOO POSITIVE (too far down).
    """
    if use_base_coords:
        # Base coordinates use robot's own Z limits; no clamping here.
        return z
    max_z = SETTINGS.coord5_max_z_mm
    if z > max_z:
        LOGGER.warning(
            "Z value %.3f mm EXCEEDS safe maximum %.3f mm in coord5 (would collide with table); clamping to %.3f",
            z, max_z, max_z,
        )
        return max_z
    return z


ROOT_DIR = Path(__file__).resolve().parent.parent
ASSETS_DIR = ROOT_DIR / "assets"


def _get_robot_session() -> RobotSession:
    if SESSION.robot_session is None:
        SESSION.robot_session = RobotSession(
            CodroidSettings(),
            flange_button_port=DEFAULT_FLANGE_BUTTON_PORT,
            flange_button_ports=DEFAULT_FLANGE_BUTTON_PORTS,
        )
    return SESSION.robot_session


def _record_pick_cycle(success: bool, start_time: Optional[float]) -> None:
    """Track pick/place performance metrics."""
    SESSION.pick_attempts += 1
    if success:
        SESSION.pick_successes += 1
    else:
        SESSION.pick_failures += 1

    if start_time is None:
        return

    duration = max(time.monotonic() - start_time, 0.0)
    SESSION.last_cycle_s = duration
    SESSION.cycle_times.append(duration)


def _set_robot_guidance(
    message: str,
    *,
    step: Optional[int] = None,
    status: Optional[str] = None,
    verification_ready: Optional[bool] = None,
    modal_suppressed: Optional[bool] = None,
) -> None:
    # Use the new state module function
    state.update_calibration_guidance(
        message,
        step=step,
        status=status,
        verification_ready=verification_ready,
        modal_suppressed=modal_suppressed
    )
    
    # Clear accuracy if status changed
    if status is not None and status not in {
        "accuracy_ready",
        "awaiting_verification",
        "verifying",
        "complete",
    }:
        state.clear_calibration_accuracy()


def _clear_robot_accuracy_guidance() -> None:
    """Reset the cached accuracy summary so the UI hides the panel."""
    SESSION.robot_calibration_accuracy = None
    state.clear_calibration_accuracy()


def _record_robot_accuracy(mean_xy_error: float, max_xy_error: float) -> dict[str, float]:
    """Store accuracy metrics computed after robot calibration."""
    accuracy = {
        "mean_xy_error": float(mean_xy_error),
        "max_xy_error": float(max_xy_error),
    }
    SESSION.robot_calibration_accuracy = accuracy
    state.set_calibration_accuracy(accuracy)
    return accuracy


def _load_robot_calibration(path: Path = DEFAULT_ROBOT_CALIBRATION_PATH) -> None:
    if not path.exists():
        return
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        LOGGER.warning("Failed to parse robot calibration data: %s", exc)
        return
    points = data.get("points", [])
    if isinstance(points, list):
        SESSION.robot_calibration_points = points
    min_z = data.get("min_z")
    if isinstance(min_z, (int, float)):
        SESSION.robot_calibration_min_z = float(min_z)
    elif SESSION.robot_calibration_points:
        SESSION.robot_calibration_min_z = min(
            float(pt.get("z", 0.0)) for pt in SESSION.robot_calibration_points
        )
    plate_reference = data.get("plate_reference_homography")
    if isinstance(plate_reference, list):
        try:
            SESSION.plate_reference_homography = np.asarray(
                plate_reference, dtype=np.float64
            )
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning(
                "Failed to load plate reference homography: %s", exc)
    SESSION.plate_to_robot_transform = None
    SESSION.plate_origin_mm = None
    LOGGER.info("Loaded robot calibration data from %s", path)
    _commit_settings(robot_calibration_path=str(path))


def _save_robot_calibration(points: list[dict], path: Path = DEFAULT_ROBOT_CALIBRATION_PATH) -> None:
    min_z = None
    if points:
        min_z = min(float(pt.get("z", 0.0)) for pt in points)
    payload = {
        "points": points,
        "min_z": min_z,
        "timestamp": time.time(),
        "coordinate_id": ROBOT_COORDINATE_ID,
    }
    if SESSION.plate_reference_homography is not None:
        payload["plate_reference_homography"] = (
            np.asarray(SESSION.plate_reference_homography,
                       dtype=np.float64).tolist()
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))
    _commit_settings(robot_calibration_path=str(path))


def _ensure_posture_axes(posture: dict, fallback: RobotPosture) -> dict:
    normalized = dict(posture)
    for axis in ("a", "b", "c"):
        if axis not in normalized:
            normalized[axis] = float(getattr(fallback, axis, 0.0))
    return normalized


def _plate_test_capture_path() -> Path:
    SETTINGS.plate_test_capture_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    return SETTINGS.plate_test_capture_dir / f"plate_test_{stamp}.json"


def _save_plate_test_capture(payload: dict) -> Path:
    path = _plate_test_capture_path()
    payload["capture_path"] = str(path)
    path.write_text(json.dumps(payload, indent=2))
    SESSION.last_plate_test_capture = str(path)
    return path


def _posture_to_dict(posture: RobotPosture) -> dict:
    return {
        "x": float(posture.x),
        "y": float(posture.y),
        "z": float(posture.z),
        "a": float(posture.a),
        "b": float(posture.b),
        "c": float(posture.c),
        "timestamp": float(posture.timestamp),
        "age_s": float(posture.age_seconds()),
    }


def _analyze_plate_test_capture(capture: dict) -> str:
    moves = capture.get("moves", [])
    if not moves:
        return "No motion samples captured."
    distances = []
    lines = []
    for move in moves:
        cmd = move.get("command_coord5")
        act = move.get("posture_coord5")
        label = move.get("label", "move")
        if not cmd or not act:
            continue
        dx = act["x"] - cmd["x"]
        dy = act["y"] - cmd["y"]
        dz = act["z"] - cmd["z"]
        dist = float(np.linalg.norm([dx, dy, dz]))
        distances.append(dist)
        lines.append(
            f"{label}: d={dist:.2f} (dx={dx:.2f}, dy={dy:.2f}, dz={dz:.2f})"
        )
    if not distances:
        return "No usable move comparisons captured."
    mean_dist = float(np.mean(distances))
    max_dist = float(np.max(distances))
    summary = [
        f"Move error mean: {mean_dist:.2f} mm",
        f"Move error max: {max_dist:.2f} mm",
    ]
    capture_path = capture.get("capture_path")
    if capture_path:
        summary.append(f"Capture saved: {capture_path}")
    return "\n".join(summary + lines[:12])


def _save_drop_target(
    drop_target_mm: Optional[Tuple[float, float]],
    orientation: Optional[Tuple[float, float, float]] = None,
    robot_xyz: Optional[Tuple[float, float, float]] = None,
    coordinate_id: Optional[int] = None,
) -> None:
    DROP_TARGET_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not drop_target_mm:
        if DROP_TARGET_PATH.exists():
            DROP_TARGET_PATH.unlink()
        _commit_settings(drop_target=None)
        return
    payload = {"x": float(drop_target_mm[0]), "y": float(drop_target_mm[1])}
    if orientation is not None:
        payload.update(
            {
                "a": float(orientation[0]),
                "b": float(orientation[1]),
                "c": float(orientation[2]),
            }
        )
    if robot_xyz is not None:
        payload.update(
            {
                "robot_x": float(robot_xyz[0]),
                "robot_y": float(robot_xyz[1]),
                "robot_z": float(robot_xyz[2]),
            }
        )
    if coordinate_id is not None:
        payload["coord_id"] = int(coordinate_id)
    DROP_TARGET_PATH.write_text(json.dumps(payload, indent=2))
    _commit_settings(drop_target=drop_target_mm, workspace_path=str(
        DEFAULT_WORKSPACE_PATH), calibration_bundle_path=str(DEFAULT_BUNDLE_PATH))


def _load_drop_target() -> None:
    if DROP_TARGET_PATH.exists():
        try:
            data = json.loads(DROP_TARGET_PATH.read_text())
            x = float(data["x"])
            y = float(data["y"])
            SESSION.drop_target_mm = (x, y)
            if all(key in data for key in ("a", "b", "c")):
                SESSION.drop_target_orientation = (
                    float(data["a"]),
                    float(data["b"]),
                    float(data["c"]),
                )
            if all(key in data for key in ("robot_x", "robot_y", "robot_z")):
                SESSION.drop_target_robot = (
                    float(data["robot_x"]),
                    float(data["robot_y"]),
                    float(data["robot_z"]),
                )
            if "coord_id" in data:
                try:
                    SESSION.drop_target_coord_id = int(data["coord_id"])
                except (TypeError, ValueError):
                    SESSION.drop_target_coord_id = None
            return
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("Failed to load drop target: %s", exc)
    if SETTINGS.drop_target:
        SESSION.drop_target_mm = SETTINGS.drop_target


def _update_robot_calibration_meta(robot_points: list[dict]) -> None:
    SESSION.robot_calibration_points = list(robot_points)
    if robot_points:
        SESSION.robot_calibration_min_z = min(
            float(pt.get("z", 0.0)) for pt in robot_points
        )
    else:
        SESSION.robot_calibration_min_z = None
    SESSION.plate_to_robot_transform = None
    SESSION.plate_origin_mm = None
    _save_robot_calibration(robot_points)


def _current_coordinate_id(session: RobotSession) -> Optional[int]:
    status = session.robot_status_snapshot()
    if not isinstance(status, dict):
        return None
    coord_id = status.get("CoordinateId")
    if coord_id is None:
        return None
    try:
        return int(coord_id)
    except (TypeError, ValueError):
        return None


async def _ensure_robot_coordinate(
    robot_api: CodroidAPI,
    expected_id: int,
    *,
    timeout_s: float = 5.0,
    require: bool = True,
) -> None:
    """Wait for the robot to report the desired coordinate frame."""
    await robot_api.set_current_coordinate_id(expected_id)
    session = _get_robot_session()
    deadline = time.monotonic() + timeout_s
    coord_id = None
    resent = False
    while time.monotonic() < deadline:
        status = session.robot_status_snapshot()
        if isinstance(status, dict):
            coord_id = status.get("CoordinateId")
        if coord_id is not None:
            try:
                if int(coord_id) == expected_id:
                    return
            except (TypeError, ValueError):
                pass
        if not resent and time.monotonic() > deadline - (timeout_s / 2):
            await robot_api.set_current_coordinate_id(expected_id)
            resent = True
        await asyncio.sleep(0.1)
    if not require:
        LOGGER.warning(
            "CoordinateId did not switch to %s within %.1fs (last=%s).",
            expected_id,
            timeout_s,
            coord_id,
        )
        return
    raise RuntimeError(
        f"Failed to switch robot to CoordinateId {expected_id}. "
        f"Current CoordinateId={coord_id}."
    )


async def _ensure_coordinate5_active(
    robot_api: CodroidAPI,
    *,
    timeout_s: float = 5.0,
) -> None:
    """Activate CoordinateId 5 before issuing motion commands."""
    await _ensure_robot_coordinate(
        robot_api, ROBOT_COORDINATE_ID, timeout_s=timeout_s
    )


def _log_move_target(
    label: str,
    pose: dict,
    *,
    coordinate_id: Optional[int] = None,
) -> None:
    suffix = f" coord_id={coordinate_id}" if coordinate_id is not None else ""
    LOGGER.info(
        "Move %s%s -> x=%.3f, y=%.3f, z=%.3f, a=%.3f, b=%.3f, c=%.3f",
        label,
        suffix,
        float(pose.get("x", 0.0)),
        float(pose.get("y", 0.0)),
        float(pose.get("z", 0.0)),
        float(pose.get("a", 0.0)),
        float(pose.get("b", 0.0)),
        float(pose.get("c", 0.0)),
    )


def _set_plate_test_state(
    message: str,
    *,
    running: bool = True,
    label: Optional[str] = None,
) -> None:
    state = {"running": running, "message": message, "timestamp": time.time()}
    if label is not None:
        state["label"] = label
    SESSION.plate_test_state = state
    SESSION.last_action = message


def _robot_posture_ready(
    max_age_s: float = SETTINGS.robot_posture_max_age_s,
) -> Tuple[bool, str]:
    session = _get_robot_session()
    if not session.posture_seen():
        return False, "Robot posture stream not available yet."
    age = session.position_snapshot().age_seconds()
    if age > max_age_s:
        return False, f"Robot posture stale ({age:.1f}s old)."
    return True, "Robot posture stream healthy."


def _plate_center_px(detections: dict) -> Optional[np.ndarray]:
    if not detections:
        return None
    centers = [det.center for det in detections.values()]
    if not centers:
        return None
    return np.mean(np.asarray(centers, dtype=np.float64), axis=0)


def _plate_is_stable(
    max_age_s: float = SETTINGS.plate_visibility_max_age_s,
    require_baseline: bool = False,
    allow_lost: bool = False,
) -> Tuple[bool, str]:
    if require_baseline and SESSION.robot_calibration_baseline_px is None:
        return False, "Calibration plate baseline not set."
    
    # Get plate status from new state module
    plate_status = state.get_plate_status_snapshot()
    
    if not plate_status.get("ok"):
        if allow_lost and SESSION.robot_calibration_baseline_px is not None:
            return True, "Plate markers not visible; waiting for re-acquire."
        return False, "Calibration plate markers are not visible."
    movement_px = plate_status.get("movement_px")
    if movement_px is not None and movement_px > SETTINGS.plate_movement_threshold_px:
        return False, f"Calibration plate moved ({movement_px:.1f}px)."
    if plate_status.get("stable") is False:
        return False, plate_status.get("message", "Calibration plate not stable.")
    return True, plate_status.get("message", "Plate markers stable.")


# PLATE_STATUS now managed by state.plate_tracking_state
# ROBOT_CALIBRATION_GUIDANCE now managed by state.calibration_stepper_state


class OperationAbort(RuntimeError):
    """Raised when an operation should abort immediately."""


class OperationPaused(RuntimeError):
    """Raised when an operation should pause and wait for resume."""


# OPERATION_STATE and ROBOT_ERROR_STATE now managed by state module
# Import helper functions for backwards compatibility
from app.calibration_ui.state import (
    _operation_snapshot,
    _update_operation_state,
    _begin_operation,
    _finish_operation,
    _pause_operation,
    _update_operation_message,
    get_operation_run_id,
    get_operation_status,
    is_operation_running,
    resume_operation as _resume_operation,
    stop_operation as _request_stop_operation,
    finish_operation as _reset_operation_state,
    get_error_snapshot as _error_snapshot,
    set_error_state as _activate_robot_error,
    clear_error_state as _clear_robot_error_state,
    update_plate_status,
    update_calibration_guidance,
    clear_calibration_accuracy,
    set_calibration_accuracy
)


def _operation_status(run_id: int) -> str:
    """Get operation status for a specific run_id."""
    current_run_id = get_operation_run_id()
    if current_run_id != run_id:
        return "superseded"
    return get_operation_status()


def _ensure_operation_active(run_id: int) -> None:
    """Check if operation is active and raise appropriate exceptions."""
    status = _operation_status(run_id)
    if status == "paused":
        raise OperationPaused("Program paused.")
    if status in {"stopping", "stopped", "idle", "superseded"}:
        raise OperationAbort("Program stopped.")


async def _wait_for_operation_resume(run_id: int) -> None:
    """Wait for operation to resume or abort if stopped."""
    while True:
        status = _operation_status(run_id)
        if status == "paused":
            await asyncio.sleep(0.2)
            continue
        if status in {"stopping", "stopped", "idle", "superseded"}:
            raise OperationAbort("Program stopped.")
        return
    if is_new:
        _pause_operation(error_type, message)
    return snapshot


def _acknowledge_robot_error() -> dict:
    """Acknowledge the current robot error using the new state module."""
    error_snapshot = state.get_error_snapshot()
    if error_snapshot.get("active"):
        state.set_error_state(
            error_type=error_snapshot.get("error_type", ""),
            message=error_snapshot.get("message", ""),
            physical_pressed=False,
            acknowledge_existing=True
        )
    return state.get_error_snapshot()


def _coerce_truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "on", "yes", "enabled"}
    return False


def _status_flag(status: dict, keys: Tuple[str, ...]) -> bool:
    for key in keys:
        if key in status and _coerce_truthy(status.get(key)):
            return True
    return False


def _alarm_contains(status: dict, keywords: Tuple[str, ...]) -> bool:
    alarms = status.get("alarms") or status.get("alarm") or []
    if isinstance(alarms, dict):
        alarms = [alarms]
    if not isinstance(alarms, list):
        alarms = [alarms]
    for alarm in alarms:
        text = str(alarm).lower()
        if any(keyword in text for keyword in keywords):
            return True
    return False


def _detect_emergency_stop_status(status: dict) -> bool:
    emergency_keys = (
        "emergency_stop",
        "emergencyStop",
        "estop",
        "e_stop",
        "EmergencyStop",
        "EStop",
    )
    if _status_flag(status, emergency_keys):
        return True
    nested = status.get("robot_status")
    if isinstance(nested, dict) and _status_flag(nested, emergency_keys):
        return True
    if _alarm_contains(status, ("emergency", "e-stop", "estop")):
        return True
    payload_text = str(status).lower()
    return "emergency" in payload_text and "stop" in payload_text


def _detect_overspeed_status(status: dict) -> bool:
    overspeed_keys = (
        "overspeed",
        "over_speed",
        "overSpeed",
        "speed_alarm",
        "overspeed_alarm",
    )
    if _status_flag(status, overspeed_keys):
        return True
    nested = status.get("robot_status")
    if isinstance(nested, dict) and _status_flag(nested, overspeed_keys):
        return True
    return _alarm_contains(status, ("overspeed", "over speed"))


def _detect_joint_protection_status(status: dict) -> bool:
    joint_keys = (
        "joint_protection",
        "jointProtection",
        "joint_collision",
        "jointCollision",
        "collision",
    )
    if _status_flag(status, joint_keys):
        return True
    nested = status.get("robot_status")
    if isinstance(nested, dict) and _status_flag(nested, joint_keys):
        return True
    return _alarm_contains(status, ("joint", "collision"))


def _detect_wrong_tool_status(status: dict) -> bool:
    tool_keys = (
        "wrong_tool",
        "wrongTool",
        "tool_error",
        "payload_error",
        "payloadError",
    )
    if _status_flag(status, tool_keys):
        return True
    nested = status.get("robot_status")
    if isinstance(nested, dict) and _status_flag(nested, tool_keys):
        return True
    return _alarm_contains(status, ("tool", "payload"))


def _detect_rescue_mode_status(status: dict) -> bool:
    state = status.get("state")
    if isinstance(state, (int, float)) and int(state) == 3:
        return True
    return False


def _detect_drag_not_allowed_status(status: dict) -> bool:
    drag_keys = (
        "drag_not_allowed",
        "dragNotAllowed",
        "drag_not_allowed_warning",
    )
    if _status_flag(status, drag_keys):
        return True
    nested = status.get("robot_status")
    if isinstance(nested, dict) and _status_flag(nested, drag_keys):
        return True
    return _alarm_contains(status, ("drag not allowed",))


def _restore_persisted_state() -> None:
    if SESSION.calibration_bundle is None and DEFAULT_BUNDLE_PATH.exists():
        try:
            SESSION.calibration_bundle = calibration.load_calibration(
                DEFAULT_BUNDLE_PATH)
            LOGGER.info("Loaded calibration bundle from %s",
                        DEFAULT_BUNDLE_PATH)
        except Exception as exc:
            LOGGER.warning(
                "Failed to load calibration bundle from %s: %s",
                DEFAULT_BUNDLE_PATH,
                exc,
            )
    if SESSION.workspace_definition is None and DEFAULT_WORKSPACE_PATH.exists():
        try:
            SESSION.workspace_definition = workspace.load_workspace(
                DEFAULT_WORKSPACE_PATH)
            LOGGER.info("Loaded workspace definition from %s",
                        DEFAULT_WORKSPACE_PATH)
        except Exception as exc:
            LOGGER.warning(
                "Failed to load workspace definition from %s: %s",
                DEFAULT_WORKSPACE_PATH,
                exc,
            )
    _load_drop_target()
    if SETTINGS.last_model:
        try:
            activate_model(Path(SETTINGS.last_model))
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("Failed to restore last model: %s", exc)
    if DEFAULT_ROBOT_CALIBRATION_PATH.exists():
        _load_robot_calibration(DEFAULT_ROBOT_CALIBRATION_PATH)


_restore_persisted_state()

# Global calibration progress tracker
CALIBRATION_PROGRESS = {"value": 0, "current_frame": 0, "max_frames": 0}


class MotionJPEGStream:
    """Manager for Motion JPEG video streaming for fullscreen video playback."""

    def __init__(self, frame_rate: float = 30.0, min_push_interval_s: float = 1 / 120):
        self.frame_rate = frame_rate
        self.frame_queue: list[bytes] = []
        self.lock = threading.Lock()
        self.max_queue_size = 3
        self.frame_count = 0
        self.min_push_interval_s = min_push_interval_s
        self._last_push_ts = 0.0
        self._fps = 0.0
        self._last_fps_ts = 0.0

    def add_frame(self, frame: np.ndarray) -> None:
        """Add a frame to the stream queue, dropping frames if pushing too fast."""
        now = time.time()
        if now - self._last_push_ts < self.min_push_interval_s:
            return
        success, buffer = cv2.imencode(".jpg", frame)
        if not success:
            return

        with self.lock:
            self.frame_queue.append(buffer.tobytes())
            self.frame_count += 1
            self._last_push_ts = now
            if self._last_fps_ts > 0.0:
                dt = now - self._last_fps_ts
                if dt > 0:
                    inst_fps = 1.0 / dt
                    self._fps = 0.9 * self._fps + 0.1 * inst_fps
            self._last_fps_ts = now
            # Keep only recent frames
            if len(self.frame_queue) > self.max_queue_size:
                self.frame_queue.pop(0)

    def get_frame(self) -> Optional[bytes]:
        """Get the most recent frame from the queue."""
        with self.lock:
            if self.frame_queue:
                return self.frame_queue[-1]
        return None

    def get_fps(self) -> float:
        with self.lock:
            return self._fps


@dataclass
class OverlayResult:
    vis: np.ndarray
    status: str
    metric: str


@dataclass
class OverlayTask:
    key: str
    runner: Callable[[np.ndarray], OverlayResult]


class FrameProcessor:
    """Background processor that keeps only the newest request and result."""

    def __init__(self, max_queue: int = 2) -> None:
        self._queue: Deque[tuple[str, np.ndarray,
                                 Callable[[np.ndarray], OverlayResult]]]
        self._queue = deque(maxlen=max_queue)
        self._lock = threading.Lock()
        self._latest: dict[str, OverlayResult] = {}
        self._running = True
        self._thread = threading.Thread(
            target=self._worker, name="FrameProcessor", daemon=True
        )
        self._thread.start()

    def submit(self, task: OverlayTask, frame: np.ndarray) -> None:
        # Drop older pending tasks and keep the newest request only.
        with self._lock:
            self._queue.clear()
            self._queue.append((task.key, frame.copy(), task.runner))

    def get_latest(self, key: str) -> Optional[OverlayResult]:
        with self._lock:
            result = self._latest.get(key)
            if result is None:
                return None
            # Copy the image so callers cannot mutate internal state.
            return OverlayResult(result.vis.copy(), result.status, result.metric)

    def stop(self) -> None:
        self._running = False
        if self._thread.is_alive():
            self._thread.join(timeout=1.0)

    def _worker(self) -> None:
        while self._running:
            task: Optional[tuple[str, np.ndarray,
                                 Callable[[np.ndarray], OverlayResult]]] = None
            with self._lock:
                if self._queue:
                    # Pop newest to drop stale work.
                    task = self._queue.pop()
            if task is None:
                time.sleep(0.003)
                continue
            key, frame, runner = task
            try:
                result = runner(frame)
            except Exception as exc:  # noqa: BLE001
                logging.error("FrameProcessor error for %s: %s",
                              key, exc, exc_info=True)
                result = OverlayResult(frame, f"Processing error: {exc}", "--")
            with self._lock:
                self._latest[key] = result


def _format_fps(capture_fps: float, stream_fps: float) -> str:
    cap_text = f"{capture_fps:.1f}" if capture_fps > 0 else "--"
    stream_text = f"{stream_fps:.1f}" if stream_fps > 0 else "--"
    return f"FPS — capture: {cap_text}, stream: {stream_text}"


MJPEG_STREAM = MotionJPEGStream()
FRAME_PROCESSOR = FrameProcessor()
RAW_STREAM = MotionJPEGStream()


def _safe_int(value: Optional[float], fallback: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return fallback


def _safe_float(value: Optional[float], fallback: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return fallback


def _safe_path(value: Optional[str], fallback: Path) -> Path:
    if value is None:
        return fallback
    candidate = str(value).strip()
    return Path(candidate) if candidate else fallback


def _recent_gripper_sync(
    key: str,
    value: Optional[float],
    *,
    window_s: float = 0.75,
) -> bool:
    sync_data = SESSION.gripper_sync
    if not sync_data or value is None:
        return False
    try:
        updated = float(sync_data.get("updated", 0.0))
        if time.time() - updated > window_s:
            return False
        synced_value = sync_data.get(key)
        if synced_value is None:
            return False
        return abs(float(value) - float(synced_value)) < 0.01
    except (TypeError, ValueError):
        return False


def _record_gripper_sync(width: Optional[float], force: Optional[float]) -> None:
    payload = dict(SESSION.gripper_sync)
    payload["updated"] = time.time()
    if width is not None:
        payload["width"] = width
    if force is not None:
        payload["force"] = force
    SESSION.gripper_sync = payload


def get_gpu_info() -> dict:
    """Get GPU status information including availability, clock speed, usage, and memory.

    Supports CUDA (NVIDIA), Metal Performance Shaders (Apple Silicon), and CPU fallback.
    """
    info = {
        "available": False,
        "name": "N/A",
        "driver_version": "N/A",
        "cuda_available": False,
        "cuda_version": "N/A",
        "mps_available": False,
        "gpu_count": 0,
        "devices": [],
    }

    # Check CUDA availability via PyTorch
    if TORCH_AVAILABLE:
        info["cuda_available"] = torch.cuda.is_available()
        if info["cuda_available"]:
            info["cuda_version"] = torch.version.cuda or "N/A"
            info["gpu_count"] = torch.cuda.device_count()
            info["available"] = True

        # Check for Metal Performance Shaders (Apple Silicon)
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            info["mps_available"] = True
            info["available"] = True
            info["name"] = "Metal Performance Shaders"
            info["gpu_count"] = 1

    # Get detailed GPU info via NVML (NVIDIA only)
    if not NVML_AVAILABLE or not info["cuda_available"]:
        return info

    try:
        pynvml.nvmlInit()
        device_count = pynvml.nvmlDeviceGetCount()
        info["available"] = device_count > 0

        for i in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            name = pynvml.nvmlDeviceGetName(handle)
            if isinstance(name, bytes):
                name = name.decode("utf-8")

            # Get memory info
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            mem_used_mb = mem_info.used / (1024 ** 2)
            mem_total_mb = mem_info.total / (1024 ** 2)
            mem_percent = (mem_info.used / mem_info.total) * \
                100 if mem_info.total > 0 else 0

            # Get utilization
            try:
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                gpu_util = util.gpu
            except Exception:
                gpu_util = 0

            # Get clock speeds
            try:
                clock_graphics = pynvml.nvmlDeviceGetClockInfo(
                    handle, pynvml.NVML_CLOCK_GRAPHICS)
                clock_mem = pynvml.nvmlDeviceGetClockInfo(
                    handle, pynvml.NVML_CLOCK_MEM)
            except Exception:
                clock_graphics = 0
                clock_mem = 0

            # Get temperature
            try:
                temp = pynvml.nvmlDeviceGetTemperature(
                    handle, pynvml.NVML_TEMPERATURE_GPU)
            except Exception:
                temp = 0

            # Get power usage
            try:
                power = pynvml.nvmlDeviceGetPowerUsage(
                    handle) / 1000.0  # mW to W
            except Exception:
                power = 0

            device_info = {
                "index": i,
                "name": name,
                "memory_used_mb": round(mem_used_mb, 1),
                "memory_total_mb": round(mem_total_mb, 1),
                "memory_percent": round(mem_percent, 1),
                "gpu_utilization": gpu_util,
                "clock_graphics_mhz": clock_graphics,
                "clock_memory_mhz": clock_mem,
                "temperature_c": temp,
                "power_watts": round(power, 1),
            }
            info["devices"].append(device_info)

            if i == 0:
                info["name"] = name
                try:
                    driver_version = pynvml.nvmlSystemGetDriverVersion()
                    if isinstance(driver_version, bytes):
                        driver_version = driver_version.decode("utf-8")
                    info["driver_version"] = driver_version
                except Exception:
                    pass

        pynvml.nvmlShutdown()
    except Exception as exc:
        # Only log at debug level for expected failures on non-NVIDIA platforms
        LOGGER.debug(
            "NVML GPU detection failed (expected on non-NVIDIA systems): %s", exc)

    return info


def format_gpu_status() -> list:
    """Format GPU status for display in the UI."""
    gpu_info = get_gpu_info()

    if not gpu_info["available"]:
        return [
            dmc.Alert(
                "No GPU detected (NVIDIA CUDA or Apple MPS not available)",
                color="yellow",
                title="GPU Status",
            )
        ]

    elements = []

    # Overall status - handle both CUDA and MPS
    if gpu_info.get("mps_available"):
        # Apple Silicon MPS
        elements.append(
            dmc.Group(
                [
                    dmc.Badge("MPS: ✓ Available", color="green"),
                    dmc.Badge(f"{gpu_info['name']}", color="blue"),
                    dmc.Badge(
                        f"{gpu_info['gpu_count']} GPU(s)", color="brand"),
                ],
                gap="xs",
            )
        )
    else:
        # NVIDIA CUDA
        cuda_status = "✓ Available" if gpu_info["cuda_available"] else "✗ Not available"
        elements.append(
            dmc.Group(
                [
                    dmc.Badge(
                        f"CUDA: {cuda_status}", color="green" if gpu_info["cuda_available"] else "red"),
                    dmc.Badge(
                        f"Driver: {gpu_info['driver_version']}", color="blue"),
                    dmc.Badge(
                        f"{gpu_info['gpu_count']} GPU(s)", color="brand"),
                ],
                gap="xs",
            )
        )

    # Per-device info
    for dev in gpu_info["devices"]:
        elements.append(
            dmc.Card(
                [
                    dmc.Text(
                        f"GPU {dev['index']}: {dev['name']}", fw=600, size="sm"),
                    dmc.Group(
                        [
                            dmc.Stack(
                                [
                                    dmc.Text("Memory", size="xs", c="dimmed"),
                                    dmc.Text(
                                        f"{dev['memory_used_mb']:.0f} / {dev['memory_total_mb']:.0f} MB ({dev['memory_percent']:.1f}%)",
                                        size="sm",
                                    ),
                                ],
                                gap=2,
                            ),
                            dmc.Stack(
                                [
                                    dmc.Text(
                                        "GPU Usage", size="xs", c="dimmed"),
                                    dmc.Text(
                                        f"{dev['gpu_utilization']}%", size="sm"),
                                ],
                                gap=2,
                            ),
                            dmc.Stack(
                                [
                                    dmc.Text("Clock", size="xs", c="dimmed"),
                                    dmc.Text(
                                        f"{dev['clock_graphics_mhz']} MHz", size="sm"),
                                ],
                                gap=2,
                            ),
                            dmc.Stack(
                                [
                                    dmc.Text("Temp", size="xs", c="dimmed"),
                                    dmc.Text(
                                        f"{dev['temperature_c']}°C", size="sm"),
                                ],
                                gap=2,
                            ),
                            dmc.Stack(
                                [
                                    dmc.Text("Power", size="xs", c="dimmed"),
                                    dmc.Text(
                                        f"{dev['power_watts']:.1f} W", size="sm"),
                                ],
                                gap=2,
                            ),
                        ],
                        gap="md",
                    ),
                ],
                withBorder=True,
                padding="sm",
                radius="md",
            )
        )

    return elements


def _frame_index(path: Path) -> int:
    match = re.search(r"calibration_color_(\d+)\.png", path.name)
    if match:
        return int(match.group(1))
    return 0


def _list_capture_frames(output_dir: Path) -> list[Path]:
    frames = list(output_dir.glob("calibration_color_*.png"))
    frames.sort(key=_frame_index)
    return frames


def _update_capture_state(output_dir: Path) -> list[Path]:
    SESSION.capture.output_dir = output_dir
    frames = _list_capture_frames(output_dir)
    SESSION.capture.frames = frames
    return frames


def _sanitize_filename(name: str, fallback: str = "model.bin") -> str:
    base = Path(name).name
    safe = re.sub(r"[^A-Za-z0-9._-]", "_", base)
    return safe or fallback


def _list_model_files(output_dir: Path = DEFAULT_MODEL_DIR) -> list[Path]:
    if not output_dir.exists():
        return []
    files = [path for path in output_dir.iterdir() if path.is_file()]
    return sorted(files, key=lambda path: path.name.lower())


def _model_options() -> list[dict]:
    options = []
    for path in _list_model_files():
        options.append({"value": str(path), "label": path.name})
    return options


def save_uploaded_model(contents: str, filename: str, output_dir: Path) -> Path:
    if "," not in contents:
        raise ValueError("Uploaded model payload is not base64 encoded.")
    _, encoded = contents.split(",", 1)
    data = base64.b64decode(encoded)
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_name = _sanitize_filename(filename or "model.bin")
    path = output_dir / safe_name
    path.write_bytes(data)
    return path


def _load_ultralytics_model(model_path: Path) -> object:
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError(
            "ultralytics is not installed. Run `uv add ultralytics` to enable YOLO inference.") from exc
    return YOLO(str(model_path))


def activate_model(model_path: Optional[Path]) -> str:
    if model_path is None:
        SESSION.active_model_path = None
        SESSION.yolo_model = None
        SESSION.model_loaded = False
        _commit_settings(last_model=None, active_model=None)
        return "No model selected."
    if not model_path.exists():
        SESSION.active_model_path = None
        SESSION.yolo_model = None
        SESSION.model_loaded = False
        raise FileNotFoundError(f"Model not found: {model_path}")
    SESSION.active_model_path = model_path
    try:
        SESSION.yolo_model = _load_ultralytics_model(model_path)
        SESSION.model_loaded = True
        SESSION.last_detection_error = None
        _commit_settings(last_model=str(model_path),
                         active_model=str(model_path))
        LOGGER.info(
            f"Model loaded and detection auto-started: {model_path.name}")
        return f"Model loaded: {model_path.name}. Detection auto-started."
    except Exception as exc:
        SESSION.yolo_model = None
        SESSION.model_loaded = False
        SESSION.last_detection_error = str(exc)
        return f"Model ready but not loaded: {exc}"


def _workspace_pixel_to_mm(
    workspace_def: workspace.WorkspaceDefinition,
    pixel_xy: Tuple[float, float],
) -> Tuple[float, float]:
    scale = float(workspace_def.x_axis_camera[0])
    if scale == 0:
        raise RuntimeError(
            "Workspace scale is zero; recompute workspace markers.")
    origin_px_x = workspace_def.origin_camera[0] / scale
    origin_px_y = workspace_def.origin_camera[1] / scale
    x_mm = (pixel_xy[0] - origin_px_x) * scale
    y_mm = (pixel_xy[1] - origin_px_y) * scale
    return x_mm, y_mm


def _pixel_to_workspace_mm(
    workspace_def: workspace.WorkspaceDefinition,
    pixel_xy: np.ndarray,
) -> Tuple[float, float]:
    bundle = SESSION.calibration_bundle
    if bundle is None or bundle.pixel_to_mm <= 0:
        return _workspace_pixel_to_mm(
            workspace_def, (float(pixel_xy[0]), float(pixel_xy[1]))
        )
    scale = float(bundle.pixel_to_mm)
    camera_xyz = np.array(
        [float(pixel_xy[0]) * scale, float(pixel_xy[1]) * scale, 0.0],
        dtype=np.float64,
    )
    origin = workspace_def.origin_camera
    basis = np.column_stack(
        [workspace_def.x_axis_camera, workspace_def.y_axis_camera]
    )
    coeffs, *_ = np.linalg.lstsq(basis, camera_xyz - origin, rcond=None)
    return float(coeffs[0]), float(coeffs[1])


def _resolve_plate_test_targets(
    layout_path: Path,
    workspace_def: workspace.WorkspaceDefinition,
) -> Tuple[list[dict], dict]:
    calibration_result = compute_plate_calibration_from_latest(layout_path)
    plate_points = calibration_result.plate.robot_calibration_points()
    if len(plate_points) < 3:
        raise RuntimeError("Calibration plate layout is missing targets 1-3.")
    plate_xy = np.array(
        [[pt.center_mm[0], pt.center_mm[1]] for pt in plate_points],
        dtype=np.float64,
    ).reshape(-1, 1, 2)
    homography = calibration_result.homography
    if homography is None:
        raise RuntimeError("Plate homography unavailable.")
    inv_h = np.linalg.inv(homography)
    pixel_xy = cv2.perspectiveTransform(plate_xy, inv_h).reshape(-1, 2)
    targets = []
    for point, px in zip(plate_points, pixel_xy):
        workspace_xy = _pixel_to_workspace_mm(workspace_def, px)
        sequence = point.sequence if point.sequence is not None else len(
            targets) + 1
        targets.append(
            {
                "sequence": int(sequence),
                "workspace_mm": workspace_xy,
                "plate_mm": (float(point.center_mm[0]), float(point.center_mm[1])),
                "pixel": (float(px[0]), float(px[1])),
            }
        )
    state.update_plate_status(
        ok=len(calibration_result.detections) >= SETTINGS.min_plate_markers,
        stable=len(calibration_result.detections) >= SETTINGS.min_plate_markers,
        message=f"Plate markers detected: {len(calibration_result.detections)}",
        count=len(calibration_result.detections),
        center_px=None,
        movement_px=None,
    )
    return targets, {
        "count": len(calibration_result.detections),
        "reprojection_error_mm": calibration_result.reprojection_error_mm,
    }


def _resolve_plate_test_coord5_targets(
    layout_path: Path,
) -> Tuple[list[dict], dict, workspace.RigidTransform, bool]:
    if not SESSION.robot_calibration_points or len(SESSION.robot_calibration_points) < 3:
        raise RuntimeError(
            "Robot calibration points are missing; re-run robot calibration.")

    calibration_result = compute_plate_calibration_from_latest(layout_path)
    plate_layout = calibration_result.plate
    plate_points = plate_layout.robot_calibration_points()
    if len(plate_points) < 3:
        raise RuntimeError("Calibration plate layout is missing targets 1-3.")

    reference_initialized = False
    if SESSION.plate_reference_homography is None:
        bundle = SESSION.calibration_bundle
        if bundle is not None and bundle.plate_homography is not None:
            SESSION.plate_reference_homography = bundle.plate_homography.copy()
        else:
            SESSION.plate_reference_homography = calibration_result.homography.copy()
            reference_initialized = True

    origin_mm = np.asarray(plate_points[0].center_mm, dtype=np.float64)
    marker_base: list[np.ndarray] = []
    marker_curr: list[np.ndarray] = []
    marker_info: dict[int, dict] = {}
    reference_h = SESSION.plate_reference_homography
    for marker_id, detection in calibration_result.detections.items():
        marker = plate_layout.markers.get(marker_id)
        if marker is None:
            continue
        base_pt = marker.center_mm - origin_mm
        curr_pt = plate.image_points_to_plate_mm(
            reference_h, np.asarray([detection.center], dtype=np.float64)
        )[0]
        marker_base.append(np.asarray(base_pt, dtype=np.float64))
        marker_curr.append(np.asarray(curr_pt - origin_mm, dtype=np.float64))
        marker_info[int(marker_id)] = {
            "plate_mm": (
                float(marker.center_mm[0]),
                float(marker.center_mm[1]),
            ),
            "base_xy": (float(base_pt[0]), float(base_pt[1])),
        }

    if len(marker_base) < 2:
        raise RuntimeError(
            "Not enough plate markers detected for offset tracking.")

    r2, t2 = _rigid_transform_2d(
        np.vstack(marker_base),
        np.vstack(marker_curr),
    )

    offset_rotation_rad = float(np.arctan2(r2[1, 0], r2[0, 0]))
    offset_rotation_deg = float(np.degrees(offset_rotation_rad))
    # Debug: log plate offset tracking transformation
    LOGGER.info("Plate offset tracking: detected %d markers", len(marker_base))
    LOGGER.info("Plate offset r2 (rotation):\n%s", r2)
    LOGGER.info("Plate offset t2 (translation): %s", t2)
    LOGGER.info("Plate offset rotation: %.2f degrees", offset_rotation_deg)

    robot_pts = np.asarray(
        [
            [
                float(pt["x"]),
                float(pt["y"]),
                float(pt["z"]),
            ]
            for pt in SESSION.robot_calibration_points[:3]
        ],
        dtype=np.float64,
    )
    baseline_transform = _rigid_transform(
        np.asarray(
            [
                [
                    float(pt.center_mm[0] - origin_mm[0]),
                    float(pt.center_mm[1] - origin_mm[1]),
                    0.0,
                ]
                for pt in plate_points[:3]
            ],
            dtype=np.float64,
        ),
        robot_pts,
    )

    # Debug logging for coordinate transformation
    LOGGER.info("Baseline transform rotation:\n%s",
                baseline_transform.rotation)
    LOGGER.info("Baseline transform translation: %s",
                baseline_transform.translation)
    LOGGER.info("Plate points (relative to P1): %s",
                [(pt.center_mm[0] - origin_mm[0], pt.center_mm[1] - origin_mm[1])
                 for pt in plate_points[:3]])
    LOGGER.info("Robot calibration points: %s",
                [(pt["x"], pt["y"], pt["z"]) for pt in SESSION.robot_calibration_points[:3]])

    targets = []
    for point in plate_points:
        base_xy = np.array(
            [
                float(point.center_mm[0] - origin_mm[0]),
                float(point.center_mm[1] - origin_mm[1]),
            ],
            dtype=np.float64,
        )
        adj_xy = (r2 @ base_xy) + t2
        sequence = point.sequence if point.sequence is not None else len(
            targets) + 1
        targets.append(
            {
                "sequence": int(sequence),
                "reference_xy": (float(base_xy[0]), float(base_xy[1])),
                "adjusted_xy": (float(adj_xy[0]), float(adj_xy[1])),
                "plate_mm": (float(point.center_mm[0]), float(point.center_mm[1])),
            }
        )

    for marker_id, meta in marker_info.items():
        base_xy = np.asarray(meta["base_xy"], dtype=np.float64)
        adj_xy = (r2 @ base_xy) + t2
        meta["reference_xy"] = (float(base_xy[0]), float(base_xy[1]))
        meta["adjusted_xy"] = (float(adj_xy[0]), float(adj_xy[1]))

    PLATE_STATUS.update(
        {
            "count": len(calibration_result.detections),
            "ok": len(calibration_result.detections) >= SETTINGS.min_plate_markers,
            "stable": len(calibration_result.detections) >= SETTINGS.min_plate_markers,
            "center_px": None,
            "movement_px": None,
            "timestamp": time.time(),
            "message": f"Plate markers detected: {len(calibration_result.detections)}",
        }
    )

    return targets, {
        "count": len(calibration_result.detections),
        "reprojection_error_mm": calibration_result.reprojection_error_mm,
        "reference_initialized": reference_initialized,
        "markers_coord5": marker_info,
        "offset_rotation_deg": offset_rotation_deg,
    }, baseline_transform, reference_initialized


async def _await_grip_status(timeout_s: float) -> bool:
    gripper = SESSION.gripper
    if gripper is None:
        return False
    t_index = GRIPPER_DEFAULTS["t_index"]
    deadline = time.monotonic() + max(timeout_s, 0.1)
    while time.monotonic() < deadline:
        try:
            if gripper.isGripped(t_index=t_index):
                return True
        except Exception as exc:  # noqa: BLE001
            LOGGER.debug("Failed to poll gripper grip state: %s", exc)
            return False
        await asyncio.sleep(0.2)
    return False


def _pose_from_workspace_mm(
    workspace_def: workspace.WorkspaceDefinition,
    x_mm: float,
    y_mm: float,
    z_offset_mm: float,
) -> dict:
    robot_xyz = workspace_def.workspace_to_robot(x_mm, y_mm, z_offset_mm)
    robot_xyz = _apply_robot_z_floor(robot_xyz)
    return {
        "x": float(robot_xyz[0]),
        "y": float(robot_xyz[1]),
        "z": float(robot_xyz[2]),
        "a": 0.0,
        "b": 0.0,
        "c": 0.0,
    }


def _coord5_transform() -> Optional[workspace.RigidTransform]:
    session = _get_robot_session()
    frame = session.coordinate_frame_snapshot()
    if isinstance(frame, dict) and frame:
        try:
            transform = _coord_frame_to_transform(frame)
            SESSION.coord5_transform = transform
            return transform
        except Exception as exc:  # noqa: BLE001
            LOGGER.debug("Failed to parse RobotCoordinate frame: %s", exc)
    if SESSION.coord5_transform is not None:
        return SESSION.coord5_transform
    if not SESSION.robot_calibration_points or len(SESSION.robot_calibration_points) < 3:
        return None
    try:
        plate_layout = plate.load_plate_layout(DEFAULT_PLATE_PATH)
        plate_points = plate_layout.robot_calibration_points()
        if len(plate_points) < 3:
            raise RuntimeError("Calibration plate layout missing points.")
        origin_mm = np.asarray(plate_points[0].center_mm, dtype=np.float64)
        plate_pts = np.asarray(
            [
                [
                    float(pt.center_mm[0] - origin_mm[0]),
                    float(pt.center_mm[1] - origin_mm[1]),
                    0.0,
                ]
                for pt in plate_points[:3]
            ],
            dtype=np.float64,
        )
        robot_pts = np.asarray(
            [
                [
                    float(pt["x"]),
                    float(pt["y"]),
                    float(pt["z"]),
                ]
                for pt in SESSION.robot_calibration_points[:3]
            ],
            dtype=np.float64,
        )
        transform = _rigid_transform(plate_pts, robot_pts)
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("Failed to build coord5 transform: %s", exc)
        return None
    SESSION.coord5_transform = transform
    return transform


def _coord_frame_to_transform(frame: dict) -> workspace.RigidTransform:
    x = _safe_float(frame.get("x"), 0.0)
    y = _safe_float(frame.get("y"), 0.0)
    z = _safe_float(frame.get("z"), 0.0)
    raw_a = frame.get("a", frame.get("rx"))
    raw_b = frame.get("b", frame.get("ry"))
    raw_c = frame.get("c", frame.get("rz"))
    a = _safe_float(raw_a, 0.0)
    b = _safe_float(raw_b, 0.0)
    c = _safe_float(raw_c, 0.0)
    max_abs = max(abs(a), abs(b), abs(c))
    use_degrees = max_abs > (2.0 * np.pi)
    rotation = _rotation_from_abc(a, b, c, degrees=use_degrees)
    translation = np.array([x, y, z], dtype=np.float64)
    return workspace.RigidTransform(rotation=rotation, translation=translation)


def _rotation_from_abc(
    a: float,
    b: float,
    c: float,
    *,
    degrees: bool = True,
) -> np.ndarray:
    if degrees:
        a = np.deg2rad(a)
        b = np.deg2rad(b)
        c = np.deg2rad(c)
    ca, sa = float(np.cos(a)), float(np.sin(a))
    cb, sb = float(np.cos(b)), float(np.sin(b))
    cc, sc = float(np.cos(c)), float(np.sin(c))
    rot_x = np.array(
        [[1.0, 0.0, 0.0], [0.0, ca, -sa], [0.0, sa, ca]],
        dtype=np.float64,
    )
    rot_y = np.array(
        [[cb, 0.0, sb], [0.0, 1.0, 0.0], [-sb, 0.0, cb]],
        dtype=np.float64,
    )
    rot_z = np.array(
        [[cc, -sc, 0.0], [sc, cc, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    return rot_z @ rot_y @ rot_x


def _invert_rigid_transform(
    transform: workspace.RigidTransform,
) -> workspace.RigidTransform:
    inv_rot = transform.rotation.T
    inv_trans = -inv_rot @ transform.translation
    return workspace.RigidTransform(rotation=inv_rot, translation=inv_trans)


def _base_to_coord5(base_xyz: np.ndarray) -> Optional[np.ndarray]:
    transform = _coord5_transform()
    if transform is None:
        return None
    inv = _invert_rigid_transform(transform)
    return inv.rotation @ base_xyz + inv.translation


def _coord5_to_base(coord5_xyz: np.ndarray) -> Optional[np.ndarray]:
    transform = _coord5_transform()
    if transform is None:
        return None
    return transform.rotation @ coord5_xyz + transform.translation


def _workspace_to_coord5_mm(
    workspace_def: workspace.WorkspaceDefinition,
    workspace_xy: Tuple[float, float],
    z_offset_mm: float = 0.0,
) -> Optional[Tuple[float, float]]:
    robot_xyz = workspace_def.workspace_to_robot(
        workspace_xy[0], workspace_xy[1], z_offset_mm
    )
    coord5_xyz = _base_to_coord5(robot_xyz)
    if coord5_xyz is None:
        return None
    return float(coord5_xyz[0]), float(coord5_xyz[1])


def _coord5_to_workspace_mm(
    workspace_def: workspace.WorkspaceDefinition,
    coord5_xy: Tuple[float, float],
) -> Optional[Tuple[float, float]]:
    coord5_xyz = np.array(
        [coord5_xy[0], coord5_xy[1], 0.0], dtype=np.float64
    )
    base_xyz = _coord5_to_base(coord5_xyz)
    if base_xyz is None:
        return None
    return _robot_to_workspace_mm(workspace_def, base_xyz)


def _pose_from_workspace_coord5_mm(
    workspace_def: workspace.WorkspaceDefinition,
    x_mm: float,
    y_mm: float,
    z_offset_mm: float,
) -> dict:
    robot_xyz = workspace_def.workspace_to_robot(x_mm, y_mm, z_offset_mm)
    coord5_xyz = _base_to_coord5(robot_xyz)
    if coord5_xyz is None:
        raise RuntimeError("Coord5 transform unavailable.")
    z_val = _clamp_z_for_coord5(float(coord5_xyz[2]), use_base_coords=False)
    return {
        "x": float(coord5_xyz[0]),
        "y": float(coord5_xyz[1]),
        "z": float(z_val),
        "a": 0.0,
        "b": 0.0,
        "c": 0.0,
    }


def _pose_from_coord5_mm(
    x_mm: float,
    y_mm: float,
    z_offset_mm: float,
) -> dict:
    z_val = _clamp_z_for_coord5(float(z_offset_mm), use_base_coords=False)
    return {
        "x": float(x_mm),
        "y": float(y_mm),
        "z": float(z_val),
        "a": 0.0,
        "b": 0.0,
        "c": 0.0,
    }


def _plate_reference_homography() -> Optional[np.ndarray]:
    if SESSION.plate_reference_homography is not None:
        return SESSION.plate_reference_homography
    bundle = SESSION.calibration_bundle
    if bundle is not None and bundle.plate_homography is not None:
        return bundle.plate_homography
    return None


def _pixel_to_plate_mm(
    pixel: Tuple[float, float],
) -> Optional[np.ndarray]:
    homography = _plate_reference_homography()
    if homography is None:
        return None
    pts = np.array([[pixel]], dtype=np.float64)
    try:
        mapped = cv2.perspectiveTransform(pts, homography)
    except Exception:
        return None
    return mapped[0, 0]


def _plate_to_robot_transform(
) -> Optional[Tuple[workspace.RigidTransform, np.ndarray]]:
    if (
        SESSION.plate_to_robot_transform is not None
        and SESSION.plate_origin_mm is not None
    ):
        return SESSION.plate_to_robot_transform, SESSION.plate_origin_mm
    if not SESSION.robot_calibration_points or len(SESSION.robot_calibration_points) < 3:
        return None
    try:
        plate_layout = plate.load_plate_layout(DEFAULT_PLATE_PATH)
        plate_points = plate_layout.robot_calibration_points()
        if len(plate_points) < 3:
            return None
        origin_mm = np.asarray(plate_points[0].center_mm, dtype=np.float64)
        plate_pts = np.asarray(
            [
                [
                    float(pt.center_mm[0] - origin_mm[0]),
                    float(pt.center_mm[1] - origin_mm[1]),
                    0.0,
                ]
                for pt in plate_points[:3]
            ],
            dtype=np.float64,
        )
        robot_pts = np.asarray(
            [
                [
                    float(pt["x"]),
                    float(pt["y"]),
                    float(pt["z"]),
                ]
                for pt in SESSION.robot_calibration_points[:3]
            ],
            dtype=np.float64,
        )
        transform = _rigid_transform(plate_pts, robot_pts)
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("Failed to build plate-to-robot transform: %s", exc)
        return None
    SESSION.plate_to_robot_transform = transform
    SESSION.plate_origin_mm = origin_mm
    return transform, origin_mm


def _pixel_to_robot_base_xyz(
    pixel: Tuple[float, float],
) -> Optional[np.ndarray]:
    bundle = SESSION.calibration_bundle
    workspace_def = SESSION.workspace_definition
    if bundle is None or workspace_def is None:
        return None
    plane = bundle.workspace_plane
    camera = bundle.camera
    pts = np.array([[pixel]], dtype=np.float64)
    try:
        undist = cv2.undistortPoints(
            pts, camera.camera_matrix, camera.distortion_coeffs
        )
    except Exception:
        return None
    ray = np.array([undist[0, 0, 0], undist[0, 0, 1], 1.0], dtype=np.float64)
    denom = float(np.dot(plane.normal, ray))
    if abs(denom) < 1e-6:
        return None
    t = -float(plane.offset) / denom
    camera_point = ray * t
    # DepthAI plane fit can be in meters or millimeters depending on stream config.
    unit_scale = 1000.0 if abs(float(plane.offset)) < 10.0 else 1.0
    camera_point_mm = camera_point * unit_scale
    transform = workspace_def.camera_to_robot
    return transform.rotation @ camera_point_mm + transform.translation


def _robot_to_workspace_mm(
    workspace_def: workspace.WorkspaceDefinition,
    robot_xyz: np.ndarray,
) -> Tuple[float, float]:
    """Map a robot XYZ point back into workspace (mm) assuming z on the plane."""
    cam_to_robot = workspace_def.camera_to_robot.as_matrix()
    robot_h = np.hstack([robot_xyz, 1.0])
    camera_h = np.linalg.inv(cam_to_robot) @ robot_h
    camera_xyz = camera_h[:3]
    origin = workspace_def.origin_camera
    basis = np.column_stack(
        [workspace_def.x_axis_camera, workspace_def.y_axis_camera])
    try:
        coeffs, *_ = np.linalg.lstsq(basis, camera_xyz - origin, rcond=None)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            f"Unable to project robot pose into workspace: {exc}") from exc
    return float(coeffs[0]), float(coeffs[1])


def _apply_robot_z_floor(robot_xyz: np.ndarray) -> np.ndarray:
    min_z = SESSION.robot_calibration_min_z
    if min_z is None or robot_xyz[2] >= min_z:
        return robot_xyz
    adjusted = np.array(robot_xyz, dtype=np.float64, copy=True)
    adjusted[2] = min_z
    return adjusted


def _run_detection(
    frame: np.ndarray,
    confidence: float,
    iou: float,
    max_detections: int,
    class_filter: Optional[str],
) -> list[dict]:
    if SESSION.yolo_model is None:
        raise RuntimeError(
            "No YOLO model loaded. Upload a model or install ultralytics.")
    classes = None
    if class_filter:
        classes = [int(val.strip())
                   for val in class_filter.split(",") if val.strip().isdigit()]

    # Crop to workspace region if workspace is defined
    crop_offset_x = 0
    crop_offset_y = 0
    detection_frame = frame

    if SESSION.workspace_definition is not None:
        try:
            corners = _workspace_corners_px(SESSION.workspace_definition)
            # Get bounding box of workspace region
            x_min = int(np.min(corners[:, 0]))
            y_min = int(np.min(corners[:, 1]))
            x_max = int(np.max(corners[:, 0]))
            y_max = int(np.max(corners[:, 1]))

            # Clamp to frame bounds
            x_min = max(0, x_min)
            y_min = max(0, y_min)
            x_max = min(frame.shape[1], x_max)
            y_max = min(frame.shape[0], y_max)

            # Crop frame to workspace region
            detection_frame = frame[y_min:y_max, x_min:x_max]
            crop_offset_x = x_min
            crop_offset_y = y_min
        except Exception as exc:
            logging.debug(
                "Could not crop to workspace, using full frame: %s", exc)

    results = SESSION.yolo_model.predict(
        detection_frame,
        conf=confidence,
        iou=iou,
        max_det=max_detections,
        classes=classes,
        verbose=False,
    )
    detections: list[dict] = []
    if not results:
        return detections
    for result in results:
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            continue
        xyxy = boxes.xyxy.cpu().numpy() if hasattr(boxes.xyxy, "cpu") else boxes.xyxy
        confs = boxes.conf.cpu().numpy() if hasattr(boxes.conf, "cpu") else boxes.conf
        clss = boxes.cls.cpu().numpy() if hasattr(boxes.cls, "cpu") else boxes.cls
        for box, conf, cls_idx in zip(xyxy, confs, clss):
            # Adjust coordinates back to full frame
            x1, y1, x2, y2 = [float(val) for val in box]
            x1 += crop_offset_x
            y1 += crop_offset_y
            x2 += crop_offset_x
            y2 += crop_offset_y
            center = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
            label = str(int(cls_idx))
            if hasattr(SESSION.yolo_model, "names"):
                label = SESSION.yolo_model.names.get(int(cls_idx), label)
            detection = {
                "bbox": (x1, y1, x2, y2),
                "center": center,
                "label": label,
                "confidence": float(conf),
            }
            if SESSION.workspace_definition is not None:
                try:
                    detection["workspace_mm"] = _workspace_pixel_to_mm(
                        SESSION.workspace_definition, center)
                except Exception:
                    detection["workspace_mm"] = None
            detections.append(detection)
    return detections


def _format_detection_list(detections: list[dict]) -> list:
    if not detections:
        return [dmc.Text("No detections yet.", size="sm", c="dimmed")]
    rows = []
    for idx, det in enumerate(detections, start=1):
        label = det.get("label", "object")
        conf = det.get("confidence", 0.0)
        coord_text = "c5: --"
        coord5_mm = _detection_coord5_mm(det, SESSION.workspace_definition)
        if coord5_mm is not None:
            coord_text = f"c5 {coord5_mm[0]:.1f}, {coord5_mm[1]:.1f}"
        rows.append(
            dmc.Text(f"{idx}. {label} ({conf:.2f}) @ {coord_text}", size="sm"))
    return rows


def _refresh_follow_detections() -> list[dict]:
    frame, _depth = get_latest_frame()
    if frame is None:
        return SESSION.detections or []
    if SESSION.yolo_model is None:
        return SESSION.detections or []
    try:
        detections = _run_detection(
            frame,
            SESSION.detection_confidence,
            SESSION.detection_iou,
            SESSION.detection_max,
            SESSION.detection_classes,
        )
        SESSION.detections = detections
        SESSION.last_detection_error = None
        return detections
    except Exception as exc:  # noqa: BLE001
        SESSION.last_detection_error = str(exc)
        LOGGER.warning("Follow detection update failed: %s", exc)
        return SESSION.detections or []


def _set_auto_state(message: str, *, running: bool) -> None:
    SESSION.auto_message = message
    SESSION.auto_running = running


def _auto_state_snapshot() -> dict:
    return {"running": SESSION.auto_running, "message": SESSION.auto_message}


def _workspace_contains(
    workspace_def: workspace.WorkspaceDefinition,
    workspace_xy: Tuple[float, float],
) -> bool:
    return (
        0.0 <= workspace_xy[0] <= workspace_def.width_mm
        and 0.0 <= workspace_xy[1] <= workspace_def.height_mm
    )


def _auto_refresh_detections(frame: np.ndarray) -> list[dict]:
    detections = _run_detection(
        frame,
        SESSION.detection_confidence,
        SESSION.detection_iou,
        SESSION.detection_max,
        SESSION.detection_classes,
    )
    SESSION.detections = detections
    SESSION.last_detection_error = None
    SESSION.auto_detection_updated_at = time.monotonic()
    return detections


def _auto_filter_targets(
    detections: list[dict],
    workspace_def: workspace.WorkspaceDefinition,
    *,
    target_label: str,
    drop_target_mm: Optional[Tuple[float, float]],
    drop_ignore_radius_mm: float,
) -> list[dict]:
    label_norm = target_label.strip().lower()
    # Log what labels are available
    available_labels = set(str(det.get("label", "")).strip().lower()
                           for det in detections if det.get("label"))
    LOGGER.info(
        f"_auto_filter_targets: Filtering {len(detections)} detections, target_label='{target_label}', available_labels={available_labels}")
    targets = []
    for i, det in enumerate(detections):
        if label_norm:
            det_label = str(det.get("label", "")).strip().lower()
            if det_label != label_norm:
                LOGGER.info(
                    f"_auto_filter_targets: Det {i} skipped - label mismatch ('{det_label}' != '{label_norm}')")
                continue
        workspace_mm = _detection_workspace_mm(det, workspace_def)
        if workspace_mm is None:
            LOGGER.info(
                f"_auto_filter_targets: Det {i} skipped - no workspace coordinates")
            continue
        # Skip workspace boundary check in auto mode - user can verify visually
        # if not _workspace_contains(workspace_def, workspace_mm):
        #     LOGGER.info(
        #         f"_auto_filter_targets: Det {i} skipped - outside workspace (coords={workspace_mm}, bounds=0-{workspace_def.width_mm}x0-{workspace_def.height_mm})")
        #     continue
        if (
            drop_target_mm is not None
            and drop_ignore_radius_mm > 0.0
            and float(
                np.linalg.norm(
                    np.array(workspace_mm) - np.array(drop_target_mm)
                )
            )
            <= drop_ignore_radius_mm
        ):
            dist = float(np.linalg.norm(
                np.array(workspace_mm) - np.array(drop_target_mm)))
            LOGGER.info(
                f"_auto_filter_targets: Det {i} skipped - too close to drop target (dist={dist:.1f}mm, threshold={drop_ignore_radius_mm}mm)")
            continue
        LOGGER.info(
            f"_auto_filter_targets: Det {i} accepted - workspace={workspace_mm}")
        targets.append(det)
    LOGGER.info(
        f"_auto_filter_targets: Result: {len(targets)} targets from {len(detections)} detections")
    return targets


def _auto_select_nearest_target(
    detections: list[dict],
    workspace_def: workspace.WorkspaceDefinition,
) -> Optional[dict]:
    if not detections:
        return None
    session = _get_robot_session()
    if session.posture_seen():
        pos = session.position_snapshot()
        coord_id = _current_coordinate_id(session)
        robot_xy = None
        if coord_id == ROBOT_COORDINATE_ID:
            robot_xy = (float(pos.x), float(pos.y))
        if robot_xy is None:
            try:
                robot_xy = _robot_to_workspace_mm(
                    workspace_def,
                    np.array([pos.x, pos.y, pos.z], dtype=float),
                )
            except Exception:
                robot_xy = None
        if robot_xy is not None:
            best_det = None
            best_dist = float("inf")
            for det in detections:
                if coord_id == ROBOT_COORDINATE_ID:
                    det_xy = _detection_coord5_mm(det, workspace_def)
                else:
                    det_xy = _detection_workspace_mm(det, workspace_def)
                if det_xy is None:
                    continue
                dist = float(
                    np.linalg.norm(
                        np.array(det_xy) - np.array(robot_xy)
                    )
                )
                if dist < best_dist:
                    best_dist = dist
                    best_det = det
            if best_det is not None:
                return best_det
    return detections[0]


def _auto_intrusion_present(
    detections: list[dict],
    workspace_def: workspace.WorkspaceDefinition,
    *,
    target_xy: Optional[Tuple[float, float]],
    drop_target_mm: Optional[Tuple[float, float]],
    drop_ignore_radius_mm: float,
    target_match_mm: float,
) -> bool:
    for det in detections:
        workspace_xy = _detection_workspace_mm(det, workspace_def)
        if workspace_xy is None:
            continue
        if not _workspace_contains(workspace_def, workspace_xy):
            continue
        if (
            drop_target_mm is not None
            and drop_ignore_radius_mm > 0.0
            and float(
                np.linalg.norm(
                    np.array(workspace_xy) - np.array(drop_target_mm)
                )
            )
            <= drop_ignore_radius_mm
        ):
            continue
        if target_xy is not None:
            if (
                float(
                    np.linalg.norm(
                        np.array(workspace_xy) - np.array(target_xy)
                    )
                )
                <= target_match_mm
            ):
                continue
        return True
    return False


def _detection_workspace_mm(
    detection: dict,
    workspace_def: workspace.WorkspaceDefinition,
) -> Optional[Tuple[float, float]]:
    workspace_mm = detection.get("workspace_mm")
    if workspace_mm is not None:
        return (float(workspace_mm[0]), float(workspace_mm[1]))
    center = detection.get("center")
    if center is None:
        return None
    try:
        workspace_mm = _workspace_pixel_to_mm(workspace_def, center)
    except Exception:  # noqa: BLE001
        return None
    return (float(workspace_mm[0]), float(workspace_mm[1]))


def _detection_coord5_mm(
    detection: dict,
    workspace_def: Optional[workspace.WorkspaceDefinition],
) -> Optional[Tuple[float, float]]:
    coord5_mm = detection.get("coord5_mm")
    if coord5_mm is not None:
        return (float(coord5_mm[0]), float(coord5_mm[1]))
    center = detection.get("center")
    if center is None:
        return None
    plate_xy = _pixel_to_plate_mm((float(center[0]), float(center[1])))
    plate_transform = _plate_to_robot_transform()
    if plate_xy is not None and plate_transform is not None:
        transform, origin_mm = plate_transform
        plate_rel = np.array(
            [
                float(plate_xy[0] - origin_mm[0]),
                float(plate_xy[1] - origin_mm[1]),
                0.0,
            ],
            dtype=np.float64,
        )
        base_xyz = transform.rotation @ plate_rel + transform.translation
        coord5_xyz = _base_to_coord5(base_xyz)
        if coord5_xyz is not None:
            coord5_xy = (float(coord5_xyz[0]), float(coord5_xyz[1]))
            detection["coord5_mm"] = coord5_xy
            detection["robot_base_mm"] = (
                float(base_xyz[0]),
                float(base_xyz[1]),
                float(base_xyz[2]),
            )
            return coord5_xy
    base_xyz = _pixel_to_robot_base_xyz((float(center[0]), float(center[1])))
    if base_xyz is not None:
        coord5_xyz = _base_to_coord5(base_xyz)
        if coord5_xyz is not None:
            coord5_xy = (float(coord5_xyz[0]), float(coord5_xyz[1]))
            detection["coord5_mm"] = coord5_xy
            detection["robot_base_mm"] = (
                float(base_xyz[0]),
                float(base_xyz[1]),
                float(base_xyz[2]),
            )
            return coord5_xy
    if workspace_def is None:
        return None
    workspace_mm = _detection_workspace_mm(detection, workspace_def)
    if workspace_mm is None:
        return None
    coord5_xy = _workspace_to_coord5_mm(workspace_def, workspace_mm)
    if coord5_xy is None:
        return None
    detection["coord5_mm"] = coord5_xy
    return coord5_xy


def _select_follow_detection(
    detections: list[dict],
    last_workspace_mm: Optional[Tuple[float, float]],
    fallback: Optional[dict],
) -> Optional[dict]:
    if not detections:
        return fallback
    if last_workspace_mm is not None:
        best_detection = None
        best_distance = float("inf")
        for detection in detections:
            workspace_mm = detection.get("workspace_mm")
            if workspace_mm is None:
                continue
            dist = float(
                np.linalg.norm(
                    np.array(workspace_mm[:2]) - np.array(last_workspace_mm)
                )
            )
            if dist < best_distance:
                best_distance = dist
                best_detection = detection
        if best_detection is not None:
            return best_detection
    return fallback or detections[0]


def _build_follow_pose_fetcher(
    workspace_def: workspace.WorkspaceDefinition,
    initial_detection: dict,
    z_offset_mm: float,
) -> Callable[[], Optional[dict]]:
    last_detection = initial_detection
    last_workspace_mm = _detection_workspace_mm(
        initial_detection, workspace_def)

    def _fetch() -> Optional[dict]:
        nonlocal last_detection, last_workspace_mm
        detections = _refresh_follow_detections()
        detection = _select_follow_detection(
            detections, last_workspace_mm, last_detection
        )
        if detection is None:
            return None
        last_detection = detection
        workspace_mm = _detection_workspace_mm(detection, workspace_def)
        if workspace_mm is None:
            workspace_mm = last_workspace_mm
        if workspace_mm is None:
            return None
        last_workspace_mm = workspace_mm
        return _pose_from_workspace_mm(
            workspace_def, workspace_mm[0], workspace_mm[1], z_offset_mm
        )

    return _fetch


async def _move_linear_follow_target(
    robot_api: CodroidAPI,
    target_fetch: Callable[[], Optional[dict]],
    *,
    label: str,
    coordinate_id: int = ROBOT_COORDINATE_ID,
    timeout_s: Optional[float] = None,
    update_interval: Optional[float] = None,
    segment_hold_s: Optional[float] = None,
    settle_s: Optional[float] = None,
    tolerance_mm: Optional[float] = None,
    require_posture: bool = True,
    run_id: Optional[int] = None,
    safety_check: Optional[Callable[[], Awaitable[None]]] = None,
) -> Optional[dict]:
    session = _get_robot_session()
    timeout_s = _safe_float(timeout_s, SETTINGS.pick_follow_timeout_s)
    update_interval = _safe_float(
        update_interval, SETTINGS.pick_follow_update_interval_s
    )
    segment_hold_s = _safe_float(
        segment_hold_s, SETTINGS.pick_follow_segment_hold_s
    )
    settle_s = _safe_float(settle_s, SETTINGS.pick_follow_settle_s)
    tolerance_mm = _safe_float(
        tolerance_mm, SETTINGS.pick_follow_tolerance_mm
    )

    last_target: Optional[dict] = None

    def _check_operation() -> None:
        if run_id is not None:
            _ensure_operation_active(run_id)

    if require_posture and not session.posture_seen():
        _check_operation()
        LOGGER.info(
            f"_move_linear_follow_target '{label}': No posture seen yet, fetching initial target")
        # Call directly, don't thread - target_fetch is synchronous and fast
        initial_target = target_fetch()
        LOGGER.info(
            f"_move_linear_follow_target '{label}': Got initial target: {initial_target}")
        if initial_target is None:
            raise RuntimeError("No target available for follow move.")
        await robot_api.move_to_cartesian_target_linear(
            robot_api.build_target_cpos(**initial_target)
        )
        return initial_target
    LOGGER.info(
        f"_move_linear_follow_target '{label}': Starting follow loop")
    await robot_api.set_target_pos_type(robot_api.config.target_pos_types.cpos)
    stable_since: Optional[float] = None
    deadline = time.monotonic() + timeout_s
    loop_iter = 0
    while True:
        loop_iter += 1
        _check_operation()
        now = time.monotonic()
        if now >= deadline:
            LOGGER.warning(
                "Follow move '%s' timed out after %.1f s.", label, timeout_s
            )
            break
        # Call directly, don't thread - target_fetch is synchronous and fast
        target = target_fetch()
        if target is not None:
            last_target = target
        if last_target is None:
            raise RuntimeError("No target available for follow move.")
        await robot_api.set_target_cpos(
            robot_api.attach_coordinate_to_cpos(
                robot_api.build_target_cpos(**last_target),
                coordinate_id=coordinate_id,
            )
        )
        await robot_api.set_robot_command(
            robot_api.config.commands.move_target_linear
        )
        segment_deadline = time.monotonic() + max(segment_hold_s, 0.0)
        while time.monotonic() < segment_deadline:
            _check_operation()
            if safety_check is not None:
                await safety_check()
            await robot_api.send_command_heartbeat()
            await asyncio.sleep(min(update_interval, 0.05))
        if session.posture_seen():
            posture = session.position_snapshot()
            curr_xyz = np.array(
                [posture.x, posture.y, posture.z], dtype=np.float64
            )
            target_xyz = np.array(
                [
                    float(last_target["x"]),
                    float(last_target["y"]),
                    float(last_target["z"]),
                ],
                dtype=np.float64,
            )
            if np.linalg.norm(curr_xyz - target_xyz) <= tolerance_mm:
                if stable_since is None:
                    stable_since = now
                elif now - stable_since >= settle_s:
                    break
            else:
                stable_since = None
        await asyncio.sleep(update_interval)
    await robot_api.clear_target_position()
    await robot_api.stop_command()
    return last_target


async def _auto_wait_for_clear(
    run_id: int,
    *,
    workspace_def: workspace.WorkspaceDefinition,
    target_xy: Optional[Tuple[float, float]],
    drop_target_mm: Optional[Tuple[float, float]],
    drop_ignore_radius_mm: float,
) -> None:
    while True:
        status = _operation_status(run_id)
        if status in {"stopping", "stopped", "idle", "superseded"}:
            raise OperationAbort("Program stopped.")
        snapshot = _operation_snapshot()
        if snapshot.get("pause_reason") != "workspace_intrusion":
            await _wait_for_operation_resume(run_id)
            return
        frame, _depth = get_latest_frame()
        detections = SESSION.detections or []
        if frame is not None and (
            time.monotonic() - SESSION.auto_detection_updated_at
            > SETTINGS.auto_detection_stale_s
        ):
            try:
                detections = await asyncio.to_thread(
                    _auto_refresh_detections, frame
                )
            except Exception as exc:  # noqa: BLE001
                SESSION.last_detection_error = str(exc)
                LOGGER.warning("Auto detection refresh failed: %s", exc)
        if not _auto_intrusion_present(
            detections,
            workspace_def,
            target_xy=target_xy,
            drop_target_mm=drop_target_mm,
            drop_ignore_radius_mm=drop_ignore_radius_mm,
            target_match_mm=SETTINGS.auto_intrusion_match_mm,
        ):
            _resume_operation("Workspace clear. Resuming.")
            return
        await asyncio.sleep(SETTINGS.auto_idle_sleep_s)


async def _auto_pick_place_loop(
    robot_api: CodroidAPI,
    *,
    run_id: int,
) -> None:
    LOGGER.info(f"_auto_pick_place_loop: Starting with run_id={run_id}")
    workspace_def = SESSION.workspace_definition
    if workspace_def is None:
        raise RuntimeError("Workspace is not defined.")
    LOGGER.info("_auto_pick_place_loop: Workspace defined")
    if SESSION.drop_target_mm is None:
        raise RuntimeError("Drop position is not defined.")
    LOGGER.info(
        f"_auto_pick_place_loop: Drop target defined at {SESSION.drop_target_mm}")
    await robot_api.power_on()
    LOGGER.info("_auto_pick_place_loop: Robot powered on")
    await _ensure_coordinate5_active(robot_api)
    LOGGER.info("_auto_pick_place_loop: Coordinate 5 active")
    SESSION.auto_waiting_at_drop = False

    def _drop_poses(
        approach_mm: float,
        drop_mm: float,
    ) -> Tuple[dict, dict]:
        drop_target_mm = SESSION.drop_target_mm
        if drop_target_mm is None:
            raise RuntimeError("Drop position is not defined.")
        if (
            SESSION.drop_target_robot is not None
            and SESSION.drop_target_coord_id == ROBOT_COORDINATE_ID
        ):
            base = SESSION.drop_target_robot
            drop_pose = _pose_from_coord5_mm(
                float(base[0]), float(base[1]), float(base[2]) + drop_mm
            )
            drop_approach = _pose_from_coord5_mm(
                float(base[0]),
                float(base[1]),
                float(base[2]) + approach_mm,
            )
        else:
            drop_pose = _pose_from_workspace_coord5_mm(
                workspace_def, drop_target_mm[0], drop_target_mm[1], drop_mm
            )
            drop_approach = _pose_from_workspace_coord5_mm(
                workspace_def, drop_target_mm[0], drop_target_mm[1], approach_mm
            )
        if SESSION.drop_target_orientation is not None:
            orient = SESSION.drop_target_orientation
            drop_pose.update(
                {"a": float(orient[0]), "b": float(
                    orient[1]), "c": float(orient[2])}
            )
            drop_approach.update(
                {"a": float(orient[0]), "b": float(
                    orient[1]), "c": float(orient[2])}
            )
        return drop_pose, drop_approach

    async def _auto_move_to_drop(
        *,
        approach_mm: float,
        drop_mm: float,
    ) -> None:
        drop_pose, drop_approach = _drop_poses(approach_mm, drop_mm)
        await _move_linear_follow_target(
            robot_api,
            lambda: drop_approach,
            label="auto_drop_approach",
            run_id=run_id,
        )
        await _move_linear_follow_target(
            robot_api,
            lambda: drop_pose,
            label="auto_drop_pose",
            run_id=run_id,
        )

    async def _auto_drop_only(
        *,
        approach_mm: float,
        drop_mm: float,
    ) -> None:
        await _auto_move_to_drop(approach_mm=approach_mm, drop_mm=drop_mm)
        _command_gripper(GRIPPER_OPEN_WIDTH_DEFAULT, "opened")
        SESSION.robot_has_object = False
        SESSION.auto_waiting_at_drop = True

    while True:
        LOGGER.info("_auto_pick_place_loop: Starting new iteration")
        try:
            _ensure_operation_active(run_id)
        except OperationPaused:
            LOGGER.info(
                "_auto_pick_place_loop: Operation paused, waiting for resume")
            await _wait_for_operation_resume(run_id)
            continue
        config = SESSION.auto_config or {}
        target_label = str(
            config.get("target_label", SETTINGS.auto_target_label)
        )
        drop_ignore_radius_mm = float(
            config.get(
                "drop_ignore_radius_mm", SETTINGS.auto_drop_ignore_radius_mm
            )
        )
        approach_mm = float(
            config.get("approach_mm", SETTINGS.auto_approach_mm)
        )
        pick_mm = float(config.get("pick_mm", SETTINGS.auto_pick_mm))
        drop_mm = float(config.get("drop_mm", SETTINGS.auto_drop_mm))
        estimate_width = bool(
            config.get("estimate_width", SETTINGS.auto_estimate_width)
        )

        frame, _depth = get_latest_frame()
        detections = SESSION.detections or []
        LOGGER.info(
            f"_auto_pick_place_loop: Got {len(detections)} detections from SESSION")
        stale_age = time.monotonic() - SESSION.auto_detection_updated_at
        LOGGER.info(
            f"_auto_pick_place_loop: Detection age={stale_age:.2f}s, using SESSION detections directly")

        LOGGER.info(
            f"_auto_pick_place_loop: About to filter targets with label='{target_label}'")
        targets = _auto_filter_targets(
            detections,
            workspace_def,
            target_label=target_label,
            drop_target_mm=SESSION.drop_target_mm,
            drop_ignore_radius_mm=drop_ignore_radius_mm,
        )
        LOGGER.info(
            f"_auto_pick_place_loop: Filtered to {len(targets)} targets (label={target_label})")
        if not targets:
            LOGGER.info(
                f"_auto_pick_place_loop: No targets found, auto_waiting_at_drop={SESSION.auto_waiting_at_drop}")
            if not SESSION.auto_waiting_at_drop:
                LOGGER.info(
                    "_auto_pick_place_loop: Moving to drop position (no object in gripper)")
                await _auto_drop_only(
                    approach_mm=approach_mm,
                    drop_mm=drop_mm,
                )
            LOGGER.info(
                f"_auto_pick_place_loop: Sleeping {SETTINGS.auto_idle_sleep_s}s before next check")
            await asyncio.sleep(SETTINGS.auto_idle_sleep_s)
            continue

        target = _auto_select_nearest_target(targets, workspace_def)
        if target is None:
            LOGGER.warning(
                "_auto_pick_place_loop: Failed to select nearest target")
            await asyncio.sleep(SETTINGS.auto_idle_sleep_s)
            continue
        LOGGER.info(
            f"_auto_pick_place_loop: Selected target with label={target.get('label')}")
        target_xy = _detection_workspace_mm(target, workspace_def)
        if target_xy is None:
            LOGGER.warning(
                "_auto_pick_place_loop: Target missing workspace coordinates")
            await asyncio.sleep(SETTINGS.auto_idle_sleep_s)
            continue
        target_coord5 = _detection_coord5_mm(target, workspace_def)
        if target_coord5 is None:
            LOGGER.warning(
                "_auto_pick_place_loop: Target missing coord5 coordinates")
            await asyncio.sleep(SETTINGS.auto_idle_sleep_s)
            continue
        LOGGER.info(
            f"_auto_pick_place_loop: Target at workspace {target_xy[:2]}, coord5 {target_coord5[:2]}")

        tracker_detection = target
        tracker_workspace = target_xy
        tracker_coord5 = target_coord5
        SESSION.auto_waiting_at_drop = False

        LOGGER.info("_auto_pick_place_loop: Starting pick & place operation")
        gripper_open_val = GRIPPER_OPEN_WIDTH_DEFAULT
        gripper_close_val = GRIPPER_CLOSE_WIDTH_DEFAULT
        if estimate_width:
            width_guess = _estimate_detection_width_mm(
                target, workspace_def
            )
            if width_guess is not None:
                gripper_close_val = max(
                    gripper_close_val, width_guess * 0.9
                )
                gripper_open_val = max(
                    gripper_open_val, width_guess + 5.0
                )

        async def _safety_check() -> None:
            frame, _depth = get_latest_frame()
            detections = SESSION.detections or []
            # Don't try to refresh detections during movement - use cached detections
            # The background detection loop keeps SESSION.detections updated
            if _auto_intrusion_present(
                detections,
                workspace_def,
                target_xy=target_xy,
                drop_target_mm=SESSION.drop_target_mm,
                drop_ignore_radius_mm=drop_ignore_radius_mm,
                target_match_mm=SETTINGS.auto_intrusion_match_mm,
            ):
                _pause_operation(
                    "workspace_intrusion",
                    "Workspace intrusion detected. Paused.",
                )
                await _auto_wait_for_clear(
                    run_id,
                    workspace_def=workspace_def,
                    target_xy=target_xy,
                    drop_target_mm=SESSION.drop_target_mm,
                    drop_ignore_radius_mm=drop_ignore_radius_mm,
                )

        def _fetch_pose(z_offset_mm: float) -> Optional[dict]:
            if tracker_coord5 is None:
                return None
            return _pose_from_coord5_mm(
                tracker_coord5[0],
                tracker_coord5[1],
                z_offset_mm,
            )

        LOGGER.info("_auto_pick_place_loop: About to move to pick approach")
        await _move_linear_follow_target(
            robot_api,
            lambda: _fetch_pose(approach_mm),
            label="auto_pick_approach",
            run_id=run_id,
            safety_check=_safety_check,
        )
        LOGGER.info(
            "_auto_pick_place_loop: Pick approach completed, moving to pick pose")
        await _move_linear_follow_target(
            robot_api,
            lambda: _fetch_pose(pick_mm),
            label="auto_pick_pose",
            run_id=run_id,
            safety_check=_safety_check,
        )
        LOGGER.info(
            "_auto_pick_place_loop: Pick pose completed, closing gripper")
        _command_gripper(gripper_close_val, "closed")
        SESSION.robot_has_object = True
        retreat_pose = _fetch_pose(approach_mm)
        if retreat_pose is not None:
            LOGGER.info("_auto_pick_place_loop: About to retreat from pick")
            await _move_linear_follow_target(
                robot_api,
                lambda: retreat_pose,
                label="auto_pick_retreat",
                run_id=run_id,
                safety_check=_safety_check,
            )
            LOGGER.info("_auto_pick_place_loop: Pick retreat completed")
        SESSION.last_pick_mm = tracker_workspace

        LOGGER.info("_auto_pick_place_loop: About to move to drop")
        drop_pose, drop_approach = _drop_poses(approach_mm, drop_mm)
        await _move_linear_follow_target(
            robot_api,
            lambda: drop_approach,
            label="auto_drop_approach",
            run_id=run_id,
            safety_check=_safety_check,
        )
        await _move_linear_follow_target(
            robot_api,
            lambda: drop_pose,
            label="auto_drop_pose",
            run_id=run_id,
            safety_check=_safety_check,
        )
        _command_gripper(gripper_open_val, "opened")
        SESSION.robot_has_object = False
        await _move_linear_follow_target(
            robot_api,
            lambda: drop_approach,
            label="auto_drop_retreat",
            run_id=run_id,
            safety_check=_safety_check,
        )
        SESSION.auto_waiting_at_drop = True
        LOGGER.info("_auto_pick_place_loop: Pick & place cycle completed, looping to find next target")


async def _auto_operation_task(run_id: int, robot_uri: Optional[str]) -> None:
    try:
        await _with_robot_api(
            robot_uri or DEFAULT_ROBOT_URI,
            lambda api: _auto_pick_place_loop(api, run_id=run_id),
        )
        message = "Auto pick & place completed."
        _set_auto_state(message, running=False)
        _finish_operation(run_id, status="idle", message=message)
    except OperationAbort as exc:
        message = f"Auto pick & place stopped: {exc}"
        _set_auto_state(message, running=False)
        _finish_operation(run_id, status="stopped", message=message)
    except Exception as exc:  # noqa: BLE001
        LOGGER.exception("Auto pick & place failed")
        message = f"Auto pick & place failed: {exc}"
        _set_auto_state(message, running=False)
        _finish_operation(run_id, status="idle", message=message)
    finally:
        SESSION.auto_task = None


def _detect_drop_marker(
    frame: np.ndarray,
    marker_id: int,
) -> Tuple[markers.MarkerDetection, Tuple[float, float]]:
    detections = markers.detect_all_markers(frame)
    if marker_id not in detections:
        raise RuntimeError(f"Drop marker {marker_id} not detected.")
    marker = detections[marker_id]
    workspace_mm = None
    if SESSION.workspace_definition is not None:
        workspace_mm = _workspace_pixel_to_mm(
            SESSION.workspace_definition, marker.center)
    if workspace_mm is None:
        raise RuntimeError(
            "Workspace is not defined; run workspace detection first.")
    return marker, workspace_mm


def _placeholder_frame(message: str, size: Tuple[int, int] = (720, 1280)) -> np.ndarray:
    height, width = size
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    cv2.putText(
        frame,
        message,
        (40, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 255),
        2,
    )
    return frame


def _encode_frame(frame: np.ndarray) -> str:
    success, buffer = cv2.imencode(".jpg", frame)
    if not success:
        raise RuntimeError("Failed to encode camera frame.")
    encoded = base64.b64encode(buffer).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def _default_transform() -> workspace.RigidTransform:
    return workspace.RigidTransform(rotation=np.eye(3), translation=np.zeros(3))


def _rigid_transform(from_pts: np.ndarray, to_pts: np.ndarray) -> workspace.RigidTransform:
    """Compute best-fit rigid transform from from_pts -> to_pts."""
    if from_pts.shape != to_pts.shape:
        raise ValueError("Point clouds must have the same shape.")
    if from_pts.shape[0] < 3:
        raise ValueError(
            "At least 3 points are required for a rigid transform.")
    a = np.asarray(from_pts, dtype=np.float64)
    b = np.asarray(to_pts, dtype=np.float64)
    centroid_a = np.mean(a, axis=0)
    centroid_b = np.mean(b, axis=0)
    aa = a - centroid_a
    bb = b - centroid_b
    h = aa.T @ bb
    u, _s, vt = np.linalg.svd(h)
    r = vt.T @ u.T
    if np.linalg.det(r) < 0:
        vt[-1, :] *= -1
        r = vt.T @ u.T
    t = centroid_b - r @ centroid_a
    return workspace.RigidTransform(rotation=r, translation=t)


def _rigid_transform_2d(
    from_pts: np.ndarray,
    to_pts: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    if from_pts.shape != to_pts.shape:
        raise ValueError("Point clouds must have the same shape.")
    if from_pts.shape[0] < 2:
        raise ValueError(
            "At least 2 points are required for a 2D rigid transform.")
    a = np.asarray(from_pts, dtype=np.float64)
    b = np.asarray(to_pts, dtype=np.float64)
    centroid_a = np.mean(a, axis=0)
    centroid_b = np.mean(b, axis=0)
    aa = a - centroid_a
    bb = b - centroid_b
    h = aa.T @ bb
    u, _s, vt = np.linalg.svd(h)
    r = vt.T @ u.T
    if np.linalg.det(r) < 0:
        vt[-1, :] *= -1
        r = vt.T @ u.T
    t = centroid_b - r @ centroid_a
    return r, t


def _ensure_plate_visible(max_age_s: float = SETTINGS.plate_visibility_max_age_s) -> None:
    ok, message = _plate_is_stable(
        max_age_s=max_age_s,
        require_baseline=False,
        allow_lost=False,
    )
    if not ok:
        raise RuntimeError(message)


def _ensure_plate_stable(
    max_age_s: float = SETTINGS.plate_visibility_max_age_s,
    *,
    allow_lost: bool = False,
) -> None:
    ok, message = _plate_is_stable(
        max_age_s=max_age_s,
        require_baseline=True,
        allow_lost=allow_lost,
    )
    if not ok:
        raise RuntimeError(message)


def _publish_raw_frame(frame: np.ndarray) -> None:
    RAW_STREAM.add_frame(frame)


def start_camera_stream(width: Optional[float], height: Optional[float], fps: Optional[float], lock_af: bool = False) -> str:
    stop_camera_stream()
    resolution = (_safe_int(width, DEFAULT_RESOLUTION[0]), _safe_int(
        height, DEFAULT_RESOLUTION[1]))
    fps_value = _safe_float(fps, DEFAULT_FPS)
    stream = DepthAIFrameStream(
        StreamConfig(resolution=resolution, fps=fps_value, lock_af=lock_af),
        frame_hook=_publish_raw_frame,
    )
    stream.start()
    SESSION.camera_stream = stream
    device_id = stream.device_id() or "unknown"
    return f"Camera stream running ({resolution[0]}x{resolution[1]} @ {fps_value:.1f} FPS, MXID {device_id})."


def stop_camera_stream() -> None:
    if SESSION.camera_stream is None:
        return
    SESSION.camera_stream.stop()
    SESSION.camera_stream = None


def get_latest_frame() -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    if SESSION.camera_stream is None or not SESSION.camera_stream.is_running():
        return None, None
    return SESSION.camera_stream.get_latest()


def capture_color_frame(output_dir: Path) -> Path:
    # Retry logic: frame might not be available immediately, so poll briefly
    max_retries = 5
    for attempt in range(max_retries):
        frame, _depth = get_latest_frame()
        if frame is not None and frame.size > 0:
            break
        if attempt < max_retries - 1:
            time.sleep(0.1)

    if frame is None or frame.size == 0:
        raise RuntimeError(
            f"Camera stream not providing frames after {max_retries} attempts.")

    output_dir.mkdir(parents=True, exist_ok=True)
    frames = _update_capture_state(output_dir)
    index = _frame_index(frames[-1]) + 1 if frames else 1
    path = output_dir / f"calibration_color_{index:02}.png"

    success = cv2.imwrite(str(path), frame)
    if not success:
        raise RuntimeError(f"Failed to write frame to {path}.")

    # Verify by reading back immediately
    verify = cv2.imread(str(path))
    if verify is None:
        raise RuntimeError(f"Failed to read back saved frame from {path}.")

    SESSION.capture.frames.append(path)
    return path


def capture_depth_snapshot(output_dir: Path) -> Path:
    _frame, depth = get_latest_frame()
    if depth is None:
        raise RuntimeError("No depth frame available from the camera stream.")
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "calibration_depth_frame.npz"
    np.savez_compressed(path, depth=depth.astype(np.float32))
    SESSION.capture.depth_frame = depth
    return path


def load_depth_snapshot(output_dir: Path) -> Optional[np.ndarray]:
    path = output_dir / "calibration_depth_frame.npz"
    if not path.exists():
        return None
    data = np.load(path, allow_pickle=True)
    depth = data.get("depth")
    if depth is None:
        return None
    return depth.astype(np.float32)


def _clear_calibration_outputs(output_dir: Path) -> None:
    if not output_dir.exists():
        return
    for pattern in ("calibration_color_*.png", "calibration_depth_frame.npz"):
        for path in output_dir.glob(pattern):
            try:
                path.unlink()
            except OSError:
                logging.warning(f"[CALIBRATE] Unable to remove {path}")


def _charuco_detection_counts(image_paths: list[Path], board_path: Path) -> Tuple[list[int], int]:
    spec = aruco_board.load_charuco_board_spec(board_path)
    board = spec.create_board()

    # Get dictionary for initial marker detection
    dictionary_id = getattr(cv2.aruco, spec.dictionary)
    dictionary = cv2.aruco.getPredefinedDictionary(dictionary_id)
    detector_params = cv2.aruco.DetectorParameters()
    detector_params.detectInvertedMarker = True  # Match live overlay behavior

    counts: list[int] = []
    usable_frames = 0
    for path in image_paths:
        image = cv2.imread(str(path))
        if image is None:
            counts.append(0)
            continue
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # First detect markers
        detector = cv2.aruco.ArucoDetector(dictionary, detector_params)
        marker_corners, marker_ids, _rejected = detector.detectMarkers(gray)

        # Then detect ChArUco corners if markers were found
        corner_count = 0
        if marker_ids is not None and len(marker_ids) > 0:
            charuco_params = cv2.aruco.CharucoParameters()
            charuco_detector = cv2.aruco.CharucoDetector(
                board, charuco_params, detector_params)
            charuco_corners, charuco_ids, _, _ = charuco_detector.detectBoard(
                gray)
            corner_count = 0 if charuco_ids is None else int(len(charuco_ids))

            if charuco_ids is not None and charuco_corners is not None:
                obj, img = board.matchImagePoints(charuco_corners, charuco_ids)
                if obj is not None and img is not None and len(obj) >= 4:
                    usable_frames += 1

        counts.append(corner_count)

    return counts, usable_frames


def compute_calibration_bundle(board_path: Path, bundle_path: Path, output_dir: Path) -> str:
    if not board_path.exists():
        raise FileNotFoundError(
            f"ChArUco board layout not found: {board_path}")
    frames = _update_capture_state(output_dir)
    if not frames:
        raise RuntimeError(
            "No calibration frames found. Capture frames before calibrating.")
    depth_frame = (
        SESSION.capture.depth_frame
        if SESSION.capture.depth_frame is not None
        else load_depth_snapshot(output_dir)
    )
    if depth_frame is None:
        raise RuntimeError(
            "No depth snapshot found. Capture a depth frame before calibrating.")
    board_spec = aruco_board.load_charuco_board_spec(board_path)
    cam = calibration.calibrate_camera_charuco(frames, board_spec)
    try:
        pixel_to_mm = calibration.charuco_board_pixel_scale(
            frames[0], board_spec)
    except Exception as exc:  # fallback to nominal pixel spacing
        LOGGER.warning(
            "Falling back to nominal pixel scale while calibrating: %s", exc
        )
        pixel_to_mm = board_spec.marker_length_mm
    plane = calibration.estimate_workspace_plane(depth_frame, cam)
    bundle = calibration.CalibrationBundle(
        camera=cam,
        workspace_plane=plane,
        pixel_to_mm=pixel_to_mm,
    )
    calibration.save_calibration(bundle, bundle_path)
    SESSION.calibration_bundle = bundle
    _commit_settings(calibration_bundle_path=str(bundle_path))
    return f"Calibration bundle saved to {bundle_path}."


def load_calibration_bundle(bundle_path: Path) -> str:
    if not bundle_path.exists():
        raise FileNotFoundError(f"Calibration bundle not found: {bundle_path}")
    SESSION.calibration_bundle = calibration.load_calibration(bundle_path)
    _commit_settings(calibration_bundle_path=str(bundle_path))
    return f"Loaded calibration bundle from {bundle_path}."


def compute_workspace_from_latest(marker_ids: Tuple[int, int], marker_size_mm: float) -> workspace.WorkspaceDefinition:
    frame, _depth = get_latest_frame()
    if frame is None:
        raise RuntimeError("Camera stream is not providing frames yet.")
    if SESSION.calibration_bundle is None:
        raise RuntimeError("Calibration bundle is not loaded.")
    detections = markers.detect_markers(frame, marker_ids)
    ws = workspace.compute_workspace(
        detections,
        SESSION.calibration_bundle,
        marker_size_mm=marker_size_mm,
        camera_to_robot=_default_transform(),
    )
    SESSION.workspace_definition = ws
    save_workspace_definition(ws, DEFAULT_WORKSPACE_PATH)
    return ws


def save_workspace_definition(
    workspace_def: workspace.WorkspaceDefinition,
    workspace_path: Path = DEFAULT_WORKSPACE_PATH,
) -> str:
    workspace.save_workspace(workspace_def, workspace_path)
    _commit_settings(workspace_path=str(workspace_path))
    return f"Workspace saved to {workspace_path}."


def load_workspace_definition(workspace_path: Path = DEFAULT_WORKSPACE_PATH) -> workspace.WorkspaceDefinition:
    if not workspace_path.exists():
        raise FileNotFoundError(
            f"Workspace definition not found: {workspace_path}")
    SESSION.workspace_definition = workspace.load_workspace(workspace_path)
    _commit_settings(workspace_path=str(workspace_path))
    return SESSION.workspace_definition


def _clear_workspace_definition(path: Path = DEFAULT_WORKSPACE_PATH) -> None:
    """Drop any cached workspace definition without deleting the saved file."""
    SESSION.workspace_definition = None


def compute_plate_calibration_from_latest(layout_path: Path) -> plate.PlateCalibration:
    frame, _depth = get_latest_frame()
    if frame is None:
        raise RuntimeError("Camera stream is not providing frames yet.")
    if not layout_path.exists():
        raise FileNotFoundError(
            f"Calibration plate layout not found: {layout_path}")
    layout = plate.load_plate_layout(layout_path)
    result = plate.estimate_plate_calibration(frame, layout)
    SESSION.plate_calibration = result
    return result


def attach_plate_calibration(bundle_path: Path, layout_path: Path) -> str:
    if SESSION.calibration_bundle is None:
        raise RuntimeError("Calibration bundle is not loaded.")
    if SESSION.plate_calibration is None:
        raise RuntimeError("Calibration plate has not been detected yet.")
    updated = calibration.CalibrationBundle(
        camera=SESSION.calibration_bundle.camera,
        workspace_plane=SESSION.calibration_bundle.workspace_plane,
        pixel_to_mm=SESSION.calibration_bundle.pixel_to_mm,
        plate_homography=SESSION.plate_calibration.homography,
        plate_layout_path=str(layout_path),
    )
    calibration.save_calibration(updated, bundle_path)
    SESSION.calibration_bundle = updated
    return f"Updated bundle saved to {bundle_path}."


async def _run_robot_plate_calibration(
    robot_uri: str,
    plate_layout: plate.CalibrationPlate,
) -> Tuple[list[dict], dict]:
    """Capture three robot poses via flange button 1 and calibrate coord ID 5."""
    session = _get_robot_session()

    def _ensure_plate_baseline() -> bool:
        if SESSION.robot_calibration_baseline_px is not None:
            return True
        center = PLATE_STATUS.get("center_px")
        if center is not None:
            SESSION.robot_calibration_baseline_px = np.asarray(
                center, dtype=np.float64
            )
            return True
        frame, _depth = get_latest_frame()
        if frame is None:
            return False
        try:
            calibration_result = plate.estimate_plate_calibration(
                frame, plate_layout)
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("Plate baseline detection failed: %s", exc)
            return False
        center_px = _plate_center_px(calibration_result.detections)
        if center_px is None:
            return False
        SESSION.robot_calibration_baseline_px = np.asarray(
            center_px, dtype=np.float64
        )
        PLATE_STATUS.update(
            {
                "count": len(calibration_result.detections),
                "ok": len(calibration_result.detections) >= SETTINGS.min_plate_markers,
                "stable": len(calibration_result.detections) >= SETTINGS.min_plate_markers,
                "center_px": (float(center_px[0]), float(center_px[1])),
                "movement_px": None,
                "timestamp": time.time(),
                "message": "Plate baseline refreshed from live detection.",
            }
        )
        return True

    async def _ensure_coordinate_id(
        robot_api: CodroidAPI,
        expected_id: int,
        timeout_s: float = 5.0,
        require: bool = True,
    ) -> None:
        await robot_api.set_current_coordinate_id(expected_id)
        deadline = time.monotonic() + timeout_s
        resent = False
        while time.monotonic() < deadline:
            status = session.robot_status_snapshot()
            coord_id = None
            if isinstance(status, dict):
                coord_id = status.get("CoordinateId")
            if coord_id is not None:
                try:
                    if int(coord_id) == expected_id:
                        return
                except (TypeError, ValueError):
                    pass
            if not resent and time.monotonic() > deadline - (timeout_s / 2):
                await robot_api.set_current_coordinate_id(expected_id)
                resent = True
            await asyncio.sleep(0.1)
        if not require:
            LOGGER.warning(
                "CoordinateId did not switch to %s within %.1fs (last=%s).",
                expected_id,
                timeout_s,
                coord_id,
            )
            return
        raise RuntimeError(
            "Calibration requires the robot to be on CoordinateId=0 (base). "
            f"Current CoordinateId={coord_id}. Switch to 0 and retry."
        )

    async def _await_flange_press_with_plate_guard(
        timeout_s: float,
    ) -> dict:
        deadline = time.monotonic() + timeout_s
        while True:
            if SESSION.robot_calibration_baseline_px is None:
                _ensure_plate_baseline()
            plate_ok, plate_message = _plate_is_stable(
                require_baseline=True,
                allow_lost=True,
            )
            if not plate_ok:
                raise RuntimeError(f"Calibration canceled: {plate_message}")
            posture_ok, posture_message = _robot_posture_ready()
            if not posture_ok:
                raise RuntimeError(f"Calibration canceled: {posture_message}")
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise asyncio.TimeoutError()
            try:
                return await session.next_flange_press(timeout=min(0.5, remaining))
            except asyncio.TimeoutError:
                continue

    async def _runner(robot_api: CodroidAPI) -> Tuple[list[dict], dict]:
        _clear_robot_accuracy_guidance()
        _set_robot_guidance(
            "Powering on robot and selecting CoordinateId=0 (base) for calibration...",
            status="running",
            step=1,
            verification_ready=False,
            modal_suppressed=False,
        )
        await robot_api.power_on()
        await _ensure_coordinate_id(robot_api, 0)
        await session.reset_flange_queue()

        robot_points: list[dict] = []
        calib_points = plate_layout.robot_calibration_points()
        if len(calib_points) < 3:
            raise RuntimeError(
                "Calibration plate layout is missing calibration points.")

        point_labels = [
            "Point 1 (Origin)",
            "Point 2 (+X)",
            "Point 3 (+X/+Y)",
        ]
        for idx in range(3):
            _ensure_plate_stable(allow_lost=True)
            point_label = point_labels[idx] if idx < len(
                point_labels) else f"Point {idx + 1}"
            _set_robot_guidance(
                "Step {step}/3: Place the gripper fingers in indentation {number}. "
                "Keep the flange vertical. When aligned, press flange button 1.".format(
                    step=idx + 1,
                    number=idx + 1,
                ),
                step=idx + 1,
                status="running",
                verification_ready=False,
            )
            LOGGER.info(
                "Step %d/3: Waiting for gripper to align with %s...",
                idx + 1,
                point_label,
            )

            try:
                posture = await _await_flange_press_with_plate_guard(timeout_s=90.0)
            except asyncio.TimeoutError as exc:
                _set_robot_guidance(
                    f"Timed out waiting for flange button at {point_label}.",
                    status="error",
                    verification_ready=False,
                )
                raise RuntimeError(
                    f"Timed out waiting for flange button 1 press at {point_label}. "
                    f"Move gripper to the groove target and press the flange button."
                ) from exc

            posture = _ensure_posture_axes(
                posture, session.position_snapshot())
            LOGGER.info(
                "Captured %s. Position: X=%.1f, Y=%.1f, Z=%.1f",
                point_label,
                posture.get("x", 0.0),
                posture.get("y", 0.0),
                posture.get("z", 0.0),
            )
            robot_points.append(posture)
            if idx < 2:
                next_label = point_labels[idx + 1]
                _set_robot_guidance(
                    (
                        f"Captured {point_label}. "
                        "Next: place the gripper in indentation "
                        f"{idx + 2} and press flange button 1."
                    ),
                    step=idx + 1,
                    status="running",
                    verification_ready=False,
                )

        _set_robot_guidance(
            f"Switching to CoordinateId {ROBOT_COORDINATE_ID} for calibration...",
            status="running",
            verification_ready=False,
        )
        await _ensure_coordinate_id(
            robot_api, ROBOT_COORDINATE_ID, timeout_s=5.0, require=False
        )

        _set_robot_guidance(
            "Computing coordinate transform from 3 calibration points...",
            status="running",
            verification_ready=False,
        )
        LOGGER.info(
            "Computing coordinate transformation from 3 calibration points...")
        cpos_points = [
            CodroidAPI.build_target_cpos(
                x=float(pt["x"]),
                y=float(pt["y"]),
                z=float(pt["z"]),
                a=float(pt["a"]),
                b=float(pt["b"]),
                c=float(pt["c"]),
            )
            for pt in robot_points
        ]
        _set_robot_guidance(
            "Waiting for robot controller to return the calibrated frame...",
            status="running",
            verification_ready=False,
        )
        LOGGER.info(
            "Requesting coordinate calibration from robot controller...")

        async def _await_listener_frame(
            snapshot: Callable[[], dict],
            timestamp: Callable[[], Optional[float]],
            since: float,
            timeout_s: float,
            label: str,
        ) -> dict:
            deadline = time.monotonic() + timeout_s
            while time.monotonic() < deadline:
                updated_at = timestamp()
                if updated_at is not None and updated_at >= since:
                    frame = snapshot()
                    if frame:
                        return frame
                await asyncio.sleep(0.1)
            raise asyncio.TimeoutError(f"Timed out waiting for {label}.")

        calibration_request_time = time.time()
        payload_points = [
            robot_api.attach_coordinate_to_cpos(
                point, coordinate_id=0
            )
            for point in cpos_points
        ]
        calibration_request = robot_api.build_message(
            message_type="Robot",
            action="CoordinateCalibration",
            data=payload_points,
        )
        await robot_api.send_message(calibration_request)

        try:
            coord_frame = await _await_listener_frame(
                session.calibration_frame_snapshot,
                session.calibration_frame_timestamp,
                calibration_request_time,
                SETTINGS.robot_coordinate_timeout_s,
                "coordinate calibration response",
            )
        except asyncio.TimeoutError as exc:
            raise RuntimeError(
                "Timed out waiting for coordinate calibration response."
            ) from exc

        if not coord_frame:
            raise RuntimeError(
                "Robot controller returned empty calibration frame.")

        def _coord_axis(keys: Tuple[str, ...]) -> float:
            for key in keys:
                if key in coord_frame:
                    return _safe_float(coord_frame.get(key), 0.0)
            return 0.0

        coord_frame["a"] = _coord_axis(("a", "rx", "roll"))
        coord_frame["b"] = _coord_axis(("b", "ry", "pitch"))
        coord_frame["c"] = _coord_axis(("c", "rz", "yaw"))

        change_request_time = time.time()
        change_payload = {
            "id": ROBOT_COORDINATE_ID,
            "x": coord_frame["x"],
            "y": coord_frame["y"],
            "z": coord_frame["z"],
            "a": coord_frame["a"],
            "b": coord_frame["b"],
            "c": coord_frame["c"],
        }
        change_request = robot_api.build_message(
            message_type="Robot",
            action="ChangeCoordinateParameter",
            data=change_payload,
        )
        await robot_api.send_message(change_request)

        publish_timeout = min(SETTINGS.robot_coordinate_timeout_s, 5.0)
        try:
            await _await_listener_frame(
                session.coordinate_frame_snapshot,
                session.coordinate_frame_timestamp,
                change_request_time,
                publish_timeout,
                "coordinate frame publish",
            )
        except asyncio.TimeoutError:
            LOGGER.warning(
                "No RobotCoordinate publish after calibration; reselecting CoordinateId."
            )
            await robot_api.set_current_coordinate_id(ROBOT_COORDINATE_ID)
            try:
                await _await_listener_frame(
                    session.coordinate_frame_snapshot,
                    session.coordinate_frame_timestamp,
                    time.time(),
                    publish_timeout,
                    "coordinate frame publish after reselect",
                )
            except asyncio.TimeoutError:
                LOGGER.warning(
                    "Timed out waiting for coordinate frame publish after reselect."
                )
        _set_robot_guidance(
            "Coordinate frame updated. Preparing verification step...",
            status="running",
            step=3,
            verification_ready=False,
        )
        LOGGER.info("Robot calibration coordinate ID 5 updated.")
        return robot_points, coord_frame or {}

    return await session.run_with_robot(robot_uri, _runner)


def _apply_robot_calibration_results(
    robot_points: list[dict],
    plate_layout: plate.CalibrationPlate,
) -> Tuple[workspace.RigidTransform, dict[str, float]]:
    plate_pts = []
    for calib_point in plate_layout.robot_calibration_points()[:3]:
        plate_pts.append([float(calib_point.center_mm[0]),
                         float(calib_point.center_mm[1]), 0.0])
    robot_pts = []
    for pt in robot_points[:3]:
        robot_pts.append([float(pt["x"]), float(
            pt["y"]), float(pt["z"])])
    plate_pts_np = np.asarray(plate_pts, dtype=np.float64)
    robot_pts_np = np.asarray(robot_pts, dtype=np.float64)
    transform = _rigid_transform(plate_pts_np, robot_pts_np)

    # Compute calibration error diagnostics
    # Transform plate points to robot coordinates using the computed transform
    transformed_plate_pts = (
        transform.rotation @ plate_pts_np.T).T + transform.translation

    # Compute position errors for each calibration point
    errors = robot_pts_np - transformed_plate_pts
    LOGGER.info("=== Calibration Error Diagnostics ===")
    LOGGER.info(
        "This shows how far the gripper was positioned from the ideal location:")
    for i, (error, plate_pt, robot_pt) in enumerate(zip(errors, plate_pts_np, robot_pts_np)):
        error_xy = np.linalg.norm(error[:2])
        error_z = abs(error[2])
        LOGGER.info(
            "Point %d: XY error=%.2f mm (X=%+.2f, Y=%+.2f), Z error=%.2f mm | "
            "Ideal=(%.1f, %.1f, %.1f) Captured=(%.1f, %.1f, %.1f)",
            i + 1,
            error_xy,
            error[0],
            error[1],
            error_z,
            transformed_plate_pts[i][0],
            transformed_plate_pts[i][1],
            transformed_plate_pts[i][2],
            robot_pt[0],
            robot_pt[1],
            robot_pt[2],
        )
    mean_xy_error = np.mean([np.linalg.norm(e[:2]) for e in errors])
    max_xy_error = np.max([np.linalg.norm(e[:2]) for e in errors])
    LOGGER.info(
        "Mean XY error: %.2f mm | Max XY error: %.2f mm | "
        "Recommendation: %s",
        mean_xy_error,
        max_xy_error,
        "Good calibration!" if max_xy_error < 2.0 else "Consider recalibrating with more careful positioning."
    )
    LOGGER.info("="*50)

    if SESSION.workspace_definition is not None:
        ws = SESSION.workspace_definition
        updated_ws = workspace.WorkspaceDefinition(
            width_mm=ws.width_mm,
            height_mm=ws.height_mm,
            origin_camera=ws.origin_camera,
            x_axis_camera=ws.x_axis_camera,
            y_axis_camera=ws.y_axis_camera,
            camera_to_robot=transform,
        )
    else:
        # Fallback: use the plate dimensions as the workspace basis.
        updated_ws = workspace.WorkspaceDefinition(
            width_mm=float(plate_layout.plate_size_mm[0]),
            height_mm=float(plate_layout.plate_size_mm[1]),
            origin_camera=np.array([0.0, 0.0, 0.0]),
            x_axis_camera=np.array([1.0, 0.0, 0.0]),
            y_axis_camera=np.array([0.0, 1.0, 0.0]),
            camera_to_robot=transform,
        )

    SESSION.workspace_definition = updated_ws
    SESSION.coord5_transform = None
    SESSION.plate_to_robot_transform = None
    SESSION.plate_origin_mm = None
    save_workspace_definition(updated_ws, DEFAULT_WORKSPACE_PATH)
    accuracy = _record_robot_accuracy(mean_xy_error, max_xy_error)
    return transform, accuracy


def _draw_label(frame: np.ndarray, text: str, origin: Tuple[int, int], color: Tuple[int, int, int]) -> None:
    cv2.putText(frame, text, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


def _workspace_corners_px(workspace_def: workspace.WorkspaceDefinition) -> np.ndarray:
    # Workspace definition stores everything in mm in camera coordinates
    # The axes are normalized direction vectors
    # We need calibration to convert mm back to pixels
    if SESSION.calibration_bundle is None:
        raise RuntimeError("Calibration bundle required to display workspace")

    scale = SESSION.calibration_bundle.pixel_to_mm
    if scale == 0:
        raise RuntimeError(
            "Workspace scale is zero; recompute workspace markers.")

    # Convert origin from mm to pixels
    origin_px = np.array([workspace_def.origin_camera[0],
                         workspace_def.origin_camera[1]]) / scale

    # Convert width and height from mm to pixels
    width_px = workspace_def.width_mm / scale
    height_px = workspace_def.height_mm / scale

    # Extract 2D axes (they are already normalized directions in mm-space)
    x_dir = np.array([workspace_def.x_axis_camera[0],
                     workspace_def.x_axis_camera[1]])
    y_dir = np.array([workspace_def.y_axis_camera[0],
                     workspace_def.y_axis_camera[1]])

    # Normalize to ensure they are unit vectors
    x_len = np.linalg.norm(x_dir)
    y_len = np.linalg.norm(y_dir)
    if x_len > 0:
        x_dir = x_dir / x_len
    if y_len > 0:
        y_dir = y_dir / y_len

    # Build rotated rectangle corners
    points = np.asarray(
        [
            origin_px,  # top-left
            origin_px + width_px * x_dir,  # top-right
            origin_px + width_px * x_dir + height_px * y_dir,  # bottom-right
            origin_px + height_px * y_dir,  # bottom-left
        ],
        dtype=np.float64,
    )
    return np.round(points).astype(int)


def _draw_dotted_polyline(
    frame: np.ndarray,
    points: np.ndarray,
    color: Tuple[int, int, int] = (0, 215, 255),
    thickness: int = 2,
    gap: int = 10,
) -> None:
    for idx in range(len(points)):
        start = points[idx]
        end = points[(idx + 1) % len(points)]
        vec = end - start
        length = float(np.linalg.norm(vec))
        if length == 0:
            continue
        direction = vec / length
        dist = 0.0
        while dist < length:
            segment_start = start + direction * dist
            dist = min(length, dist + gap)
            segment_end = start + direction * dist
            cv2.line(
                frame,
                tuple(np.round(segment_start).astype(int)),
                tuple(np.round(segment_end).astype(int)),
                color,
                thickness,
                lineType=cv2.LINE_AA,
            )
            dist += gap


def _apply_workspace_overlay(frame: np.ndarray) -> None:
    if SESSION.workspace_definition is None:
        return
    try:
        corners_px = _workspace_corners_px(SESSION.workspace_definition)
    except Exception as exc:
        logging.debug("Workspace overlay skipped: %s", exc)
        return
    _draw_dotted_polyline(frame, corners_px)


def draw_charuco_overlay(frame: np.ndarray, board_path: Path) -> Tuple[np.ndarray, str, str]:
    vis = frame.copy()
    if not board_path.exists():
        status = "ChArUco layout not found."
        return vis, status, "ChArUco corners: --"

    try:
        board_spec = aruco_board.load_charuco_board_spec(board_path)
        board = board_spec.create_board()
        gray = cv2.cvtColor(vis, cv2.COLOR_BGR2GRAY)

        # Get dictionary and detector
        dictionary_id = getattr(cv2.aruco, board_spec.dictionary)
        dictionary = cv2.aruco.getPredefinedDictionary(dictionary_id)
        detector_params = cv2.aruco.DetectorParameters()
        detector_params.detectInvertedMarker = True  # Enable inverted marker detection

        # Detect markers
        detector = cv2.aruco.ArucoDetector(dictionary, detector_params)
        marker_corners, marker_ids, rejected = detector.detectMarkers(gray)

        marker_count = 0 if marker_ids is None else len(marker_ids)

        if marker_ids is not None:
            vis = cv2.aruco.drawDetectedMarkers(
                vis, marker_corners, marker_ids)

        # Try ChArUco detection
        corner_count = 0

        if marker_ids is not None and len(marker_ids) > 0:
            charuco_params = cv2.aruco.CharucoParameters()
            charuco_detector = cv2.aruco.CharucoDetector(
                board, charuco_params, detector_params)
            charuco_corners, charuco_ids, _, _ = charuco_detector.detectBoard(
                gray)
            corner_count = 0 if charuco_ids is None else len(charuco_ids)
            if charuco_ids is not None:
                vis = cv2.aruco.drawDetectedCornersCharuco(
                    vis, charuco_corners, charuco_ids)

        _draw_label(vis, f"Markers: {marker_count}", (10, 30), (255, 255, 255))
        _draw_label(vis, f"Corners: {corner_count}", (10, 55), (255, 255, 255))
        status = f"Detected {marker_count} markers and {corner_count} corners."
        metric = f"ChArUco corners: {corner_count}"
        return vis, status, metric
    except Exception as exc:
        logging.error(f"ChArUco detection error: {exc}", exc_info=True)
        _draw_label(vis, f"Error: {str(exc)[:50]}", (10, 30), (255, 0, 0))
        return vis, f"Detection error: {exc}", "ChArUco corners: --"


# Global buffer for smoothing workspace rectangle (reduce jitter)
_workspace_rect_buffer = []
_WORKSPACE_SMOOTH_FRAMES = 5


def draw_workspace_overlay(
    frame: np.ndarray,
    marker_ids: Tuple[int, int],
    marker_size_mm: float,
) -> Tuple[np.ndarray, str, str]:
    global _workspace_rect_buffer
    vis = frame.copy()
    try:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect ArUco markers with inverted color support
        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        detector_params = cv2.aruco.DetectorParameters()
        detector_params.detectInvertedMarker = True

        detector = cv2.aruco.ArucoDetector(dictionary, detector_params)
        marker_corners, marker_ids_detected, _rejected = detector.detectMarkers(
            gray)

        if marker_ids_detected is None or len(marker_ids_detected) == 0:
            raise RuntimeError(
                "No markers detected. Ensure markers are in frame and high-contrast.")

        # Map detected markers by ID
        detected_map = {}
        for corners, mid in zip(marker_corners, marker_ids_detected.flatten()):
            pts = corners.reshape((4, 2))
            center = tuple(np.mean(pts, axis=0).tolist())
            detected_map[int(mid)] = {
                'corners': pts,
                'center': center,
                'id': int(mid)
            }

        # Check if we have both required markers
        if marker_ids[0] not in detected_map or marker_ids[1] not in detected_map:
            found_ids = sorted(detected_map.keys())
            raise RuntimeError(
                f"Missing required markers {marker_ids}. Found: {found_ids}")

        marker_a_center = None
        marker_b_center = None

        # Draw detected markers with colors
        colors = {marker_ids[0]: (0, 255, 0), marker_ids[1]: (0, 140, 255)}
        for mid, color in colors.items():
            if mid in detected_map:
                det = detected_map[mid]
                pts = det['corners'].astype(int)
                cv2.polylines(vis, [pts], isClosed=True,
                              color=color, thickness=2)
                center = tuple(np.round(det['center']).astype(int))
                cv2.circle(vis, center, 4, color, -1)
                cv2.putText(vis, f"ID {mid}", center,
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                if mid == marker_ids[0]:
                    marker_a_center = center
                elif mid == marker_ids[1]:
                    marker_b_center = center

        # Draw rotated rectangle between the two marker centers
        if marker_a_center is not None and marker_b_center is not None:
            # Use marker_a orientation to define the rectangle axes
            marker_a_corners = detected_map[marker_ids[0]]['corners']

            # X-axis: direction from left edge to right edge of marker_a
            # top-right - top-left
            x_axis_px = marker_a_corners[1] - marker_a_corners[0]
            x_axis_norm = x_axis_px / \
                np.linalg.norm(x_axis_px) if np.linalg.norm(
                    x_axis_px) > 0 else np.array([1.0, 0.0])

            # Y-axis: perpendicular to x-axis (rotate 90 degrees)
            y_axis_norm = np.array([-x_axis_norm[1], x_axis_norm[0]])

            # Diagonal from marker_a to marker_b
            diagonal = np.array(marker_b_center) - np.array(marker_a_center)

            # Project diagonal onto the rotated axes
            width_px = np.dot(diagonal, x_axis_norm)
            height_px = np.dot(diagonal, y_axis_norm)

            # Compute the four corners of the rotated rectangle
            # Start at marker_a center and build rectangle using the projected dimensions
            origin = np.array(marker_a_center, dtype=float)
            corner_tl = origin  # top-left at marker_a
            corner_tr = origin + width_px * x_axis_norm  # move right
            corner_br = origin + width_px * x_axis_norm + height_px * y_axis_norm  # diagonal
            corner_bl = origin + height_px * y_axis_norm  # move down

            # Compute current rectangle corners
            rect_corners_float = np.array(
                [corner_tl, corner_tr, corner_br, corner_bl], dtype=np.float64)

            # Apply temporal smoothing to reduce jitter
            _workspace_rect_buffer.append(rect_corners_float)
            if len(_workspace_rect_buffer) > _WORKSPACE_SMOOTH_FRAMES:
                _workspace_rect_buffer.pop(0)

            # Average the buffered corner positions
            if len(_workspace_rect_buffer) > 0:
                rect_corners_smoothed = np.mean(_workspace_rect_buffer, axis=0)
                rect_corners = np.round(rect_corners_smoothed).astype(np.int32)
            else:
                rect_corners = np.round(rect_corners_float).astype(np.int32)

            # Draw the rotated rectangle
            cv2.polylines(vis, [rect_corners], isClosed=True,
                          color=(0, 255, 255), thickness=2)

            # Draw diagonal for reference
            cv2.line(vis, tuple(rect_corners[0]), tuple(
                rect_corners[2]), (0, 255, 255), 1)

        status = "Workspace markers detected."

        # Compute workspace metrics if calibration is available
        metric = "Workspace: --"
        if SESSION.calibration_bundle is not None:
            try:
                # Create a simple detections object for workspace computation
                det_a = type('MarkerDetection', (), {
                    'marker_id': marker_ids[0],
                    'corners': detected_map[marker_ids[0]]['corners'],
                    'center': detected_map[marker_ids[0]]['center']
                })()
                det_b = type('MarkerDetection', (), {
                    'marker_id': marker_ids[1],
                    'corners': detected_map[marker_ids[1]]['corners'],
                    'center': detected_map[marker_ids[1]]['center']
                })()
                detections = type('WorkspaceMarkers', (), {
                    'marker_a': det_a,
                    'marker_b': det_b
                })()

                ws = workspace.compute_workspace(
                    detections,
                    SESSION.calibration_bundle,
                    marker_size_mm=marker_size_mm,
                    camera_to_robot=_default_transform(),
                )
                metric = f"Workspace: {ws.width_mm:.1f} x {ws.height_mm:.1f} mm"
                _draw_label(vis, metric, (10, 30), (255, 255, 255))
            except Exception as exc:
                metric = f"Workspace: {exc}"

        return vis, status, metric

    except Exception as exc:
        return vis, f"Workspace detection: {exc}", "Workspace: --"


def draw_plate_overlay(frame: np.ndarray, layout_path: Path) -> Tuple[np.ndarray, str, str]:
    vis = frame.copy()
    if not layout_path.exists():
        PLATE_STATUS.update(
            {
                "count": 0,
                "ok": False,
                "stable": False,
                "center_px": None,
                "movement_px": None,
                "timestamp": time.time(),
                "message": "Layout not found",
            }
        )
        return vis, "Calibration plate layout not found.", "Plate error: --"
    plate_layout = plate.load_plate_layout(layout_path)
    try:
        calibration_result = plate.estimate_plate_calibration(
            frame, plate_layout)
    except Exception as exc:
        PLATE_STATUS.update(
            {
                "count": 0,
                "ok": False,
                "stable": False,
                "center_px": None,
                "movement_px": None,
                "timestamp": time.time(),
                "message": str(exc),
            }
        )
        return vis, f"Plate detection: {exc}", "Plate error: --"

    for det in calibration_result.detections.values():
        pts = det.corners.astype(int)
        cv2.polylines(vis, [pts], isClosed=True,
                      color=(255, 204, 0), thickness=2)
        center = tuple(np.round(det.center).astype(int))
        cv2.circle(vis, center, 4, (255, 204, 0), -1)
        cv2.putText(vis, f"ID {det.marker_id}", center,
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 204, 0), 2)

    # Compute coord5 positions for calibration points if robot calibration is available
    point_coords_text = ""
    plate_points = plate_layout.robot_calibration_points()
    if (
        len(plate_points) >= 3
        and SESSION.robot_calibration_points
        and len(SESSION.robot_calibration_points) >= 3
        and len(calibration_result.detections) >= 2
    ):
        try:
            origin_mm = np.asarray(plate_points[0].center_mm, dtype=np.float64)
            # Build 2D rigid transform from reference layout to current camera detection
            reference_h = SESSION.plate_reference_homography
            if reference_h is None:
                reference_h = calibration_result.homography
            marker_base: list[np.ndarray] = []
            marker_curr: list[np.ndarray] = []
            for marker_id, detection in calibration_result.detections.items():
                marker = plate_layout.markers.get(marker_id)
                if marker is None:
                    continue
                base_pt = marker.center_mm - origin_mm
                curr_pt = plate.image_points_to_plate_mm(
                    reference_h, np.asarray(
                        [detection.center], dtype=np.float64)
                )[0]
                marker_base.append(np.asarray(base_pt, dtype=np.float64))
                marker_curr.append(np.asarray(
                    curr_pt - origin_mm, dtype=np.float64))
            if len(marker_base) >= 2:
                r2, t2 = _rigid_transform_2d(
                    np.vstack(marker_base), np.vstack(marker_curr)
                )
                # Compute adjusted coord5 for each calibration point and draw them on the frame
                point_labels = ["P1", "P2", "P3"]
                # Cyan, Magenta, Light Green
                point_colors = [
                    (0, 255, 255), (255, 100, 255), (100, 255, 100)]
                point_coords_parts = []

                # Convert calibration point positions from plate mm to pixel coordinates
                homography = calibration_result.homography
                if homography is not None:
                    for i, point in enumerate(plate_points[:3]):
                        # Get plate mm coordinates
                        plate_mm_xy = np.array([[
                            float(point.center_mm[0]),
                            float(point.center_mm[1])
                        ]], dtype=np.float32).reshape(-1, 1, 2)

                        # Transform to pixel coordinates using homography inverse
                        inv_h = np.linalg.inv(homography)
                        pixel_xy = cv2.perspectiveTransform(
                            plate_mm_xy, inv_h).reshape(-1, 2)
                        pixel_x, pixel_y = int(round(pixel_xy[0][0])), int(
                            round(pixel_xy[0][1]))

                        # Compute coord5 position for text display
                        base_xy = np.array([
                            float(point.center_mm[0] - origin_mm[0]),
                            float(point.center_mm[1] - origin_mm[1]),
                        ], dtype=np.float64)
                        adj_xy = (r2 @ base_xy) + t2
                        label = point_labels[i] if i < len(
                            point_labels) else f"P{i+1}"
                        color = point_colors[i] if i < len(
                            point_colors) else (255, 255, 255)

                        # Draw circle at calibration point location
                        cv2.circle(vis, (pixel_x, pixel_y), 8, color, 2)
                        cv2.circle(vis, (pixel_x, pixel_y), 2, color, -1)

                        # Draw label with coord5 position
                        text = f"{label}"
                        text_size = cv2.getTextSize(
                            text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                        text_x = pixel_x - text_size[0] // 2
                        text_y = pixel_y - 12
                        # Draw text background
                        cv2.rectangle(vis,
                                      (text_x - 2, text_y - text_size[1] - 2),
                                      (text_x + text_size[0] + 2, text_y + 2),
                                      (0, 0, 0), -1)
                        cv2.putText(vis, text, (text_x, text_y),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                        point_coords_parts.append(
                            f"{label}:({adj_xy[0]:.1f},{adj_xy[1]:.1f})"
                        )
                    point_coords_text = " · ".join(point_coords_parts)
        except Exception:
            # Silently ignore errors in coord5 computation for overlay
            pass

    metric = f"Reprojection error: {calibration_result.reprojection_error_mm:.2f} mm"
    if point_coords_text:
        metric = f"{metric} · {point_coords_text}"
    _draw_label(vis, metric, (10, 30), (255, 255, 255))
    status = f"Plate markers detected: {len(calibration_result.detections)}"
    center_px = _plate_center_px(calibration_result.detections)
    center_list = None
    if center_px is not None:
        center_list = [float(center_px[0]), float(center_px[1])]
    movement_px = None
    stable = len(calibration_result.detections) >= SETTINGS.min_plate_markers
    baseline_px = SESSION.robot_calibration_baseline_px
    if baseline_px is not None and center_px is not None:
        movement_px = float(np.linalg.norm(center_px - baseline_px))
        if movement_px > SETTINGS.plate_movement_threshold_px:
            stable = False
            status = f"Plate moved ({movement_px:.1f}px); hold it steady."
    PLATE_STATUS.update(
        {
            "count": len(calibration_result.detections),
            "ok": len(calibration_result.detections) >= SETTINGS.min_plate_markers,
            "stable": stable,
            "center_px": center_list,
            "movement_px": movement_px,
            "timestamp": time.time(),
            "message": status,
        }
    )
    return vis, status, metric


def _refresh_plate_status(frame: np.ndarray, layout_path: Path) -> None:
    """Update PLATE_STATUS without modifying the displayed overlay frame."""
    if not layout_path.exists():
        PLATE_STATUS.update(
            {
                "count": 0,
                "ok": False,
                "stable": False,
                "center_px": None,
                "movement_px": None,
                "timestamp": time.time(),
                "message": "Layout not found",
            }
        )
        return
    plate_layout = plate.load_plate_layout(layout_path)
    try:
        calibration_result = plate.estimate_plate_calibration(
            frame, plate_layout)
    except Exception as exc:
        PLATE_STATUS.update(
            {
                "count": 0,
                "ok": False,
                "stable": False,
                "center_px": None,
                "movement_px": None,
                "timestamp": time.time(),
                "message": str(exc),
            }
        )
        return

    status = f"Plate markers detected: {len(calibration_result.detections)}"
    center_px = _plate_center_px(calibration_result.detections)
    center_list = None
    if center_px is not None:
        center_list = [float(center_px[0]), float(center_px[1])]
    movement_px = None
    stable = len(calibration_result.detections) >= SETTINGS.min_plate_markers
    baseline_px = SESSION.robot_calibration_baseline_px
    if baseline_px is not None and center_px is not None:
        movement_px = float(np.linalg.norm(center_px - baseline_px))
        if movement_px > SETTINGS.plate_movement_threshold_px:
            stable = False
            status = f"Plate moved ({movement_px:.1f}px); hold it steady."
    PLATE_STATUS.update(
        {
            "count": len(calibration_result.detections),
            "ok": len(calibration_result.detections) >= SETTINGS.min_plate_markers,
            "stable": stable,
            "center_px": center_list,
            "movement_px": movement_px,
            "timestamp": time.time(),
            "message": status,
        }
    )


def draw_detection_overlay(frame: np.ndarray) -> Tuple[np.ndarray, str, str]:
    vis = frame.copy()
    workspace_def = SESSION.workspace_definition
    coord5_frame = _get_robot_session().coordinate_frame_snapshot()
    coord5_frame_ok = bool(coord5_frame)

    # Run detection continuously if enabled and model is loaded
    if SESSION.model_loaded and SESSION.yolo_model is not None:
        try:
            detections = _run_detection(
                frame,
                SESSION.detection_confidence,
                SESSION.detection_iou,
                SESSION.detection_max,
                SESSION.detection_classes,
            )
            SESSION.detections = detections
            SESSION.last_detection_error = None
        except Exception as exc:
            SESSION.last_detection_error = str(exc)
            LOGGER.warning("Continuous detection failed: %s", exc)

    detections = SESSION.detections or []
    if not detections:
        msg = "Detection waiting..." if SESSION.model_loaded else "No detections available (model not loaded)."
        return vis, msg, "Detections: 0"

    for det in detections:
        x1, y1, x2, y2 = [int(val) for val in det["bbox"]]
        label = det.get("label", "object")
        conf = det.get("confidence", 0.0)
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 180, 255), 2)
        cv2.putText(
            vis,
            f"{label} {conf:.2f}",
            (x1, max(0, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 180, 255),
            2,
        )
        coord5_xy = _detection_coord5_mm(det, workspace_def)
        base_xy = None
        base_xyz = det.get("robot_base_mm")
        if base_xyz is None:
            base_xyz = _pixel_to_robot_base_xyz(
                (float(det["center"][0]), float(det["center"][1]))
            ) if "center" in det else None
        if base_xyz is not None:
            base_xy = (float(base_xyz[0]), float(base_xyz[1]))
        if coord5_xy is not None:
            cv2.putText(
                vis,
                f"c5 {coord5_xy[0]:.1f}, {coord5_xy[1]:.1f}",
                (x1, min(vis.shape[0] - 5, y2 + 18)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 180, 255),
                1,
            )
        if base_xy is not None:
            cv2.putText(
                vis,
                f"base {base_xy[0]:.1f}, {base_xy[1]:.1f}",
                (x1, min(vis.shape[0] - 5, y2 + 36)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 180, 255),
                1,
            )
    metric = f"Detections: {len(detections)}"
    _draw_label(vis, metric, (10, 30), (255, 255, 255))
    coord_note = "coord5: live" if coord5_frame_ok else "coord5: missing"
    status = "Detection active." if SESSION.model_loaded else "Detection overlay ready."
    status = f"{status} · {coord_note}"
    return vis, status, metric


def draw_drop_marker_overlay(frame: np.ndarray, marker_id: int) -> Tuple[np.ndarray, str, str]:
    vis = frame.copy()
    try:
        marker, workspace_mm = _detect_drop_marker(frame, marker_id)
    except Exception as exc:
        return vis, f"Drop marker: {exc}", "Drop marker: --"
    pts = marker.corners.astype(int)
    cv2.polylines(vis, [pts], isClosed=True, color=(255, 128, 0), thickness=2)
    center = tuple(np.round(marker.center).astype(int))
    cv2.circle(vis, center, 4, (255, 128, 0), -1)
    cv2.putText(
        vis,
        f"Drop ID {marker.marker_id}",
        center,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 128, 0),
        2,
    )
    metric = f"Drop marker: {workspace_mm[0]:.1f}, {workspace_mm[1]:.1f} mm"
    _draw_label(vis, metric, (10, 30), (255, 255, 255))
    return vis, "Drop marker detected.", metric


def draw_monitor_overlay(frame: np.ndarray, marker_id: int) -> Tuple[np.ndarray, str, str, str]:
    vis, status, detection_metric = draw_detection_overlay(frame)
    vis, drop_status, drop_metric = draw_drop_marker_overlay(vis, marker_id)
    status = f"{status} | {drop_status}"
    return vis, status, detection_metric, drop_metric


def build_header() -> dmc.AppShellHeader:
    branding = dmc.Text("Estun Codroid Pick-and-Place", className="app-title")
    badges = dmc.Group(
        [
            dmc.Badge("Camera: off", id="header-camera-badge",
                      color="gray", variant="light"),
            dmc.Badge("Robot: idle", id="header-robot-badge",
                      color="gray", variant="light"),
            dmc.Badge("Gripper: off", id="header-gripper-badge",
                      color="gray", variant="light"),
        ],
        className="header-badges",
    )
    actions = dmc.Group(
        [
            badges,
            dmc.Button("Setup", id="open-setup", variant="filled",
                       color="brand", radius="xl", size="md"),
        ],
        className="header-actions",
        align="center",
    )
    return dmc.AppShellHeader(
        dmc.Group(
            [
                dmc.Group(
                    [
                        dmc.Image(
                            src="/assets/profi-robotics.webp",
                            alt="Profi Robotics",
                            w=64,
                            h=64,
                            className="header-logo",
                        ),
                        branding,
                    ],
                    align="center",
                    gap="sm",
                ),
                actions,
            ],
            justify="space-between",
            align="center",
            className="header-bar",
        ),
        className="app-header",
        px="lg",
        h=84,
    )


def build_footer() -> dmc.AppShellFooter:
    stop_button = dmc.Button(
        "STOP",
        id="safety-stop",
        color="red",
        size="md",
        radius="xl",
        variant="filled",
        fullWidth=False,
        style={
            "fontWeight": 800,
            "letterSpacing": "0.06em",
            "padding": "14px 26px",
            "minWidth": "210px",
            "fontSize": "16px",
            "backgroundImage": "repeating-linear-gradient(45deg, #ff4747 0, #ff4747 12px, #ffd447 12px, #ffd447 24px)",
            "color": "#1f1f1f",
            "border": "2px solid #8b0000",
        },
    )
    power_toggle = dmc.Switch(
        id="robot-power-toggle",
        label="Robot power",
        checked=False,
        color="red",
        size="md",
    )
    rescue_button = dmc.Button(
        "Enable rescue mode",
        id="rescue-mode-enable",
        color="orange",
        variant="outline",
        size="sm",
        radius="md",
    )
    rescue_exit_button = dmc.Button(
        "Exit rescue mode",
        id="rescue-mode-exit",
        color="orange",
        variant="light",
        size="sm",
        radius="md",
        style={"display": "none"},
    )
    rescue_status = dmc.Text(
        id="rescue-mode-status",
        size="xs",
        c="dimmed",
    )

    robot_speed_slider = dmc.Slider(
        id="robot-speed-slider",
        min=0,
        max=100,
        step=5,
        value=25,
        labelAlwaysOn=False,
        size="sm",
        marks=[
            {"value": 0, "label": "0"},
            {"value": 50, "label": "50"},
            {"value": 100, "label": "100"},
        ],
        style={"flexGrow": 1, "minWidth": "220px"},
    )

    robot_speed_controls = dmc.Group(
        [
            dmc.ActionIcon(
                "-",
                id="robot-speed-decrease",
                variant="filled",
                color="gray",
                radius="xl",
                size="md",
            ),
            robot_speed_slider,
            dmc.ActionIcon(
                "+",
                id="robot-speed-increase",
                variant="filled",
                color="gray",
                radius="xl",
                size="md",
            ),
        ],
        gap="sm",
        align="center",
        justify="flex-start",
        style={"flexGrow": 1},
    )

    auto_start_button = dmc.Button(
        "Start Auto",
        id="auto-start",
        color="brand",
        size="sm",
        radius="md",
        variant="filled",
        leftSection=dmc.ActionIcon(
            "▶",
            size="sm",
            variant="transparent",
            color="white",
        ),
        style={
            "fontWeight": 700,
            "minWidth": "110px",
        },
    )

    auto_stop_button = dmc.Button(
        "Stop Auto",
        id="auto-stop",
        color="orange",
        size="sm",
        radius="md",
        variant="filled",
        leftSection=dmc.ActionIcon(
            "■",
            size="sm",
            variant="transparent",
            color="white",
        ),
        style={
            "fontWeight": 700,
            "minWidth": "110px",
        },
    )

    program_pause_button = dmc.Button(
        "Pause",
        id="program-pause",
        color="yellow",
        size="sm",
        radius="md",
        variant="filled",
        disabled=True,
        leftSection=dmc.ActionIcon(
            "⏸",
            size="sm",
            variant="transparent",
            color="black",
        ),
        style={
            "fontWeight": 700,
            "minWidth": "110px",
        },
    )

    program_stop_button = dmc.Button(
        "Stop",
        id="program-stop",
        color="red",
        size="sm",
        radius="md",
        variant="outline",
        disabled=True,
        style={
            "fontWeight": 700,
            "minWidth": "90px",
        },
    )

    program_reset_button = dmc.Button(
        "Reset",
        id="program-reset",
        color="gray",
        size="sm",
        radius="md",
        variant="light",
        disabled=True,
        style={
            "fontWeight": 700,
            "minWidth": "90px",
        },
    )

    stop_column = dmc.Stack(
        [
            dmc.Text("Emergency stop", fw=700, size="sm", c="red"),
            stop_button,
            power_toggle,
            rescue_button,
            rescue_exit_button,
            rescue_status,
        ],
        gap=6,
        align="center",
    )

    auto_control_column = dmc.Stack(
        [
            dmc.Text("Auto pick & place", fw=700, size="sm", c="blue"),
            dmc.Group(
                [auto_start_button, auto_stop_button],
                gap="xs",
            ),
        ],
        gap=6,
        align="center",
    )

    program_control_column = dmc.Stack(
        [
            dmc.Text("Program control", fw=700, size="sm", c="orange"),
            dmc.Group(
                [program_pause_button, program_stop_button, program_reset_button],
                gap="xs",
            ),
            dmc.Text(id="program-status-text", size="xs", c="dimmed"),
        ],
        gap=4,
        align="center",
    )

    robot_speed_block = dmc.Stack(
        [
            dmc.Text("Robot speed", fw=600, size="sm"),
            robot_speed_controls,
        ],
        gap=4,
        align="flex-start",
        style={"minWidth": "280px"},
    )

    fine_width_slider = dmc.Slider(
        id="footer-gripper-width",
        min=0,
        max=40,
        step=0.5,
        value=10.0,
        labelAlwaysOn=False,
        size="sm",
        marks=[],
        style={"minWidth": "220px"},
    )

    fine_force_slider = dmc.Slider(
        id="footer-gripper-force",
        min=GRIPPER_FORCE_MIN,
        max=GRIPPER_FORCE_MAX,
        step=5,
        value=GRIPPER_FORCE_DEFAULT,
        labelAlwaysOn=False,
        size="sm",
        marks=[],
        style={"minWidth": "220px"},
    )

    fine_speed_slider = dmc.Slider(
        id="footer-gripper-speed",
        min=10,
        max=100,
        step=5,
        value=60,
        labelAlwaysOn=False,
        size="sm",
        marks=[],
        style={"minWidth": "220px"},
    )

    width_controls = dmc.Group(
        [
            dmc.ActionIcon("-", id="footer-gripper-width-decrease",
                           variant="light", color="gray", radius="xl", size="md"),
            fine_width_slider,
            dmc.ActionIcon("+", id="footer-gripper-width-increase",
                           variant="light", color="gray", radius="xl", size="md"),
        ],
        gap="xs",
        align="center",
        justify="flex-start",
        className="footer-slider-controls",
    )

    force_controls = dmc.Group(
        [
            dmc.ActionIcon("-", id="footer-gripper-force-decrease",
                           variant="light", color="gray", radius="xl", size="md"),
            fine_force_slider,
            dmc.ActionIcon("+", id="footer-gripper-force-increase",
                           variant="light", color="gray", radius="xl", size="md"),
        ],
        gap="xs",
        align="center",
        justify="flex-start",
        className="footer-slider-controls",
    )

    speed_controls = dmc.Group(
        [
            dmc.ActionIcon("-", id="footer-gripper-speed-decrease",
                           variant="light", color="gray", radius="xl", size="md"),
            fine_speed_slider,
            dmc.ActionIcon("+", id="footer-gripper-speed-increase",
                           variant="light", color="gray", radius="xl", size="md"),
        ],
        gap="xs",
        align="center",
        justify="flex-start",
        className="footer-slider-controls",
    )

    gripper_block = dmc.Stack(
        [
            dmc.Text("Gripper control", fw=600, size="sm"),
            dmc.Text(id="footer-gripper-status", size="sm", c="dimmed"),
            dmc.Text(id="footer-gripper-meta", size="xs", c="dimmed"),
            html.Div(
                [
                    dmc.Text("Width (mm)", size="xs", fw=600),
                    width_controls,
                ],
                className="footer-fine-row",
            ),
            html.Div(
                [
                    dmc.Text("Force (N)", size="xs", fw=600),
                    force_controls,
                ],
                className="footer-fine-row",
            ),
            html.Div(
                [
                    dmc.Text("Speed (%)", size="xs", fw=600),
                    speed_controls,
                ],
                className="footer-fine-row",
            ),
        ],
        gap=6,
        align="flex-start",
        style={"flexGrow": 1, "minWidth": "320px"},
    )

    robot_position_text = dmc.Text(
        id="footer-robot-position",
        size="sm",
        className="footer-robot-position",
    )

    flange_indicators = html.Div(
        [
            html.Div(id="footer-flange-indicator-1",
                     className="footer-flange-indicator"),
            html.Div(id="footer-flange-indicator-2",
                     className="footer-flange-indicator"),
            html.Div(id="footer-flange-indicator-3",
                     className="footer-flange-indicator"),
            html.Div(id="footer-flange-indicator-4",
                     className="footer-flange-indicator"),
        ],
        className="footer-flange-indicators",
    )

    flange_block = html.Div(
        [
            dmc.Text("Flange press", fw=600, size="sm"),
            flange_indicators,
        ],
        className="footer-flange-block",
    )

    footer_toggle = dmc.Button(
        "Hide footer",
        id="footer-toggle",
        variant="light",
        color="gray",
        size="sm",
        radius="xl",
    )

    footer_grid = html.Div(
        [robot_speed_block, gripper_block],
        className="footer-body-grid",
    )

    telemetry_block = dmc.Stack(
        [
            dmc.Text("Robot telemetry", fw=600, size="xs"),
            robot_position_text,
        ],
        gap=2,
    )

    status_group = dmc.Group(
        [telemetry_block, flange_block],
        gap="md",
        align="center",
        className="footer-status-group",
        wrap="wrap",
    )

    rescue_joint_rows = []
    for joint_idx in range(1, 7):
        rescue_joint_rows.append(
            dmc.Group(
                [
                    dmc.Text(f"J{joint_idx}", size="xs", fw=600),
                    dmc.Button(
                        "-",
                        id=f"rescue-joint-{joint_idx}-minus",
                        variant="outline",
                        color="gray",
                        size="xs",
                    ),
                    dmc.Button(
                        "+",
                        id=f"rescue-joint-{joint_idx}-plus",
                        variant="outline",
                        color="gray",
                        size="xs",
                    ),
                ],
                gap="xs",
                align="center",
            )
        )

    rescue_controls = dmc.Stack(
        [
            dmc.Text("Rescue joint control", fw=700, size="sm", c="orange"),
            dmc.Stack(rescue_joint_rows, gap=4),
            dmc.Text(id="rescue-jog-status", size="xs", c="dimmed"),
        ],
        id="robot-controls-rescue",
        gap=6,
        align="flex-start",
        style={"display": "none"},
    )

    standard_controls = dmc.Group(
        [auto_control_column, program_control_column],
        id="robot-controls-standard",
        gap="lg",
        align="center",
        wrap="wrap",
    )

    footer_header = html.Div(
        [
            dmc.Group(
                [stop_column, standard_controls, rescue_controls],
                gap="lg",
                align="center",
                wrap="wrap",
            ),
            dmc.Group(
                [status_group, footer_toggle],
                gap="md",
                align="center",
                wrap="wrap",
            ),
        ],
        className="footer-header",
    )

    footer_body = html.Div(
        footer_grid,
        className="footer-body",
        id="footer-body",
    )

    return dmc.AppShellFooter(
        dmc.Container(
            dmc.Card(
                [footer_header, footer_body],
                className="footer-panel",
                padding="sm",
            ),
            fluid=True,
            px="lg",
            py="sm",
        ),
        id="app-footer",
        className="app-footer",
    )


def build_setup_drawer() -> dmc.Drawer:
    return dmc.Drawer(
        id="setup-drawer",
        opened=False,
        title="Setup & Connections",
        padding="md",
        size="lg",
        zIndex=2000,
        withOverlay=True,
        overlayProps={"opacity": 0.55, "color": "#000"},
        withinPortal=True,
        children=dmc.Stack(
            [
                dmc.Text("Camera connection", fw=600),
                dmc.Group(
                    [
                        dmc.Button("Scan for camera",
                                   id="camera-scan", variant="light"),
                        dmc.Switch(
                            id="camera-enable",
                            label="Enable live camera stream",
                            checked=False,
                            color="teal",
                        ),
                    ],
                    gap="md",
                ),
                dmc.Group(
                    [
                        dmc.NumberInput(
                            id="camera-width",
                            label="Width",
                            value=DEFAULT_RESOLUTION[0],
                            min=320,
                            max=3840,
                        ),
                        dmc.NumberInput(
                            id="camera-height",
                            label="Height",
                            value=DEFAULT_RESOLUTION[1],
                            min=240,
                            max=2160,
                        ),
                        dmc.NumberInput(
                            id="camera-fps-input",
                            label="FPS",
                            value=DEFAULT_FPS,
                            min=1,
                            max=120,
                            step=1,
                        ),
                    ],
                    grow=True,
                ),
                dmc.Switch(
                    id="camera-lock-af",
                    label="Lock autofocus",
                    checked=False,
                    color="teal",
                    size="md",
                ),
                dmc.Text(id="camera-status-text", size="sm", c="dimmed"),
                html.Div(id="camera-device-info"),
                dmc.Divider(),
                dmc.Text("Robot connection", fw=600),
                dmc.TextInput(
                    id="robot-uri",
                    label="WebSocket URI",
                    value=DEFAULT_ROBOT_URI,
                ),
                dmc.NumberInput(
                    id="settings-robot-coordinate-id",
                    label="Robot coordinate ID",
                    value=SETTINGS.robot_coordinate_id,
                    description="Coordinate system used for plate + pick controls.",
                    min=0,
                    max=10,
                    size="sm",
                ),
                dmc.Group(
                    [
                        dmc.Button("Test robot connection",
                                   id="robot-test", variant="light"),
                        dmc.Button(
                            "Disconnect robot",
                            id="robot-disconnect",
                            variant="outline",
                            color="red",
                        ),
                    ],
                    gap="sm",
                ),
                dmc.Text(id="robot-status-text", size="sm", c="dimmed"),
                dmc.Divider(),
                dmc.Text("OnRobot 2FG7 Gripper", fw=600),
                dmc.TextInput(
                    id="gripper-cb-ip",
                    label="Compute Box IP Address",
                    value=DEFAULT_GRIPPER_IP,
                    placeholder="192.168.101.95",
                    description="IP address of the OnRobot Compute Box",
                ),
                dmc.Group(
                    [
                        dmc.Switch(
                            id="gripper-enable",
                            label="Enable gripper connection",
                            checked=True,
                            color="teal",
                        ),
                        dmc.Button("Test gripper",
                                   id="gripper-test", variant="light"),
                    ],
                    gap="md",
                ),
                dmc.Text(id="gripper-status-text", size="sm", c="dimmed"),
                html.Div(id="gripper-device-info"),
                dmc.Stack(
                    [
                        dmc.Text("Finger orientation", fw=600, size="sm"),
                        dmc.SegmentedControl(
                            id="gripper-orientation",
                            data=[
                                {"label": "Inward", "value": "inward"},
                                {"label": "Outward", "value": "outward"},
                            ],
                            value="inward",
                            size="sm",
                            color="brand",
                            radius="md",
                        ),
                        dmc.Text(
                            id="gripper-orientation-status",
                            size="xs",
                            c="dimmed",
                        ),
                    ],
                    gap=4,
                ),
                dmc.Divider(),
                dmc.Text("Calibration & settings", fw=600),
                dmc.Grid(
                    [
                        dmc.GridCol(
                            dmc.NumberInput(
                                id="settings-marker-a",
                                label="Marker ID A",
                                value=SETTINGS.marker_ids[0],
                                min=0,
                                max=49,
                                size="sm",
                            ),
                            span=4,
                        ),
                        dmc.GridCol(
                            dmc.NumberInput(
                                id="settings-marker-b",
                                label="Marker ID B",
                                value=SETTINGS.marker_ids[1],
                                min=0,
                                max=49,
                                size="sm",
                            ),
                            span=4,
                        ),
                        dmc.GridCol(
                            dmc.NumberInput(
                                id="settings-marker-size",
                                label="Marker size (mm)",
                                value=SETTINGS.marker_size_mm,
                                min=1,
                                max=200,
                                size="sm",
                            ),
                            span=4,
                        ),
                        dmc.GridCol(
                            dmc.TextInput(
                                id="settings-calib-path",
                                label="Calibration bundle path",
                                value=str(DEFAULT_BUNDLE_PATH),
                                size="sm",
                            ),
                            span=12,
                        ),
                        dmc.GridCol(
                            dmc.TextInput(
                                id="settings-workspace-path",
                                label="Workspace path",
                                value=str(DEFAULT_WORKSPACE_PATH),
                                size="sm",
                            ),
                            span=12,
                        ),
                        dmc.GridCol(
                            dmc.TextInput(
                                id="settings-robot-calib-path",
                                label="Robot calibration path",
                                value=str(DEFAULT_ROBOT_CALIBRATION_PATH),
                                size="sm",
                            ),
                            span=12,
                        ),
                    ],
                    gutter="sm",
                ),
                dmc.Text("Auto pick & place defaults", fw=600),
                dmc.Grid(
                    [
                        dmc.GridCol(
                            dmc.TextInput(
                                id="settings-auto-target-label",
                                label="Target label",
                                value=SETTINGS.auto_target_label,
                                size="sm",
                                description="YOLO label the auto routine should pick.",
                            ),
                            span=12,
                        ),
                        dmc.GridCol(
                            dmc.NumberInput(
                                id="settings-auto-drop-ignore-radius",
                                label="Drop ignore radius (mm)",
                                value=SETTINGS.auto_drop_ignore_radius_mm,
                                min=0,
                                max=200,
                                step=5,
                                size="sm",
                            ),
                            span=4,
                        ),
                        dmc.GridCol(
                            dmc.NumberInput(
                                id="settings-auto-intrusion-match",
                                label="Intrusion match (mm)",
                                value=SETTINGS.auto_intrusion_match_mm,
                                min=10,
                                max=200,
                                step=1,
                                size="sm",
                            ),
                            span=4,
                        ),
                        dmc.GridCol(
                            dmc.NumberInput(
                                id="settings-auto-detection-stale",
                                label="Detection stale (s)",
                                value=SETTINGS.auto_detection_stale_s,
                                min=0.05,
                                max=2.0,
                                step=0.05,
                                size="sm",
                            ),
                            span=4,
                        ),
                        dmc.GridCol(
                            dmc.NumberInput(
                                id="settings-auto-idle-sleep",
                                label="Idle sleep (s)",
                                value=SETTINGS.auto_idle_sleep_s,
                                min=0.05,
                                max=1.0,
                                step=0.05,
                                size="sm",
                            ),
                            span=4,
                        ),
                        dmc.GridCol(
                            dmc.Switch(
                                id="settings-auto-estimate-width",
                                label="Estimate width",
                                checked=SETTINGS.auto_estimate_width,
                                size="sm",
                            ),
                            span=4,
                        ),
                        dmc.GridCol(
                            dmc.NumberInput(
                                id="settings-auto-approach-height",
                                label="Approach height (mm)",
                                value=SETTINGS.auto_approach_mm,
                                min=-200,
                                max=0,
                                step=1,
                                size="sm",
                            ),
                            span=4,
                        ),
                        dmc.GridCol(
                            dmc.NumberInput(
                                id="settings-auto-pick-height",
                                label="Pick height (mm)",
                                value=SETTINGS.auto_pick_mm,
                                min=-50,
                                max=200,
                                step=1,
                                size="sm",
                            ),
                            span=4,
                        ),
                        dmc.GridCol(
                            dmc.NumberInput(
                                id="settings-auto-drop-height",
                                label="Drop height (mm)",
                                value=SETTINGS.auto_drop_mm,
                                min=-50,
                                max=200,
                                step=1,
                                size="sm",
                            ),
                            span=4,
                        ),
                    ],
                    gutter="sm",
                ),
                dmc.Text("Pick follow tuning", fw=600),
                dmc.Grid(
                    [
                        dmc.GridCol(
                            dmc.NumberInput(
                                id="settings-pick-follow-timeout",
                                label="Timeout (s)",
                                description="Max follow loop duration.",
                                value=SETTINGS.pick_follow_timeout_s,
                                min=1,
                                max=60,
                                step=1,
                                size="sm",
                            ),
                            span=4,
                        ),
                        dmc.GridCol(
                            dmc.NumberInput(
                                id="settings-pick-follow-interval",
                                label="Update interval (s)",
                                value=SETTINGS.pick_follow_update_interval_s,
                                min=0.01,
                                max=0.5,
                                step=0.01,
                                size="sm",
                            ),
                            span=4,
                        ),
                        dmc.GridCol(
                            dmc.NumberInput(
                                id="settings-pick-follow-segment",
                                label="Segment hold (s)",
                                value=SETTINGS.pick_follow_segment_hold_s,
                                min=0.01,
                                max=1.0,
                                step=0.01,
                                size="sm",
                            ),
                            span=4,
                        ),
                        dmc.GridCol(
                            dmc.NumberInput(
                                id="settings-pick-follow-settle",
                                label="Settle time (s)",
                                value=SETTINGS.pick_follow_settle_s,
                                min=0.01,
                                max=5.0,
                                step=0.05,
                                size="sm",
                            ),
                            span=4,
                        ),
                        dmc.GridCol(
                            dmc.NumberInput(
                                id="settings-pick-follow-tolerance",
                                label="Tolerance (mm)",
                                value=SETTINGS.pick_follow_tolerance_mm,
                                min=0.1,
                                max=20.0,
                                step=0.1,
                                size="sm",
                            ),
                            span=4,
                        ),
                    ],
                    gutter="sm",
                ),
                dmc.Text("Safety guards", fw=600),
                dmc.Grid(
                    [
                        dmc.GridCol(
                            dmc.NumberInput(
                                id="settings-rescue-jog-hold",
                                label="Rescue jog hold (s)",
                                value=SETTINGS.rescue_jog_hold_s,
                                min=0.1,
                                max=1.5,
                                step=0.05,
                                size="sm",
                            ),
                            span=4,
                        ),
                        dmc.GridCol(
                            dmc.NumberInput(
                                id="settings-coord5-max-z",
                                label="Coord5 max Z (mm)",
                                value=SETTINGS.coord5_max_z_mm,
                                description="Positive Z moves downward; clamp to safe range.",
                                min=-50,
                                max=50,
                                step=1,
                                size="sm",
                            ),
                            span=4,
                        ),
                    ],
                    gutter="sm",
                ),
                dmc.Group(
                    [
                        dmc.Button("Save settings",
                                   id="settings-save", color="teal"),
                        dmc.Button("Reset to defaults", id="settings-reset",
                                   variant="outline", color="red"),
                        dmc.Text(id="settings-status", size="sm", c="dimmed"),
                    ],
                    justify="space-between",
                    align="center",
                ),
                dmc.Group(
                    [
                        dmc.Button("Close", id="setup-close",
                                   variant="subtle"),
                    ],
                    justify="flex-end",
                ),
            ],
            gap="sm",
        ),
    )


def build_robot_calibration_modal() -> dmc.Modal:
    return dmc.Modal(
        id="robot-calibration-modal",
        opened=False,
        title="Robot calibration guide",
        centered=True,
        size="lg",
        withCloseButton=False,
        closeOnClickOutside=False,
        children=dmc.Stack(
            [
                dmc.Stepper(
                    id="robot-calibration-modal-stepper",
                    active=0,
                    orientation="vertical",
                    children=[
                        dmc.StepperStep(
                            label="Point 1",
                            description="Indentation 1 (origin)",
                        ),
                        dmc.StepperStep(
                            label="Point 2",
                            description="Indentation 2 (+X)",
                        ),
                        dmc.StepperStep(
                            label="Point 3",
                            description="Indentation 3 (+X/+Y)",
                        ),
                        dmc.StepperStep(
                            label="Verification",
                            description="Safe motion check",
                        ),
                    ],
                ),
                dmc.Card(
                    id="robot-calibration-accuracy-panel",
                    style={"display": "none"},
                    withBorder=True,
                    radius="md",
                    padding="sm",
                    children=dmc.Stack(
                        [
                            dmc.Text(
                                "",
                                id="robot-calibration-accuracy-text",
                                size="sm",
                                c="dimmed",
                            ),
                            dmc.Group(
                                [
                                    dmc.Button(
                                        "Repeat calibration",
                                        id="robot-calibration-retry",
                                        variant="outline",
                                        color="red",
                                        size="xs",
                                        disabled=True,
                                    ),
                                    dmc.Button(
                                        "Proceed to verification",
                                        id="robot-calibration-continue",
                                        color="teal",
                                        size="xs",
                                        disabled=True,
                                    ),
                                ],
                                justify="space-between",
                            ),
                        ],
                        gap="sm",
                    ),
                ),
                dmc.Alert(
                    id="robot-calibration-modal-alert",
                    title="Calibration status",
                    color="blue",
                    variant="light",
                    children="Waiting for calibration to start.",
                ),
                dmc.Group(
                    [
                        dmc.Button(
                            "Run verification move",
                            id="robot-calibration-verify",
                            color="teal",
                            disabled=True,
                        ),
                        dmc.Button(
                            "Close",
                            id="robot-calibration-modal-close",
                            variant="outline",
                        ),
                    ],
                    justify="flex-end",
                ),
            ],
            gap="md",
        ),
    )


def build_emergency_stop_modal() -> dmc.Modal:
    return dmc.Modal(
        id="emergency-stop-modal",
        opened=False,
        centered=True,
        size="xl",
        withCloseButton=False,
        closeOnClickOutside=False,
        overlayProps={"opacity": 0.65, "color": "#000"},
        children=dmc.Stack(
            [
                dmc.Text(
                    "Emergency stop triggered",
                    fw=700,
                    size="xl",
                    c="red",
                ),
                dmc.Text(
                    "The current program has been paused. Clear the physical E-stop "
                    "and acknowledge to continue.",
                    size="sm",
                    c="dimmed",
                ),
                dmc.Alert(
                    id="emergency-stop-modal-text",
                    title="Emergency stop status",
                    color="red",
                    variant="light",
                    children="Emergency stop detected.",
                ),
                dmc.Group(
                    [
                        dmc.Button(
                            "Clear",
                            id="emergency-ack",
                            color="red",
                            size="md",
                            disabled=True,
                        )
                    ],
                    justify="flex-end",
                ),
            ],
            gap="md",
        ),
    )


def build_overspeed_modal() -> dmc.Modal:
    return dmc.Modal(
        id="overspeed-modal",
        opened=False,
        centered=True,
        size="xl",
        withCloseButton=False,
        closeOnClickOutside=False,
        overlayProps={"opacity": 0.65, "color": "#000"},
        children=dmc.Stack(
            [
                dmc.Text(
                    "Overspeed detected",
                    fw=700,
                    size="xl",
                    c="orange",
                ),
                dmc.Text(
                    "The program has been paused. Choose how to proceed.",
                    size="sm",
                    c="dimmed",
                ),
                dmc.Alert(
                    id="overspeed-modal-text",
                    title="Overspeed status",
                    color="orange",
                    variant="light",
                    children="Overspeed alarm detected.",
                ),
                dmc.Group(
                    [
                        dmc.Button(
                            "Clear",
                            id="overspeed-clear",
                            color="orange",
                            size="md",
                        ),
                        dmc.Button(
                            "Stay paused",
                            id="overspeed-stay-paused",
                            color="yellow",
                            size="md",
                            variant="outline",
                        ),
                        dmc.Button(
                            "Clear & resume",
                            id="overspeed-continue",
                            color="teal",
                            size="md",
                            variant="outline",
                        ),
                        dmc.Button(
                            "Cancel & reset",
                            id="overspeed-reset",
                            color="red",
                            size="md",
                            variant="outline",
                        ),
                    ],
                    justify="flex-end",
                ),
            ],
            gap="md",
        ),
    )


def build_joint_protection_modal() -> dmc.Modal:
    return dmc.Modal(
        id="joint-protection-modal",
        opened=False,
        centered=True,
        size="xl",
        withCloseButton=False,
        closeOnClickOutside=False,
        overlayProps={"opacity": 0.65, "color": "#000"},
        children=dmc.Stack(
            [
                dmc.Text(
                    "Joint protection detected",
                    fw=700,
                    size="xl",
                    c="orange",
                ),
                dmc.Text(
                    "The program has been paused. Clear the protection alarm to continue.",
                    size="sm",
                    c="dimmed",
                ),
                dmc.Alert(
                    id="joint-protection-modal-text",
                    title="Joint protection status",
                    color="orange",
                    variant="light",
                    children="Joint protection warning detected.",
                ),
                dmc.Group(
                    [
                        dmc.Button(
                            "Clear",
                            id="joint-protection-clear",
                            color="orange",
                            size="md",
                        ),
                        dmc.Button(
                            "Stay paused",
                            id="joint-protection-stay-paused",
                            color="gray",
                            size="md",
                            variant="outline",
                        ),
                    ],
                    justify="flex-end",
                ),
            ],
            gap="md",
        ),
    )


def build_wrong_tool_modal() -> dmc.Modal:
    return dmc.Modal(
        id="wrong-tool-modal",
        opened=False,
        centered=True,
        size="xl",
        withCloseButton=False,
        closeOnClickOutside=False,
        overlayProps={"opacity": 0.65, "color": "#000"},
        children=dmc.Stack(
            [
                dmc.Text(
                    "Tool/payload mismatch",
                    fw=700,
                    size="xl",
                    c="red",
                ),
                dmc.Text(
                    "The program has been paused. Clear the tool error or enter rescue mode.",
                    size="sm",
                    c="dimmed",
                ),
                dmc.Alert(
                    id="wrong-tool-modal-text",
                    title="Tool error status",
                    color="red",
                    variant="light",
                    children="Tool/payload mismatch detected.",
                ),
                dmc.Group(
                    [
                        dmc.Button(
                            "Enter rescue mode",
                            id="wrong-tool-rescue",
                            color="orange",
                            size="md",
                            variant="outline",
                        ),
                        dmc.Button(
                            "Clear",
                            id="wrong-tool-clear",
                            color="red",
                            size="md",
                        ),
                    ],
                    justify="flex-end",
                ),
            ],
            gap="md",
        ),
    )


def build_drag_not_allowed_modal() -> dmc.Modal:
    return dmc.Modal(
        id="drag-not-allowed-modal",
        opened=False,
        centered=True,
        size="xl",
        withCloseButton=False,
        closeOnClickOutside=False,
        overlayProps={"opacity": 0.65, "color": "#000"},
        children=dmc.Stack(
            [
                dmc.Text(
                    "Drag not allowed",
                    fw=700,
                    size="xl",
                    c="orange",
                ),
                dmc.Text(
                    "Manual dragging is blocked. Clear the warning to continue.",
                    size="sm",
                    c="dimmed",
                ),
                dmc.Alert(
                    id="drag-not-allowed-modal-text",
                    title="Drag warning status",
                    color="orange",
                    variant="light",
                    children="Drag not allowed warning detected.",
                ),
                dmc.Group(
                    [
                        dmc.Button(
                            "Clear",
                            id="drag-not-allowed-clear",
                            color="orange",
                            size="md",
                        ),
                        dmc.Button(
                            "Stay paused",
                            id="drag-not-allowed-stay-paused",
                            color="gray",
                            size="md",
                            variant="outline",
                        ),
                    ],
                    justify="flex-end",
                ),
            ],
            gap="md",
        ),
    )


def _calibration_stepper_stack() -> dmc.Stepper:
    return dmc.Stepper(
        id="calibration-stepper",
        active=0,
        orientation="vertical",
        mt="sm",
        children=[
            dmc.StepperStep(
                label="Camera calibration",
                description="Intrinsics + workspace plane",
                children=dmc.Stack(
                    [
                        dmc.Text(
                            "Capture the ChArUco board frames to calibrate the camera.",
                            size="sm",
                            c="dimmed",
                        ),
                        dmc.Card(
                            dmc.Group(
                                [
                                    dmc.ThemeIcon(
                                        dmc.Text("✕", size="lg"),
                                        size="lg",
                                        radius="md",
                                        color="red",
                                        id="markers-detected-icon",
                                    ),
                                    dmc.Stack(
                                        [
                                            dmc.Text("Markers detected",
                                                     size="sm", fw=600),
                                            dmc.Text(
                                                "Position the ChArUco board in view",
                                                size="xs",
                                                c="dimmed",
                                            ),
                                        ],
                                        gap="xs",
                                    ),
                                ],
                                gap="md",
                            ),
                            p="md",
                            withBorder=True,
                        ),
                        dmc.Button(
                            "Calibrate",
                            id="calibrate-camera",
                            fullWidth=True,
                            disabled=True,
                            color="teal",
                        ),
                        dmc.Button(
                            "Use saved calibration",
                            id="use-existing-calibration",
                            fullWidth=True,
                            variant="outline",
                            disabled=True,
                        ),
                        dmc.Box(
                            id="camera-calibration-action",
                            style={"minHeight": "0px"},
                        ),
                        dmc.Text(
                            id="calibration-progress-text",
                            size="sm",
                            c="dimmed",
                            style={"display": "none",
                                   "marginBottom": "8px"},
                        ),
                        dmc.Progress(
                            id="calibration-progress",
                            value=0,
                            size="sm",
                            striped=True,
                            animated=True,
                            style={"display": "none"},
                        ),
                        dmc.Text(
                            id="camera-calibration-metrics", size="sm"),
                    ],
                    gap="md",
                ),
            ),
            dmc.StepperStep(
                label="Robot calibration",
                description="Detect plate markers",
                children=dmc.Stack(
                    [
                        dmc.Text(
                            "Detect the calibration plate and attach the homography.",
                            size="sm",
                            c="dimmed",
                        ),
                        dmc.Alert(
                            id="plate-detection-summary",
                            title="Plate visibility",
                            color="yellow",
                            children="Waiting for plate markers...",
                            variant="light",
                        ),
                        dmc.Button(
                            "Begin calibrating robot",
                            id="start-robot-calibration",
                            variant="filled",
                            color="teal",
                            disabled=True,
                        ),
                        dmc.Button(
                            "Use saved calibration",
                            id="use-existing-robot-calibration",
                            fullWidth=True,
                            variant="outline",
                            color="brand",
                            disabled=True,
                        ),
                        dmc.Button(
                            "Run verification move",
                            id="run-verification-inline",
                            fullWidth=True,
                            variant="outline",
                            color="teal",
                            disabled=True,
                        ),
                        dmc.Text(id="current-robot-position",
                                 size="xs", c="dimmed"),
                        dmc.Text(id="plate-action-status",
                                 size="sm", c="dimmed"),
                        dmc.Text(id="plate-metrics", size="sm"),
                        dmc.Text(
                            id="robot-calibration-status",
                            size="sm",
                            c="teal",
                        ),
                    ],
                    gap="xs",
                ),
            ),
        ],
    )


def _calibration_loaded_body() -> dmc.Stack:
    return dmc.Stack(
        [
            dmc.Text("Calibration data loaded", fw=600, size="lg"),
            dmc.Text(
                "Camera and robot calibration data already exist. Reset calibration to start over.",
                size="sm",
                c="dimmed",
            ),
            dmc.Text(
                "Loading new calibration artifacts will delete the existing bundle and plate data.",
                size="xs",
                c="dimmed",
            ),
            dmc.Button(
                "Reset calibration",
                id="calibration-reset-button",
                variant="outline",
                color="red",
            ),
            html.Div(id="calibration-metadata"),
        ],
        gap="sm",
    )


def build_stepper_card() -> dmc.Card:
    stepper_visible = SESSION.calibration_menu_open
    stepper_style = {} if stepper_visible else {"display": "none"}
    loaded_style = {} if not stepper_visible else {"display": "none"}
    nav_style = stepper_style
    return dmc.Card(
        [
            dmc.Group(
                [
                    dmc.Text("Calibration Steps", fw=600, size="lg"),
                    dmc.Group(
                        [
                            dmc.Button("Back", id="step-prev",
                                       variant="light", size="sm"),
                            dmc.Button("Next", id="step-next",
                                       size="sm"),
                        ],
                        gap="xs",
                        id="calibration-nav-actions",
                        style=nav_style,
                    ),
                ],
                justify="space-between",
                align="center",
            ),
            dmc.Divider(),
            html.Div(
                id="calibration-stepper-body",
                children=_calibration_stepper_stack(),
                style=stepper_style,
            ),
            html.Div(
                id="calibration-loaded-body",
                children=_calibration_loaded_body(),
                style=loaded_style,
            ),
        ],
        withBorder=True,
        radius="lg",
        className="panel-card",
    )


def build_workspace_step_content() -> dmc.Stack:
    return dmc.Stack(
        [
            dmc.Text(
                "Use the two corner markers to define the workarea size.",
                size="sm",
                c="dimmed",
            ),
            dmc.Button(
                "Compute workspace", id="compute-workspace", variant="light"),
            dmc.Button(
                "Use saved workspace",
                id="use-existing-workspace",
                variant="outline",
                style={"display": "none"},
            ),
            dmc.Button(
                "Clear workspace",
                id="clear-workspace",
                variant="outline",
                color="red",
                size="xs",
            ),
            dmc.Text(id="workspace-action-status", size="sm", c="dimmed"),
            dmc.Text(id="workspace-metrics", size="sm"),
        ],
        gap="xs",
    )


def build_camera_panel(title: str = "Camera View") -> dmc.Card:
    placeholder_src = _encode_frame(
        _placeholder_frame("Camera stream offline"))
    return dmc.Card(
        [
            dmc.Group(
                [
                    dmc.Text(title, fw=600, size="lg"),
                    dmc.Group(
                        [
                            dmc.SegmentedControl(
                                id="overlay-mode-select",
                                data=[
                                    {"label": "Auto", "value": "auto"},
                                    {"label": "Calib", "value": "calibration"},
                                    {"label": "Workspace", "value": "workspace"},
                                    {"label": "Plate", "value": "plate"},
                                    {"label": "Detections", "value": "detection"},
                                    {"label": "Off", "value": "none"},
                                ],
                                value="auto",
                                size="xs",
                                color="brand",
                                radius="md",
                            ),
                            dmc.ActionIcon(
                                html.Span(
                                    "⛶", style={"fontSize": "1.2rem"}, title="Fullscreen"),
                                id="fullscreen-btn",
                                variant="light",
                            ),
                        ],
                        gap="sm",
                    ),
                ],
                justify="space-between",
                align="center",
            ),
            dmc.Text(id="overlay-status", size="sm", c="dimmed"),
            dmc.Text(id="camera-fps", size="sm", c="dimmed"),
            html.Div(
                id="camera-feed-container",
                children=html.Img(
                    id="camera-feed",
                    src="/video_feed",  # Use MJPEG stream to avoid per-frame HTTP polling
                    style={
                        "width": "100%",
                        "height": "auto",
                        "maxHeight": "100%",
                        "objectFit": "contain",
                        "borderRadius": "16px",
                        "border": "1px solid rgba(240, 65, 0, 0.25)",
                        "marginTop": "8px",
                        "backgroundColor": "#0b0f14",
                    },
                ),
                style={"position": "relative",
                       "height": "100%", "overflow": "hidden"}
            ),
        ],
        withBorder=True,
        radius="lg",
        padding="sm",
        className="panel-card camera-panel",
    )


def build_settings_card() -> dmc.Card:
    return dmc.Card(
        [
            dmc.Group(
                [
                    dmc.Text("Settings", fw=600, size="lg"),
                    dmc.Text(id="settings-status", size="sm", c="dimmed"),
                ],
                justify="space-between",
                align="center",
            ),
            dmc.Divider(),
            dmc.Grid(
                [
                    dmc.GridCol(
                        dmc.TextInput(
                            id="settings-robot-host",
                            label="Robot host",
                            placeholder="192.168.101.100",
                            value=SETTINGS.robot_host,
                            size="sm",
                        ),
                        span=3,
                    ),
                    dmc.GridCol(
                        dmc.NumberInput(
                            id="settings-robot-port",
                            label="Robot port",
                            value=SETTINGS.robot_port,
                            size="sm",
                            min=1,
                        ),
                        span=3,
                    ),
                    dmc.GridCol(
                        dmc.NumberInput(
                            id="settings-user-port",
                            label="User WS port",
                            value=SETTINGS.user_port,
                            size="sm",
                            min=1,
                        ),
                        span=3,
                    ),
                    dmc.GridCol(
                        dmc.TextInput(
                            id="settings-gripper-ip",
                            label="Gripper IP",
                            placeholder="192.168.101.180",
                            value=SETTINGS.gripper_ip,
                            size="sm",
                        ),
                        span=3,
                    ),
                    dmc.GridCol(
                        dmc.Select(
                            id="settings-gripper-type",
                            label="Gripper type",
                            data=GRIPPER_PROFILE_OPTIONS,
                            value=SETTINGS.gripper_type,
                            size="sm",
                            clearable=False,
                        ),
                        span=3,
                    ),
                    dmc.GridCol(
                        dmc.NumberInput(
                            id="settings-marker-a",
                            label="Marker ID A",
                            value=SETTINGS.marker_ids[0],
                            min=0,
                            max=49,
                            size="sm",
                        ),
                        span=3,
                    ),
                    dmc.GridCol(
                        dmc.NumberInput(
                            id="settings-marker-b",
                            label="Marker ID B",
                            value=SETTINGS.marker_ids[1],
                            min=0,
                            max=49,
                            size="sm",
                        ),
                        span=3,
                    ),
                    dmc.GridCol(
                        dmc.NumberInput(
                            id="settings-marker-size",
                            label="Marker size (mm)",
                            value=SETTINGS.marker_size_mm,
                            min=1,
                            max=200,
                            size="sm",
                        ),
                        span=3,
                    ),
                    dmc.GridCol(
                        dmc.TextInput(
                            id="settings-calib-path",
                            label="Calibration bundle path",
                            value=str(DEFAULT_BUNDLE_PATH),
                            size="sm",
                        ),
                        span=6,
                    ),
                    dmc.GridCol(
                        dmc.TextInput(
                            id="settings-workspace-path",
                            label="Workspace path",
                            value=str(DEFAULT_WORKSPACE_PATH),
                            size="sm",
                        ),
                        span=6,
                    ),
                    dmc.GridCol(
                        dmc.TextInput(
                            id="settings-robot-calib-path",
                            label="Robot calibration path",
                            value=str(DEFAULT_ROBOT_CALIBRATION_PATH),
                            size="sm",
                        ),
                        span=6,
                    ),
                    dmc.GridCol(
                        dmc.NumberInput(
                            id="settings-camera-calib-frames",
                            label="Camera calib frames",
                            value=SETTINGS.camera_calibration_frame_count,
                            min=1,
                            max=120,
                            step=1,
                            size="sm",
                            description="Frames captured for each camera calibration run.",
                        ),
                        span=4,
                    ),
                    dmc.GridCol(
                        dmc.NumberInput(
                            id="settings-camera-calib-delay",
                            label="Frame delay (s)",
                            value=SETTINGS.camera_calibration_frame_delay_s,
                            min=0.0,
                            max=1.0,
                            step=0.05,
                            size="sm",
                            description="Wait between captures to reduce motion blur.",
                        ),
                        span=4,
                    ),
                    dmc.GridCol(
                        dmc.NumberInput(
                            id="settings-camera-calib-min-usable",
                            label="Min usable frames",
                            value=SETTINGS.camera_calibration_min_usable_frames,
                            min=1,
                            max=30,
                            step=1,
                            size="sm",
                            description="Calibration failures when usable frames below this.",
                        ),
                        span=4,
                    ),
                ],
                gutter="sm",
            ),
            dmc.Group(
                [
                    dmc.Button("Save settings",
                               id="settings-save", color="teal"),
                    dmc.Button("Reset to defaults", id="settings-reset",
                               variant="outline", color="red"),
                ],
                justify="flex-end",
            ),
        ],
        withBorder=True,
        radius="lg",
        className="panel-card",
    )


def build_calibration_tab() -> dmc.Stack:
    return dmc.Stack([build_stepper_card()], gap="md")


def build_detection_tab() -> dmc.Stack:
    gpu_card = dmc.Card(
        [
            dmc.Text("GPU Status", fw=600, size="lg"),
            dmc.Stack(id="gpu-status-display", gap="xs",
                      children=format_gpu_status()),
        ],
        withBorder=True,
        radius="lg",
        className="panel-card",
    )

    model_options = _model_options()
    has_models = len(model_options) > 0

    if has_models:
        model_select_placeholder = "Select a model"
        upload_button_variant = "light"
    else:
        model_select_placeholder = "No models available - upload a model first"
        upload_button_variant = "gradient"  # Highlighted

    model_card = dmc.Card(
        [
            dmc.Text("Model Loader", fw=600, size="lg"),
            dmc.Text(
                "Upload a YOLO model (.pt/.onnx) to enable detection." if not has_models else "Detection starts automatically when a model is loaded.", size="sm", c="dimmed"),
            dcc.Upload(
                id="model-upload",
                children=dmc.Button("Upload YOLO model",
                                    variant=upload_button_variant),
                multiple=False,
            ),
            dmc.Select(
                id="model-select",
                label="Active model",
                data=model_options,
                value=str(
                    SESSION.active_model_path) if SESSION.active_model_path else None,
                placeholder=model_select_placeholder,
                searchable=True,
                clearable=True,
                disabled=not has_models,
            ),
            dmc.Button("Load selected model",
                       id="load-model", variant="outline", disabled=not has_models),
            dmc.Text(id="model-status", size="sm", c="dimmed"),
        ],
        withBorder=True,
        radius="lg",
        className="panel-card",
    )

    detection_card = dmc.Card(
        [
            dmc.Text("Detection Controls", fw=600, size="lg"),
            dmc.Text("Detection runs automatically when a model is loaded.",
                     size="sm", c="dimmed", style={"marginBottom": "10px"}),
            dmc.Group(
                [
                    dmc.NumberInput(
                        id="detection-confidence",
                        label="Confidence",
                        value=0.4,
                        min=0.0,
                        max=1.0,
                        step=0.05,
                    ),
                    dmc.NumberInput(
                        id="detection-iou",
                        label="IOU",
                        value=0.45,
                        min=0.0,
                        max=1.0,
                        step=0.05,
                    ),
                ],
                grow=True,
            ),
            dmc.Group(
                [
                    dmc.NumberInput(
                        id="detection-max",
                        label="Max detections",
                        value=10,
                        min=1,
                        max=200,
                    ),
                    dmc.TextInput(
                        id="detection-classes",
                        label="Class filter (comma IDs)",
                        placeholder="e.g. 0, 32",
                    ),
                ],
                grow=True,
            ),
            dmc.Text(id="detection-status", size="sm", c="dimmed"),
            dmc.Text(id="detection-metrics", size="sm"),
        ],
        withBorder=True,
        radius="lg",
        className="panel-card",
    )

    detections_card = dmc.Card(
        [
            dmc.Text("Detected Objects", fw=600, size="lg"),
            dmc.Stack(id="detection-list", gap="xs"),
        ],
        withBorder=True,
        radius="lg",
        className="panel-card",
    )

    return dmc.Stack([gpu_card, model_card, detection_card, detections_card], gap="md")


def build_pick_place_tab() -> dmc.Stack:
    # Improved drop location controls with better button styling
    drop_card = dmc.Card(
        [
            dmc.Text("Drop Target", fw=600, size="lg"),
            dmc.Text("First teach a drop position, then use return to go back to it.",
                     size="sm", c="dimmed"),
            dmc.Button(
                "Teach drop position (use current robot pose)",
                id="teach-drop-position",
                variant="filled",
                className="action-btn",
                color="green",
            ),
            dmc.Button(
                "↩ Return to drop position",
                id="return-to-drop-position",
                variant="outline",
                className="action-btn-outline",
            ),
            dmc.Text(id="drop-position-metrics", size="sm"),
            dmc.NumberInput(
                id="drop-height",
                label="Drop height offset (mm)",
                value=10.0,
                min=-50,
                max=200,
                description="Height above taught drop position (positive = higher)",
            ),
        ],
        withBorder=True,
        radius="lg",
        className="panel-card",
    )

    # Merged Pick/Drop Actions (combines manual control with action card)
    action_card = dmc.Card(
        [
            dmc.Text("Pick & Drop Controls", fw=600, size="lg"),
            build_tracking_mode_controls(),  # Add shared tracking mode controls
            # Hidden select kept for callbacks that expect a pick target selector
            dmc.Select(
                id="pick-detection-select",
                label="Detection to pick",
                data=[],
                value=None,
                placeholder="(hidden) Pick detection",
                clearable=True,
                style={"display": "none"},
            ),
            dmc.Group(
                [
                    dmc.Button("Pick closest object", id="manual-pick-closest",
                               variant="filled", color="teal"),
                    dmc.Button("Drop", id="execute-drop", variant="light"),
                ],
                grow=True,
            ),
            dmc.Space(h="md"),  # Add space for visual separation
            dmc.Button("🔄 Pick & Drop closest",
                       id="manual-pick-place-target",
                       variant="filled",
                       color="blue",
                       size="md"),
            dmc.Group([
                dmc.Button("Reset if stuck",
                           id="reset-operation-lock",
                           variant="subtle",
                           color="red",
                           size="xs"),
            ], justify="center", style={"marginTop": "8px"}),
            dmc.Text(id="pick-place-status", size="sm", c="dimmed"),
        ],
        withBorder=True,
        radius="lg",
        className="panel-card",
    )

    plate_test_card = dmc.Card(
        [
            dmc.Text("Plate Pick & Place Test", fw=600, size="lg"),
            dmc.Text(
                "Uses calibration plate targets 1-3 and re-tracks the plate pose before running.",
                size="sm",
                c="dimmed",
            ),
            build_tracking_mode_controls(),  # Use shared tracking controls
            dmc.Group(
                [
                    dmc.NumberInput(
                        id="plate-test-approach-height",
                        label="Approach height (mm)",
                        description="Negative = UP from table (safe in coord5)",
                        value=-30.0,
                        min=-200,
                        max=0,
                    ),
                    dmc.NumberInput(
                        id="plate-test-pick-height",
                        label="Pick height (mm)",
                        value=0.0,
                        min=-50,
                        max=200,
                    ),
                ],
                grow=True,
            ),
            dmc.Group(
                [
                    dmc.NumberInput(
                        id="plate-test-drop-height",
                        label="Drop height offset (mm)",
                        value=10.0,
                        min=-50,
                        max=200,
                        description="Height above taught drop position",
                    ),
                    dmc.NumberInput(
                        id="plate-test-grip-timeout",
                        label="Grip timeout (s)",
                        value=SETTINGS.plate_test_grip_timeout_s,
                        min=0.5,
                        max=10.0,
                        step=0.5,
                    ),
                ],
                grow=True,
            ),
            dmc.Text(
                f"Gripper close uses {GRIPPER_FORCE_MIN} N and {GRIPPER_SPEED_MIN}% speed.",
                size="xs",
                c="dimmed",
            ),
            dmc.Button(
                "Run plate test",
                id="plate-test-run",
                color="teal",
            ),
            dmc.Text(id="plate-test-status", size="sm", c="dimmed"),
            dcc.Textarea(
                id="plate-test-analysis",
                value="",
                readOnly=True,
                style={
                    "width": "100%",
                    "minHeight": "120px",
                    "marginTop": "6px",
                },
            ),
        ],
        withBorder=True,
        radius="lg",
        className="panel-card",
    )

    auto_card = dmc.Card(
        [
            dmc.Text("Auto pick & place", fw=600, size="lg"),
            dmc.Text(
                "Auto picks the nearest target and pauses if another object enters the workspace.",
                size="sm",
                c="dimmed",
            ),
            dmc.Group(
                [
                    dmc.TextInput(
                        id="auto-target-class",
                        label="Target class",
                        value=SETTINGS.auto_target_label,
                        description="YOLO label to pick (case-insensitive).",
                    ),
                    dmc.NumberInput(
                        id="auto-drop-ignore-radius",
                        label="Drop ignore radius (mm)",
                        value=SETTINGS.auto_drop_ignore_radius_mm,
                        min=0,
                        max=200,
                        step=5,
                        description="Detections inside this radius around the drop zone are ignored.",
                    ),
                ],
                grow=True,
            ),
        ],
        withBorder=True,
        radius="lg",
        className="panel-card",
    )

    accordion_items = [
        dmc.AccordionItem(
            [
                dmc.AccordionControl("Detect"),
                dmc.AccordionPanel(build_detection_tab()),
            ],
            value="detect",
        ),
        dmc.AccordionItem(
            [
                dmc.AccordionControl("Drop Location"),
                dmc.AccordionPanel(drop_card),
            ],
            value="drop-target",
        ),
        dmc.AccordionItem(
            [
                dmc.AccordionControl("Auto pick & place"),
                dmc.AccordionPanel(auto_card),
            ],
            value="auto",
        ),
        dmc.AccordionItem(
            [
                dmc.AccordionControl("Plate Pick & Place Test"),
                dmc.AccordionPanel(plate_test_card),
            ],
            value="plate-test",
        ),
    ]

    return dmc.Stack(
        [
            dmc.Accordion(
                accordion_items,
                multiple=True,
                value=[],
                variant="separated",
            ),
            action_card,  # Pick & Drop Controls (merged from manual control)
        ],
        gap="md",
    )


def build_application_tab() -> dmc.Stack:
    return dmc.Stack(
        [
            dmc.Tabs(
                [
                    dmc.TabsList(
                        [
                            dmc.TabsTab("Workspace", value="workspace"),
                            dmc.TabsTab("Pick & Place", value="pick-place"),
                        ],
                        style={
                            "width": "100%",
                            "justifyContent": "flex-start",
                            "marginBottom": "12px",
                        },
                    ),
                    dmc.TabsPanel(
                        build_workspace_step_content(), value="workspace"
                    ),
                    dmc.TabsPanel(
                        build_pick_place_tab(), value="pick-place"
                    ),
                ],
                id="application-tabs",
                value="pick-place",
                variant="pills",
                color="brand",
                style={"width": "100%"},
            )
        ],
        gap="md",
    )


def build_monitoring_tab() -> dmc.Stack:
    summary_card = dmc.Card(
        [
            dmc.Text("System Summary", fw=600, size="lg"),
            dmc.Stack(
                [
                    dmc.Text("Camera: off", size="sm"),
                    dmc.Text("Robot: offline", size="sm"),
                    dmc.Text("Gripper: offline", size="sm"),
                    dmc.Text("Calibration bundle: missing", size="sm"),
                    dmc.Text("Workspace: pending", size="sm"),
                ],
                id="monitor-system-summary",
                gap="xs",
            ),
        ],
        withBorder=True,
        radius="lg",
        className="panel-card",
    )
    detection_card = dmc.Card(
        [
            dmc.Text("Detection Summary", fw=600, size="lg"),
            dmc.Stack(
                [
                    dmc.Text("Model: none", size="sm"),
                    dmc.Text("Detections: 0", size="sm"),
                ],
                id="monitor-detection-summary",
                gap="xs",
            ),
        ],
        withBorder=True,
        radius="lg",
        className="panel-card",
    )
    pick_card = dmc.Card(
        [
            dmc.Text("Pick & Drop Summary", fw=600, size="lg"),
            dmc.Stack(
                [
                    dmc.Text("Holding object: no", size="sm"),
                    dmc.Text("Last action: No recent actions.", size="sm"),
                    dmc.Text("Processed objects: 0", size="sm"),
                    dmc.Text("Attempts: 0 · Failures: 0", size="sm"),
                    dmc.Text("Success rate: n/a", size="sm"),
                    dmc.Text("Avg cycle: n/a", size="sm"),
                    dmc.Text("Last cycle: n/a", size="sm"),
                ],
                id="monitor-pick-summary",
                gap="xs",
            ),
        ],
        withBorder=True,
        radius="lg",
        className="panel-card",
    )

    return dmc.Stack([summary_card, detection_card, pick_card], gap="md")


def build_layout() -> dmc.MantineProvider:
    brand_palette = [
        "#ffe8de",
        "#ffcdbd",
        "#ffac91",
        "#ff8a61",
        "#ff6b36",
        "#f04100",
        "#d23a00",
        "#b03200",
        "#8a2700",
        "#611a00",
    ]
    theme = {
        "fontFamily": "Space Grotesk, IBM Plex Sans, sans-serif",
        "headings": {"fontFamily": "Space Grotesk, IBM Plex Sans, sans-serif"},
        "primaryColor": "brand",
        "colors": {"brand": brand_palette},
    }
    return dmc.MantineProvider(
        theme=theme,
        withGlobalClasses=True,
        withCssVariables=True,
        children=html.Div(
            [
                dmc.NotificationContainer(
                    id="app-notifications", position="top-right", limit=3),
                dcc.Store(id="camera-connection-state"),
                dcc.Store(id="robot-connection-state"),
                dcc.Store(
                    id="gripper-connection-state",
                    data={"status": "offline", "message": ""},
                ),
                dcc.Store(
                    id="notification-state",
                    data={"camera": None, "robot": None},
                ),
                dcc.Store(id="calibration-progress-store",
                          data={"progress": 0}),
                dcc.Store(
                    id="calibration-menu-open-store",
                    data=SESSION.calibration_menu_open,
                ),
                dcc.Store(
                    id="plate-detection-state",
                    data={
                        "count": 0,
                        "ok": False,
                        "stable": False,
                        "center_px": None,
                        "movement_px": None,
                        "timestamp": 0.0,
                        "message": "",
                    },
                ),
                dcc.Store(
                    id="robot-calibration-state",
                    data={
                        "status": "idle",
                        "message": "",
                        "step": 0,
                        "verification_ready": False,
                        "accuracy": None,
                        "accuracy_ready": False,
                        "accuracy_passed": False,
                    },
                ),
                dcc.Store(
                    id="auto-operation-state",
                    data={"running": False, "message": ""},
                ),
                dcc.Store(
                    id="auto-config-state",
                    data={
                        "target_label": SETTINGS.auto_target_label,
                        "drop_ignore_radius_mm": SETTINGS.auto_drop_ignore_radius_mm,
                    },
                ),
                dcc.Store(
                    id="robot-operation-state",
                    data=_operation_snapshot(),
                ),
                dcc.Store(
                    id="robot-error-state",
                    data=_error_snapshot(),
                ),
                dcc.Store(
                    id="robot-rescue-state",
                    data={"active": False, "timestamp": 0.0},
                ),
                dcc.Store(
                    id="prerequisites-state",
                    data={
                        "camera_calibrated": False,
                        "workspace_defined": False,
                        "robot_calibrated": False,
                        "model_loaded": False,
                        "gripper_ready": False,
                        "drop_target_defined": False,
                    },
                ),
                dcc.Store(
                    id="footer-visibility",
                    data={"collapsed": False},
                ),
                dcc.Store(
                    id="robot-speed-state",
                    data={"value": 25},
                ),
                dcc.Store(
                    id="robot-power-state",
                    data={"value": None, "timestamp": 0.0},
                ),
                dcc.Interval(id="auto-init", interval=400,
                             n_intervals=0, max_intervals=1),
                dcc.Interval(id="robot-status-interval",
                             interval=400, n_intervals=0),
                dcc.Interval(
                    id="monitor-interval",
                    interval=1000,
                    n_intervals=0,
                ),
                dcc.Interval(
                    id="camera-interval",
                    interval=33,  # ~30 FPS UI poll; stream push happens each tick
                    n_intervals=0,
                    disabled=True,
                ),
                dcc.Interval(id="calibration-progress-interval",
                             interval=100, n_intervals=0),
                html.Div(id="camera-refresh-container",
                         style={"display": "none"}),
                build_setup_drawer(),
                build_robot_calibration_modal(),
                build_emergency_stop_modal(),
                build_overspeed_modal(),
                build_joint_protection_modal(),
                build_wrong_tool_modal(),
                build_drag_not_allowed_modal(),
                dmc.AppShell(
                    padding="md",
                    className="app-shell",
                    header={"height": 90},
                    footer={"height": 340},
                    children=[
                        build_header(),
                        dmc.AppShellMain(
                            dmc.Container(
                                dmc.Grid(
                                    [
                                        dmc.GridCol(
                                            html.Div(
                                                dmc.Tabs(
                                                    [
                                                        dmc.TabsList(
                                                            [
                                                                dmc.TabsTab(
                                                                    "Applications", value="applications"),
                                                                dmc.TabsTab(
                                                                    "Monitoring", value="monitoring"),
                                                                dmc.TabsTab(
                                                                    "Calibration", value="calibration"),
                                                            ],
                                                            style={
                                                                "width": "100%",
                                                                "justifyContent": "flex-start",
                                                                "marginTop": "0px",
                                                                "marginBottom": "12px",
                                                            },
                                                        ),
                                                        dmc.TabsPanel(
                                                            build_application_tab(), value="applications"),
                                                        dmc.TabsPanel(
                                                            build_monitoring_tab(), value="monitoring"),
                                                        dmc.TabsPanel(
                                                            build_calibration_tab(), value="calibration"),
                                                    ],
                                                    id="main-tabs",
                                                    value="applications",
                                                    variant="pills",
                                                    color="brand",
                                                    className="main-tabs",
                                                    style={
                                                        "width": "100%",
                                                        "marginBottom": "6px",
                                                    },
                                                ),
                                                className="tabs-scroll-area",
                                            ),
                                            span={"base": 12, "md": 4},
                                            className="tabs-column",
                                        ),
                                        dmc.GridCol(
                                            dmc.Stack(
                                                [build_camera_panel()], gap="md"),
                                            span={"base": 12, "md": 8},
                                            className="camera-column",
                                        ),
                                    ],
                                    gutter="md",
                                    className="layout-grid",
                                ),
                                fluid=True,
                                px=0,
                                mx=0,
                                mt="sm",
                                className="main-panel",
                            ),
                            id="app-main",
                            className="app-main footer-expanded",
                            style={"overflow": "hidden"},
                        ),
                        build_footer(),
                    ],
                ),
            ]
        ),
    )


def _format_device_list(devices: list[object]) -> html.Ul:
    if not devices:
        return html.Ul([html.Li("No OAK-D devices detected.")])
    items = []
    for idx, device in enumerate(devices, start=1):
        name = None
        mx_id_fn = getattr(device, "getMxId", None)
        device_id_fn = getattr(device, "getDeviceId", None)
        if callable(mx_id_fn):
            name = mx_id_fn()
        elif callable(device_id_fn):
            name = device_id_fn()
        else:
            name = str(device)
        state = getattr(device, "state", None)
        items.append(html.Li(f"{idx}. {name} ({state})"))
    return html.Ul(items)


async def _probe_robot(uri: str) -> dict:
    session = _get_robot_session()
    return await session.probe(uri or DEFAULT_ROBOT_URI)


async def _with_robot_api(uri: str, action: Callable[[CodroidAPI], Awaitable[None]]) -> None:
    session = _get_robot_session()
    await session.run_with_robot(uri or DEFAULT_ROBOT_URI, action)


async def _ensure_robot_coord5_for_uri(uri: Optional[str]) -> None:
    """Switch the robot to CoordinateId 5 via the provided URI."""
    async def _activate(api: CodroidAPI) -> None:
        await _ensure_coordinate5_active(api)

    await _with_robot_api(uri or DEFAULT_ROBOT_URI, _activate)


def _run_charuco_task(board_path: Path) -> Callable[[np.ndarray], OverlayResult]:
    def _runner(frame: np.ndarray) -> OverlayResult:
        vis, status, metric = draw_charuco_overlay(frame, board_path)
        _apply_workspace_overlay(vis)
        return OverlayResult(vis, status, metric)

    return _runner


def _run_workspace_task(marker_ids: Tuple[int, int], marker_size_mm: float) -> Callable[[np.ndarray], OverlayResult]:
    def _runner(frame: np.ndarray) -> OverlayResult:
        vis, status, metric = draw_workspace_overlay(
            frame, marker_ids, marker_size_mm)
        _apply_workspace_overlay(vis)
        return OverlayResult(vis, status, metric)

    return _runner


def _run_plate_task(layout_path: Path) -> Callable[[np.ndarray], OverlayResult]:
    def _runner(frame: np.ndarray) -> OverlayResult:
        vis, status, metric = draw_plate_overlay(frame, layout_path)
        _apply_workspace_overlay(vis)
        return OverlayResult(vis, status, metric)

    return _runner


def _run_detection_task() -> Callable[[np.ndarray], OverlayResult]:
    def _runner(frame: np.ndarray) -> OverlayResult:
        vis, status, metric = draw_detection_overlay(frame)
        _apply_workspace_overlay(vis)
        return OverlayResult(vis, status, metric)

    return _runner


def _run_drop_marker_task(marker_id: int) -> Callable[[np.ndarray], OverlayResult]:
    def _runner(frame: np.ndarray) -> OverlayResult:
        vis, status, metric = draw_drop_marker_overlay(frame, marker_id)
        _apply_workspace_overlay(vis)
        return OverlayResult(vis, status, metric)

    return _runner


def _select_overlay_task(
    tab: str,
    calibration_step: int,
    application_tab: Optional[str],
    board_path: Path,
    marker_ids: Tuple[int, int],
    marker_size_mm: float,
    plate_path: Path,
) -> Optional[OverlayTask]:
    if tab == "calibration":
        if calibration_step == 0:
            return OverlayTask("calibration-charuco", _run_charuco_task(board_path))
        return OverlayTask("calibration-plate", _run_plate_task(plate_path))
    if tab == "applications":
        if application_tab == "workspace":
            return OverlayTask(
                "application-workspace",
                _run_workspace_task(marker_ids, marker_size_mm),
            )
        if application_tab == "pick-place":
            return OverlayTask("pick-place-detection", _run_detection_task())
    return None


def register_callbacks(app: Dash) -> None:
    # Client-side toggle for the Setup drawer (faster UI response).
    app.clientside_callback(
        """
        function(openClicks, closeClicks, opened) {
            const ctx = dash_clientside.callback_context;
            if (!ctx.triggered || ctx.triggered.length === 0) {
                return opened;
            }
            const trigger = ctx.triggered[0].prop_id.split('.')[0];
            if (trigger === "open-setup") {
                return true;
            }
            if (trigger === "setup-close") {
                return false;
            }
            // Fallback: compare counts.
            const o = openClicks || 0;
            const c = closeClicks || 0;
            if (o > c) return true;
            if (c > o) return false;
            return opened;
        }
        """,
        Output("setup-drawer", "opened"),
        Input("open-setup", "n_clicks"),
        Input("setup-close", "n_clicks"),
        State("setup-drawer", "opened"),
        prevent_initial_call=False,
    )

    # Auto-refresh camera image (enabled only when camera is live)
    # (Client refresh removed; MJPEG stream handles updates)

    @app.callback(
        Output("calibration-stepper", "active"),
        Input("step-prev", "n_clicks"),
        Input("step-next", "n_clicks"),
        Input("use-existing-calibration", "n_clicks"),
        State("calibration-stepper", "active"),
    )
    def _update_stepper(
        prev_clicks: int,
        next_clicks: int,
        use_saved_calib: int,
        active: Optional[int],
    ) -> int:
        current = active or 0
        trigger = callback_context.triggered_id
        if trigger == "step-next":
            return min(current + 1, 1)
        if trigger == "step-prev":
            return max(current - 1, 0)
        if trigger == "use-existing-calibration":
            return 1  # Advance to robot calibration step
        return current

    @app.callback(
        Output("calibration-stepper-body", "style"),
        Output("calibration-loaded-body", "style"),
        Output("calibration-nav-actions", "style"),
        Input("calibration-menu-open-store", "data"),
    )
    def _toggle_calibration_menu(opened: bool) -> Tuple[dict[str, str], dict[str, str], dict[str, str]]:
        stepper_style = {} if opened else {"display": "none"}
        loaded_style = {} if not opened else {"display": "none"}
        return stepper_style, loaded_style, stepper_style

    @app.callback(
        Output("calibration-menu-open-store", "data"),
        Input("calibration-reset-button", "n_clicks"),
        Input("calibration-progress-store", "data"),
        Input("robot-calibration-state", "data"),
        State("calibration-menu-open-store", "data"),
    )
    def _sync_calibration_menu_state(
        reset_clicks: Optional[int],
        _progress: dict,
        _robot_state: dict,
        current_open: Optional[bool],
    ) -> bool | None:
        trigger = callback_context.triggered_id
        if trigger == "calibration-reset-button" and reset_clicks:
            _reset_calibration_assets()
            return True
        desired = not _calibration_data_complete()
        if current_open is None or desired == current_open:
            return no_update
        return desired

    @app.callback(
        Output("calibration-metadata", "children"),
        Input("calibration-menu-open-store", "data"),
        Input("calibration-progress-store", "data"),
        Input("robot-calibration-state", "data"),
    )
    def _update_calibration_metadata(
        opened: Optional[bool],
        _progress: dict,
        _robot_state: dict,
    ) -> list[Any] | str:
        if not _calibration_data_complete():
            return ""
        entries = _calibration_metadata_entries()
        if not entries:
            return dmc.Text("Calibration metadata unavailable.", size="xs", c="dimmed")
        return [dmc.Text(entry, size="xs", c="dimmed") for entry in entries]

    @app.callback(
        Output("camera-device-info", "children"),
        Input("camera-scan", "n_clicks"),
        prevent_initial_call=True,
    )
    def _scan_camera(_clicks: int) -> html.Ul:
        devices = list_oak_devices()
        return _format_device_list(devices)

    @app.callback(
        Output("camera-status-text", "children"),
        Output("header-camera-badge", "children"),
        Output("header-camera-badge", "color"),
        Output("camera-enable", "checked"),
        Output("camera-connection-state", "data"),
        Output("camera-interval", "disabled"),
        Input("camera-enable", "checked"),
        Input("auto-init", "n_intervals"),
        Input("camera-lock-af", "checked"),
        State("camera-width", "value"),
        State("camera-height", "value"),
        State("camera-fps-input", "value"),
    )
    def _toggle_camera(
        checked: bool,
        _init: int,
        lock_af: Optional[bool],
        width: Optional[float],
        height: Optional[float],
        fps: Optional[float],
    ) -> Tuple[str, str, str, bool, dict, bool]:
        trigger = callback_context.triggered_id
        desired_checked = checked
        stream_running = SESSION.camera_stream is not None and SESSION.camera_stream.is_running()

        if trigger == "camera-lock-af" and stream_running:
            try:
                assert SESSION.camera_stream is not None
                SESSION.camera_stream.set_focus_lock(bool(lock_af))
                state = "locked" if lock_af else "unlocked"
                status = f"Camera stream running; focus {state}."
                return (
                    status,
                    "Camera: live",
                    "brand",
                    True,
                    {"status": "live", "message": status},
                    False,
                )
            except Exception as exc:  # noqa: BLE001
                error = f"Camera focus control error: {exc}"
                return (
                    error,
                    "Camera: error",
                    "red",
                    True,
                    {"status": "error", "message": error},
                    False,
                )

        if trigger == "auto-init":
            desired_checked = True
        if desired_checked:
            try:
                status = start_camera_stream(width, height, fps, bool(lock_af))
                return (
                    status,
                    "Camera: live",
                    "brand",
                    True,
                    {"status": "live", "message": status},
                    False,
                )
            except Exception as exc:
                stop_camera_stream()
                error = f"Camera error: {exc}"
                return (
                    error,
                    "Camera: error",
                    "yellow",
                    False,
                    {"status": "error", "message": error},
                    True,
                )
        stop_camera_stream()
        stopped = "Camera stream stopped."
        return (
            stopped,
            "Camera: off",
            "gray",
            False,
            {"status": "off", "message": stopped},
            True,
        )

    @app.callback(
        Output("robot-speed-slider", "value"),
        Input("robot-speed-decrease", "n_clicks"),
        Input("robot-speed-increase", "n_clicks"),
        State("robot-speed-slider", "value"),
        prevent_initial_call=True,
    )
    def _nudge_robot_speed(dec_clicks: Optional[int], inc_clicks: Optional[int], current: Optional[float]) -> float:
        current_value = float(current or 0.0)
        trigger = callback_context.triggered_id
        step = 5.0
        if trigger == "robot-speed-decrease":
            return max(0.0, current_value - step)
        if trigger == "robot-speed-increase":
            return min(100.0, current_value + step)
        return current_value

    @app.callback(
        Output("robot-speed-state", "data"),
        Input("robot-speed-slider", "value"),
        State("robot-connection-state", "data"),
        State("robot-uri", "value"),
        prevent_initial_call=True,
    )
    async def _apply_robot_speed(
        value: Optional[float],
        robot_state: Optional[dict],
        robot_uri: Optional[str],
    ) -> dict:
        if value is None:
            return no_update
        rate = max(0.0, min(1.0, float(value) / 100.0))
        now = time.time()
        if abs(rate - SESSION.robot_speed_rate) < 0.001:
            return no_update

        SESSION.robot_speed_rate = rate
        if (robot_state or {}).get("status") != "online":
            LOGGER.info("Robot speed requested: %.0f%% (robot offline)", value)
            return {"status": "offline", "value": value, "rate": rate, "updated": now}

        async def _set_speed(api: CodroidAPI) -> None:
            await api.set_manual_move_rate(rate)

        try:
            await _with_robot_api(robot_uri or DEFAULT_ROBOT_URI, _set_speed)
            SESSION.robot_speed_last_set = now
            SESSION.last_action = f"Robot speed set to {rate:.2f}"
            LOGGER.info("Robot speed set to %.0f%%", value)
            return {"status": "ok", "value": value, "rate": rate, "updated": now}
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("Failed to set robot speed")
            return {"status": "error", "value": value, "rate": rate, "error": str(exc), "updated": now}

    @app.callback(
        Output("robot-status-text", "children"),
        Output("header-robot-badge", "children"),
        Output("header-robot-badge", "color"),
        Output("robot-connection-state", "data"),
        Input("robot-test", "n_clicks"),
        Input("robot-disconnect", "n_clicks"),
        Input("auto-init", "n_intervals"),
        State("robot-uri", "value"),
        prevent_initial_call=True,
    )
    async def _test_robot(
        _clicks: int,
        _disconnect_clicks: int,
        _init: int,
        uri: str,
    ) -> Tuple[str, str, str, dict]:
        trigger = callback_context.triggered_id
        session = _get_robot_session()
        if trigger == "robot-disconnect":
            await session.close()
            SESSION.robot_connection_attempted = False
            SESSION.robot_last_connection_error = None
            session.clear_position()
            _set_robot_guidance(
                "Robot calibration idle. Click 'Begin calibrating robot' to start.",
                status="idle",
                step=0,
                verification_ready=False,
            )
            status = "Robot disconnected."
            return status, "Robot: offline", "gray", {"status": "offline", "message": status}
        if trigger == "auto-init":
            if session.is_connected(uri or DEFAULT_ROBOT_URI):
                target = session.resolve_robot_target(uri or DEFAULT_ROBOT_URI)
                status = f"Robot connected to {target['host']}:{target['port']}."
                SESSION.robot_connection_attempted = True
                SESSION.robot_last_connection_error = None
                return status, "Robot: online", "brand", {"status": "online", "message": status}
            if SESSION.robot_connection_attempted:
                if SESSION.robot_last_connection_error:
                    error = SESSION.robot_last_connection_error
                    return error, "Robot: error", "yellow", {"status": "error", "message": error}
                return no_update, no_update, no_update, no_update
        try:
            response = await _probe_robot(uri)
            if not response:
                raise RuntimeError("Robot probe returned no response.")
            host = response.get("host", "unknown")
            port = response.get("port", "unknown")
            status = f"Robot connected to {host}:{port}."
            SESSION.robot_connection_attempted = True
            SESSION.robot_last_connection_error = None
            return status, "Robot: online", "brand", {"status": "online", "message": status}
        except Exception as exc:
            error = f"Robot connection failed: {exc}"
            SESSION.robot_connection_attempted = True
            SESSION.robot_last_connection_error = error
            return error, "Robot: error", "yellow", {"status": "error", "message": error}

    def _command_gripper(
        target_width: float,
        action_label: str,
        *,
        force: Optional[int] = None,
        speed: Optional[int] = None,
        prevent_overclose: bool = False,
    ) -> str:
        """Move gripper to target width and return status text."""
        if SESSION.gripper is None or not SESSION.gripper.isConnected(t_index=GRIPPER_DEFAULTS["t_index"]):
            return "Gripper is not connected. Enable gripper in Setup menu."

        try:
            safe_width = float(target_width)
            min_width, max_width = _resolve_gripper_limits()
            if min_width > max_width:
                min_width, max_width = max_width, min_width
            if prevent_overclose:
                current_width = SESSION.gripper.get_ext_width(
                    t_index=GRIPPER_DEFAULTS["t_index"]
                )
                if isinstance(current_width, (int, float)) and current_width >= 0:
                    if safe_width < current_width:
                        safe_width = current_width
            safe_width = min(max(safe_width, min_width), max_width)

            applied_force = _safe_int(force, GRIPPER_FORCE_DEFAULT)
            applied_force = min(
                max(applied_force, GRIPPER_FORCE_MIN), GRIPPER_FORCE_MAX
            )
            applied_speed = _safe_int(speed, 80)

            SESSION.gripper.grip(
                t_index=GRIPPER_DEFAULTS["t_index"],
                t_width=safe_width,
                n_force=applied_force,
                p_speed=applied_speed,
                f_wait=True,
            )
            SESSION.last_action = (
                f"Gripper {action_label} to {safe_width:.1f} mm @ "
                f"{applied_force} N / {applied_speed}% speed."
            )
            return SESSION.last_action
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("Gripper command failed")
            return f"Gripper action failed: {exc}"

    def _resolve_gripper_limits() -> Tuple[float, float]:
        min_width = SESSION.gripper_min_width
        max_width = SESSION.gripper_max_width
        if SESSION.gripper is not None:
            t_index = GRIPPER_DEFAULTS["t_index"]
            status_snapshot = SESSION.gripper.get_status_snapshot(
                t_index=t_index)
            if status_snapshot:
                snapshot_min = status_snapshot.get("min_external_width")
                snapshot_max = status_snapshot.get("max_external_width")
                if isinstance(snapshot_min, (int, float)) and snapshot_min >= 0:
                    min_width = float(snapshot_min)
                    SESSION.gripper_min_width = min_width
                if isinstance(snapshot_max, (int, float)) and snapshot_max >= 0:
                    max_width = float(snapshot_max)
                    SESSION.gripper_max_width = max_width
            if min_width is None:
                try:
                    value = SESSION.gripper.get_min_ext_width(t_index=t_index)
                except Exception as exc:  # noqa: BLE001
                    LOGGER.debug("Failed to read gripper min width: %s", exc)
                else:
                    if isinstance(value, (int, float)) and value >= 0:
                        min_width = float(value)
            if max_width is None:
                try:
                    value = SESSION.gripper.get_max_ext_width(t_index=t_index)
                except Exception as exc:  # noqa: BLE001
                    LOGGER.debug("Failed to read gripper max width: %s", exc)
                else:
                    if isinstance(value, (int, float)) and value >= 0:
                        max_width = float(value)
        if min_width is None:
            min_width = GRIPPER_CLOSE_WIDTH_DEFAULT
        if max_width is None:
            max_width = GRIPPER_OPEN_WIDTH_DEFAULT
        if min_width > max_width:
            LOGGER.warning(
                "Gripper width limits inverted (min=%.1f, max=%.1f); swapping.",
                min_width,
                max_width,
            )
            min_width, max_width = max_width, min_width
        return min_width, max_width

    def _prepare_gripper_for_calibration() -> None:
        """Set gripper to calibration defaults and remember prior width."""
        SESSION.robot_calibration_gripper_width = None
        gripper = SESSION.gripper
        if gripper is not None:
            try:
                width = gripper.get_ext_width(
                    t_index=GRIPPER_DEFAULTS["t_index"])
            except Exception as exc:  # noqa: BLE001
                LOGGER.debug(
                    "Unable to read gripper width before calibration: %s", exc)
            else:
                if isinstance(width, (int, float)):
                    SESSION.robot_calibration_gripper_width = float(width)
        response = _command_gripper(
            GRIPPER_CALIBRATION_WIDTH_MM,
            "prepping for calibration",
            force=GRIPPER_FORCE_MIN,
            speed=GRIPPER_SPEED_MIN,
        )
        LOGGER.debug("Calibration gripper prep: %s", response)

    def _restore_calibration_gripper_width() -> None:
        """Restore gripper width saved prior to calibration."""
        width = SESSION.robot_calibration_gripper_width
        if width is None:
            return
        response = _command_gripper(
            width,
            "restored after calibration",
            force=GRIPPER_FORCE_DEFAULT,
            speed=GRIPPER_SPEED_DEFAULT,
            prevent_overclose=True,
        )
        LOGGER.debug("Calibration gripper restored: %s", response)
        SESSION.robot_calibration_gripper_width = None

    @app.callback(
        Output("gripper-status-text", "children"),
        Output("gripper-device-info", "children"),
        Output("header-gripper-badge", "children"),
        Output("header-gripper-badge", "color"),
        Output("gripper-connection-state", "data"),
        Input("gripper-test", "n_clicks"),
        Input("gripper-enable", "checked"),
        Input("auto-init", "n_intervals"),
        State("gripper-cb-ip", "value"),
    )
    def _handle_gripper(
        _test_clicks: int,
        enabled: bool,
        _init: int,
        cb_ip: str,
    ) -> Tuple[str, html.Div, str, str, dict]:
        """Handle gripper connection and testing."""
        trigger = callback_context.triggered_id

        if not enabled and trigger == "gripper-enable":
            # Disable gripper
            if SESSION.gripper is not None:
                SESSION.gripper = None
                SESSION.gripper_device = None
            return (
                "Gripper disabled",
                html.Div(),
                "Gripper: off",
                "gray",
                {"status": "disabled", "message": "Gripper disabled"},
            )

        # Auto-connect once on load
        if trigger == "auto-init":
            enabled = True
        if not enabled:
            return no_update, no_update, no_update, no_update, no_update

        # Enable or test gripper
        try:
            if SESSION.gripper_device is None or SESSION.gripper_device.Global_cbip != cb_ip:
                # Create new connection
                device = GripperDevice(Global_cbip=cb_ip)
                cb = device.getCB()

                if cb is None:
                    return (
                        f"Failed to connect to Compute Box at {cb_ip}",
                        html.Div(
                            dmc.Alert(
                                "Cannot reach the gripper control box. Check power and network.",
                                title="Gripper Unavailable",
                                color="yellow",
                            )
                        ),
                        "Gripper: error",
                        "yellow",
                        {
                            "status": "error",
                            "message": f"Failed to connect to Compute Box at {cb_ip}",
                        },
                    )

                SESSION.gripper_device = device
                SESSION.gripper = TwoFG7Gripper(device)

            # Check gripper connection
            try:
                is_connected = SESSION.gripper.isConnected(
                    t_index=GRIPPER_DEFAULTS["t_index"]
                )
            except Exception as exc:  # noqa: BLE001
                SESSION.gripper = None
                SESSION.gripper_device = None
                return (
                    f"Failed to query gripper at {cb_ip}: {exc}",
                    html.Div(
                        dmc.Alert(
                            "Could not reach the gripper. Verify the Compute Box IP and network connection.",
                            title="Gripper Unavailable",
                            color="yellow",
                        )
                    ),
                    "Gripper: error",
                    "yellow",
                    {
                        "status": "error",
                        "message": f"Failed to query gripper at {cb_ip}: {exc}",
                    },
                )

            if not is_connected:
                return (
                    f"No 2FG7 gripper detected at Compute Box {cb_ip}",
                    html.Div(
                        dmc.Alert(
                            "Ensure the 2FG7 gripper is connected to the Compute Box "
                            "and powered on. Check the tool index setting if using "
                            "a dual gripper setup.",
                            title="Gripper Not Detected",
                            color="yellow",
                        )
                    ),
                    "Gripper: error",
                    "yellow",
                    {
                        "status": "error",
                        "message": f"No 2FG7 gripper detected at Compute Box {cb_ip}",
                    },
                )

            # Get gripper status and information
            status_text = f"2FG7 connected at {cb_ip}"
            t_index = GRIPPER_DEFAULTS["t_index"]
            SESSION.gripper.start_status_stream()

            # Query gripper state
            width = SESSION.gripper.get_ext_width(t_index=t_index)
            force = SESSION.gripper.get_force(t_index=t_index)
            is_gripped = SESSION.gripper.isGripped(t_index=t_index)
            SESSION.robot_has_object = bool(is_gripped)

            info_card = html.Div(
                dmc.Card(
                    [
                        dmc.Group(
                            [
                                dmc.Stack(
                                    [
                                        dmc.Text(
                                            "Width", size="sm", c="dimmed"),
                                        dmc.Text(
                                            f"{width:.1f} mm" if isinstance(
                                                width, (int, float)) and width > -2 else "N/A",
                                            fw=500,
                                        ),
                                    ],
                                    gap=0,
                                ),
                                dmc.Stack(
                                    [
                                        dmc.Text(
                                            "Force", size="sm", c="dimmed"),
                                        dmc.Text(
                                            f"{force:.1f} N" if isinstance(
                                                force, (int, float)) and force > -2 else "N/A",
                                            fw=500,
                                        ),
                                    ],
                                    gap=0,
                                ),
                                dmc.Stack(
                                    [
                                        dmc.Text(
                                            "Status", size="sm", c="dimmed"),
                                        dmc.Badge(
                                            "Gripped" if is_gripped else "Open",
                                            color="teal" if is_gripped else "gray",
                                            variant="light",
                                        ),
                                    ],
                                    gap=0,
                                ),
                            ],
                            grow=True,
                        ),
                    ],
                    withBorder=True,
                    padding="sm",
                    mt="xs",
                )
            )

            return (
                status_text,
                info_card,
                "Gripper: online",
                "brand",
                {"status": "online", "message": status_text},
            )

        except Exception as exc:
            LOGGER.exception("Gripper test failed")
            return (
                f"Gripper test failed: {exc}",
                html.Div(),
                "Gripper: error",
                "red",
                {"status": "error", "message": f"Gripper test failed: {exc}"},
            )

    @app.callback(
        Output("footer-gripper-width", "disabled"),
        Output("footer-gripper-force", "disabled"),
        Output("footer-gripper-speed", "disabled"),
        Output("footer-gripper-width-decrease", "disabled"),
        Output("footer-gripper-width-increase", "disabled"),
        Output("footer-gripper-force-decrease", "disabled"),
        Output("footer-gripper-force-increase", "disabled"),
        Output("footer-gripper-speed-decrease", "disabled"),
        Output("footer-gripper-speed-increase", "disabled"),
        Input("gripper-connection-state", "data"),
    )
    def _sync_footer_gripper_disabled(
        gripper_state: Optional[dict],
    ) -> Tuple[bool, bool, bool, bool, bool, bool, bool, bool, bool]:
        status = (gripper_state or {}).get("status", "offline")
        disabled = status != "online" or bool(SESSION.gripper_busy)
        return (
            disabled,
            disabled,
            disabled,
            disabled,
            disabled,
            disabled,
            disabled,
            disabled,
            disabled,
        )

    @app.callback(
        Output("footer-gripper-width", "min", allow_duplicate=True),
        Output("footer-gripper-width", "max", allow_duplicate=True),
        Input("gripper-connection-state", "data"),
        prevent_initial_call=True,
    )
    def _sync_gripper_width_limits(gripper_state: Optional[dict]) -> Tuple[float, float]:
        if (gripper_state or {}).get("status") != "online":
            return no_update, no_update
        if SESSION.gripper is None:
            return no_update, no_update
        min_width, max_width = _resolve_gripper_limits()
        return min_width, max_width

    @app.callback(
        Output("footer-visibility", "data"),
        Output("app-footer", "className"),
        Output("footer-toggle", "children"),
        Output("app-main", "className"),
        Output("footer-body", "style"),
        Input("footer-toggle", "n_clicks"),
        Input("robot-operation-state", "data"),
        State("footer-visibility", "data"),
    )
    def _toggle_footer_visibility(
        n_clicks: Optional[int],
        operation_state: Optional[dict],
        state: Optional[dict],
    ) -> Tuple[dict, str, str, str, dict]:
        def _main_class(collapsed: bool) -> str:
            base = "app-main footer-collapsed" if collapsed else "app-main footer-expanded"
            status = (operation_state or {}).get("status", "idle")
            if status == "running":
                return f"{base} operation-running"
            if status in {"paused", "stopping"}:
                return f"{base} operation-paused"
            return base

        collapsed = bool((state or {}).get("collapsed", False))
        trigger = callback_context.triggered_id
        if not n_clicks or trigger != "footer-toggle":
            # honor current state without toggling on first render or non-toggle updates
            class_name = "app-footer is-collapsed" if collapsed else "app-footer"
            label = "Show footer" if collapsed else "Hide footer"
            main_class = _main_class(collapsed)
            style = {"display": "none"} if collapsed else {}
            return state or {"collapsed": False}, class_name, label, main_class, style

        new_collapsed = not collapsed
        class_name = "app-footer is-collapsed" if new_collapsed else "app-footer"
        label = "Show footer" if new_collapsed else "Hide footer"
        main_class = _main_class(new_collapsed)
        style = {"display": "none"} if new_collapsed else {}
        return {"collapsed": new_collapsed}, class_name, label, main_class, style

    @app.callback(
        Output("footer-robot-position", "children"),
        Input("robot-connection-state", "data"),
        Input("robot-status-interval", "n_intervals"),
    )
    def _update_footer_robot_position(
        robot_conn_state: Optional[dict],
        _intervals: int,
    ) -> str:
        session = _get_robot_session()
        if not session.posture_seen():
            if (robot_conn_state or {}).get("status") == "online":
                return "X=--.--- Y=--.--- Z=--.---  A=--.--- B=--.--- C=--.---"
            return "X=--.--- Y=--.--- Z=--.---  A=--.--- B=--.--- C=--.---"

        pos = session.position_snapshot()
        text = (
            f"X={pos.x:.3f} Y={pos.y:.3f} Z={pos.z:.3f}  "
            f"A={pos.a:.3f} B={pos.b:.3f} C={pos.c:.3f}"
        )
        age_s = pos.age_seconds()
        if age_s > 2.0:
            text = f"{text} · stale {age_s:.1f}s"
        return text

    @app.callback(
        Output("footer-flange-indicator-1", "className"),
        Output("footer-flange-indicator-2", "className"),
        Output("footer-flange-indicator-3", "className"),
        Output("footer-flange-indicator-4", "className"),
        Input("robot-status-interval", "n_intervals"),
    )
    def _update_footer_flange_indicators(_intervals: int) -> Tuple[str, str, str, str]:
        session = _get_robot_session()
        states = session.flange_button_states()
        classes = []
        for port in DEFAULT_FLANGE_BUTTON_PORTS:
            active = states.get(port, 0) == 1
            classes.append(
                "footer-flange-indicator active" if active else "footer-flange-indicator"
            )
        return tuple(classes)

    @app.callback(
        Output("model-status", "children"),
        Output("model-select", "data"),
        Output("model-select", "value"),
        Input("model-upload", "contents"),
        Input("load-model", "n_clicks"),
        State("model-upload", "filename"),
        State("model-select", "value"),
        prevent_initial_call=True,
    )
    def _handle_model_actions(
        contents: Optional[str],
        _load_clicks: int,
        filename: Optional[str],
        selected: Optional[str],
    ) -> Tuple[str, list, Optional[str]]:
        trigger = callback_context.triggered_id
        options = _model_options()
        if trigger == "model-upload" and contents:
            try:
                model_path = save_uploaded_model(
                    contents, filename or "model.bin", DEFAULT_MODEL_DIR)
                status = activate_model(model_path)
                options = _model_options()
                return status, options, str(model_path)
            except Exception as exc:
                return f"Model upload failed: {exc}", options, selected
        if trigger == "load-model":
            if selected:
                try:
                    status = activate_model(Path(selected))
                except Exception as exc:
                    status = f"Model load failed: {exc}"
                return status, options, selected
            status = activate_model(None)
            return status, options, None
        return no_update, options, no_update

    @app.callback(
        Output("gpu-status-display", "children"),
        Input("camera-interval", "n_intervals"),
    )
    def _update_gpu_status(_n: int) -> list:
        """Update GPU status display periodically."""
        return format_gpu_status()

    @app.callback(
        Output("settings-status", "children"),
        Output("settings-robot-host", "value"),
        Output("settings-robot-coordinate-id", "value"),
        Output("settings-gripper-ip", "value"),
        Output("settings-gripper-type", "value"),
        Output("settings-marker-a", "value"),
        Output("settings-marker-b", "value"),
        Output("settings-marker-size", "value"),
        Output("settings-calib-path", "value"),
        Output("settings-workspace-path", "value"),
        Output("settings-robot-calib-path", "value"),
        Output("settings-auto-target-label", "value"),
        Output("settings-auto-drop-ignore-radius", "value"),
        Output("settings-auto-intrusion-match", "value"),
        Output("settings-auto-detection-stale", "value"),
        Output("settings-auto-idle-sleep", "value"),
        Output("settings-auto-estimate-width", "checked"),
        Output("settings-auto-approach-height", "value"),
        Output("settings-auto-pick-height", "value"),
        Output("settings-auto-drop-height", "value"),
        Output("settings-camera-calib-frames", "value"),
        Output("settings-camera-calib-delay", "value"),
        Output("settings-camera-calib-min-usable", "value"),
        Output("settings-pick-follow-timeout", "value"),
        Output("settings-pick-follow-interval", "value"),
        Output("settings-pick-follow-segment", "value"),
        Output("settings-pick-follow-settle", "value"),
        Output("settings-pick-follow-tolerance", "value"),
        Output("settings-rescue-jog-hold", "value"),
        Output("settings-coord5-max-z", "value"),
        Input("settings-save", "n_clicks"),
        Input("settings-reset", "n_clicks"),
        State("settings-robot-host", "value"),
        State("settings-robot-coordinate-id", "value"),
        State("settings-gripper-ip", "value"),
        State("settings-gripper-type", "value"),
        State("settings-marker-a", "value"),
        State("settings-marker-b", "value"),
        State("settings-marker-size", "value"),
        State("settings-calib-path", "value"),
        State("settings-workspace-path", "value"),
        State("settings-robot-calib-path", "value"),
        State("settings-auto-target-label", "value"),
        State("settings-auto-drop-ignore-radius", "value"),
        State("settings-auto-intrusion-match", "value"),
        State("settings-auto-detection-stale", "value"),
        State("settings-auto-idle-sleep", "value"),
        State("settings-auto-estimate-width", "checked"),
        State("settings-auto-approach-height", "value"),
        State("settings-auto-pick-height", "value"),
        State("settings-auto-drop-height", "value"),
        State("settings-camera-calib-frames", "value"),
        State("settings-camera-calib-delay", "value"),
        State("settings-camera-calib-min-usable", "value"),
        State("settings-pick-follow-timeout", "value"),
        State("settings-pick-follow-interval", "value"),
        State("settings-pick-follow-segment", "value"),
        State("settings-pick-follow-settle", "value"),
        State("settings-pick-follow-tolerance", "value"),
        State("settings-rescue-jog-hold", "value"),
        State("settings-coord5-max-z", "value"),
        prevent_initial_call=True,
    )
    def _handle_settings(
        _save: int,
        _reset: int,
        robot_host: Optional[str],
        robot_coordinate_id: Optional[float],
        gripper_ip: Optional[str],
        gripper_type: Optional[str],
        marker_a: Optional[float],
        marker_b: Optional[float],
        marker_size: Optional[float],
        calib_path: Optional[str],
        workspace_path: Optional[str],
        robot_calib_path: Optional[str],
        auto_target_label: Optional[str],
        auto_drop_ignore_radius: Optional[float],
        auto_intrusion_match: Optional[float],
        auto_detection_stale: Optional[float],
        auto_idle_sleep: Optional[float],
        auto_estimate_width: Optional[bool],
        auto_approach_height: Optional[float],
        auto_pick_height: Optional[float],
        auto_drop_height: Optional[float],
        camera_calib_frames: Optional[float],
        camera_calib_delay: Optional[float],
        camera_calib_min_usable: Optional[float],
        pick_follow_timeout: Optional[float],
        pick_follow_interval: Optional[float],
        pick_follow_segment: Optional[float],
        pick_follow_settle: Optional[float],
        pick_follow_tolerance: Optional[float],
        rescue_jog_hold: Optional[float],
        coord5_max_z: Optional[float],
    ) -> Tuple[
        str,
        Optional[str],
        int,
        Optional[str],
        Optional[str],
        Optional[int],
        Optional[int],
        Optional[float],
        str,
        str,
        str,
        str,
        float,
        float,
        float,
        float,
        bool,
        float,
        float,
        float,
        int,
        float,
        int,
        float,
        float,
        float,
        float,
        float,
    ]:
        trigger = callback_context.triggered_id
        global CODROID_DEFAULTS, DEFAULT_ROBOT_URI, GRIPPER_DEFAULTS, DEFAULT_GRIPPER_IP
        global DEFAULT_BUNDLE_PATH, DEFAULT_WORKSPACE_PATH, DEFAULT_ROBOT_CALIBRATION_PATH
        global ROBOT_COORDINATE_ID
        default_marker_a = 1
        default_marker_b = 2
        default_marker_size = 40.0
        default_calib = str(DEFAULT_BUNDLE_PATH)
        default_workspace = str(DEFAULT_WORKSPACE_PATH)
        default_robot_calib = str(DEFAULT_ROBOT_CALIBRATION_PATH)
        default_gripper_ip = app_settings.DEFAULT_GRIPPER_IP
        default_gripper_type = DEFAULT_GRIPPER_TYPE
        default_auto_target_label = app_settings._BASE_AUTO_TARGET_LABEL
        default_auto_drop_ignore_radius = app_settings._BASE_AUTO_DROP_IGNORE_RADIUS_MM
        default_auto_intrusion_match = app_settings._BASE_AUTO_INTRUSION_MATCH_MM
        default_auto_detection_stale = app_settings._BASE_AUTO_DETECTION_STALE_S
        default_auto_idle_sleep = app_settings._BASE_AUTO_IDLE_SLEEP_S
        default_auto_estimate_width = app_settings._BASE_AUTO_ESTIMATE_WIDTH
        default_auto_approach_height = app_settings._BASE_AUTO_APPROACH_MM
        default_auto_pick_height = app_settings._BASE_AUTO_PICK_MM
        default_auto_drop_height = app_settings._BASE_AUTO_DROP_MM

        default_camera_calib_frames = app_settings._BASE_CAMERA_CALIBRATION_FRAMES
        default_camera_calib_delay = app_settings._BASE_CAMERA_CALIBRATION_DELAY_S
        default_camera_calib_min_usable = app_settings._BASE_CAMERA_CALIBRATION_MIN_USABLE_FRAMES
        default_pick_follow_timeout = app_settings._BASE_PICK_FOLLOW_TIMEOUT_S
        default_pick_follow_interval = app_settings._BASE_PICK_FOLLOW_UPDATE_S
        default_pick_follow_segment = app_settings._BASE_PICK_FOLLOW_SEGMENT_S
        default_pick_follow_settle = app_settings._BASE_PICK_FOLLOW_SETTLE_S
        default_pick_follow_tolerance = app_settings._BASE_PICK_FOLLOW_TOLERANCE_MM
        default_rescue_jog_hold = app_settings._BASE_RESCUE_JOG_HOLD_S
        default_coord5_max_z = app_settings._BASE_COORD5_MAX_Z_MM
        default_robot_coordinate_id = app_settings._BASE_ROBOT_COORDINATE_ID
        status_text = "Settings saved."
        if trigger == "settings-reset":
            status_text = "Settings reset to defaults (env + built-ins)."
            host_val = None
            gripper_ip_val = None
            gripper_type_val = default_gripper_type
            marker_a_val = default_marker_a
            marker_b_val = default_marker_b
            marker_size_val = default_marker_size
            calib_path_val = default_calib
            workspace_path_val = default_workspace
            robot_calib_path_val = default_robot_calib
            target_label_val = default_auto_target_label
            drop_ignore_radius_val = default_auto_drop_ignore_radius
            intrusion_match_val = default_auto_intrusion_match
            detection_stale_val = default_auto_detection_stale
            idle_sleep_val = default_auto_idle_sleep
            estimate_width_val = default_auto_estimate_width
            approach_height_val = default_auto_approach_height
            pick_height_val = default_auto_pick_height
            drop_height_val = default_auto_drop_height
            robot_coordinate_id_val = default_robot_coordinate_id
            pick_follow_timeout_val = default_pick_follow_timeout
            pick_follow_interval_val = default_pick_follow_interval
            pick_follow_segment_val = default_pick_follow_segment
            pick_follow_settle_val = default_pick_follow_settle
            pick_follow_tolerance_val = default_pick_follow_tolerance
            rescue_jog_hold_val = default_rescue_jog_hold
            coord5_max_z_val = default_coord5_max_z
        else:
            host_val = (robot_host or "").strip() or None
            gripper_ip_val = (gripper_ip or "").strip() or None
            gripper_type_val = (gripper_type or "").strip() or default_gripper_type
            if gripper_type_val not in GRIPPER_PROFILE_KEYS:
                gripper_type_val = default_gripper_type
            marker_a_val = int(marker_a) if marker_a is not None else default_marker_a
            marker_b_val = int(marker_b) if marker_b is not None else default_marker_b
            marker_size_val = float(marker_size) if marker_size is not None else default_marker_size
            calib_path_val = (calib_path or "").strip() or default_calib
            workspace_path_val = (workspace_path or "").strip() or default_workspace
            robot_calib_path_val = (robot_calib_path or "").strip() or default_robot_calib
            target_label_val = (auto_target_label or "").strip() or default_auto_target_label
            drop_ignore_radius_val = _safe_float(auto_drop_ignore_radius, default_auto_drop_ignore_radius)
            intrusion_match_val = _safe_float(auto_intrusion_match, default_auto_intrusion_match)
            detection_stale_val = _safe_float(auto_detection_stale, default_auto_detection_stale)
            idle_sleep_val = _safe_float(auto_idle_sleep, default_auto_idle_sleep)
            estimate_width_val = (
                bool(auto_estimate_width)
                if auto_estimate_width is not None
                else default_auto_estimate_width
            )
            approach_height_val = _safe_float(auto_approach_height, default_auto_approach_height)
            pick_height_val = _safe_float(auto_pick_height, default_auto_pick_height)
            drop_height_val = _safe_float(auto_drop_height, default_auto_drop_height)
            robot_coordinate_id_val = _safe_int(
                robot_coordinate_id, default_robot_coordinate_id
            )
            pick_follow_timeout_val = _safe_float(
                pick_follow_timeout, default_pick_follow_timeout
            )
            pick_follow_interval_val = _safe_float(
                pick_follow_interval, default_pick_follow_interval
            )
            pick_follow_segment_val = _safe_float(
                pick_follow_segment, default_pick_follow_segment
            )
            pick_follow_settle_val = _safe_float(
                pick_follow_settle, default_pick_follow_settle
            )
            pick_follow_tolerance_val = _safe_float(
                pick_follow_tolerance, default_pick_follow_tolerance
            )
            rescue_jog_hold_val = _safe_float(
                rescue_jog_hold, default_rescue_jog_hold
            )
            coord5_max_z_val = _safe_float(
                coord5_max_z, default_coord5_max_z
            )

        _commit_settings(
            robot_host=host_val,
            gripper_ip=gripper_ip_val,
            gripper_type=gripper_type_val,
            marker_ids=(marker_a_val, marker_b_val),
            marker_size_mm=marker_size_val,
            calibration_bundle_path=calib_path_val,
            workspace_path=workspace_path_val,
            robot_calibration_path=robot_calib_path_val,
            auto_target_label=target_label_val,
            auto_drop_ignore_radius_mm=drop_ignore_radius_val,
            auto_intrusion_match_mm=intrusion_match_val,
            auto_detection_stale_s=detection_stale_val,
            auto_idle_sleep_s=idle_sleep_val,
            auto_estimate_width=estimate_width_val,
            auto_approach_mm=approach_height_val,
            auto_pick_mm=pick_height_val,
            auto_drop_mm=drop_height_val,
            pick_follow_timeout_s=pick_follow_timeout_val,
            pick_follow_update_interval_s=pick_follow_interval_val,
            pick_follow_segment_hold_s=pick_follow_segment_val,
            pick_follow_settle_s=pick_follow_settle_val,
            pick_follow_tolerance_mm=pick_follow_tolerance_val,
            rescue_jog_hold_s=rescue_jog_hold_val,
            coord5_max_z_mm=coord5_max_z_val,
            robot_coordinate_id=robot_coordinate_id_val,
        )

        app_settings.reload_defaults()
        CODROID_DEFAULTS = app_settings.CODROID_DEFAULTS
        DEFAULT_ROBOT_URI = app_settings.DEFAULT_ROBOT_URI
        GRIPPER_DEFAULTS = app_settings.GRIPPER_DEFAULTS
        DEFAULT_GRIPPER_IP = app_settings.DEFAULT_GRIPPER_IP
        _refresh_gripper_constants()
        _refresh_session_auto_config_defaults()
        set_asset_paths(
            calibration_bundle_path=calib_path_val,
            workspace_path=workspace_path_val,
            robot_calibration_path=robot_calib_path_val,
        )
        DEFAULT_BUNDLE_PATH = app_settings.DEFAULT_BUNDLE_PATH
        DEFAULT_WORKSPACE_PATH = app_settings.DEFAULT_WORKSPACE_PATH
        DEFAULT_ROBOT_CALIBRATION_PATH = app_settings.DEFAULT_ROBOT_CALIBRATION_PATH
        SESSION.marker_ids = (marker_a_val, marker_b_val)
        SESSION.marker_size_mm = marker_size_val
        ROBOT_COORDINATE_ID = SETTINGS.robot_coordinate_id

        host_display = host_val or CODROID_DEFAULTS.get("host")
        gripper_ip_display = gripper_ip_val or app_settings.DEFAULT_GRIPPER_IP

        return (
            status_text,
            host_display,
            robot_coordinate_id_val,
            gripper_ip_display,
            gripper_type_val,
            marker_a_val,
            marker_b_val,
            marker_size_val,
            calib_path_val,
            workspace_path_val,
            robot_calib_path_val,
            target_label_val,
            drop_ignore_radius_val,
            intrusion_match_val,
            detection_stale_val,
            idle_sleep_val,
            estimate_width_val,
            approach_height_val,
            pick_height_val,
            drop_height_val,
            camera_calib_frames,
            camera_calib_delay,
            camera_calib_min_usable,
            pick_follow_timeout_val,
            pick_follow_interval_val,
            pick_follow_segment_val,
            pick_follow_settle_val,
            pick_follow_tolerance_val,
            rescue_jog_hold_val,
            coord5_max_z_val,
        )

    @app.callback(
        Output("detection-confidence", "value"),
        Input("detection-confidence", "value"),
        Input("detection-iou", "value"),
        Input("detection-max", "value"),
        Input("detection-classes", "value"),
        State("detection-confidence", "value"),
        prevent_initial_call=True,
    )
    def _update_detection_parameters(
        confidence: Optional[float],
        iou: Optional[float],
        max_detections: Optional[float],
        class_filter: Optional[str],
        current_confidence: Optional[float],
    ) -> float:
        """Store detection parameters when they change."""
        SESSION.detection_confidence = _safe_float(confidence, 0.4)
        SESSION.detection_iou = _safe_float(iou, 0.45)
        SESSION.detection_max = _safe_int(max_detections, 10)
        SESSION.detection_classes = class_filter
        return SESSION.detection_confidence

    @app.callback(
        Output("detection-status", "children"),
        Output("detection-list", "children"),
        Output("pick-detection-select", "data"),
        Output("pick-detection-select", "value"),
        Input("run-detection", "n_clicks"),
        State("detection-confidence", "value"),
        State("detection-iou", "value"),
        State("detection-max", "value"),
        State("detection-classes", "value"),
        prevent_initial_call=True,
    )
    def _run_detection_action(
        _clicks: int,
        confidence: Optional[float],
        iou: Optional[float],
        max_detections: Optional[float],
        class_filter: Optional[str],
    ) -> Tuple[str, list, list, Optional[str]]:
        frame, _depth = get_latest_frame()
        if frame is None:
            return (
                "Camera stream is not running.",
                _format_detection_list([]),
                [],
                None,
            )
        conf_value = _safe_float(confidence, 0.4)
        iou_value = _safe_float(iou, 0.45)
        max_value = _safe_int(max_detections, 10)
        try:
            detections = _run_detection(
                frame, conf_value, iou_value, max_value, class_filter)
            SESSION.detections = detections
            SESSION.last_detection_error = None
            SESSION.last_action = "Detection run completed."
        except Exception as exc:
            SESSION.detections = []
            SESSION.last_detection_error = str(exc)
            return (
                f"Detection failed: {exc}",
                _format_detection_list([]),
                [],
                None,
            )

        options = []
        for idx, det in enumerate(detections):
            label = det.get("label", "object")
            options.append({"value": str(idx), "label": f"{idx + 1}: {label}"})
        selected = options[0]["value"] if options else None
        return (
            f"Detection complete: {len(detections)} object(s).",
            _format_detection_list(detections),
            options,
            selected,
        )

    @app.callback(
        Output("detection-list", "children", allow_duplicate=True),
        Output("pick-detection-select", "data", allow_duplicate=True),
        Output("detection-metrics", "children", allow_duplicate=True),
        Input("camera-interval", "n_intervals"),
        prevent_initial_call=True,
    )
    def _update_detection_display(
        _n: int,
    ) -> Tuple[list, list, str]:
        """Update detection display from continuous detection results.

        Detection updates are produced automatically when a model is loaded;
        if no model is loaded skip updating the display.
        """
        if not SESSION.model_loaded:
            return no_update, no_update, no_update

        detections = SESSION.detections or []
        detection_list = _format_detection_list(detections)

        options = []
        for idx, det in enumerate(detections):
            label = det.get("label", "object")
            conf = det.get("confidence", 0.0)
            options.append(
                {"value": str(idx), "label": f"{idx + 1}: {label} ({conf:.2f})"})

        metrics = f"Live: {len(detections)} object(s)"
        if SESSION.last_detection_error:
            metrics = f"Error: {SESSION.last_detection_error}"

        return detection_list, options, metrics

    @app.callback(
        Output("auto-config-state", "data"),
        Input("auto-target-class", "value"),
        Input("auto-drop-ignore-radius", "value"),
        Input("plate-test-approach-height", "value"),
        Input("plate-test-pick-height", "value"),
        Input("plate-test-drop-height", "value"),
    )
    def _update_auto_config(
        target_label: Optional[str],
        drop_ignore_radius: Optional[float],
        approach_height: Optional[float],
        pick_height: Optional[float],
        drop_height: Optional[float],
    ) -> dict:
        config = {
            "target_label": (target_label or SETTINGS.auto_target_label),
            "drop_ignore_radius_mm": _safe_float(
                drop_ignore_radius, SETTINGS.auto_drop_ignore_radius_mm
            ),
            "approach_mm": _safe_float(
                approach_height, SETTINGS.auto_approach_mm
            ),
            "pick_mm": _safe_float(pick_height, SETTINGS.auto_pick_mm),
            "drop_mm": _safe_float(drop_height, SETTINGS.auto_drop_mm),
            "estimate_width": bool(
                SESSION.auto_config.get(
                    "estimate_width", SETTINGS.auto_estimate_width
                )
            ),
        }
        SESSION.auto_config.update(config)
        return config

    @app.callback(
        Output("pick-place-status", "children"),
        Output("drop-position-metrics", "children", allow_duplicate=True),
        Input("teach-drop-position", "n_clicks"),
        Input("return-to-drop-position", "n_clicks"),
        Input("execute-drop", "n_clicks"),
        Input("manual-pick-closest", "n_clicks"),
        Input("manual-pick-place-target", "n_clicks"),
        State("pick-detection-select", "value"),
        State("plate-test-approach-height", "value"),
        State("plate-test-pick-height", "value"),
        State("plate-test-drop-height", "value"),
        State("robot-uri", "value"),
        prevent_initial_call=True,
    )
    async def _handle_pick_place(
        _teach_drop: int,
        _return_to_drop: int,
        _drop_action: int,
        _manual_pick: int,
        _manual_pick_place: int,
        selected_detection: Optional[str],
        approach_height: Optional[float],
        pick_height: Optional[float],
        drop_height: Optional[float],
        robot_uri: Optional[str],
    ) -> Tuple[str, str]:
        trigger = callback_context.triggered_id
        LOGGER.info(f"Pick/place callback triggered: {trigger}")

        # Prevent multiple rapid operations
        if SESSION.operation_in_progress:
            return f"Operation already in progress, please wait...", no_update

        frame, _depth = get_latest_frame()
        drop_text = no_update
        gripper_min_width, gripper_max_width = _resolve_gripper_limits()
        gripper_open_val = GRIPPER_OPEN_WIDTH_DEFAULT
        gripper_close_val = GRIPPER_CLOSE_WIDTH_DEFAULT
        if trigger is None:
            return no_update, no_update
        if frame is None and trigger not in {"teach-drop-position", "return-to-drop-position"}:
            return "Camera stream is not running.", drop_text

        workspace_def = SESSION.workspace_definition
        # Negative = UP from table in coord5
        approach_mm = _safe_float(approach_height, -30.0)
        pick_mm = _safe_float(pick_height, 0.0)
        drop_mm = _safe_float(drop_height, 0.0)

        def _estimate_width_mm(det: dict) -> Optional[float]:
            estimate_enabled = bool(
                SESSION.auto_config.get("estimate_width", False))
            if not estimate_enabled or workspace_def is None:
                return None
            bbox = det.get("bbox")
            if not bbox:
                return None
            x1, y1, x2, y2 = bbox
            width_px = max(abs(x2 - x1), abs(y2 - y1))
            if width_px <= 0:
                return None
            x_scale = np.linalg.norm(workspace_def.x_axis_camera)
            y_scale = np.linalg.norm(workspace_def.y_axis_camera)
            denom = (x_scale + y_scale) / 2.0
            if denom <= 0:
                return None
            mm_per_px = 1.0 / denom
            return width_px * mm_per_px

        if trigger == "teach-drop-position":
            if workspace_def is None:
                return "Workspace is not defined; compute it first.", drop_text
            try:
                await _ensure_robot_coord5_for_uri(robot_uri)
            except Exception as exc:  # noqa: BLE001
                return f"Failed to set CoordinateId 5: {exc}", drop_text
            session = _get_robot_session()
            if not session.posture_seen():
                return "Robot posture stream not available yet.", drop_text
            pos = session.position_snapshot()
            coord_id = _current_coordinate_id(session)
            try:
                if coord_id == ROBOT_COORDINATE_ID:
                    drop_mm_xy = _coord5_to_workspace_mm(
                        workspace_def, (float(pos.x), float(pos.y))
                    )
                    if drop_mm_xy is None:
                        raise RuntimeError(
                            "Coord5 transform unavailable; check calibration.")
                else:
                    robot_xyz = np.array([pos.x, pos.y, pos.z], dtype=float)
                    drop_mm_xy = _robot_to_workspace_mm(
                        workspace_def, robot_xyz
                    )
            except Exception as exc:  # noqa: BLE001
                return f"Failed to record drop position: {exc}", drop_text
            SESSION.drop_target_mm = drop_mm_xy
            SESSION.drop_target_orientation = (pos.a, pos.b, pos.c)
            SESSION.drop_target_robot = (pos.x, pos.y, pos.z)
            SESSION.drop_target_coord_id = coord_id
            _save_drop_target(
                drop_mm_xy,
                SESSION.drop_target_orientation,
                SESSION.drop_target_robot,
                coordinate_id=coord_id,
            )
            _save_settings(SETTINGS.model_copy(
                update={"drop_target": drop_mm_xy}))
            SESSION.last_action = "Drop position taught."
            drop_text = (
                f"Drop at {drop_mm_xy[0]:.1f}, {drop_mm_xy[1]:.1f} mm (z={pos.z:.1f}) · "
                f"a={pos.a:.1f}, b={pos.b:.1f}, c={pos.c:.1f}"
            )
            return "Drop position recorded from robot pose.", drop_text

        if trigger == "return-to-drop-position":
            if SESSION.drop_target_robot is None:
                return "No drop position has been taught yet.", drop_text
            try:
                await _ensure_robot_coord5_for_uri(robot_uri)
            except Exception as exc:  # noqa: BLE001
                return f"Failed to set CoordinateId 5: {exc}", drop_text
            session = _get_robot_session()
            target_coord_id = SESSION.drop_target_coord_id or 5
            if target_coord_id != _current_coordinate_id(session):
                return f"Robot is not in coordinate system {target_coord_id}; switch coordinate systems first.", drop_text
            drop_robot_xyz = SESSION.drop_target_robot
            drop_orient = SESSION.drop_target_orientation or (0.0, 0.0, 0.0)
            try:
                async def _return_to_drop(robot_api: CodroidAPI) -> None:
                    await robot_api.power_on()
                    await _ensure_coordinate5_active(robot_api)
                    target_pose = {
                        "x": float(drop_robot_xyz[0]),
                        "y": float(drop_robot_xyz[1]),
                        "z": float(drop_robot_xyz[2]),
                        "a": float(drop_orient[0]),
                        "b": float(drop_orient[1]),
                        "c": float(drop_orient[2]),
                    }
                    _log_move_target(
                        "return_drop", target_pose, coordinate_id=ROBOT_COORDINATE_ID
                    )

                    # Use the same hold pattern as plate test movements
                    await robot_api.set_target_pos_type(
                        robot_api.config.target_pos_types.cpos
                    )
                    await robot_api.set_target_cpos(
                        robot_api.attach_coordinate_to_cpos(
                            robot_api.build_target_cpos(**target_pose),
                            coordinate_id=ROBOT_COORDINATE_ID,
                        )
                    )
                    await robot_api.set_robot_command(
                        robot_api.config.commands.move_target_linear
                    )

                    # Hold with heartbeats for 5 seconds to ensure movement completes
                    hold_seconds = 5.0
                    heartbeat_interval = 0.2
                    deadline = time.monotonic() + hold_seconds
                    while time.monotonic() < deadline:
                        await robot_api.send_command_heartbeat()
                        await asyncio.sleep(heartbeat_interval)

                    await robot_api.clear_target_position()
                    await robot_api.stop_command()

                await _with_robot_api(robot_uri or DEFAULT_ROBOT_URI, _return_to_drop)
                SESSION.last_action = "Returned to drop position."
                return "Robot moved to drop position.", drop_text
            except Exception as exc:  # noqa: BLE001
                return f"Failed to move to drop position: {exc}", drop_text

        # Gripper open/close UI was removed; gripper commands are managed
        # elsewhere. Ignore triggers for those actions here.

        if workspace_def is None:
            return "Workspace is not defined; run workspace calibration first.", drop_text
        # Negative = UP from table in coord5
        approach_mm = _safe_float(approach_height, -30.0)
        pick_mm = _safe_float(pick_height, 0.0)
        drop_mm = _safe_float(drop_height, 0.0)
        LOGGER.info(
            f"Pick/drop heights: approach={approach_mm}mm, pick={pick_mm}mm, drop={drop_mm}mm")

        # Get dynamic gripper limits instead of using hardcoded values
        if SESSION.gripper is not None and SESSION.gripper.isConnected(t_index=GRIPPER_DEFAULTS["t_index"]):
            try:
                max_gripper_width = SESSION.gripper.get_max_ext_width(
                    t_index=GRIPPER_DEFAULTS["t_index"])
                min_gripper_width = SESSION.gripper.get_min_ext_width(
                    t_index=GRIPPER_DEFAULTS["t_index"])

                if isinstance(max_gripper_width, (int, float)) and max_gripper_width > 0:
                    # Open to maximum width for reliable release
                    gripper_open_val = max_gripper_width
                    LOGGER.info(
                        f"Using dynamic gripper max width: {gripper_open_val} mm")
                else:
                    gripper_open_val = GRIPPER_OPEN_WIDTH_DEFAULT
                    LOGGER.warning(
                        f"Could not get dynamic gripper width, using default: {gripper_open_val} mm")

                if isinstance(min_gripper_width, (int, float)) and min_gripper_width > 0:
                    gripper_close_val = min_gripper_width  # Close to minimum width for secure grip
                    LOGGER.info(
                        f"Using dynamic gripper min width: {gripper_close_val} mm")
                else:
                    gripper_close_val = GRIPPER_CLOSE_WIDTH_DEFAULT
                    LOGGER.warning(
                        f"Could not get dynamic gripper min width, using default: {gripper_close_val} mm")
            except Exception as e:
                LOGGER.warning(
                    f"Error getting dynamic gripper limits: {e}, using defaults")
                gripper_open_val = GRIPPER_OPEN_WIDTH_DEFAULT
                gripper_close_val = GRIPPER_CLOSE_WIDTH_DEFAULT
        else:
            LOGGER.warning("Gripper not connected, using default widths")
            gripper_open_val = GRIPPER_OPEN_WIDTH_DEFAULT
            gripper_close_val = GRIPPER_CLOSE_WIDTH_DEFAULT
        if not SESSION.detections and trigger in {"execute-pick", "execute-pick-drop", "manual-pick-closest", "manual-pick-place-target"}:
            return "No detections available; run detection first.", drop_text

        if trigger in {"execute-pick", "execute-pick-drop"} and selected_detection is None:
            return "Select a detection to pick.", drop_text

        def _select_detection_for_action() -> dict:
            if trigger in {"manual-pick-closest", "manual-pick-place-target"}:
                session = _get_robot_session()
                pos = session.position_snapshot()
                coord_id = _current_coordinate_id(session)
                try:
                    if coord_id == ROBOT_COORDINATE_ID:
                        ref_xy = (float(pos.x), float(pos.y))
                    else:
                        ref_xy = _robot_to_workspace_mm(
                            workspace_def,
                            np.array([pos.x, pos.y, pos.z], dtype=float),
                        )
                except Exception:
                    ref_xy = None
                best_det = SESSION.detections[0]
                best_dist = float("inf")
                for det in SESSION.detections:
                    if coord_id == ROBOT_COORDINATE_ID:
                        det_xy = _detection_coord5_mm(det, workspace_def)
                    else:
                        det_xy = det.get("workspace_mm")
                    if det_xy is None:
                        continue
                    dist = np.linalg.norm(
                        np.array(det_xy[:2]) - np.array(ref_xy or det_xy[:2]))
                    if dist < best_dist:
                        best_dist = dist
                        best_det = det
                return best_det
            det_idx = int(selected_detection)
            return SESSION.detections[det_idx]

        try:
            detection = _select_detection_for_action()
        except Exception:
            return "Invalid detection selection.", drop_text

        center = detection.get("center")
        if center is None:
            return "Detection is missing center coordinates.", drop_text
        workspace_mm = detection.get("workspace_mm")
        if workspace_mm is None:
            workspace_mm = _workspace_pixel_to_mm(workspace_def, center)
        coord5_xy = _detection_coord5_mm(detection, workspace_def)
        if coord5_xy is None:
            return "Detection missing robot coordinates; check plate markers.", drop_text

        width_guess = _estimate_width_mm(detection)
        if width_guess is not None:
            gripper_close_val = max(gripper_close_val, width_guess * 0.9)
            gripper_open_val = max(gripper_open_val, width_guess + 5.0)

        try:
            pick_pose = _pose_from_coord5_mm(
                coord5_xy[0], coord5_xy[1], pick_mm
            )
            approach_pose = _pose_from_coord5_mm(
                coord5_xy[0], coord5_xy[1], approach_mm
            )
        except Exception as exc:  # noqa: BLE001
            return f"Failed to map pick target: {exc}", drop_text
        SESSION.last_pick_mm = workspace_mm
        tracker_detection = detection
        tracker_workspace = _detection_workspace_mm(
            detection, workspace_def
        ) or workspace_mm
        tracker_coord5 = coord5_xy

        def _resolve_drop_target() -> Tuple[float, float]:
            if SESSION.drop_target_mm:
                return SESSION.drop_target_mm
            raise RuntimeError("No drop location taught yet.")

        async def _pick_action(robot_api: CodroidAPI) -> None:
            LOGGER.info("_pick_action: Starting")
            # Check if operation is still active
            await _wait_for_operation_resume(run_id)
            await robot_api.power_on()
            await _ensure_coordinate5_active(robot_api)
            nonlocal tracker_detection, tracker_workspace

            def _fetch_pose(z_offset_mm: float) -> Optional[dict]:
                if tracker_coord5 is None:
                    return None
                return _pose_from_coord5_mm(
                    tracker_coord5[0],
                    tracker_coord5[1],
                    z_offset_mm,
                )

            _log_move_target(
                "pick_approach", approach_pose, coordinate_id=ROBOT_COORDINATE_ID
            )
            await _move_linear_follow_target(
                robot_api,
                lambda: _fetch_pose(approach_mm),
                label="pick_approach_follow",
            )
            _log_move_target(
                "pick_pose", pick_pose, coordinate_id=ROBOT_COORDINATE_ID
            )
            await _move_linear_follow_target(
                robot_api,
                lambda: _fetch_pose(pick_mm),
                label="pick_pose_follow",
            )
            # Check before gripper operation
            await _wait_for_operation_resume(run_id)
            # Use higher force for gripping objects
            # Double the force for better grip
            grip_force = min(GRIPPER_FORCE_DEFAULT * 2, GRIPPER_FORCE_MAX)
            gripper_result = _command_gripper(
                gripper_close_val, "closed", force=grip_force)
            LOGGER.info(
                f"_pick_action: Gripper close command result: {gripper_result}")
            if "not connected" in gripper_result.lower() or "error" in gripper_result.lower():
                LOGGER.warning(
                    f"Gripper command may have failed: {gripper_result}")
            else:
                # Verify gripper closed properly
                if SESSION.gripper is not None and SESSION.gripper.isConnected(t_index=GRIPPER_DEFAULTS["t_index"]):
                    try:
                        # Gripper command already waits for completion (f_wait=True)
                        actual_width = SESSION.gripper.get_ext_width(
                            t_index=GRIPPER_DEFAULTS["t_index"])
                        LOGGER.info(
                            f"_pick_action: Gripper actual width after close command: {actual_width} mm")
                    except Exception as e:
                        LOGGER.warning(
                            f"_pick_action: Error checking gripper state: {e}")

            retreat_pose = _fetch_pose(approach_mm) or approach_pose
            _log_move_target(
                "pick_retreat", retreat_pose, coordinate_id=ROBOT_COORDINATE_ID
            )
            await robot_api.set_target_cpos(
                robot_api.attach_coordinate_to_cpos(
                    robot_api.build_target_cpos(**retreat_pose),
                    coordinate_id=ROBOT_COORDINATE_ID,
                )
            )
            await robot_api.set_robot_command(robot_api.config.commands.move_target_linear)

            # Hold with heartbeats for 2 seconds to ensure movement completes
            hold_seconds = 2.0
            heartbeat_interval = 0.2
            deadline = time.monotonic() + hold_seconds
            while time.monotonic() < deadline:
                await robot_api.send_command_heartbeat()
                await asyncio.sleep(heartbeat_interval)

            await robot_api.clear_target_position()
            await robot_api.stop_command()
            SESSION.robot_has_object = True
            if tracker_workspace is not None:
                SESSION.last_pick_mm = tracker_workspace
            LOGGER.info("_pick_action: Completed successfully")

        async def _drop_action(robot_api: CodroidAPI, drop_target_mm: Tuple[float, float]) -> None:
            LOGGER.info(f"_drop_action: Starting with target {drop_target_mm}")
            # Check if operation is still active
            await _wait_for_operation_resume(run_id)
            await robot_api.power_on()
            await _ensure_coordinate5_active(robot_api)

            # Use the stored robot coordinates directly (same as "Return to drop position")
            # This ensures we drop at the exact same location where the drop position was taught
            if SESSION.drop_target_robot is not None:
                base = SESSION.drop_target_robot
                drop_pose = {
                    "x": float(base[0]),
                    "y": float(base[1]),
                    "z": float(base[2]) + drop_mm,
                    "a": 0.0, "b": 0.0, "c": 0.0
                }
                drop_approach = {
                    "x": float(base[0]),
                    "y": float(base[1]),
                    "z": float(base[2]) + approach_mm,
                    "a": 0.0, "b": 0.0, "c": 0.0
                }
                LOGGER.info(
                    f"_drop_action: Using stored robot coords - base: {base}")
            else:
                # Fallback to workspace coordinate transformation (shouldn't normally happen)
                drop_pose = _pose_from_workspace_coord5_mm(
                    workspace_def, drop_target_mm[0], drop_target_mm[1], drop_mm
                )
                drop_approach = _pose_from_workspace_coord5_mm(
                    workspace_def, drop_target_mm[0], drop_target_mm[1], approach_mm
                )
                LOGGER.warning(
                    "_drop_action: No stored robot coords, using workspace transformation")

            LOGGER.info(
                f"_drop_action: Drop coords - pose: {drop_pose}, approach: {drop_approach}")

            # Apply stored orientation if available
            if SESSION.drop_target_orientation is not None:
                drop_pose.update(
                    {
                        "a": float(SESSION.drop_target_orientation[0]),
                        "b": float(SESSION.drop_target_orientation[1]),
                        "c": float(SESSION.drop_target_orientation[2]),
                    }
                )
                drop_approach.update(
                    {
                        "a": float(SESSION.drop_target_orientation[0]),
                        "b": float(SESSION.drop_target_orientation[1]),
                        "c": float(SESSION.drop_target_orientation[2]),
                    }
                )
            _log_move_target(
                "drop_approach", drop_approach, coordinate_id=ROBOT_COORDINATE_ID
            )
            LOGGER.info("_drop_action: Executing drop_approach movement...")
            await robot_api.set_target_pos_type(robot_api.config.target_pos_types.cpos)
            await robot_api.set_target_cpos(
                robot_api.attach_coordinate_to_cpos(
                    robot_api.build_target_cpos(**drop_approach),
                    coordinate_id=ROBOT_COORDINATE_ID,
                )
            )
            await robot_api.set_robot_command(robot_api.config.commands.move_target_linear)

            # Hold with heartbeats for 2 seconds to ensure movement completes
            hold_seconds = 2.0
            heartbeat_interval = 0.2
            deadline = time.monotonic() + hold_seconds
            while time.monotonic() < deadline:
                await robot_api.send_command_heartbeat()
                await asyncio.sleep(heartbeat_interval)

            await robot_api.clear_target_position()
            await robot_api.stop_command()
            LOGGER.info("_drop_action: Drop_approach movement completed")

            # Check before next movement
            await _wait_for_operation_resume(run_id)
            _log_move_target(
                "drop_pose", drop_pose, coordinate_id=ROBOT_COORDINATE_ID
            )
            LOGGER.info("_drop_action: Executing drop_pose movement...")
            await robot_api.set_target_pos_type(robot_api.config.target_pos_types.cpos)
            await robot_api.set_target_cpos(
                robot_api.attach_coordinate_to_cpos(
                    robot_api.build_target_cpos(**drop_pose),
                    coordinate_id=ROBOT_COORDINATE_ID,
                )
            )
            await robot_api.set_robot_command(robot_api.config.commands.move_target_linear)

            deadline = time.monotonic() + hold_seconds
            while time.monotonic() < deadline:
                await robot_api.send_command_heartbeat()
                await asyncio.sleep(heartbeat_interval)

            await robot_api.clear_target_position()
            await robot_api.stop_command()
            LOGGER.info("_drop_action: Drop_pose movement completed")

            # Check before gripper operation
            await _wait_for_operation_resume(run_id)
            LOGGER.info(
                f"_drop_action: Opening gripper to {gripper_open_val} mm...")
# Use standard gripper command for opening
            gripper_result = _command_gripper(gripper_open_val, "opened")

            LOGGER.info(
                f"_drop_action: Gripper open command result: {gripper_result}")
            if "not connected" in gripper_result.lower() or "error" in gripper_result.lower():
                LOGGER.warning(
                    f"Gripper command may have failed: {gripper_result}")
            else:
                # Poll gripper busy status instead of fixed wait
                if SESSION.gripper is not None:
                    max_polls = 20  # 2 seconds max at 0.1s intervals
                    poll_count = 0
                    while poll_count < max_polls:
                        try:
                            busy = SESSION.gripper.isBusy(
                                t_index=GRIPPER_DEFAULTS["t_index"])
                            if not busy:
                                break
                        except Exception:
                            break
                        await asyncio.sleep(0.1)
                        poll_count += 1
                    LOGGER.info(
                        f"_drop_action: Gripper ready after {poll_count * 0.1:.1f}s")

                # Verify gripper actually opened
                if SESSION.gripper is not None and SESSION.gripper.isConnected(t_index=GRIPPER_DEFAULTS["t_index"]):
                    try:
                        actual_width = SESSION.gripper.get_ext_width(
                            t_index=GRIPPER_DEFAULTS["t_index"])
                        LOGGER.info(
                            f"_drop_action: Gripper actual width after open command: {actual_width} mm")

                        # Check if gripper actually opened (should be close to target width)
                        if isinstance(actual_width, (int, float)) and actual_width > 0:
                            width_diff = abs(actual_width - gripper_open_val)
                            if width_diff > 5.0:  # If more than 5mm difference, retry
                                LOGGER.warning(
                                    f"_drop_action: Gripper width mismatch (target: {gripper_open_val}, actual: {actual_width}), retrying...")
                                # Get maximum gripper width and try to open to that
                                try:
                                    max_width = SESSION.gripper.get_max_ext_width(
                                        t_index=GRIPPER_DEFAULTS["t_index"])
                                    if isinstance(max_width, (int, float)) and max_width > actual_width:
                                        retry_width = max_width
                                        LOGGER.info(
                                            f"_drop_action: Retrying with maximum gripper width: {retry_width} mm")
                                    else:
                                        retry_width = gripper_open_val + 10.0  # Fallback: try 10mm wider
                                        LOGGER.info(
                                            f"_drop_action: Retrying with wider opening: {retry_width} mm")
                                except Exception as e:
                                    retry_width = gripper_open_val + 10.0
                                    LOGGER.warning(
                                        f"_drop_action: Could not get max width ({e}), trying {retry_width} mm")

                                retry_result = _command_gripper(
                                    retry_width, "opened (retry wider)", force=GRIPPER_FORCE_MAX)
                                LOGGER.info(
                                    f"_drop_action: Gripper retry result: {retry_result}")
                                # Gripper command already waits for completion

                                # Check again
                                final_width = SESSION.gripper.get_ext_width(
                                    t_index=GRIPPER_DEFAULTS["t_index"])
                                LOGGER.info(
                                    f"_drop_action: Final gripper width: {final_width} mm")
                            else:
                                LOGGER.info(
                                    "_drop_action: Gripper opened successfully")
                        else:
                            LOGGER.warning(
                                f"_drop_action: Could not read gripper width: {actual_width}")
                    except Exception as e:
                        LOGGER.warning(
                            f"_drop_action: Error verifying gripper state: {e}")
                else:
                    LOGGER.warning(
                        "_drop_action: Cannot verify gripper state - not connected")

                LOGGER.info("_drop_action: Gripper wait completed")

            # Check before retreat movement
            await _wait_for_operation_resume(run_id)
            _log_move_target(
                "drop_retreat", drop_approach, coordinate_id=ROBOT_COORDINATE_ID
            )
            LOGGER.info("_drop_action: Executing drop_retreat movement...")

            # Re-establish robot command state after gripper operations
            await robot_api.set_target_pos_type(robot_api.config.target_pos_types.cpos)
            await robot_api.set_target_cpos(
                robot_api.attach_coordinate_to_cpos(
                    robot_api.build_target_cpos(**drop_approach),
                    coordinate_id=ROBOT_COORDINATE_ID,
                )
            )
            await robot_api.set_robot_command(robot_api.config.commands.move_target_linear)

            # Hold with heartbeats for 5 seconds to ensure movement completes
            deadline = time.monotonic() + hold_seconds
            while time.monotonic() < deadline:
                await robot_api.send_command_heartbeat()
                await asyncio.sleep(heartbeat_interval)

            await robot_api.clear_target_position()
            await robot_api.stop_command()
            LOGGER.info("_drop_action: Drop_retreat movement completed")
            # Keep robot powered for faster subsequent operations
            # await robot_api.power_off()  # Disabled to keep robot ready
            LOGGER.info(
                "_drop_action: Drop completed, robot staying powered for next operation")
            SESSION.robot_has_object = False

        async def _pick_drop_action(robot_api: CodroidAPI) -> None:
            LOGGER.info("Starting pick & drop sequence")
            # Check if operation is still active
            await _wait_for_operation_resume(run_id)
            try:
                await _pick_action(robot_api)
                LOGGER.info("Pick action completed, starting drop")
                await _wait_for_operation_resume(run_id)  # Check before drop
                drop_target_mm = _resolve_drop_target()
                LOGGER.info(f"Drop target resolved: {drop_target_mm}")
                await _drop_action(robot_api, drop_target_mm)
                LOGGER.info("Drop action completed successfully")
            except Exception as exc:
                LOGGER.exception("Exception in pick & drop sequence")
                raise

        pick_cycle = trigger in {
            "manual-pick-closest",
            "manual-pick-place-target",
        }
        cycle_started_at = time.monotonic() if pick_cycle else None

        try:
            if trigger in {"manual-pick-closest"}:
                SESSION.operation_in_progress = True
                run_id = _begin_operation(
                    source="manual",
                    label="Manual Pick Closest",
                    message="Picking closest object..."
                )
                try:
                    await _with_robot_api(
                        robot_uri or DEFAULT_ROBOT_URI, _pick_action)
                    SESSION.last_action = "Pick command sent."
                    if pick_cycle:
                        _record_pick_cycle(True, cycle_started_at)
                    _finish_operation(run_id, status="success",
                                      message="Pick completed successfully.")
                except Exception as e:
                    _finish_operation(run_id, status="error",
                                      message=f"Pick failed: {e}")
                    raise
                finally:
                    SESSION.operation_in_progress = False
                return "Pick command sent.", drop_text
            if trigger == "execute-drop":
                drop_target_mm = _resolve_drop_target()
                SESSION.operation_in_progress = True
                run_id = _begin_operation(
                    source="manual",
                    label="Manual Drop",
                    message="Dropping object..."
                )

                try:
                    async def _drop_only(robot_api: CodroidAPI) -> None:
                        await _drop_action(robot_api, drop_target_mm)

                    await _with_robot_api(
                        robot_uri or DEFAULT_ROBOT_URI, _drop_only)
                    SESSION.last_action = "Drop command sent."
                    drop_text = f"Drop at {drop_target_mm[0]:.1f}, {drop_target_mm[1]:.1f} mm"
                    _finish_operation(run_id, status="success",
                                      message="Drop completed successfully.")
                except Exception as e:
                    _finish_operation(run_id, status="error",
                                      message=f"Drop failed: {e}")
                    raise
                finally:
                    SESSION.operation_in_progress = False
                return "Drop command sent.", drop_text
            if trigger in {"manual-pick-place-target"}:
                if SESSION.drop_target_mm is None:
                    return "Teach a drop position first.", drop_text
                SESSION.operation_in_progress = True
                run_id = _begin_operation(
                    source="manual",
                    label="Manual Pick & Drop",
                    message="Picking and dropping object..."
                )
                try:
                    await _with_robot_api(robot_uri or DEFAULT_ROBOT_URI, _pick_drop_action)
                    SESSION.last_action = "Pick & drop command sent."
                    drop_text = (
                        f"Drop at {SESSION.drop_target_mm[0]:.1f}, {SESSION.drop_target_mm[1]:.1f} mm"
                        if SESSION.drop_target_mm
                        else drop_text
                    )
                    if pick_cycle:
                        _record_pick_cycle(True, cycle_started_at)
                    _finish_operation(
                        run_id, status="success", message="Pick & drop completed successfully.")
                except Exception as e:
                    _finish_operation(run_id, status="error",
                                      message=f"Pick & drop failed: {e}")
                    raise
                finally:
                    SESSION.operation_in_progress = False
                return "Pick & drop command sent.", drop_text
            # Release action removed from this UI; handled elsewhere.
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("Pick/place action failed")
            SESSION.operation_in_progress = False
            if pick_cycle:
                _record_pick_cycle(False, cycle_started_at)
            return f"Pick/place action failed: {exc}", drop_text

        return no_update, drop_text

    @app.callback(
        [
            Output("manual-pick-closest", "disabled"),
            Output("execute-drop", "disabled"),
            Output("manual-pick-place-target", "disabled"),
            Output("reset-operation-lock", "disabled"),
        ],
        Input("camera-interval", "n_intervals"),
        prevent_initial_call=True,
    )
    def _update_button_states(_n: int) -> Tuple[bool, bool, bool, bool]:
        """Disable buttons during operations to prevent rapid clicking."""
        disabled = SESSION.operation_in_progress
        # Only enable reset when operation is in progress
        reset_disabled = not SESSION.operation_in_progress
        return disabled, disabled, disabled, reset_disabled

    @app.callback(
        Output("pick-place-status", "children", allow_duplicate=True),
        Input("reset-operation-lock", "n_clicks"),
        prevent_initial_call=True,
    )
    def _reset_operation_lock(_clicks: int) -> str:
        """Emergency reset for stuck operations."""
        SESSION.operation_in_progress = False
        LOGGER.info("Operation lock manually reset by user")
        return "Operation lock reset."

    @app.callback(
        Output("plate-test-status", "children"),
        Output("plate-test-analysis", "value"),
        Input("plate-test-run", "n_clicks"),
        State("plate-test-tracking-mode", "value"),
        State("plate-test-segment-min", "value"),
        State("plate-test-segment-max", "value"),
        State("plate-test-segment-fraction", "value"),
        State("plate-test-approach-height", "value"),
        State("plate-test-pick-height", "value"),
        State("plate-test-drop-height", "value"),
        State("plate-test-grip-timeout", "value"),
        State("robot-uri", "value"),
        State("robot-connection-state", "data"),
        State("gripper-connection-state", "data"),
        prevent_initial_call=True,
    )
    async def _run_plate_pick_place_test(
        _clicks: int,
        tracking_mode: Optional[str],
        segment_min: Optional[float],
        segment_max: Optional[float],
        segment_fraction: Optional[float],
        approach_height: Optional[float],
        pick_height: Optional[float],
        drop_height: Optional[float],
        grip_timeout_s: Optional[float],
        robot_uri: Optional[str],
        robot_state: Optional[dict],
        gripper_state: Optional[dict],
    ) -> Tuple[str, str]:
        run_id: Optional[int] = None

        def _plate_test_abort(message: str) -> Tuple[str, str]:
            _set_plate_test_state(message, running=False)
            if run_id is not None:
                _finish_operation(run_id, status="stopped", message=message)
            return message, ""

        if (robot_state or {}).get("status") != "online":
            return _plate_test_abort(
                "Robot is not connected. Click 'Test robot connection' first."
            )
        if (gripper_state or {}).get("status") != "online":
            return _plate_test_abort(
                "Gripper is not connected. Enable gripper in Setup menu."
            )
        if (
            SESSION.gripper is None
            or not SESSION.gripper.isConnected(
                t_index=GRIPPER_DEFAULTS["t_index"]
            )
        ):
            return _plate_test_abort(
                "Gripper session is not active; re-enable the gripper."
            )
        if SESSION.workspace_definition is None:
            return _plate_test_abort("Workspace is not defined; compute it first.")
        if SESSION.drop_target_mm is None:
            return _plate_test_abort("Teach a drop position first.")
        if SESSION.robot_calibration_min_z is None:
            return _plate_test_abort(
                "Robot calibration is missing; run robot calibration first."
            )
        frame, _depth = get_latest_frame()
        if frame is None:
            return _plate_test_abort("Camera stream is not running.")

        try:
            run_id = _begin_operation("plate_test", "Plate test")
        except RuntimeError as exc:
            return _plate_test_abort(str(exc))

        # Negative = UP from table in coord5
        approach_mm = _safe_float(approach_height, -30.0)
        pick_mm = _safe_float(pick_height, 0.0)
        drop_mm = _safe_float(drop_height, 0.0)
        grip_timeout = _safe_float(grip_timeout_s, SETTINGS.plate_test_grip_timeout_s)
        grip_timeout = max(grip_timeout, 0.5)
        segment_min = _safe_float(segment_min, SETTINGS.segment_min)
        segment_max = _safe_float(segment_max, SETTINGS.segment_max)
        segment_fraction = _safe_float(
            segment_fraction, SETTINGS.segment_fraction)
        segment_min = float(np.clip(segment_min, 1.0, 200.0))
        segment_max = float(np.clip(segment_max, 5.0, 500.0))
        segment_fraction = float(np.clip(segment_fraction, 0.1, 1.0))
        if segment_max < segment_min:
            segment_min, segment_max = segment_max, segment_min
        workspace_def = SESSION.workspace_definition
        session = _get_robot_session()
        if not session.posture_seen():
            return _plate_test_abort("Robot posture stream not available yet.")
        tracking_mode = (tracking_mode or "move").strip().lower()
        use_segment_tracking = tracking_mode == "segment"
        use_base_coordinates = SETTINGS.plate_test_use_base_coordinates
        if use_base_coordinates:
            if approach_mm < drop_mm:
                LOGGER.info(
                    "Clamping plate-test approach height from %.3f to %.3f (base coords).",
                    approach_mm,
                    drop_mm,
                )
                approach_mm = drop_mm
        else:
            if approach_mm > drop_mm:
                LOGGER.info(
                    "Clamping plate-test approach height from %.3f to %.3f (coord5).",
                    approach_mm,
                    drop_mm,
                )
                approach_mm = drop_mm
        if (
            SESSION.drop_target_orientation is not None
            and (
                not use_base_coordinates
                or SESSION.drop_target_coord_id in (None, 0)
            )
        ):
            orientation = {
                "a": float(SESSION.drop_target_orientation[0]),
                "b": float(SESSION.drop_target_orientation[1]),
                "c": float(SESSION.drop_target_orientation[2]),
            }
        else:
            posture = session.position_snapshot()
            orientation = {
                "a": float(posture.a),
                "b": float(posture.b),
                "c": float(posture.c),
            }

        _set_plate_test_state("Plate test starting...", running=True)
        _update_operation_message("Plate test starting...")
        try:
            (
                targets,
                plate_meta,
                baseline_transform,
                reference_initialized,
            ) = _resolve_plate_test_coord5_targets(DEFAULT_PLATE_PATH)
            target_sequence_order = [int(target["sequence"])
                                     for target in targets]
            targets_by_sequence: dict[int, dict] = {
                int(target["sequence"]): dict(target) for target in targets
            }
        except Exception as exc:  # noqa: BLE001
            return _plate_test_abort(f"Plate detection failed: {exc}")

        if not targets:
            return _plate_test_abort("No calibration targets found on the plate.")

        drop_target_mm = SESSION.drop_target_mm
        poscfg = CodroidAPI.build_poscfg(
            mode=int(SESSION.robot_calibration_points[0].get("mode", 0))
        )
        # Create two orientations: one aligned with plate (for picking), one for drop (original)
        drop_orientation = orientation.copy()
        plate_orientation = orientation.copy()
        plate_rotation_total = 0.0

        def _update_plate_alignment() -> None:
            nonlocal plate_rotation_total
            # Extract plate rotation angle (Z-axis rotation) from baseline transform
            # The baseline_transform.rotation is the rotation from plate frame to robot base frame
            # We extract the yaw angle (rotation around Z axis) for the 6th axis alignment
            plate_rotation_rad = np.arctan2(
                baseline_transform.rotation[1, 0],
                baseline_transform.rotation[0, 0],
            )
            plate_rotation_deg = np.degrees(plate_rotation_rad)
            offset_rotation_deg = float(
                plate_meta.get("offset_rotation_deg", 0.0))
            plate_rotation_total = float(
                plate_rotation_deg + offset_rotation_deg)
            plate_orientation["c"] = float(
                orientation["c"] + plate_rotation_total)
            LOGGER.info("Plate rotation angle: %.2f degrees",
                        plate_rotation_deg)
            LOGGER.info("Plate rotation total: %.2f degrees",
                        plate_rotation_total)

        _update_plate_alignment()

        def _base_to_coord(base_xyz: np.ndarray) -> np.ndarray:
            inv_rot = baseline_transform.rotation.T
            inv_trans = -inv_rot @ baseline_transform.translation
            return inv_rot @ base_xyz + inv_trans

        def _coord_to_base(coord_xyz: np.ndarray) -> np.ndarray:
            return (
                baseline_transform.rotation @ coord_xyz
                + baseline_transform.translation
            )

        plate_test_coord_id = 0 if use_base_coordinates else ROBOT_COORDINATE_ID

        def _resolve_coord5_pose(
            posture: RobotPosture,
            target: Optional[dict] = None,
        ) -> Tuple[np.ndarray, str]:
            raw_xyz = np.array(
                [float(posture.x), float(posture.y), float(posture.z)],
                dtype=np.float64,
            )
            coord_id = _current_coordinate_id(session)
            if coord_id == ROBOT_COORDINATE_ID:
                return raw_xyz, "robot_coord5"
            converted_xyz = _base_to_coord(raw_xyz)
            if coord_id == 0:
                return converted_xyz, "base_to_coord"
            if target is not None:
                target_xyz = np.array(
                    [float(target["x"]), float(
                        target["y"]), float(target["z"])],
                    dtype=np.float64,
                )
                if np.linalg.norm(raw_xyz - target_xyz) <= np.linalg.norm(
                    converted_xyz - target_xyz
                ):
                    return raw_xyz, "robot_coord5_guess"
                return converted_xyz, "base_to_coord_guess"
            return raw_xyz, "robot_coord5_unknown"

        def _coord5_pose_from_base(base_pose: dict) -> dict:
            base_xyz = np.array(
                [base_pose["x"], base_pose["y"], base_pose["z"]],
                dtype=np.float64,
            )
            coord_xyz = _base_to_coord(base_xyz)
            coord_pose = dict(base_pose)
            coord_pose["x"] = float(coord_xyz[0])
            coord_pose["y"] = float(coord_xyz[1])
            coord_pose["z"] = float(coord_xyz[2])
            return coord_pose

        def _base_pose_from_coord5(coord_pose: dict) -> dict:
            coord_xyz = np.array(
                [coord_pose["x"], coord_pose["y"], coord_pose["z"]],
                dtype=np.float64,
            )
            base_xyz = _coord_to_base(coord_xyz)
            base_pose = dict(coord_pose)
            base_pose["x"] = float(base_xyz[0])
            base_pose["y"] = float(base_xyz[1])
            base_pose["z"] = float(base_xyz[2])
            return base_pose

        if (
            SESSION.drop_target_robot is not None
            and SESSION.drop_target_coord_id == ROBOT_COORDINATE_ID
        ):
            # Clamp Z to safe minimum when using coordinate system 5
            drop_z = _clamp_z_for_coord5(
                float(SESSION.drop_target_robot[2]) + drop_mm,
                use_base_coords=use_base_coordinates,
            )
            approach_z = _clamp_z_for_coord5(
                float(SESSION.drop_target_robot[2]) + approach_mm,
                use_base_coords=use_base_coordinates,
            )
            drop_pose_coord5 = {
                "x": float(SESSION.drop_target_robot[0]),
                "y": float(SESSION.drop_target_robot[1]),
                "z": drop_z,
                "poscfg": poscfg,
            }
            drop_pose_coord5.update(drop_orientation)
            drop_approach_coord5 = {
                "x": float(SESSION.drop_target_robot[0]),
                "y": float(SESSION.drop_target_robot[1]),
                "z": approach_z,
                "poscfg": poscfg,
            }
            drop_approach_coord5.update(drop_orientation)
            drop_pose_base = _base_pose_from_coord5(drop_pose_coord5)
            drop_approach_base = _base_pose_from_coord5(drop_approach_coord5)
        else:
            if SESSION.drop_target_robot is not None:
                drop_pose_base = {
                    "x": float(SESSION.drop_target_robot[0]),
                    "y": float(SESSION.drop_target_robot[1]),
                    "z": float(SESSION.drop_target_robot[2]) + drop_mm,
                    "poscfg": poscfg,
                }
                drop_pose_base.update(drop_orientation)
                drop_approach_base = {
                    "x": float(SESSION.drop_target_robot[0]),
                    "y": float(SESSION.drop_target_robot[1]),
                    "z": float(SESSION.drop_target_robot[2]) + approach_mm,
                    "poscfg": poscfg,
                }
                drop_approach_base.update(drop_orientation)
            else:
                drop_pose_base = _pose_from_workspace_mm(
                    workspace_def,
                    drop_target_mm[0],
                    drop_target_mm[1],
                    drop_mm,
                )
                drop_pose_base["poscfg"] = poscfg
                drop_pose_base.update(drop_orientation)
                drop_approach_base = _pose_from_workspace_mm(
                    workspace_def,
                    drop_target_mm[0],
                    drop_target_mm[1],
                    approach_mm,
                )
                drop_approach_base["poscfg"] = poscfg
                drop_approach_base.update(drop_orientation)
            drop_pose_coord5 = _coord5_pose_from_base(drop_pose_base)
            drop_approach_coord5 = _coord5_pose_from_base(drop_approach_base)
            # Clamp Z to safe minimum after conversion to coord5
            drop_pose_coord5["z"] = _clamp_z_for_coord5(
                drop_pose_coord5["z"], use_base_coords=use_base_coordinates
            )
            drop_approach_coord5["z"] = _clamp_z_for_coord5(
                drop_approach_coord5["z"], use_base_coords=use_base_coordinates
            )
        drop_pose = drop_pose_base if use_base_coordinates else drop_pose_coord5
        drop_approach = (
            drop_approach_base if use_base_coordinates else drop_approach_coord5
        )
        gripper_min_width, gripper_max_width = _resolve_gripper_limits()
        summary = {"attempts": 0, "successes": 0, "details": []}

        capture: dict = {
            "timestamp": time.time(),
            "coordinate_id": plate_test_coord_id,
            "approach_height_mm": float(approach_mm),
            "pick_height_mm": float(pick_mm),
            "drop_height_mm": float(drop_mm),
            "grip_timeout_s": float(grip_timeout),
            "targets": [dict(target) for target in targets],
            "plate_meta": plate_meta,
            "drop_target": {
                "workspace_mm": drop_target_mm,
                "coord5_pose": drop_pose_coord5,
                "coord5_approach": drop_approach_coord5,
            },
            "moves": [],
            "grip_events": [],
        }

        async def _refresh_plate_targets() -> bool:
            nonlocal baseline_transform, plate_meta, reference_initialized, targets, targets_by_sequence

            try:
                (
                    new_targets,
                    new_plate_meta,
                    new_baseline,
                    new_reference_initialized,
                ) = await asyncio.to_thread(
                    _resolve_plate_test_coord5_targets, DEFAULT_PLATE_PATH
                )
            except Exception as exc:  # noqa: BLE001
                LOGGER.warning("Plate test dynamic update skipped: %s", exc)
                return False

            reference_initialized = reference_initialized or bool(
                new_reference_initialized
            )
            plate_meta = new_plate_meta
            capture["plate_meta"] = plate_meta
            plate_meta["reference_initialized"] = reference_initialized
            baseline_transform = new_baseline
            targets = new_targets
            updated_targets = {
                int(target["sequence"]): dict(target)
                for target in new_targets
            }
            targets_by_sequence.update(updated_targets)
            capture["targets"] = [dict(target) for target in new_targets]
            _update_plate_alignment()
            return True

        def _record_move(
            label: str, command_pose: dict, command_coord_id: int
        ) -> None:
            posture = session.position_snapshot()
            coord5_xyz, coord5_basis = _resolve_coord5_pose(
                posture, target=command_pose
            )
            coord_id = _current_coordinate_id(session)
            if command_coord_id == ROBOT_COORDINATE_ID:
                command_coord5 = command_pose
            else:
                command_coord5 = _coord5_pose_from_base(command_pose)
            capture["moves"].append(
                {
                    "label": label,
                    "command_coord5": {
                        "x": float(command_coord5["x"]),
                        "y": float(command_coord5["y"]),
                        "z": float(command_coord5["z"]),
                        "a": float(command_coord5.get("a", 0.0)),
                        "b": float(command_coord5.get("b", 0.0)),
                        "c": float(command_coord5.get("c", 0.0)),
                    },
                    "posture_base": _posture_to_dict(posture),
                    "posture_coord5": {
                        "x": float(coord5_xyz[0]),
                        "y": float(coord5_xyz[1]),
                        "z": float(coord5_xyz[2]),
                        "a": float(posture.a),
                        "b": float(posture.b),
                        "c": float(posture.c),
                    },
                    "posture_coord5_basis": coord5_basis,
                    "posture_coordinate_id": coord_id,
                }
            )

        async def _run_sequence(robot_api: CodroidAPI) -> None:
            await _wait_for_operation_resume(run_id)
            await robot_api.power_on()
            await robot_api.set_current_coordinate_id(plate_test_coord_id)
            speed_rate = max(float(SESSION.robot_speed_rate), 0.1)
            await robot_api.set_manual_move_rate(speed_rate)
            capture["robot_speed_rate"] = speed_rate
            speed_percent = int(round(speed_rate * 100))

            def _update_step(label: str) -> None:
                message = f"Plate test: {label.replace('_', ' ')} (speed {speed_percent}%)"
                _set_plate_test_state(message, running=True, label=label)
                _update_operation_message(message, label=label)

            def _pose_xyz(
                posture: RobotPosture,
                command_coord_id: int,
                target: dict,
            ) -> np.ndarray:
                if command_coord_id == ROBOT_COORDINATE_ID:
                    coord5_xyz, _coord_basis = _resolve_coord5_pose(
                        posture, target=target
                    )
                    return coord5_xyz
                return np.array(
                    [
                        float(posture.x),
                        float(posture.y),
                        float(posture.z),
                    ],
                    dtype=np.float64,
                )

            def _travel_z(
                z_a: float, z_b: float, *, use_base: bool
            ) -> float:
                if use_base:
                    return float(max(z_a, z_b))
                return float(min(z_a, z_b))

            def _plate_pose_for_target(
                target: dict,
                z_value: float,
                *,
                use_travel_z: bool,
            ) -> dict:
                coord_xy = target.get("adjusted_xy", target["reference_xy"])
                coord = np.array(
                    [coord_xy[0], coord_xy[1], z_value], dtype=np.float64
                )
                pose_coord5 = {
                    "x": float(coord[0]),
                    "y": float(coord[1]),
                    "z": _clamp_z_for_coord5(
                        float(coord[2]), use_base_coords=use_base_coordinates
                    ),
                    "poscfg": poscfg,
                }
                pose_coord5.update(plate_orientation)
                base = _coord_to_base(coord)
                pose_base = {
                    "x": float(base[0]),
                    "y": float(base[1]),
                    "z": float(base[2]),
                    "poscfg": poscfg,
                }
                pose_base.update(plate_orientation)
                pose = pose_base if use_base_coordinates else pose_coord5
                if use_travel_z:
                    travel_z = _travel_z(
                        float(drop_approach["z"]),
                        float(pose["z"]),
                        use_base=use_base_coordinates,
                    )
                    pose = dict(pose)
                    pose["z"] = travel_z
                return pose

            async def _move_linear_follow_plate_target(
                sequence: int,
                z_value: float,
                *,
                label: str,
                use_travel_z: bool,
                timeout_s: float = SETTINGS.plate_test_move_timeout_s,
                update_interval: float = SETTINGS.plate_test_follow_update_s,
                settle_s: float = SETTINGS.plate_test_settle_s,
                segment_hold_s: float = SETTINGS.plate_test_follow_segment_s,
            ) -> dict:
                await _wait_for_operation_resume(run_id)
                target = targets_by_sequence.get(sequence)
                if target is None:
                    raise RuntimeError(
                        f"Plate test target {sequence} missing from pose data."
                    )
                command_pose = _plate_pose_for_target(
                    target, z_value, use_travel_z=use_travel_z
                )
                _log_move_target(
                    label, command_pose, coordinate_id=plate_test_coord_id
                )
                await robot_api.set_target_pos_type(
                    robot_api.config.target_pos_types.cpos
                )
                stable_since: Optional[float] = None
                deadline = time.monotonic() + timeout_s
                last_pose = command_pose
                while True:
                    await _wait_for_operation_resume(run_id)
                    now = time.monotonic()
                    if now >= deadline:
                        LOGGER.warning(
                            "Plate follow move '%s' timed out after %.1f s.",
                            label,
                            timeout_s,
                        )
                        break
                    refreshed = await _refresh_plate_targets()
                    if not refreshed:
                        LOGGER.warning(
                            "Plate follow move '%s': pose refresh failed.",
                            label,
                        )
                    target = targets_by_sequence.get(sequence)
                    if target is not None:
                        last_pose = _plate_pose_for_target(
                            target, z_value, use_travel_z=use_travel_z
                        )
                    await robot_api.set_target_cpos(
                        robot_api.attach_coordinate_to_cpos(
                            robot_api.build_target_cpos(**last_pose),
                            coordinate_id=plate_test_coord_id,
                        )
                    )
                    await robot_api.set_robot_command(
                        robot_api.config.commands.move_target_linear
                    )
                    segment_deadline = time.monotonic() + max(segment_hold_s, 0.0)
                    while time.monotonic() < segment_deadline:
                        await robot_api.send_command_heartbeat()
                        await asyncio.sleep(min(update_interval, 0.05))
                    posture = session.position_snapshot()
                    curr_xyz = _pose_xyz(
                        posture, plate_test_coord_id, last_pose
                    )
                    target_xyz = np.array(
                        [
                            float(last_pose["x"]),
                            float(last_pose["y"]),
                            float(last_pose["z"]),
                        ],
                        dtype=np.float64,
                    )
                    if (
                        abs(curr_xyz[0] - target_xyz[0])
                        <= SETTINGS.plate_test_tolerance_mm
                        and abs(curr_xyz[1] - target_xyz[1])
                        <= SETTINGS.plate_test_tolerance_mm
                        and abs(curr_xyz[2] - target_xyz[2])
                        <= SETTINGS.plate_test_z_tolerance_mm
                    ):
                        if settle_s <= 0:
                            break
                        if stable_since is None:
                            stable_since = now
                        elif now - stable_since >= settle_s:
                            break
                    else:
                        stable_since = None
                    await asyncio.sleep(update_interval)
                await robot_api.clear_target_position()
                await robot_api.stop_command()
                _record_move(label, last_pose, plate_test_coord_id)
                return last_pose

            async def _segment_follow_plate_target(
                sequence: int,
                z_value: float,
                *,
                label: str,
                use_travel_z: bool,
                segment_min_mm: float,
                segment_max_mm: float,
                segment_fraction: float,
                timeout_s: float = SETTINGS.plate_test_move_timeout_s,
                update_interval: float = SETTINGS.plate_test_follow_update_s,
                segment_hold_s: float = SETTINGS.plate_test_follow_segment_s,
                settle_s: float = SETTINGS.plate_test_settle_s,
            ) -> dict:
                await _wait_for_operation_resume(run_id)
                target = targets_by_sequence.get(sequence)
                if target is None:
                    raise RuntimeError(
                        f"Plate test target {sequence} missing from pose data."
                    )
                last_pose = _plate_pose_for_target(
                    target, z_value, use_travel_z=use_travel_z
                )
                _log_move_target(
                    label, last_pose, coordinate_id=plate_test_coord_id
                )
                deadline = time.monotonic() + timeout_s
                stable_since: Optional[float] = None

                while True:
                    await _wait_for_operation_resume(run_id)
                    now = time.monotonic()
                    if now >= deadline:
                        LOGGER.warning(
                            "Plate segment follow '%s' timed out after %.1f s.",
                            label,
                            timeout_s,
                        )
                        break
                    refreshed = await _refresh_plate_targets()
                    if not refreshed:
                        LOGGER.warning(
                            "Plate segment follow '%s': pose refresh failed.",
                            label,
                        )
                    target = targets_by_sequence.get(sequence)
                    if target is not None:
                        last_pose = _plate_pose_for_target(
                            target, z_value, use_travel_z=use_travel_z
                        )
                    posture = session.position_snapshot()
                    curr_xyz = _pose_xyz(
                        posture, plate_test_coord_id, last_pose
                    )
                    target_xyz = np.array(
                        [
                            float(last_pose["x"]),
                            float(last_pose["y"]),
                            float(last_pose["z"]),
                        ],
                        dtype=np.float64,
                    )
                    delta = target_xyz - curr_xyz
                    planar_dist = float(np.hypot(delta[0], delta[1]))
                    if (
                        abs(delta[0]) <= SETTINGS.plate_test_tolerance_mm
                        and abs(delta[1]) <= SETTINGS.plate_test_tolerance_mm
                        and abs(delta[2]) <= SETTINGS.plate_test_z_tolerance_mm
                    ):
                        if settle_s <= 0:
                            break
                        if stable_since is None:
                            stable_since = now
                        elif now - stable_since >= settle_s:
                            break
                    else:
                        stable_since = None
                    step_ratio = 1.0
                    if planar_dist > 0:
                        step_len = float(
                            np.clip(
                                planar_dist * segment_fraction,
                                segment_min_mm,
                                segment_max_mm,
                            )
                        )
                        step_ratio = min(1.0, step_len / planar_dist)
                    step_pose = dict(last_pose)
                    if planar_dist > 0:
                        step_pose["x"] = float(
                            curr_xyz[0] + delta[0] * step_ratio
                        )
                        step_pose["y"] = float(
                            curr_xyz[1] + delta[1] * step_ratio
                        )
                    await robot_api.set_target_pos_type(
                        robot_api.config.target_pos_types.cpos
                    )
                    await robot_api.set_target_cpos(
                        robot_api.attach_coordinate_to_cpos(
                            robot_api.build_target_cpos(**step_pose),
                            coordinate_id=plate_test_coord_id,
                        )
                    )
                    await robot_api.set_robot_command(
                        robot_api.config.commands.move_target_linear
                    )
                    segment_deadline = time.monotonic() + max(segment_hold_s, 0.0)
                    while time.monotonic() < segment_deadline:
                        await robot_api.send_command_heartbeat()
                        await asyncio.sleep(min(update_interval, 0.05))
                    await asyncio.sleep(update_interval)

                await robot_api.clear_target_position()
                await robot_api.stop_command()
                _record_move(label, last_pose, plate_test_coord_id)
                return last_pose

            async def _await_motion_settle(
                target: dict,
                *,
                label: str,
                command_coord_id: int,
                timeout_s: float = SETTINGS.plate_test_move_timeout_s,
                settle_s: float = SETTINGS.plate_test_settle_s,
                motion_eps_mm: float = SETTINGS.plate_test_motion_eps_mm,
            ) -> None:
                deadline = time.monotonic() + timeout_s
                start = session.position_snapshot()
                start_xyz = _pose_xyz(start, command_coord_id, target)
                target_xyz = np.array(
                    [
                        float(target["x"]),
                        float(target["y"]),
                        float(target["z"]),
                    ],
                    dtype=np.float64,
                )
                if (
                    abs(float(start_xyz[0]) - float(target_xyz[0]))
                    <= SETTINGS.plate_test_tolerance_mm
                    and abs(float(start_xyz[1]) - float(target_xyz[1]))
                    <= SETTINGS.plate_test_tolerance_mm
                    and abs(float(start_xyz[2]) - float(target_xyz[2]))
                    <= SETTINGS.plate_test_z_tolerance_mm
                ):
                    return
                last_xyz = start_xyz
                moved = False
                settle_deadline: Optional[float] = None
                while time.monotonic() < deadline:
                    _ensure_operation_active(run_id)
                    await robot_api.send_command_heartbeat()
                    await asyncio.sleep(0.2)
                    curr = session.position_snapshot()
                    curr_xyz = _pose_xyz(curr, command_coord_id, target)
                    base_delta = float(np.linalg.norm(curr_xyz - last_xyz))
                    if base_delta > motion_eps_mm:
                        moved = True
                        settle_deadline = time.monotonic() + settle_s
                    if moved and settle_deadline is not None:
                        if base_delta > motion_eps_mm:
                            settle_deadline = time.monotonic() + settle_s
                        elif time.monotonic() >= settle_deadline:
                            return
                    last_xyz = curr_xyz
                total = float(np.linalg.norm(last_xyz - start_xyz))
                if not moved:
                    status = session.robot_status_snapshot()
                    LOGGER.warning(
                        "Plate test move timeout at %s: no motion detected (status=%s).",
                        label,
                        status,
                    )
                    return
                raise RuntimeError(
                    f"Plate test move timeout at {label}: motion did not settle "
                    f"(total={total:.2f} mm)."
                )

            async def _move_linear_with_hold(
                command_pose: dict,
                hold_seconds: float = 0.0,
                heartbeat_interval: float = 0.2,
                label: str = "move",
            ) -> None:
                while True:
                    await _wait_for_operation_resume(run_id)
                    _update_step(label)
                    _log_move_target(
                        label, command_pose, coordinate_id=plate_test_coord_id
                    )
                    await robot_api.set_target_pos_type(
                        robot_api.config.target_pos_types.cpos
                    )
                    await robot_api.set_target_cpos(
                        robot_api.attach_coordinate_to_cpos(
                            robot_api.build_target_cpos(**command_pose),
                            coordinate_id=plate_test_coord_id,
                        )
                    )
                    await robot_api.set_robot_command(
                        robot_api.config.commands.move_target_linear
                    )
                    try:
                        await _await_motion_settle(
                            command_pose,
                            label=label,
                            command_coord_id=plate_test_coord_id,
                        )
                    except OperationPaused:
                        continue
                    try:
                        if hold_seconds > 0:
                            deadline = time.monotonic() + hold_seconds
                            while time.monotonic() < deadline:
                                _ensure_operation_active(run_id)
                                await robot_api.send_command_heartbeat()
                                await asyncio.sleep(heartbeat_interval)
                    except OperationPaused:
                        continue
                    await robot_api.clear_target_position()
                    await robot_api.stop_command()
                    _record_move(label, command_pose, plate_test_coord_id)
                    return

            await _move_linear_with_hold(
                drop_approach,
                hold_seconds=SETTINGS.plate_test_move_hold_s,
                label="drop_approach_start",
            )
            await _move_linear_with_hold(
                drop_pose,
                hold_seconds=SETTINGS.plate_test_move_hold_s,
                label="return_to_drop_pose",
            )
            _command_gripper(gripper_max_width, "opened")
            await _move_linear_with_hold(
                drop_approach,
                hold_seconds=SETTINGS.plate_test_move_hold_s,
                label="drop_approach_ready",
            )

            for sequence in target_sequence_order:
                await _wait_for_operation_resume(run_id)
                refreshed = await _refresh_plate_targets()
                if not refreshed:
                    LOGGER.warning(
                        "Plate test target %s: unable to refresh plate pose, using last known position.",
                        sequence,
                    )
                target = targets_by_sequence.get(sequence)
                if target is None:
                    raise RuntimeError(
                        f"Plate test target {sequence} missing from latest pose data."
                    )
                # Use adjusted coordinates: accounts for current plate position as tracked by camera
                coord_xy = target.get("adjusted_xy", target["reference_xy"])
                plate_mm = target.get("plate_mm")

                # Debug logging to diagnose coordinate issues
                LOGGER.info(
                    "Target %d: plate_mm=(%.1f, %.1f), reference_xy=(%.1f, %.1f), adjusted_xy=(%.1f, %.1f), using %s",
                    sequence,
                    plate_mm[0] if plate_mm else 0,
                    plate_mm[1] if plate_mm else 0,
                    target["reference_xy"][0],
                    target["reference_xy"][1],
                    target["adjusted_xy"][0],
                    target["adjusted_xy"][1],
                    "coord5" if not use_base_coordinates else "base",
                )

                approach_coord = np.array(
                    [coord_xy[0], coord_xy[1], approach_mm], dtype=np.float64
                )
                pick_coord = np.array(
                    [coord_xy[0], coord_xy[1], pick_mm], dtype=np.float64
                )
                # Clamp Z to safe minimum when using coordinate system 5
                approach_pose_coord5 = {
                    "x": float(approach_coord[0]),
                    "y": float(approach_coord[1]),
                    "z": _clamp_z_for_coord5(float(approach_coord[2]), use_base_coords=use_base_coordinates),
                    "poscfg": poscfg,
                }
                pick_pose_coord5 = {
                    "x": float(pick_coord[0]),
                    "y": float(pick_coord[1]),
                    "z": _clamp_z_for_coord5(float(pick_coord[2]), use_base_coords=use_base_coordinates),
                    "poscfg": poscfg,
                }
                approach_pose_coord5.update(plate_orientation)
                pick_pose_coord5.update(plate_orientation)
                approach_base = _coord_to_base(approach_coord)
                pick_base = _coord_to_base(pick_coord)
                approach_pose_base = {
                    "x": float(approach_base[0]),
                    "y": float(approach_base[1]),
                    "z": float(approach_base[2]),
                    "poscfg": poscfg,
                }
                pick_pose_base = {
                    "x": float(pick_base[0]),
                    "y": float(pick_base[1]),
                    "z": float(pick_base[2]),
                    "poscfg": poscfg,
                }
                approach_pose_base.update(plate_orientation)
                pick_pose_base.update(plate_orientation)
                approach_pose = (
                    approach_pose_base
                    if use_base_coordinates
                    else approach_pose_coord5
                )
                pick_pose = (
                    pick_pose_base if use_base_coordinates else pick_pose_coord5
                )

                travel_z = _travel_z(
                    float(drop_approach["z"]),
                    float(approach_pose["z"]),
                    use_base=use_base_coordinates,
                )
                cycle_started_at = time.monotonic()
                if use_segment_tracking:
                    await _segment_follow_plate_target(
                        sequence,
                        approach_mm,
                        label=f"target_{sequence}_transit_to_pick",
                        use_travel_z=True,
                        segment_min_mm=segment_min,
                        segment_max_mm=segment_max,
                        segment_fraction=segment_fraction,
                    )
                    last_pick_pose = await _segment_follow_plate_target(
                        sequence,
                        pick_mm,
                        label=f"target_{sequence}_pick",
                        use_travel_z=False,
                        segment_min_mm=segment_min,
                        segment_max_mm=segment_max,
                        segment_fraction=segment_fraction,
                    )
                else:
                    await _move_linear_follow_plate_target(
                        sequence,
                        approach_mm,
                        label=f"target_{sequence}_transit_to_pick",
                        use_travel_z=True,
                        settle_s=0.0,
                    )
                    last_pick_pose = await _move_linear_follow_plate_target(
                        sequence,
                        pick_mm,
                        label=f"target_{sequence}_pick",
                        use_travel_z=False,
                        settle_s=0.0,
                    )
                await _wait_for_operation_resume(run_id)
                _command_gripper(
                    gripper_min_width,
                    "closed",
                    force=GRIPPER_FORCE_MIN,
                    speed=GRIPPER_SPEED_MIN,
                )
                await _wait_for_operation_resume(run_id)
                gripped = await _await_grip_status(grip_timeout)
                capture["grip_events"].append(
                    {
                        "sequence": int(sequence),
                        "gripped": bool(gripped),
                        "timeout_s": float(grip_timeout),
                    }
                )
                summary["attempts"] += 1
                if plate_mm is not None:
                    SESSION.last_pick_mm = (plate_mm[0], plate_mm[1])

                if gripped:
                    summary["successes"] += 1
                    SESSION.robot_has_object = True
                    lift_pose = (last_pick_pose or pick_pose).copy()
                    lift_pose["z"] = travel_z
                    await _move_linear_with_hold(
                        lift_pose,
                        hold_seconds=SETTINGS.plate_test_move_hold_s,
                        label=f"target_{sequence}_lift",
                    )
                    drop_travel_pose = drop_approach.copy()
                    drop_travel_pose["z"] = travel_z
                    await _move_linear_with_hold(
                        drop_travel_pose,
                        hold_seconds=SETTINGS.plate_test_move_hold_s,
                        label=f"target_{sequence}_transit_to_drop",
                    )
                    await _move_linear_with_hold(
                        drop_approach,
                        hold_seconds=SETTINGS.plate_test_move_hold_s,
                        label=f"target_{sequence}_drop_approach",
                    )
                    await _move_linear_with_hold(
                        drop_pose,
                        hold_seconds=SETTINGS.plate_test_move_hold_s,
                        label=f"target_{sequence}_drop_pose",
                    )
                    await _wait_for_operation_resume(run_id)
                    _command_gripper(gripper_max_width, "opened")
                    SESSION.robot_has_object = False
                    await _move_linear_with_hold(
                        drop_approach,
                        hold_seconds=SETTINGS.plate_test_move_hold_s,
                        label=f"target_{sequence}_drop_retreat",
                    )
                    summary["details"].append(f"{sequence}: dropped")
                    _record_pick_cycle(True, cycle_started_at)
                else:
                    await _wait_for_operation_resume(run_id)
                    _command_gripper(gripper_max_width, "opened")
                    lift_pose = (last_pick_pose or pick_pose).copy()
                    lift_pose["z"] = travel_z
                    await _move_linear_with_hold(
                        lift_pose,
                        hold_seconds=SETTINGS.plate_test_move_hold_s,
                        label=f"target_{sequence}_empty_lift",
                    )
                    drop_travel_pose = drop_approach.copy()
                    drop_travel_pose["z"] = travel_z
                    await _move_linear_with_hold(
                        drop_travel_pose,
                        hold_seconds=SETTINGS.plate_test_move_hold_s,
                        label=f"target_{sequence}_empty_transit_to_drop",
                    )
                    await _move_linear_with_hold(
                        drop_approach,
                        hold_seconds=SETTINGS.plate_test_move_hold_s,
                        label=f"target_{sequence}_return_to_drop",
                    )
                    summary["details"].append(f"{sequence}: empty")
                    _record_pick_cycle(False, cycle_started_at)

            await _wait_for_operation_resume(run_id)
            _command_gripper(gripper_max_width, "opened")
            await robot_api.power_off()

        try:
            await _with_robot_api(
                robot_uri or DEFAULT_ROBOT_URI, _run_sequence
            )
        except OperationAbort as exc:
            message = f"Plate test stopped: {exc}"
            _set_plate_test_state(message, running=False)
            _finish_operation(run_id, status="stopped", message=message)
            return message, ""
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("Plate pick & place test failed")
            message = f"Plate test failed: {exc}"
            _set_plate_test_state(message, running=False)
            _finish_operation(run_id, status="idle", message=message)
            return message, ""

        attempts = summary["attempts"]
        successes = summary["successes"]
        detail = ", ".join(summary["details"])
        SESSION.last_action = (
            f"Plate test complete: {successes}/{attempts} picked."
        )
        capture["summary"] = summary
        _save_plate_test_capture(capture)
        analysis = _analyze_plate_test_capture(capture)
        plate_note = (
            f"Plate markers: {plate_meta['count']} · "
            f"Reproj: {plate_meta['reprojection_error_mm']:.2f} mm"
        )
        if plate_meta.get("reference_initialized"):
            plate_note = f"{plate_note} · Plate reference set"
        message = (
            f"Plate test complete: {successes}/{attempts} picked. {plate_note}"
        )
        if detail:
            message = f"{message} ({detail})"
        _set_plate_test_state(message, running=False)
        _finish_operation(run_id, status="idle", message=message)
        return message, analysis

    @app.callback(
        Output("plate-test-status", "children", allow_duplicate=True),
        Input("monitor-interval", "n_intervals"),
        prevent_initial_call=True,
    )
    def _refresh_plate_test_status(_ticks: int) -> str:
        state = SESSION.plate_test_state or {}
        if state.get("running"):
            return state.get("message", "")
        return no_update

    @app.callback(
        Output("footer-gripper-status", "children"),
        Input("footer-gripper-width", "value"),
        Input("footer-gripper-force", "value"),
        Input("footer-gripper-speed", "value"),
        prevent_initial_call=True,
    )
    def _footer_gripper_actions(
        footer_width: Optional[float],
        footer_force: Optional[float],
        footer_speed: Optional[float],
    ) -> str:
        try:
            if SESSION.gripper_busy:
                return "Gripper is busy; wait for motion to finish."
            trigger = callback_context.triggered_id
            if trigger == "footer-gripper-width":
                if _recent_gripper_sync("width", footer_width):
                    return no_update
            if trigger == "footer-gripper-force":
                if _recent_gripper_sync("force", footer_force):
                    return no_update

            applied_force = _safe_int(footer_force, GRIPPER_FORCE_DEFAULT)
            applied_force = min(
                max(applied_force, GRIPPER_FORCE_MIN), GRIPPER_FORCE_MAX
            )
            applied_speed = _safe_int(footer_speed, 80)
            if trigger in {"footer-gripper-width", "footer-gripper-force", "footer-gripper-speed"}:
                width = _safe_float(
                    footer_width, GRIPPER_CLOSE_WIDTH_DEFAULT)
                return _command_gripper(
                    width,
                    "moved",
                    force=applied_force,
                    speed=applied_speed,
                )
            return no_update
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("Footer gripper action failed")
            return f"Robot action failed: {exc}"

    @app.callback(
        Output("footer-gripper-width", "value", allow_duplicate=True),
        Output("footer-gripper-force", "value", allow_duplicate=True),
        Output("footer-gripper-speed", "value", allow_duplicate=True),
        Input("footer-gripper-width-decrease", "n_clicks"),
        Input("footer-gripper-width-increase", "n_clicks"),
        Input("footer-gripper-force-decrease", "n_clicks"),
        Input("footer-gripper-force-increase", "n_clicks"),
        Input("footer-gripper-speed-decrease", "n_clicks"),
        Input("footer-gripper-speed-increase", "n_clicks"),
        State("footer-gripper-width", "value"),
        State("footer-gripper-force", "value"),
        State("footer-gripper-speed", "value"),
        State("footer-gripper-width", "min"),
        State("footer-gripper-width", "max"),
        prevent_initial_call=True,
    )
    def _step_footer_gripper_controls(
        _width_dec: int,
        _width_inc: int,
        _force_dec: int,
        _force_inc: int,
        _speed_dec: int,
        _speed_inc: int,
        width: Optional[float],
        force: Optional[float],
        speed: Optional[float],
        width_min: Optional[float],
        width_max: Optional[float],
    ) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        trigger = callback_context.triggered_id
        if trigger is None:
            return no_update, no_update, no_update
        if SESSION.gripper_busy:
            return no_update, no_update, no_update

        width_value = _safe_float(width, GRIPPER_CLOSE_WIDTH_DEFAULT)
        force_value = _safe_float(force, GRIPPER_FORCE_DEFAULT)
        speed_value = _safe_float(speed, GRIPPER_SPEED_DEFAULT)

        if trigger == "footer-gripper-width-decrease":
            width_value -= 0.5
        elif trigger == "footer-gripper-width-increase":
            width_value += 0.5
        elif trigger == "footer-gripper-force-decrease":
            force_value -= 5
        elif trigger == "footer-gripper-force-increase":
            force_value += 5
        elif trigger == "footer-gripper-speed-decrease":
            speed_value -= 5
        elif trigger == "footer-gripper-speed-increase":
            speed_value += 5

        if width_min is not None:
            width_value = max(width_value, float(width_min))
        if width_max is not None:
            width_value = min(width_value, float(width_max))
        force_value = min(
            max(force_value, GRIPPER_FORCE_MIN), GRIPPER_FORCE_MAX)
        speed_value = min(max(speed_value, 0), 100)

        return width_value, force_value, speed_value

    @app.callback(
        Output("app-notifications", "children", allow_duplicate=True),
        Output("robot-error-state", "data", allow_duplicate=True),
        Input("safety-stop", "n_clicks"),
        State("robot-uri", "value"),
        prevent_initial_call=True,
    )
    async def _handle_safety_stop(
        _clicks: int,
        robot_uri: Optional[str],
    ) -> Tuple[dmc.Notification, dict]:
        try:
            async def _power_off(api: CodroidAPI) -> None:
                await api.stop_command()
                await api.power_off()

            await _with_robot_api(robot_uri or DEFAULT_ROBOT_URI, _power_off)
            SESSION.last_action = "Emergency stop: power off command sent."
            error_state = _activate_robot_error(
                "emergency_stop",
                message="Emergency stop triggered from UI.",
                physical_pressed=False,
            )
            return dmc.Notification(
                action="show",
                title="Emergency stop",
                message="Stop command + power off sent to robot controller.",
                color="red",
                autoClose=5000,
            ), error_state
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("Emergency stop failed")
            error_state = _activate_robot_error(
                "emergency_stop",
                message="Emergency stop triggered from UI.",
                physical_pressed=False,
            )
            return dmc.Notification(
                action="show",
                title="Emergency stop failed",
                message=str(exc),
                color="red",
                autoClose=7000,
            ), error_state

    @app.callback(
        Output("app-notifications", "children", allow_duplicate=True),
        Input("robot-power-toggle", "checked"),
        State("robot-uri", "value"),
        State("robot-power-state", "data"),
        prevent_initial_call=True,
    )
    async def _handle_power_toggle(
        checked: bool,
        robot_uri: Optional[str],
        power_state: Optional[dict],
    ) -> dmc.Notification:
        if power_state and power_state.get("value") == checked:
            return no_update
        try:
            async def _apply_power(api: CodroidAPI) -> None:
                if checked:
                    await api.power_on()
                else:
                    await api.power_off()

            await _with_robot_api(robot_uri or DEFAULT_ROBOT_URI, _apply_power)
            action = "Power on" if checked else "Power off"
            SESSION.last_action = f"{action} command sent."
            return dmc.Notification(
                action="show",
                title="Robot power",
                message=f"{action} command sent to robot controller.",
                color="red" if not checked else "teal",
                autoClose=4000,
            )
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("Power toggle failed")
            return dmc.Notification(
                action="show",
                title="Robot power failed",
                message=str(exc),
                color="red",
                autoClose=7000,
            )

    @app.callback(
        Output("footer-gripper-width", "value"),
        Output("footer-gripper-force", "value"),
        Output("footer-gripper-width", "min", allow_duplicate=True),
        Output("footer-gripper-width", "max", allow_duplicate=True),
        Output("footer-gripper-meta", "children"),
        Output("gripper-orientation", "value"),
        Input("robot-status-interval", "n_intervals"),
        State("gripper-connection-state", "data"),
        State("footer-gripper-width", "value"),
        State("footer-gripper-force", "value"),
        prevent_initial_call=True,
    )
    def _sync_footer_gripper_values(
        _intervals: int,
        gripper_state: Optional[dict],
        current_width: Optional[float],
        current_force: Optional[float],
    ) -> Tuple[
        Optional[float],
        Optional[float],
        Optional[float],
        Optional[float],
        str,
        Optional[str],
    ]:
        if (gripper_state or {}).get("status") != "online":
            return no_update, no_update, no_update, no_update, no_update, no_update
        if SESSION.gripper is None:
            return no_update, no_update, no_update, no_update, no_update, no_update

        status_snapshot = None
        try:
            t_index = GRIPPER_DEFAULTS["t_index"]
            if SESSION.gripper.get_status_snapshot(t_index=t_index) is None:
                SESSION.gripper.start_status_stream()
            status_snapshot = SESSION.gripper.get_status_snapshot(
                t_index=t_index)
            if status_snapshot:
                width = status_snapshot.get("external_width")
                force = status_snapshot.get("force")
                min_width = status_snapshot.get("min_external_width")
                max_width = status_snapshot.get("max_external_width")
                busy = status_snapshot.get("busy")
                gripped = status_snapshot.get("grip_detected")
                status = status_snapshot.get("status")
                outward = status_snapshot.get("finger_orientation_outward")
                orientation = None
                if isinstance(outward, bool):
                    orientation = "outward" if outward else "inward"
                elif isinstance(outward, str):
                    cleaned = outward.strip().lower()
                    if cleaned in {"true", "1", "yes", "on"}:
                        orientation = "outward"
                    elif cleaned in {"false", "0", "no", "off"}:
                        orientation = "inward"
            else:
                width = SESSION.gripper.get_ext_width(t_index=t_index)
                force = SESSION.gripper.get_force(t_index=t_index)
                min_width = SESSION.gripper.get_min_ext_width(t_index=t_index)
                max_width = SESSION.gripper.get_max_ext_width(t_index=t_index)
                busy = SESSION.gripper.isBusy(t_index=t_index)
                gripped = SESSION.gripper.isGripped(t_index=t_index)
                status = SESSION.gripper.getStatus(t_index=t_index)
                orientation = SESSION.gripper.get_finger_orientation_label(
                    t_index=t_index
                )
        except Exception as exc:  # noqa: BLE001
            LOGGER.debug("Failed to poll gripper values: %s", exc)
            return no_update, no_update, no_update, no_update, no_update, no_update

        width_update: Optional[float] = no_update
        force_update: Optional[float] = no_update
        if isinstance(width, (int, float)) and width > -2:
            if current_width is None or abs(width - current_width) > 0.05:
                width_update = float(width)
        if (
            isinstance(force, (int, float))
            and GRIPPER_FORCE_MIN <= force <= GRIPPER_FORCE_MAX
        ):
            if current_force is None or abs(force - current_force) > 0.1:
                force_update = float(force)

        if (
            isinstance(force, (int, float))
            and force < 0
            and status_snapshot
            and isinstance(status_snapshot.get("max_force"), (int, float))
        ):
            # Avoid displaying negative force values when using status snapshots.
            force = None

        min_update: Optional[float] = no_update
        max_update: Optional[float] = no_update
        if isinstance(min_width, (int, float)) and min_width >= 0:
            if (
                SESSION.gripper_min_width is None
                or abs(min_width - SESSION.gripper_min_width) > 0.1
            ):
                min_update = float(min_width)
                SESSION.gripper_min_width = float(min_width)
        if isinstance(max_width, (int, float)) and max_width >= 0:
            if (
                SESSION.gripper_max_width is None
                or abs(max_width - SESSION.gripper_max_width) > 0.1
            ):
                max_update = float(max_width)
                SESSION.gripper_max_width = float(max_width)

        width_value = None if width_update is no_update else float(
            width_update)
        force_value = None if force_update is no_update else float(
            force_update)
        _record_gripper_sync(width_value, force_value)

        if isinstance(force, (int, float)) and force >= 0:
            SESSION.gripper_force = float(force)
        if isinstance(busy, (bool, int, float)) and busy != -2:
            SESSION.gripper_busy = bool(busy)
        if isinstance(gripped, (bool, int, float)) and gripped != -2:
            SESSION.gripper_grip_detected = bool(gripped)
        if isinstance(status, (int, float)) and status != -2:
            SESSION.gripper_status_code = int(status)
        previous_orientation = SESSION.gripper_orientation
        if orientation in {"inward", "outward"}:
            SESSION.gripper_orientation = orientation

        busy_text = (
            "unknown"
            if SESSION.gripper_busy is None
            else ("yes" if SESSION.gripper_busy else "no")
        )
        grip_text = (
            "unknown"
            if SESSION.gripper_grip_detected is None
            else ("yes" if SESSION.gripper_grip_detected else "no")
        )
        meta_text = (
            f"Orientation: {SESSION.gripper_orientation or 'unknown'}"
            f" · Busy: {busy_text}"
            f" · Grip: {grip_text}"
        )
        if isinstance(force, (int, float)) and force >= 0:
            meta_text = f"{meta_text} · Force: {float(force):.0f} N"

        orientation_update: Optional[str] = no_update
        if orientation in {"inward", "outward"} and orientation != previous_orientation:
            orientation_update = orientation

        width_min = (
            SESSION.gripper_min_width
            if SESSION.gripper_min_width is not None
            else None
        )
        width_max = (
            SESSION.gripper_max_width
            if SESSION.gripper_max_width is not None
            else None
        )
        if width_update is no_update and width_min is not None and width_max is not None:
            if isinstance(current_width, (int, float)):
                clamped = min(max(current_width, width_min), width_max)
                if abs(clamped - current_width) > 0.05:
                    width_update = float(clamped)

        return (
            width_update,
            force_update,
            min_update,
            max_update,
            meta_text,
            orientation_update,
        )

    @app.callback(
        Output("gripper-orientation-status", "children"),
        Output("footer-gripper-width", "min", allow_duplicate=True),
        Output("footer-gripper-width", "max", allow_duplicate=True),
        Input("gripper-orientation", "value"),
        State("gripper-connection-state", "data"),
        prevent_initial_call=True,
    )
    def _apply_gripper_orientation(
        orientation: Optional[str],
        gripper_state: Optional[dict],
    ) -> Tuple[str, Optional[float], Optional[float]]:
        if (gripper_state or {}).get("status") != "online":
            return "Gripper is offline.", no_update, no_update
        if SESSION.gripper is None:
            return "Gripper is unavailable.", no_update, no_update
        if orientation not in {"inward", "outward"}:
            return no_update, no_update, no_update
        if SESSION.gripper_orientation == orientation:
            return no_update, no_update, no_update
        try:
            t_index = GRIPPER_DEFAULTS["t_index"]
            result = SESSION.gripper.set_finger_orientation(
                t_index=t_index,
                orientation=orientation,
            )
            if isinstance(result, (int, float)) and result < 0:
                return "Failed to update gripper orientation.", no_update, no_update
            SESSION.gripper_orientation = orientation
            SESSION.gripper_min_width = None
            SESSION.gripper_max_width = None
            min_width = SESSION.gripper.get_min_ext_width(t_index=t_index)
            max_width = SESSION.gripper.get_max_ext_width(t_index=t_index)
            min_update = (
                float(min_width)
                if isinstance(min_width, (int, float)) and min_width >= 0
                else no_update
            )
            max_update = (
                float(max_width)
                if isinstance(max_width, (int, float)) and max_width >= 0
                else no_update
            )
            if min_update is not no_update:
                SESSION.gripper_min_width = min_update
            if max_update is not no_update:
                SESSION.gripper_max_width = max_update
            return f"Orientation set to {orientation}.", min_update, max_update
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("Failed to set gripper orientation")
            return f"Failed to set orientation: {exc}", no_update, no_update

    @app.callback(
        Output("holding-status", "children"),
        Input("robot-status-interval", "n_intervals"),
        State("gripper-connection-state", "data"),
    )
    def _update_holding_badge(
        _intervals: int, gripper_state: Optional[dict]
    ) -> str:
        status = (gripper_state or {}).get("status", "offline")
        if status != "online" or SESSION.gripper is None:
            return "Holding object: unknown (gripper offline)"
        try:
            t_index = GRIPPER_DEFAULTS["t_index"]
            is_gripped = bool(SESSION.gripper.isGripped(t_index=t_index))
            SESSION.robot_has_object = is_gripped
            return f"Holding object: {'yes' if is_gripped else 'no'}"
        except Exception as exc:  # noqa: BLE001
            LOGGER.debug("Unable to query gripper grip state: %s", exc)
            return "Holding object: unknown"

    @app.callback(
        Output("monitor-system-summary", "children"),
        Output("monitor-detection-summary", "children"),
        Output("monitor-pick-summary", "children"),
        Output("use-existing-calibration", "style"),
        Output("use-existing-workspace", "style"),
        Output("use-existing-calibration", "disabled"),
        Output("use-existing-robot-calibration", "disabled"),
        Input("monitor-interval", "n_intervals"),
        Input("camera-interval", "n_intervals"),
        State("camera-connection-state", "data"),
        State("robot-connection-state", "data"),
        State("gripper-connection-state", "data"),
        State("robot-power-state", "data"),
        State("robot-speed-state", "data"),
        State("settings-calib-path", "value"),
        State("settings-robot-calib-path", "value"),
    )
    def _update_monitoring_summary(
        _monitor_ticks: int,
        _camera_ticks: int,
        camera_state: Optional[dict],
        robot_state: Optional[dict],
        gripper_state: Optional[dict],
        robot_power: Optional[dict],
        robot_speed: Optional[dict],
        bundle_path_value: Optional[str],
        robot_calib_path_value: Optional[str],
    ) -> Tuple[list, list, list, dict, dict, bool, bool]:
        camera_status = (camera_state or {}).get("status", "off")
        robot_status = (robot_state or {}).get("status", "offline")
        gripper_status = (gripper_state or {}).get("status", "offline")
        calibration_state = "loaded" if SESSION.calibration_bundle is not None else "missing"
        workspace_state = "ready" if SESSION.workspace_definition is not None else "pending"
        plate_state = "ready" if SESSION.plate_calibration is not None else "pending"
        model_state = SESSION.active_model_path.name if SESSION.active_model_path else "none"
        detection_count = len(SESSION.detections)
        last_action = SESSION.last_action or "No recent actions."

        stream = SESSION.camera_stream
        stream_fps = stream.get_fps() if stream else 0.0
        stream_cfg_fps = stream.config.fps if stream else DEFAULT_FPS
        res_w, res_h = stream.config.resolution if stream else DEFAULT_RESOLUTION

        system_items = [
            dmc.Text(
                f"Camera: {camera_status} · {res_w}x{res_h} @ {stream_fps:.1f}/{stream_cfg_fps:g} FPS",
                size="sm",
            ),
            dmc.Text(
                f"Robot: {robot_status} · speed {int((robot_speed or {}).get('value', SESSION.robot_speed_rate*100))}%",
                size="sm",
            ),
            dmc.Text(
                f"Power: {('on' if (robot_power or {}).get('value') else 'off')}",
                size="sm",
            ),
            dmc.Text(f"Calibration bundle: {calibration_state}", size="sm"),
            dmc.Text(f"Workspace: {workspace_state}", size="sm"),
            dmc.Text(f"Plate calibration: {plate_state}", size="sm"),
            dmc.Text(
                f"Gripper: {gripper_status}",
                size="sm",
            ),
        ]

        detection_items = [
            dmc.Text(f"Model: {model_state}", size="sm"),
            dmc.Text(f"Detections: {detection_count}", size="sm"),
        ]
        if SESSION.last_detection_error:
            detection_items.append(
                dmc.Text(
                    f"Last error: {SESSION.last_detection_error}",
                    size="sm",
                    c="red",
                )
            )

        attempts = SESSION.pick_attempts
        successes = SESSION.pick_successes
        failures = SESSION.pick_failures
        processed = successes
        success_rate = (successes / attempts * 100.0) if attempts else None
        avg_cycle = (
            sum(SESSION.cycle_times) / len(SESSION.cycle_times)
            if SESSION.cycle_times
            else None
        )
        pick_items = [
            dmc.Text(
                f"Holding object: {'yes' if SESSION.robot_has_object else 'no'}",
                size="sm",
            ),
            dmc.Text(f"Last action: {last_action}", size="sm"),
            dmc.Text(f"Processed objects: {processed}", size="sm"),
            dmc.Text(
                f"Attempts: {attempts} · Failures: {failures}",
                size="sm",
            ),
            dmc.Text(
                f"Success rate: {success_rate:.1f}%"
                if success_rate is not None
                else "Success rate: n/a",
                size="sm",
            ),
            dmc.Text(
                f"Avg cycle: {avg_cycle:.2f}s"
                if avg_cycle is not None
                else "Avg cycle: n/a",
                size="sm",
            ),
            dmc.Text(
                f"Last cycle: {SESSION.last_cycle_s:.2f}s"
                if SESSION.last_cycle_s is not None
                else "Last cycle: n/a",
                size="sm",
            ),
        ]
        if SESSION.last_pick_mm:
            pick_items.append(
                dmc.Text(
                    f"Last pick: {SESSION.last_pick_mm[0]:.1f}, {SESSION.last_pick_mm[1]:.1f} mm",
                    size="sm",
                )
            )
        if SESSION.drop_target_mm:
            pick_items.append(
                dmc.Text(
                    f"Drop target: {SESSION.drop_target_mm[0]:.1f}, {SESSION.drop_target_mm[1]:.1f} mm",
                    size="sm",
                )
            )
        if SESSION.drop_target_orientation:
            pick_items.append(
                dmc.Text(
                    "Drop orientation: "
                    f"a={SESSION.drop_target_orientation[0]:.1f}, "
                    f"b={SESSION.drop_target_orientation[1]:.1f}, "
                    f"c={SESSION.drop_target_orientation[2]:.1f}",
                    size="sm",
                )
            )
        if SESSION.drop_target_robot:
            pick_items.append(
                dmc.Text(
                    "Drop robot pose: "
                    f"x={SESSION.drop_target_robot[0]:.1f}, "
                    f"y={SESSION.drop_target_robot[1]:.1f}, "
                    f"z={SESSION.drop_target_robot[2]:.1f}",
                    size="sm",
                )
            )
        bundle_path = _safe_path(bundle_path_value, DEFAULT_BUNDLE_PATH)
        bundle_style = {}
        workspace_style = {} if DEFAULT_WORKSPACE_PATH.exists() else {
            "display": "none"}
        bundle_disabled = not bundle_path.exists()
        robot_path = _safe_path(robot_calib_path_value,
                                DEFAULT_ROBOT_CALIBRATION_PATH)
        robot_disabled = not robot_path.exists()

        return system_items, detection_items, pick_items, bundle_style, workspace_style, bundle_disabled, robot_disabled

    @app.callback(
        Output("app-notifications", "sendNotifications"),
        Output("notification-state", "data"),
        Input("camera-connection-state", "data"),
        Input("robot-connection-state", "data"),
        State("notification-state", "data"),
    )
    def _notify_connections(
        camera_state: Optional[dict],
        robot_state: Optional[dict],
        previous_state: Optional[dict],
    ) -> Tuple[list, dict]:
        prev = previous_state or {"camera": None, "robot": None}
        current = {"camera": prev.get("camera"), "robot": prev.get("robot")}
        notifications = []

        def _append_notification(name: str, state: Optional[dict]) -> None:
            if not state:
                return
            status = state.get("status")
            if prev.get(name) == status:
                return
            current[name] = status
            color = "teal"
            title = f"{name.capitalize()} connected"
            if status in ("error", "off"):
                color = "red" if status == "error" else "gray"
                title = f"{name.capitalize()} disconnected"
            notifications.append(
                {
                    "action": "show",
                    "id": f"{name}-connection",
                    "title": title,
                    "message": state.get("message", ""),
                    "color": color,
                    "autoClose": 4500,
                }
            )

        _append_notification("camera", camera_state)
        _append_notification("robot", robot_state)

        if not notifications:
            return no_update, prev
        return notifications, current

    @app.callback(
        Output("safety-stop", "disabled"),
        Output("robot-power-toggle", "disabled"),
        Output("robot-speed-slider", "disabled"),
        Output("robot-speed-decrease", "disabled"),
        Output("robot-speed-increase", "disabled"),
        Input("robot-connection-state", "data"),
    )
    def _sync_robot_controls(robot_state: Optional[dict]) -> Tuple[bool, bool, bool, bool, bool]:
        status = (robot_state or {}).get("status") or "offline"
        disabled = status != "online"
        return disabled, disabled, disabled, disabled, disabled

    @app.callback(
        Output("camera-calibration-action", "children"),
        Output("calibration-progress-store", "data"),
        Input("calibrate-camera", "n_clicks"),
        Input("use-existing-calibration", "n_clicks"),
        State("settings-calib-path", "value"),
        prevent_initial_call=True,
    )
    def _handle_camera_calibration_actions(
        _calibrate: int,
        _use_existing: int,
        bundle_path: str,
    ) -> dmc.Alert:
        trigger = callback_context.triggered_id

        output_path = DEFAULT_CAPTURE_DIR
        board_path = DEFAULT_BOARD_PATH
        bundle_path = _safe_path(bundle_path, DEFAULT_BUNDLE_PATH)

        if trigger == "use-existing-calibration":
            try:
                result = load_calibration_bundle(bundle_path)
                alert = dmc.Alert(
                    result,
                    icon="✓",
                    title="Using saved calibration",
                    color="green",
                )
                return alert, {"progress": 0}
            except Exception as exc:
                logging.error(
                    f"[CALIBRATE] Error loading saved calibration: {exc}", exc_info=True)
                alert = dmc.Alert(
                    str(exc)[:200],
                    icon="!",
                    title="Load error",
                    color="red",
                )
                return alert, {"progress": 0}

        if trigger == "calibrate-camera":
            try:
                # Clear prior calibration artifacts on disk to avoid mixing old frames
                _clear_calibration_outputs(output_path)

                # Clear previous captures
                SESSION.capture.frames = []
                SESSION.capture.depth_frame = None

                # Capture frames for more robust calibration
                num_frames = max(1, SETTINGS.camera_calibration_frame_count)
                captured_frames = []
                CALIBRATION_PROGRESS["max_frames"] = num_frames
                CALIBRATION_PROGRESS["value"] = 1  # Start progress
                for i in range(num_frames):
                    captured = capture_color_frame(output_path)
                    captured_frames.append(captured.name)
                    # Update progress (1-30 frames = 3-98%)
                    frame_num = i + 1
                    CALIBRATION_PROGRESS["current_frame"] = frame_num
                    CALIBRATION_PROGRESS["value"] = int(
                        (frame_num / num_frames) * 95) + 3
                    # Small delay between captures to avoid motion blur
                    if (
                        i < num_frames - 1
                        and SETTINGS.camera_calibration_frame_delay_s > 0.0
                    ):
                        time.sleep(SETTINGS.camera_calibration_frame_delay_s)

                # Capture depth snapshot
                depth_path = capture_depth_snapshot(output_path)

                # Pre-flight check: ensure the saved frames actually contain ChArUco corners
                detection_counts, usable_frames = _charuco_detection_counts(
                    _list_capture_frames(output_path), board_path
                )
                if usable_frames < SETTINGS.camera_calibration_min_usable_frames:
                    raise RuntimeError(
                        "No usable ChArUco detections in captured frames. "
                        f"Corner counts: {detection_counts}. "
                        "Ensure the board is visible, sharp, and well-lit."
                    )

                # Compute calibration bundle
                result = compute_calibration_bundle(
                    board_path, bundle_path, output_path)

                alert = dmc.Alert(
                    children=[
                        dmc.Stack(
                            [
                                dmc.Text(
                                    f"✓ Captured {len(captured_frames)} RGB frames", size="sm"
                                ),
                                dmc.Text(
                                    "✓ Captured 1 depth snapshot", size="sm"),
                                dmc.Text(
                                    f"✓ Detected {usable_frames}/{num_frames} usable frames", size="sm"),
                                dmc.Text(result, size="sm", fw=600, c="teal"),
                            ],
                            gap="xs",
                        ),
                    ],
                    icon="✓",
                    title="Calibration complete",
                    color="green",
                )
                # Reset progress tracker
                CALIBRATION_PROGRESS["value"] = 0
                CALIBRATION_PROGRESS["current_frame"] = 0
                return alert, {"progress": 0}
            except Exception as exc:
                logging.error(
                    f"[CALIBRATE] Error during calibration: {exc}", exc_info=True)
                alert = dmc.Alert(
                    str(exc)[:200],
                    icon="!",
                    title="Calibration error",
                    color="red",
                )
                # Reset progress tracker on error
                CALIBRATION_PROGRESS["value"] = 0
                CALIBRATION_PROGRESS["current_frame"] = 0
                return alert, {"progress": 0}

        return dmc.Alert("", icon="•", title="Ready", color="gray"), {"progress": 0}

    @app.callback(
        Output("robot-calibration-state", "data", allow_duplicate=True),
        Input("use-existing-robot-calibration", "n_clicks"),
        State("settings-robot-calib-path", "value"),
        prevent_initial_call=True,
    )
    def _load_saved_robot_calibration(
        _clicks: int,
        robot_calib_path_value: Optional[str],
    ) -> dict:
        """Load persisted robot calibration data instead of running the guided flow."""
        path = _safe_path(robot_calib_path_value,
                          DEFAULT_ROBOT_CALIBRATION_PATH)
        if not path.exists():
            message = f"Saved robot calibration not found: {path}"
            _set_robot_guidance(message, status="error",
                                verification_ready=False)
            return {"status": "error", "message": message}
        try:
            _load_robot_calibration(path)
            guidance = (
                "Saved robot calibration loaded. Place a calibration plate on the workspace and run the verification move to confirm."
            )
            _set_robot_guidance(
                guidance,
                status="awaiting_verification",
                verification_ready=True,
                step=5,
                modal_suppressed=True,
            )
            return {
                "status": "awaiting_verification",
                "message": guidance,
                "step": 5,
                "verification_ready": True,
            }
        except Exception as exc:
            message = f"Failed to load saved robot calibration: {exc}"
            _set_robot_guidance(
                message, status="error", verification_ready=False, modal_suppressed=True
            )
            return {"status": "error", "message": message}

    @app.callback(
        Output("workspace-action-status", "children"),
        Input("compute-workspace", "n_clicks"),
        Input("use-existing-workspace", "n_clicks"),
        Input("clear-workspace", "n_clicks"),
        State("settings-marker-a", "value"),
        State("settings-marker-b", "value"),
        State("settings-marker-size", "value"),
        prevent_initial_call=True,
    )
    def _handle_workspace_action(
        _clicks: int,
        _use_existing: int,
        _clear: int,
        marker_a: Optional[float],
        marker_b: Optional[float],
        marker_size: Optional[float],
    ) -> str:
        marker_ids = (
            _safe_int(marker_a, SESSION.marker_ids[0]),
            _safe_int(marker_b, SESSION.marker_ids[1]),
        )
        marker_size_mm = _safe_float(
            marker_size, SESSION.marker_size_mm)
        try:
            # Load calibration bundle from disk if not already in memory
            if SESSION.calibration_bundle is None and DEFAULT_BUNDLE_PATH.exists():
                load_calibration_bundle(DEFAULT_BUNDLE_PATH)
            trigger = callback_context.triggered_id
            if trigger == "clear-workspace":
                _clear_workspace_definition(DEFAULT_WORKSPACE_PATH)
                return "Workspace definition cleared; robot movement unrestricted."
            if trigger == "use-existing-workspace":
                ws = load_workspace_definition(DEFAULT_WORKSPACE_PATH)
                return (
                    f"Workspace loaded from {DEFAULT_WORKSPACE_PATH}: "
                    f"{ws.width_mm:.1f} x {ws.height_mm:.1f} mm"
                )
            ws = compute_workspace_from_latest(marker_ids, marker_size_mm)
            return (
                f"Workspace computed and saved ({DEFAULT_WORKSPACE_PATH}): "
                f"{ws.width_mm:.1f} x {ws.height_mm:.1f} mm"
            )
        except Exception as exc:
            return f"Workspace error: {exc}"

    @app.callback(
        Output("plate-action-status", "children"),
        Output("plate-metrics", "children", allow_duplicate=True),
        Input("camera-interval", "n_intervals"),
        prevent_initial_call=True,
    )
    def _handle_plate_actions(
        _ticks: int,
    ) -> Tuple[str, str]:
        return no_update, no_update

    @app.callback(
        Output("overlay-status", "children"),
        Output("camera-fps", "children"),
        Output("camera-calibration-metrics", "children"),
        Output("workspace-metrics", "children"),
        Output("plate-metrics", "children"),
        Output("detection-metrics", "children"),
        Output("plate-detection-state", "data"),
        Output("calibrate-camera", "disabled"),
        Output("markers-detected-icon", "color"),
        Input("camera-interval", "n_intervals"),
        State("main-tabs", "value"),
        State("calibration-stepper", "active"),
        State("application-tabs", "value"),
        State("settings-marker-a", "value"),
        State("settings-marker-b", "value"),
        State("settings-marker-size", "value"),
        State("overlay-mode-select", "value"),
    )
    def _update_camera_view(
        _ticks: int,
        active_tab: Optional[str],
        calibration_step: Optional[int],
        application_tab: Optional[str],
        marker_a: Optional[float],
        marker_b: Optional[float],
        marker_size: Optional[float],
        overlay_mode: Optional[str],
    ) -> Tuple[str, str, str, str, str, str, dict, bool, str]:
        frame, _depth = get_latest_frame()

        def _camera_calibration_state(
            metrics_text: str, step: int
        ) -> Tuple[bool, str]:
            if step != 0:
                return True, "gray"
            if not metrics_text or "ChArUco corners:" not in metrics_text:
                return True, "gray"
            try:
                count = int(metrics_text.split(":")[-1].strip().split()[0])
                is_disabled = count < 16
                return is_disabled, "green" if not is_disabled else "red"
            except (ValueError, IndexError):
                return True, "gray"

        plate_state = {
            "count": 0,
            "ok": False,
            "stable": False,
            "center_px": None,
            "movement_px": None,
            "timestamp": time.time(),
            "message": "Waiting for plate markers",
        }
        capture_fps = SESSION.camera_stream.get_fps() if SESSION.camera_stream else 0.0
        overlay_fps_text = _format_fps(capture_fps, MJPEG_STREAM.get_fps())
        # if _ticks % 30 == 0:
        #     logging.debug(
        #         "_update_camera_view tick; frame available=%s", frame is not None)

        if frame is None:
            # Don't add placeholder to MJPEG stream - let video element show its poster image
            # Only log every 30 calls to reduce noise
            if _ticks % 30 == 0:
                is_running = SESSION.camera_stream and SESSION.camera_stream.is_running()
                logging.info(
                    "No frame from camera. Stream running: %s", is_running)
            return (
                "Camera stream is stopped.",
                overlay_fps_text,
                "ChArUco corners: --",
                "Workspace: --",
                "Plate error: --",
                "Detections: --",
                plate_state,
                True,
                "gray",
            )

        # Check if this is a new frame by comparing object identity
        current_frame_id = id(frame)
        is_new_frame = current_frame_id != SESSION.last_frame_id
        if is_new_frame:
            SESSION.last_frame_id = current_frame_id

        # Frame received, dispatch to background processor for heavy overlays.
        tab = active_tab or "applications"
        calibration_step = calibration_step or 0
        application_tab = application_tab or "workspace"
        board_path = DEFAULT_BOARD_PATH
        plate_path = DEFAULT_PLATE_PATH
        marker_ids = (
            _safe_int(marker_a, SESSION.marker_ids[0]),
            _safe_int(marker_b, SESSION.marker_ids[1]),
        )
        marker_size_mm = _safe_float(
            marker_size, SESSION.marker_size_mm)

        requested_overlay = overlay_mode or "auto"
        task = _select_overlay_task(
            tab,
            calibration_step,
            application_tab,
            board_path,
            marker_ids,
            marker_size_mm,
            plate_path,
        )
        if requested_overlay != "auto":
            if requested_overlay == "calibration":
                task = OverlayTask("manual-calibration",
                                   _run_charuco_task(board_path))
            elif requested_overlay == "workspace":
                task = OverlayTask(
                    "manual-workspace",
                    _run_workspace_task(marker_ids, marker_size_mm),
                )
            elif requested_overlay == "plate":
                task = OverlayTask("manual-plate", _run_plate_task(plate_path))
            elif requested_overlay == "detection":
                task = OverlayTask("manual-detection", _run_detection_task())
            elif requested_overlay == "none":
                task = None
        overlay: Optional[OverlayResult] = None
        if task is not None:
            FRAME_PROCESSOR.submit(task, frame)
            overlay = FRAME_PROCESSOR.get_latest(task.key)
            if calibration_step == 1 and task.key != "calibration-plate":
                _refresh_plate_status(frame, plate_path)
        elif calibration_step == 1:
            _refresh_plate_status(frame, plate_path)

        vis = overlay.vis if overlay else frame
        status_text = overlay.status if overlay else "No overlay applied."
        metric_text = overlay.metric if overlay else "--"
        mode_label = {
            "auto": "Overlay: Auto",
            "calibration": "Overlay: Calibration",
            "workspace": "Overlay: Workspace",
            "plate": "Overlay: Plate",
            "detection": "Overlay: Detections",
            "none": "Overlay: Off",
        }.get(requested_overlay, "Overlay: Auto")
        status_text = f"{mode_label} · {status_text}"
        plate_state = PLATE_STATUS.copy()

        if tab == "calibration":
            if calibration_step == 0:
                if is_new_frame:
                    MJPEG_STREAM.add_frame(vis)
                overlay_fps_text = _format_fps(
                    capture_fps, MJPEG_STREAM.get_fps())
                calibrate_disabled, marker_color = _camera_calibration_state(
                    metric_text, calibration_step
                )
                return (
                    status_text,
                    overlay_fps_text,
                    metric_text,
                    "Workspace: --",
                    "Plate error: --",
                    "Detections: --",
                    plate_state,
                    calibrate_disabled,
                    marker_color,
                )
            if is_new_frame:
                MJPEG_STREAM.add_frame(vis)
            overlay_fps_text = _format_fps(
                capture_fps, MJPEG_STREAM.get_fps())
            metrics_text = "ChArUco corners: --"
            calibrate_disabled, marker_color = _camera_calibration_state(
                metrics_text, calibration_step
            )
            return (
                status_text,
                overlay_fps_text,
                metrics_text,
                "Workspace: --",
                metric_text,
                "Detections: --",
                plate_state,
                calibrate_disabled,
                marker_color,
            )

        if tab == "applications":
            if is_new_frame:
                MJPEG_STREAM.add_frame(vis)
            overlay_fps_text = _format_fps(
                capture_fps, MJPEG_STREAM.get_fps())
            metrics_text = "ChArUco corners: --"
            calibrate_disabled, marker_color = _camera_calibration_state(
                metrics_text, calibration_step
            )
            return (
                status_text,
                overlay_fps_text,
                metrics_text,
                metric_text if application_tab == "workspace" else "Workspace: --",
                "Plate error: --",
                metric_text if application_tab == "pick-place" else "Detections: --",
                plate_state,
                calibrate_disabled,
                marker_color,
            )

        # Monitoring or any other tab: keep stream alive and surface metrics.
        if is_new_frame:
            MJPEG_STREAM.add_frame(vis)
        overlay_fps_text = _format_fps(capture_fps, MJPEG_STREAM.get_fps())
        metrics_text = "ChArUco corners: --"
        calibrate_disabled, marker_color = _camera_calibration_state(
            metrics_text, calibration_step
        )
        return (
            status_text,
            overlay_fps_text,
            metrics_text,
            "Workspace: --",
            "Plate error: --",
            "Detections: --",
            plate_state,
            calibrate_disabled,
            marker_color,
        )

    @app.callback(
        Output("plate-detection-summary", "children"),
        Output("plate-detection-summary", "color"),
        Output("start-robot-calibration", "disabled"),
        Output("start-robot-calibration", "variant"),
        Input("plate-detection-state", "data"),
        Input("robot-connection-state", "data"),
        State("robot-calibration-state", "data"),
    )
    def _update_plate_summary(
        plate_state: Optional[dict],
        robot_conn_state: Optional[dict],
        calib_state: Optional[dict],
    ) -> Tuple[str, str, bool, str]:
        state = plate_state or {}
        count = state.get("count", 0)
        ok = bool(state.get("ok"))
        stable = bool(state.get("stable", ok))
        calib_status = (calib_state or {}).get("status")
        in_progress = calib_status in {
            "running",
            "awaiting_verification",
            "verifying",
        }
        if state.get_calibration_guidance_snapshot().get("status") in {
            "running",
            "awaiting_verification",
            "verifying",
        }:
            in_progress = True
        robot_online = (robot_conn_state or {}).get("status") == "online"
        posture_ready, posture_message = _robot_posture_ready()

        # Determine message based on the actual blocking issue
        if in_progress:
            if calib_status == "awaiting_verification":
                message = "Awaiting verification move."
                color = "yellow"
            elif calib_status == "verifying":
                message = "Verification in progress..."
                color = "blue"
            else:
                message = "Calibration in progress..."
                color = "blue"
        elif not robot_online:
            message = "Robot not connected. Click 'Test robot connection' to establish connection."
            color = "red"
        elif not ok:
            message = state.get("message") or f"Detected {count} marker(s)."
            color = "yellow"
        elif not stable:
            message = state.get(
                "message") or "Calibration plate moved; hold it steady."
            color = "yellow"
        elif not posture_ready:
            message = posture_message
            color = "yellow"
        else:
            message = "Plate markers stable. Ready to calibrate."
            color = "green"

        disabled = not ok or not stable or not posture_ready or in_progress or not robot_online
        variant = "filled" if not disabled else "light"
        return message, color, disabled, variant

    @app.callback(
        Output("run-verification-inline", "disabled"),
        Input("plate-detection-state", "data"),
        Input("robot-calibration-state", "data"),
        Input("robot-connection-state", "data"),
    )
    def _sync_verification_inline_button(
        plate_state: Optional[dict],
        calibration_state: Optional[dict],
        robot_state: Optional[dict],
    ) -> bool:
        plate_ok = bool((plate_state or {}).get("ok"))
        plate_stable = bool((plate_state or {}).get(
            "stable", plate_ok))
        guidance = calibration_state or {}
        status = guidance.get("status")
        in_progress = status in {"running",
                                 "awaiting_verification", "verifying"}
        robot_online = (robot_state or {}).get("status") == "online"
        has_calibration_points = len(SESSION.robot_calibration_points) >= 3
        return not (
            plate_ok
            and plate_stable
            and robot_online
            and has_calibration_points
            and not in_progress
        )

    @app.callback(
        Output("current-robot-position", "children"),
        Input("robot-connection-state", "data"),
        Input("robot-status-interval", "n_intervals"),
    )
    def _update_robot_position_display(
        robot_conn_state: Optional[dict],
        _intervals: int,
    ) -> str:
        """Display current robot position with real-time updates."""
        session = _get_robot_session()
        pos = session.position_snapshot()
        age_s = pos.age_seconds()

        # Show position if we have recent data (from any source)
        if session.posture_seen() and age_s < 2.0:
            return (
                f"🤖 Robot position: "
                f"X={pos.x:.1f}, "
                f"Y={pos.y:.1f}, "
                f"Z={pos.z:.1f}"
            )

        # Check if robot is connected via test connection
        if (robot_conn_state or {}).get("status") == "online":
            if not session.posture_seen():
                return "Robot: Awaiting posture stream..."
            return f"Robot: Position stale (last seen {age_s:.1f}s ago)"

        return "Robot: --"

    @app.callback(
        Output("robot-power-state", "data"),
        Input("robot-status-interval", "n_intervals"),
        State("robot-connection-state", "data"),
        State("robot-uri", "value"),
        State("robot-power-state", "data"),
        prevent_initial_call=True,
    )
    async def _sync_robot_power_state(
        _intervals: int,
        robot_state: Optional[dict],
        robot_uri: Optional[str],
        current_state: Optional[dict],
    ) -> dict:
        if (robot_state or {}).get("status") != "online":
            return no_update
        now = time.time()
        if now - SESSION.robot_status_last_poll < 1.0:
            return no_update

        async def _poll(api: CodroidAPI) -> None:
            await api.read_system_data()
            await api.read_global_data()

        try:
            await _with_robot_api(robot_uri or DEFAULT_ROBOT_URI, _poll)
            SESSION.robot_status_last_poll = now
        except Exception as exc:  # noqa: BLE001
            LOGGER.debug("Robot power poll failed: %s", exc)
            return no_update

        session = _get_robot_session()
        power_on = session.robot_power_on()
        if power_on is None:
            return no_update
        desired = {"value": bool(power_on), "timestamp": time.time()}
        if current_state and current_state.get("value") == desired["value"]:
            return no_update
        return desired

    @app.callback(
        Output("robot-power-toggle", "checked"),
        Input("robot-power-state", "data"),
        State("robot-power-toggle", "checked"),
        prevent_initial_call=True,
    )
    def _sync_power_toggle(
        power_state: Optional[dict],
        current: Optional[bool],
    ) -> bool:
        if not power_state or power_state.get("value") is None:
            return no_update
        desired = bool(power_state.get("value"))
        if current is None or current != desired:
            return desired
        return no_update

    @app.callback(
        Output("robot-error-state", "data"),
        Input("robot-status-interval", "n_intervals"),
        State("robot-error-state", "data"),
        prevent_initial_call=True,
    )
    def _sync_robot_error_state(
        _intervals: int,
        current_state: Optional[dict],
    ) -> dict:
        session = _get_robot_session()
        status = session.robot_status_snapshot()
        status_age = session.robot_status_age_seconds()
        if not status or status_age is None or status_age > 2.0:
            return no_update

        warning = session.robot_warning_snapshot()
        warning_age = session.robot_warning_age_seconds()
        error_msg = session.robot_error_snapshot()
        error_age = session.robot_error_age_seconds()

        emergency_active = _detect_emergency_stop_status(status)
        overspeed_active = _detect_overspeed_status(status)
        joint_active = _detect_joint_protection_status(status)
        wrong_tool_active = _detect_wrong_tool_status(status)
        drag_active = _detect_drag_not_allowed_status(status)

        if warning and warning_age is not None and warning_age <= 2.0:
            emergency_active = emergency_active or CodroidAPI.is_emergency_button_warning(
                warning
            )
            overspeed_active = overspeed_active or CodroidAPI.is_overspeed_warning(
                warning
            )
            joint_active = joint_active or CodroidAPI.is_joint_protection_warning(
                warning
            )
            drag_active = drag_active or CodroidAPI.is_drag_not_allowed_warning(
                warning
            )

        if error_msg and error_age is not None and error_age <= 5.0:
            wrong_tool_active = wrong_tool_active or CodroidAPI.is_wrong_tool_error(
                error_msg
            )

        physical_pressed = session.emergency_button_active()

        if emergency_active:
            snapshot = _activate_robot_error(
                "emergency_stop",
                message="Emergency stop reported by robot controller.",
                physical_pressed=physical_pressed,
            )
        elif overspeed_active:
            snapshot = _activate_robot_error(
                "overspeed",
                message="Overspeed alarm reported by robot controller.",
                physical_pressed=False,
            )
        elif joint_active:
            snapshot = _activate_robot_error(
                "joint_protection",
                message="Joint protection alarm reported by robot controller.",
                physical_pressed=False,
            )
        elif wrong_tool_active:
            snapshot = _activate_robot_error(
                "wrong_tool",
                message="Tool/payload mismatch reported by robot controller.",
                physical_pressed=False,
            )
        elif drag_active:
            snapshot = _activate_robot_error(
                "drag_not_allowed",
                message="Drag not allowed warning reported by robot controller.",
                physical_pressed=False,
            )
        else:
            if (current_state or {}).get("error_type") == "emergency_stop":
                snapshot = _update_robot_error_state(
                    physical_pressed=physical_pressed
                )
            else:
                snapshot = _error_snapshot()

        if current_state == snapshot:
            return no_update
        return snapshot

    @app.callback(
        Output("robot-rescue-state", "data"),
        Input("robot-status-interval", "n_intervals"),
        State("robot-rescue-state", "data"),
        prevent_initial_call=True,
    )
    def _sync_robot_rescue_state(
        _intervals: int,
        current_state: Optional[dict],
    ) -> dict:
        session = _get_robot_session()
        status = session.robot_status_snapshot()
        status_age = session.robot_status_age_seconds()
        if not status or status_age is None or status_age > 2.0:
            return no_update
        active = _detect_rescue_mode_status(status)
        if current_state and current_state.get("active") == active:
            return no_update
        return {"active": active, "timestamp": time.time()}

    @app.callback(
        Output("robot-controls-standard", "style"),
        Output("robot-controls-rescue", "style"),
        Output("rescue-mode-status", "children"),
        Output("rescue-mode-enable", "disabled"),
        Output("rescue-mode-exit", "style"),
        Input("robot-rescue-state", "data"),
    )
    def _toggle_rescue_controls(
        rescue_state: Optional[dict],
    ) -> Tuple[dict, dict, str, bool, dict]:
        active = bool((rescue_state or {}).get("active"))
        if active:
            return (
                {"display": "none"},
                {"display": "block"},
                "Rescue mode active.",
                True,
                {"display": "inline-flex"},
            )
        return (
            {"display": "flex"},
            {"display": "none"},
            "Rescue mode idle.",
            False,
            {"display": "none"},
        )

    @app.callback(
        Output("emergency-stop-modal", "opened"),
        Output("emergency-stop-modal-text", "children"),
        Output("emergency-ack", "disabled"),
        Input("robot-error-state", "data"),
    )
    def _update_emergency_modal(
        error_state: Optional[dict],
    ) -> Tuple[bool, str, bool]:
        state = error_state or {}
        active = bool(state.get("active")) and state.get(
            "error_type") == "emergency_stop"
        message = state.get("message") or "Emergency stop detected."
        physical_pressed = bool(state.get("physical_pressed"))
        return active, message, physical_pressed

    @app.callback(
        Output("overspeed-modal", "opened"),
        Output("overspeed-modal-text", "children"),
        Input("robot-error-state", "data"),
    )
    def _update_overspeed_modal(
        error_state: Optional[dict],
    ) -> Tuple[bool, str]:
        state = error_state or {}
        active = bool(state.get("active")) and state.get(
            "error_type") == "overspeed"
        message = state.get("message") or "Overspeed alarm detected."
        return active, message

    @app.callback(
        Output("joint-protection-modal", "opened"),
        Output("joint-protection-modal-text", "children"),
        Input("robot-error-state", "data"),
    )
    def _update_joint_protection_modal(
        error_state: Optional[dict],
    ) -> Tuple[bool, str]:
        state = error_state or {}
        active = bool(state.get("active")) and state.get(
            "error_type") == "joint_protection"
        message = state.get("message") or "Joint protection warning detected."
        return active, message

    @app.callback(
        Output("wrong-tool-modal", "opened"),
        Output("wrong-tool-modal-text", "children"),
        Input("robot-error-state", "data"),
    )
    def _update_wrong_tool_modal(
        error_state: Optional[dict],
    ) -> Tuple[bool, str]:
        state = error_state or {}
        active = bool(state.get("active")) and state.get(
            "error_type") == "wrong_tool"
        message = state.get("message") or "Tool/payload mismatch detected."
        return active, message

    @app.callback(
        Output("drag-not-allowed-modal", "opened"),
        Output("drag-not-allowed-modal-text", "children"),
        Input("robot-error-state", "data"),
    )
    def _update_drag_not_allowed_modal(
        error_state: Optional[dict],
    ) -> Tuple[bool, str]:
        state = error_state or {}
        active = bool(state.get("active")) and state.get(
            "error_type") == "drag_not_allowed"
        message = state.get("message") or "Drag not allowed warning detected."
        return active, message

    @app.callback(
        Output("robot-error-state", "data", allow_duplicate=True),
        Output("app-notifications", "children", allow_duplicate=True),
        Input("emergency-ack", "n_clicks"),
        State("robot-error-state", "data"),
        State("robot-uri", "value"),
        prevent_initial_call=True,
    )
    async def _ack_emergency_stop(
        _clicks: int,
        error_state: Optional[dict],
        robot_uri: Optional[str],
    ) -> Tuple[dict, dmc.Notification]:
        state = error_state or {}
        if state.get("error_type") != "emergency_stop":
            return no_update, no_update
        if state.get("physical_pressed"):
            return no_update, no_update

        async def _clear(api: CodroidAPI) -> None:
            await api.clear_emergency_stop()

        try:
            await _with_robot_api(robot_uri or DEFAULT_ROBOT_URI, _clear)
            snapshot = _acknowledge_robot_error()
            return (
                snapshot,
                dmc.Notification(
                    action="show",
                    title="Emergency stop acknowledged",
                    message="Recovery sequence sent. Program remains paused.",
                    color="red",
                    autoClose=5000,
                ),
            )
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("Failed to clear emergency stop")
            return (
                no_update,
                dmc.Notification(
                    action="show",
                    title="Emergency stop clear failed",
                    message=str(exc),
                    color="red",
                    autoClose=7000,
                ),
            )

    @app.callback(
        Output("robot-error-state", "data", allow_duplicate=True),
        Output("app-notifications", "children", allow_duplicate=True),
        Input("overspeed-clear", "n_clicks"),
        Input("overspeed-continue", "n_clicks"),
        Input("overspeed-stay-paused", "n_clicks"),
        Input("overspeed-reset", "n_clicks"),
        State("robot-error-state", "data"),
        State("robot-uri", "value"),
        prevent_initial_call=True,
    )
    async def _handle_overspeed_response(
        _clear_clicks: int,
        _continue_clicks: int,
        _stay_clicks: int,
        _reset_clicks: int,
        error_state: Optional[dict],
        robot_uri: Optional[str],
    ) -> Tuple[dict, dmc.Notification]:
        state = error_state or {}
        if state.get("error_type") != "overspeed":
            return no_update, no_update

        trigger = callback_context.triggered_id

        async def _clear(api: CodroidAPI) -> None:
            await api.clear_overspeed()

        if trigger == "overspeed-clear":
            try:
                await _with_robot_api(robot_uri or DEFAULT_ROBOT_URI, _clear)
                snapshot = _acknowledge_robot_error()
                _pause_operation("overspeed", "Paused after overspeed clear.")
                return (
                    snapshot,
                    dmc.Notification(
                        action="show",
                        title="Overspeed cleared",
                        message="Recovery sequence sent. Program remains paused.",
                        color="orange",
                        autoClose=5000,
                    ),
                )
            except Exception as exc:  # noqa: BLE001
                LOGGER.exception("Failed to clear overspeed")
                return (
                    no_update,
                    dmc.Notification(
                        action="show",
                        title="Overspeed clear failed",
                        message=str(exc),
                        color="red",
                        autoClose=7000,
                    ),
                )
        if trigger == "overspeed-continue":
            try:
                await _with_robot_api(robot_uri or DEFAULT_ROBOT_URI, _clear)
                _resume_operation("Resuming after overspeed.")
                snapshot = _acknowledge_robot_error()
                return (
                    snapshot,
                    dmc.Notification(
                        action="show",
                        title="Overspeed cleared",
                        message="Recovery sequence sent. Program resumed.",
                        color="teal",
                        autoClose=5000,
                    ),
                )
            except Exception as exc:  # noqa: BLE001
                LOGGER.exception("Failed to clear overspeed")
                return (
                    no_update,
                    dmc.Notification(
                        action="show",
                        title="Overspeed clear failed",
                        message=str(exc),
                        color="red",
                        autoClose=7000,
                    ),
                )
        if trigger == "overspeed-stay-paused":
            snapshot = _acknowledge_robot_error()
            _pause_operation(
                "overspeed", "Paused after overspeed acknowledgment.")
            return (
                snapshot,
                dmc.Notification(
                    action="show",
                    title="Overspeed acknowledged",
                    message="Program remains paused.",
                    color="yellow",
                    autoClose=4500,
                ),
            )
        if trigger == "overspeed-reset":
            _request_stop_operation("Program canceled after overspeed.")

            async def _stop_motion(api: CodroidAPI) -> None:
                await api.stop_command()

            try:
                await _with_robot_api(robot_uri or DEFAULT_ROBOT_URI, _stop_motion)
            except Exception as exc:  # noqa: BLE001
                LOGGER.debug("Stop command failed: %s", exc)
            _reset_operation_state("Program reset after overspeed.")
            snapshot = _acknowledge_robot_error()
            return (
                snapshot,
                dmc.Notification(
                    action="show",
                    title="Program reset",
                    message="Overspeed acknowledged and program reset.",
                    color="red",
                    autoClose=5000,
                ),
            )

        return no_update, no_update

    @app.callback(
        Output("robot-error-state", "data", allow_duplicate=True),
        Output("app-notifications", "children", allow_duplicate=True),
        Input("joint-protection-clear", "n_clicks"),
        Input("joint-protection-stay-paused", "n_clicks"),
        State("robot-error-state", "data"),
        State("robot-uri", "value"),
        prevent_initial_call=True,
    )
    async def _handle_joint_protection_response(
        _clear_clicks: int,
        _stay_clicks: int,
        error_state: Optional[dict],
        robot_uri: Optional[str],
    ) -> Tuple[dict, dmc.Notification]:
        state = error_state or {}
        if state.get("error_type") != "joint_protection":
            return no_update, no_update

        trigger = callback_context.triggered_id

        async def _clear(api: CodroidAPI) -> None:
            await api.clear_joint_protection()

        if trigger == "joint-protection-clear":
            try:
                await _with_robot_api(robot_uri or DEFAULT_ROBOT_URI, _clear)
                snapshot = _acknowledge_robot_error()
                _pause_operation(
                    "joint_protection", "Paused after joint protection clear."
                )
                return (
                    snapshot,
                    dmc.Notification(
                        action="show",
                        title="Joint protection cleared",
                        message="Recovery sequence sent. Program remains paused.",
                        color="orange",
                        autoClose=5000,
                    ),
                )
            except Exception as exc:  # noqa: BLE001
                LOGGER.exception("Failed to clear joint protection")
                return (
                    no_update,
                    dmc.Notification(
                        action="show",
                        title="Joint protection clear failed",
                        message=str(exc),
                        color="red",
                        autoClose=7000,
                    ),
                )
        if trigger == "joint-protection-stay-paused":
            snapshot = _acknowledge_robot_error()
            _pause_operation(
                "joint_protection",
                "Paused after joint protection acknowledgment.",
            )
            return (
                snapshot,
                dmc.Notification(
                    action="show",
                    title="Joint protection acknowledged",
                    message="Program remains paused.",
                    color="yellow",
                    autoClose=4500,
                ),
            )

        return no_update, no_update

    @app.callback(
        Output("robot-error-state", "data", allow_duplicate=True),
        Output("app-notifications", "children", allow_duplicate=True),
        Input("wrong-tool-clear", "n_clicks"),
        Input("wrong-tool-rescue", "n_clicks"),
        State("robot-error-state", "data"),
        State("robot-uri", "value"),
        prevent_initial_call=True,
    )
    async def _handle_wrong_tool_response(
        _clear_clicks: int,
        _rescue_clicks: int,
        error_state: Optional[dict],
        robot_uri: Optional[str],
    ) -> Tuple[dict, dmc.Notification]:
        state = error_state or {}
        if state.get("error_type") != "wrong_tool":
            return no_update, no_update

        trigger = callback_context.triggered_id

        async def _clear(api: CodroidAPI) -> None:
            await api.clear_tool_error()

        async def _enter_rescue(api: CodroidAPI) -> None:
            await api.enter_rescue_mode()

        if trigger == "wrong-tool-clear":
            try:
                await _with_robot_api(robot_uri or DEFAULT_ROBOT_URI, _clear)
                snapshot = _acknowledge_robot_error()
                _pause_operation(
                    "wrong_tool", "Paused after tool error clear.")
                return (
                    snapshot,
                    dmc.Notification(
                        action="show",
                        title="Tool error cleared",
                        message="Recovery sequence sent. Program remains paused.",
                        color="red",
                        autoClose=5000,
                    ),
                )
            except Exception as exc:  # noqa: BLE001
                LOGGER.exception("Failed to clear tool error")
                return (
                    no_update,
                    dmc.Notification(
                        action="show",
                        title="Tool error clear failed",
                        message=str(exc),
                        color="red",
                        autoClose=7000,
                    ),
                )
        if trigger == "wrong-tool-rescue":
            try:
                await _with_robot_api(robot_uri or DEFAULT_ROBOT_URI, _enter_rescue)
                return (
                    no_update,
                    dmc.Notification(
                        action="show",
                        title="Rescue mode requested",
                        message="Rescue mode command sent.",
                        color="orange",
                        autoClose=5000,
                    ),
                )
            except Exception as exc:  # noqa: BLE001
                LOGGER.exception("Failed to enter rescue mode")
                return (
                    no_update,
                    dmc.Notification(
                        action="show",
                        title="Rescue mode failed",
                        message=str(exc),
                        color="red",
                        autoClose=7000,
                    ),
                )

        return no_update, no_update

    @app.callback(
        Output("robot-error-state", "data", allow_duplicate=True),
        Output("app-notifications", "children", allow_duplicate=True),
        Input("drag-not-allowed-clear", "n_clicks"),
        Input("drag-not-allowed-stay-paused", "n_clicks"),
        State("robot-error-state", "data"),
        State("robot-uri", "value"),
        prevent_initial_call=True,
    )
    async def _handle_drag_not_allowed_response(
        _clear_clicks: int,
        _stay_clicks: int,
        error_state: Optional[dict],
        robot_uri: Optional[str],
    ) -> Tuple[dict, dmc.Notification]:
        state = error_state or {}
        if state.get("error_type") != "drag_not_allowed":
            return no_update, no_update

        trigger = callback_context.triggered_id

        async def _clear(api: CodroidAPI) -> None:
            await api.clear_drag_not_allowed()

        if trigger == "drag-not-allowed-clear":
            try:
                await _with_robot_api(robot_uri or DEFAULT_ROBOT_URI, _clear)
                snapshot = _acknowledge_robot_error()
                _pause_operation(
                    "drag_not_allowed", "Paused after drag warning clear."
                )
                return (
                    snapshot,
                    dmc.Notification(
                        action="show",
                        title="Drag warning cleared",
                        message="Recovery sequence sent. Program remains paused.",
                        color="orange",
                        autoClose=5000,
                    ),
                )
            except Exception as exc:  # noqa: BLE001
                LOGGER.exception("Failed to clear drag warning")
                return (
                    no_update,
                    dmc.Notification(
                        action="show",
                        title="Drag warning clear failed",
                        message=str(exc),
                        color="red",
                        autoClose=7000,
                    ),
                )
        if trigger == "drag-not-allowed-stay-paused":
            snapshot = _acknowledge_robot_error()
            _pause_operation(
                "drag_not_allowed",
                "Paused after drag warning acknowledgment.",
            )
            return (
                snapshot,
                dmc.Notification(
                    action="show",
                    title="Drag warning acknowledged",
                    message="Program remains paused.",
                    color="yellow",
                    autoClose=4500,
                ),
            )

        return no_update, no_update

    @app.callback(
        Output("app-notifications", "children", allow_duplicate=True),
        Input("rescue-mode-enable", "n_clicks"),
        State("robot-uri", "value"),
        State("robot-connection-state", "data"),
        State("robot-rescue-state", "data"),
        prevent_initial_call=True,
    )
    async def _enable_rescue_mode(
        _clicks: int,
        robot_uri: Optional[str],
        robot_state: Optional[dict],
        rescue_state: Optional[dict],
    ) -> dmc.Notification:
        if (robot_state or {}).get("status") != "online":
            return dmc.Notification(
                action="show",
                title="Rescue mode unavailable",
                message="Robot is not connected.",
                color="red",
                autoClose=5000,
            )
        if (rescue_state or {}).get("active"):
            return dmc.Notification(
                action="show",
                title="Rescue mode",
                message="Rescue mode is already active.",
                color="orange",
                autoClose=4000,
            )

        async def _enter(api: CodroidAPI) -> None:
            await api.enter_rescue_mode()

        try:
            await _with_robot_api(robot_uri or DEFAULT_ROBOT_URI, _enter)
            return dmc.Notification(
                action="show",
                title="Rescue mode requested",
                message="Rescue mode command sent.",
                color="orange",
                autoClose=5000,
            )
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("Failed to enter rescue mode")
            return dmc.Notification(
                action="show",
                title="Rescue mode failed",
                message=str(exc),
                color="red",
                autoClose=7000,
            )

    @app.callback(
        Output("app-notifications", "children", allow_duplicate=True),
        Input("rescue-mode-exit", "n_clicks"),
        State("robot-uri", "value"),
        State("robot-rescue-state", "data"),
        prevent_initial_call=True,
    )
    async def _exit_rescue_mode(
        _clicks: int,
        robot_uri: Optional[str],
        rescue_state: Optional[dict],
    ) -> dmc.Notification:
        if not (rescue_state or {}).get("active"):
            return dmc.Notification(
                action="show",
                title="Rescue mode",
                message="Rescue mode is not active.",
                color="gray",
                autoClose=4000,
            )

        async def _exit(api: CodroidAPI) -> None:
            await api.exit_rescue_mode()

        try:
            await _with_robot_api(robot_uri or DEFAULT_ROBOT_URI, _exit)
            return dmc.Notification(
                action="show",
                title="Rescue mode exit requested",
                message="Exit command sent.",
                color="orange",
                autoClose=5000,
            )
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("Failed to exit rescue mode")
            return dmc.Notification(
                action="show",
                title="Exit rescue mode failed",
                message=str(exc),
                color="red",
                autoClose=7000,
            )

    @app.callback(
        Output("rescue-jog-status", "children"),
        Input("rescue-joint-1-minus", "n_clicks"),
        Input("rescue-joint-1-plus", "n_clicks"),
        Input("rescue-joint-2-minus", "n_clicks"),
        Input("rescue-joint-2-plus", "n_clicks"),
        Input("rescue-joint-3-minus", "n_clicks"),
        Input("rescue-joint-3-plus", "n_clicks"),
        Input("rescue-joint-4-minus", "n_clicks"),
        Input("rescue-joint-4-plus", "n_clicks"),
        Input("rescue-joint-5-minus", "n_clicks"),
        Input("rescue-joint-5-plus", "n_clicks"),
        Input("rescue-joint-6-minus", "n_clicks"),
        Input("rescue-joint-6-plus", "n_clicks"),
        State("robot-uri", "value"),
        State("robot-rescue-state", "data"),
        prevent_initial_call=True,
    )
    async def _handle_rescue_joint_jog(
        _j1_minus: int,
        _j1_plus: int,
        _j2_minus: int,
        _j2_plus: int,
        _j3_minus: int,
        _j3_plus: int,
        _j4_minus: int,
        _j4_plus: int,
        _j5_minus: int,
        _j5_plus: int,
        _j6_minus: int,
        _j6_plus: int,
        robot_uri: Optional[str],
        rescue_state: Optional[dict],
    ) -> str:
        if not (rescue_state or {}).get("active"):
            return "Rescue mode is not active."

        trigger = callback_context.triggered_id
        if not trigger:
            return no_update

        mapping = {
            "rescue-joint-1-minus": (1, -1),
            "rescue-joint-1-plus": (1, 1),
            "rescue-joint-2-minus": (2, -1),
            "rescue-joint-2-plus": (2, 1),
            "rescue-joint-3-minus": (3, -1),
            "rescue-joint-3-plus": (3, 1),
            "rescue-joint-4-minus": (4, -1),
            "rescue-joint-4-plus": (4, 1),
            "rescue-joint-5-minus": (5, -1),
            "rescue-joint-5-plus": (5, 1),
            "rescue-joint-6-minus": (6, -1),
            "rescue-joint-6-plus": (6, 1),
        }
        if trigger not in mapping:
            return no_update
        axis, direction = mapping[trigger]

        async def _jog(api: CodroidAPI) -> None:
            await api.set_robot_command(api.config.commands.manual_mode)
            await api.hold_jog(
                api.config.jog_modes.joint,
                axis=axis,
                direction=direction,
                hold_seconds=SETTINGS.rescue_jog_hold_s,
                heartbeat_interval=0.25,
            )

        try:
            await _with_robot_api(robot_uri or DEFAULT_ROBOT_URI, _jog)
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("Rescue joint jog failed")
            return f"Rescue jog failed: {exc}"
        sign = "+" if direction > 0 else "-"
        return f"Jogged J{axis}{sign} for {SETTINGS.rescue_jog_hold_s:.2f}s."

    @app.callback(
        Output("robot-calibration-status", "children"),
        Input("robot-status-interval", "n_intervals"),
    )
    def _update_robot_calibration_guidance(_intervals: int) -> str:
        return state.get_calibration_guidance_snapshot().get("message", "")

    def _build_robot_calibration_state_payload() -> dict[str, Any]:
        guidance = state.get_calibration_guidance_snapshot()
        return {
            "status": guidance.get("status", "idle"),
            "message": guidance.get("message", ""),
            "step": guidance.get("step", 0),
            "verification_ready": guidance.get("verification_ready", False),
            "accuracy": guidance.get("accuracy"),
            "accuracy_ready": guidance.get("accuracy_ready", False),
            "accuracy_passed": guidance.get("accuracy_passed", False),
        }

    @app.callback(
        Output("robot-calibration-state", "data", allow_duplicate=True),
        Input("robot-status-interval", "n_intervals"),
        State("robot-calibration-state", "data"),
        prevent_initial_call=True,
    )
    def _sync_robot_calibration_state(_intervals: int, current: Optional[dict]) -> dict:
        desired = _build_robot_calibration_state_payload()
        if current == desired:
            return no_update
        return desired

    @app.callback(
        Output("robot-calibration-modal", "opened"),
        Input("robot-calibration-state", "data"),
        Input("robot-calibration-modal-close", "n_clicks"),
    )
    def _toggle_robot_calibration_modal(
        calibration_state: Optional[dict],
        _close_clicks: int,
    ) -> bool:
        status = (calibration_state or {}).get("status", "idle")
        trigger = callback_context.triggered_id
        should_show_modal = not ROBOT_CALIBRATION_GUIDANCE.get(
            "modal_suppressed", False)
        if status in {"running", "accuracy_ready", "awaiting_verification", "verifying"} and should_show_modal:
            return True
        if status in {"complete", "error", "canceled"}:
            return trigger != "robot-calibration-modal-close"
        if trigger == "robot-calibration-modal-close":
            return False
        return False

    @app.callback(
        Output("robot-calibration-modal-stepper", "active"),
        Output("robot-calibration-modal-alert", "children"),
        Output("robot-calibration-modal-alert", "color"),
        Output("robot-calibration-verify", "disabled"),
        Output("robot-calibration-modal-stepper", "children"),
        Input("robot-calibration-state", "data"),
    )
    def _update_robot_calibration_modal(
        calibration_state: Optional[dict],
    ) -> Tuple[int, str, str, bool, list[dmc.StepperStep]]:
        state = calibration_state or {}
        status = state.get("status", "idle")
        step = int(state.get("step", 0) or 0)
        message = state.get("message", "")
        active = max(0, min(4, step - 1)) if step else 0
        color_map = {
            "running": "blue",
            "awaiting_verification": "yellow",
            "accuracy_ready": "orange",
            "verifying": "blue",
            "complete": "green",
            "error": "red",
            "canceled": "red",
        }
        color = color_map.get(status, "gray")
        verify_enabled = (
            status == "awaiting_verification"
            and state.get("verification_ready", False)
        )
        verification_completed = status == "complete"

        def _step_state(is_done: bool) -> Optional[str]:
            return "completed" if is_done else None

        stepper_children = [
            dmc.StepperStep(
                label="Point 1",
                description="Indentation 1 (origin)",
                state=_step_state(step > 1),
            ),
            dmc.StepperStep(
                label="Point 2",
                description="Indentation 2 (+X)",
                state=_step_state(step > 2),
            ),
            dmc.StepperStep(
                label="Point 3",
                description="Indentation 3 (+X/+Y)",
                state=_step_state(step > 3),
            ),
            dmc.StepperStep(
                label="Accuracy",
                description="Calibration accuracy check",
                state=_step_state(step > 4 or bool(
                    state.get("accuracy_ready"))),
            ),
            dmc.StepperStep(
                label="Verification",
                description="Safe motion check",
                state=_step_state(verification_completed),
            ),
        ]
        return (
            active,
            message or "Waiting for calibration to start.",
            color,
            not verify_enabled,
            stepper_children,
        )

    @app.callback(
        Output("robot-calibration-accuracy-panel", "style"),
        Output("robot-calibration-accuracy-text", "children"),
        Output("robot-calibration-accuracy-text", "c"),
        Output("robot-calibration-retry", "disabled"),
        Output("robot-calibration-continue", "disabled"),
        Output("robot-calibration-continue", "children"),
        Input("robot-calibration-state", "data"),
    )
    def _sync_robot_calibration_accuracy_panel(
        calibration_state: Optional[dict],
    ) -> Tuple[dict[str, Any], str, str, bool, bool, str]:
        state = calibration_state or {}
        accuracy_ready = bool(state.get("accuracy_ready"))
        accuracy = state.get("accuracy") or {}
        if not accuracy_ready or not accuracy:
            return (
                {"display": "none"},
                "",
                "dimmed",
                True,
                True,
                "Proceed to verification",
            )
        max_err = float(accuracy.get("max_xy_error", 0.0))
        mean_err = float(accuracy.get("mean_xy_error", 0.0))
        passed = bool(state.get("accuracy_passed"))
        text = (
            f"Calibration accuracy: max XY error {max_err:.2f} mm · "
            f"mean XY error {mean_err:.2f} mm."
        )
        if passed:
            text += f" Within {SETTINGS.verification_tolerance_mm:.2f} mm threshold."
            continue_label = "Proceed to verification"
            text_color = "green"
        else:
            text += (
                f" Above {SETTINGS.verification_tolerance_mm:.2f} mm threshold; "
                "consider repeating calibration or proceed with caution."
            )
            continue_label = "Proceed with warning"
            text_color = "red"
        return (
            {"display": "flex", "flexDirection": "column", "gap": "0.6rem"},
            text,
            text_color,
            False,
            False,
            continue_label,
        )

    @app.callback(
        Output("robot-calibration-state", "data", allow_duplicate=True),
        Input("robot-calibration-continue", "n_clicks"),
        State("robot-calibration-state", "data"),
        prevent_initial_call=True,
    )
    def _advance_to_verification_after_accuracy(
        _clicks: int,
        calibration_state: Optional[dict],
    ) -> dict:
        guidance = calibration_state or {}
        if not guidance.get("accuracy_ready"):
            return no_update
        _set_robot_guidance(
            "Verification ready. Run the verification move when ready.",
            status="awaiting_verification",
            step=5,
            verification_ready=True,
            modal_suppressed=False,
        )
        return _build_robot_calibration_state_payload()

    @app.callback(
        Output("robot-calibration-state", "data", allow_duplicate=True),
        Input("robot-calibration-retry", "n_clicks"),
        prevent_initial_call=True,
    )
    def _reset_calibration_after_accuracy(_clicks: int) -> dict:
        _clear_robot_accuracy_guidance()
        _set_robot_guidance(
            "Calibration reset. Close this dialog and restart from the main panel.",
            status="idle",
            step=0,
            verification_ready=False,
            modal_suppressed=True,
        )
        return _build_robot_calibration_state_payload()

    @app.callback(
        Output("robot-calibration-state", "data", allow_duplicate=True),
        Input("start-robot-calibration", "n_clicks"),
        State("plate-detection-state", "data"),
        State("robot-connection-state", "data"),
        State("robot-uri", "value"),
        prevent_initial_call=True,
    )
    async def _start_robot_calibration(
        _clicks: int,
        plate_state: Optional[dict],
        robot_state: Optional[dict],
        robot_uri: Optional[str],
    ) -> dict:
        LOGGER.info("=== Robot calibration button clicked ===")
        LOGGER.info(f"Plate state: {plate_state}")
        LOGGER.info(f"Robot URI: {robot_uri}")

        if (robot_state or {}).get("status") != "online":
            _set_robot_guidance(
                "Robot not connected. Click 'Test robot connection' to establish connection.",
                status="error",
                verification_ready=False,
            )
            return {"status": "error", "message": "Robot not connected."}

        plate_ok, plate_message = _plate_is_stable(require_baseline=False)
        if not plate_ok:
            LOGGER.warning("Plate not ready: %s", plate_message)
            _set_robot_guidance(
                plate_message,
                status="error",
                verification_ready=False,
            )
            return {"status": "error", "message": plate_message}

        posture_ok, posture_message = _robot_posture_ready()
        if not posture_ok:
            _set_robot_guidance(
                posture_message,
                status="error",
                verification_ready=False,
            )
            return {"status": "error", "message": posture_message}

        _prepare_gripper_for_calibration()

        LOGGER.info("Plate state OK, starting calibration procedure...")
        try:
            layout_path = DEFAULT_PLATE_PATH
            LOGGER.info(f"Loading plate layout from: {layout_path}")
            plate_layout = plate.load_plate_layout(layout_path)
            LOGGER.info("Plate layout loaded successfully")

            center = (plate_state or {}).get("center_px")
            if center is None:
                raise RuntimeError(
                    "Plate center not available; keep markers visible.")
            SESSION.robot_calibration_baseline_px = np.asarray(
                center, dtype=np.float64)
            _ensure_plate_stable()
            LOGGER.info("Plate stability baseline set")

            _set_robot_guidance(
                "Robot calibration started. Step 1/3: Place the gripper fingers in "
                "indentation 1, keep the flange vertical, and press flange button 1.",
                step=1,
                status="running",
                verification_ready=False,
            )
            SESSION.robot_calibration_points = []
            LOGGER.info(
                f"Starting robot plate calibration with URI: {robot_uri or DEFAULT_ROBOT_URI}")
            robot_points, _coord_frame = await _run_robot_plate_calibration(
                robot_uri or DEFAULT_ROBOT_URI, plate_layout)
            LOGGER.info(
                f"Robot plate calibration completed. Captured {len(robot_points)} points.")

            LOGGER.info("Applying robot calibration results...")
            _, accuracy = _apply_robot_calibration_results(
                robot_points,
                plate_layout,
            )
            try:
                plate_result = compute_plate_calibration_from_latest(
                    layout_path)
                SESSION.plate_reference_homography = plate_result.homography.copy()
                LOGGER.info(
                    "Captured plate reference homography for plate test.")
            except Exception as exc:  # noqa: BLE001
                LOGGER.warning(
                    "Failed to capture plate reference homography: %s", exc
                )
            _update_robot_calibration_meta(robot_points)
            SESSION.robot_calibration_baseline_px = None
            LOGGER.info("Calibration results applied successfully")
            mean_err = accuracy.get("mean_xy_error", 0.0)
            max_err = accuracy.get("max_xy_error", 0.0)
            passes_threshold = max_err <= SETTINGS.verification_tolerance_mm
            message = (
                f"Calibration accuracy: max XY error {max_err:.2f} mm · "
                f"mean XY error {mean_err:.2f} mm."
            )
            if passes_threshold:
                message = (
                    f"{message} Within {SETTINGS.verification_tolerance_mm:.2f} mm "
                    "threshold. Proceed to verification."
                )
            else:
                message = (
                    f"{message} Above {SETTINGS.verification_tolerance_mm:.2f} mm "
                    "threshold; consider repeating calibration or proceed with caution."
                )
            _set_robot_guidance(
                message,
                status="accuracy_ready",
                step=4,
                verification_ready=False,
                modal_suppressed=False,
            )
            return {
                "status": "accuracy_ready",
                "message": message,
                "step": 4,
                "verification_ready": False,
                "accuracy": accuracy,
                "accuracy_ready": True,
                "accuracy_passed": passes_threshold,
            }
        except Exception as exc:
            SESSION.robot_calibration_baseline_px = None
            _restore_calibration_gripper_width()
            LOGGER.error("Robot calibration failed: %s", exc, exc_info=True)
            _set_robot_guidance(
                f"Robot calibration failed: {exc}",
                status="error",
                verification_ready=False,
            )
            return {"status": "error", "message": str(exc)}

    @app.callback(
        Output("robot-calibration-state", "data", allow_duplicate=True),
        Input("robot-calibration-verify", "n_clicks"),
        Input("run-verification-inline", "n_clicks"),
        State("robot-uri", "value"),
        prevent_initial_call=True,
    )
    async def _run_robot_calibration_verification(
        _clicks: int,
        robot_uri: Optional[str],
    ) -> dict:
        run_id: Optional[int] = None
        if not SESSION.robot_calibration_points or len(SESSION.robot_calibration_points) < 3:
            message = "Verification requires three calibration points."
            _set_robot_guidance(message, status="error",
                                step=5, verification_ready=False)
            return {"status": "error", "message": message, "step": 5}

        points = SESSION.robot_calibration_points[:3]
        min_z = SESSION.robot_calibration_min_z
        if min_z is None:
            min_z = min(float(pt.get("z", 0.0)) for pt in points)
            SESSION.robot_calibration_min_z = min_z

        session = _get_robot_session()
        posture_ok, posture_message = _robot_posture_ready()
        if not posture_ok:
            _set_robot_guidance(
                posture_message,
                status="error",
                step=5,
                verification_ready=False,
            )
            return {"status": "error", "message": posture_message, "step": 5}
        try:
            run_id = _begin_operation("verification", "Verification move")
        except RuntimeError as exc:
            message = f"Verification blocked: {exc}"
            _set_robot_guidance(
                message,
                status="error",
                step=5,
                verification_ready=False,
            )
            return {"status": "error", "message": message, "step": 5}
        _set_robot_guidance(
            "Verification move running. Keep clear of the robot.",
            status="verifying",
            step=5,
            verification_ready=False,
            modal_suppressed=False,
        )

        async def _verification_motion(api: CodroidAPI) -> None:
            await _wait_for_operation_resume(run_id)
            _update_operation_message("Verification move: powering on robot.")
            await api.power_on()
            await api.set_current_coordinate_id(ROBOT_COORDINATE_ID)

            # Compute coordinate system 5 positions from plate layout
            try:
                plate_layout = plate.load_plate_layout(
                    DEFAULT_PLATE_LAYOUT_PATH)
                plate_points = plate_layout.robot_calibration_points()
                if len(plate_points) < 3:
                    raise RuntimeError(
                        "Plate layout missing calibration points.")
                # Positions relative to Point 1 (origin)
                origin_mm = np.asarray(
                    plate_points[0].center_mm, dtype=np.float64)
                coord5_positions = [
                    np.array([
                        float(pt.center_mm[0] - origin_mm[0]),
                        float(pt.center_mm[1] - origin_mm[1]),
                        0.0,
                    ], dtype=np.float64)
                    for pt in plate_points[:3]
                ]
            except Exception as exc:
                LOGGER.warning(
                    "Failed to load plate layout for verification, using base coordinates: %s", exc)
                # Fallback to base coordinates if plate layout unavailable
                await api.set_current_coordinate_id(0)
                coord5_positions = None

            async def _await_motion_then_settle(
                *,
                label: str,
                timeout_s: float = SETTINGS.verification_move_timeout_s,
                settle_s: float = SETTINGS.verification_settle_s,
                motion_eps_mm: float = SETTINGS.verification_motion_eps_mm,
            ) -> None:
                deadline = time.monotonic() + timeout_s
                start = session.position_snapshot()
                last = start
                moved = False
                settle_deadline: Optional[float] = None
                while time.monotonic() < deadline:
                    _ensure_operation_active(run_id)
                    await api.send_command_heartbeat()
                    await asyncio.sleep(0.2)
                    curr = session.position_snapshot()
                    delta = np.linalg.norm(
                        [
                            float(curr.x) - float(last.x),
                            float(curr.y) - float(last.y),
                            float(curr.z) - float(last.z),
                        ]
                    )
                    if delta > motion_eps_mm:
                        moved = True
                        settle_deadline = time.monotonic() + settle_s
                    if moved and settle_deadline is not None:
                        if delta > motion_eps_mm:
                            settle_deadline = time.monotonic() + settle_s
                        elif time.monotonic() >= settle_deadline:
                            return
                    last = curr
                total = np.linalg.norm(
                    [
                        float(last.x) - float(start.x),
                        float(last.y) - float(start.y),
                        float(last.z) - float(start.z),
                    ]
                )
                if not moved:
                    raise RuntimeError(
                        f"Verification move timeout at {label}: no motion detected."
                    )
                raise RuntimeError(
                    f"Verification move timeout at {label}: motion did not settle "
                    f"(total={total:.2f} mm)."
                )

            async def _move_linear_with_hold(
                target: dict,
                *,
                label: str,
                hold_seconds: float = SETTINGS.verification_hold_seconds,
                coord_id: int,
            ) -> None:
                while True:
                    await _wait_for_operation_resume(run_id)
                    _update_operation_message(
                        f"Verification move: {label.replace('_', ' ')}",
                        label=label,
                    )
                    _log_move_target(label, target, coordinate_id=coord_id)
                    await api.set_target_pos_type(api.config.target_pos_types.cpos)
                    if coord_id == ROBOT_COORDINATE_ID:
                        await api.set_target_cpos(
                            api.attach_coordinate_to_cpos(
                                target, coordinate_id=ROBOT_COORDINATE_ID)
                        )
                    else:
                        await api.set_target_cpos(target)
                    await api.set_robot_command(api.config.commands.move_target_linear)
                    try:
                        await _await_motion_then_settle(label=label)
                    except OperationPaused:
                        continue
                    try:
                        if hold_seconds > 0:
                            deadline = time.monotonic() + hold_seconds
                            while time.monotonic() < deadline:
                                _ensure_operation_active(run_id)
                                await api.send_command_heartbeat()
                                await asyncio.sleep(0.2)
                    except OperationPaused:
                        continue
                    await api.clear_target_position()
                    await api.stop_command()
                    return

            def _safe_z(value: float) -> float:
                return max(float(value), float(min_z))

            def _target(point: dict, z_value: float, coord5_pos: Optional[np.ndarray] = None) -> dict:
                poscfg = CodroidAPI.build_poscfg(
                    mode=int(point.get("mode", 0))
                )
                if coord5_pos is not None:
                    # Use coordinate system 5 positions (relative to Point 1)
                    return api.build_target_cpos(
                        x=float(coord5_pos[0]),
                        y=float(coord5_pos[1]),
                        z=_safe_z(z_value),
                        a=float(point.get("a", 0.0)),
                        b=float(point.get("b", 0.0)),
                        c=float(point.get("c", 0.0)),
                        poscfg=poscfg,
                    )
                else:
                    # Fallback to base coordinates
                    return api.build_target_cpos(
                        x=float(point["x"]),
                        y=float(point["y"]),
                        z=_safe_z(z_value),
                        a=float(point.get("a", 0.0)),
                        b=float(point.get("b", 0.0)),
                        c=float(point.get("c", 0.0)),
                        poscfg=poscfg,
                    )

            p1, p2, p3 = points
            z1 = _safe_z(p1.get("z", 0.0))
            z2 = _safe_z(p2.get("z", 0.0))
            z3 = _safe_z(p3.get("z", 0.0))
            raise_1 = z1 + SETTINGS.verification_raise_mm
            raise_2 = z2 + SETTINGS.verification_raise_mm
            raise_3 = z3 + SETTINGS.verification_raise_mm

            # Determine which coordinate system to use
            coord_id = 0 if coord5_positions is None else ROBOT_COORDINATE_ID
            c5p1 = coord5_positions[0] if coord5_positions is not None else None
            c5p2 = coord5_positions[1] if coord5_positions is not None else None
            c5p3 = coord5_positions[2] if coord5_positions is not None else None

            await _move_linear_with_hold(_target(p3, raise_3, c5p3), label="p3_raise", coord_id=coord_id)
            await _move_linear_with_hold(_target(p1, raise_1, c5p1), label="p1_raise", coord_id=coord_id)
            await _move_linear_with_hold(_target(p1, z1, c5p1), label="p1_down", coord_id=coord_id)
            await _move_linear_with_hold(_target(p1, raise_1, c5p1), label="p1_raise_return", coord_id=coord_id)
            await _move_linear_with_hold(_target(p2, raise_2, c5p2), label="p2_raise", coord_id=coord_id)
            await _move_linear_with_hold(_target(p2, z2, c5p2), label="p2_down", coord_id=coord_id)
            await _move_linear_with_hold(_target(p2, raise_2, c5p2), label="p2_raise_return", coord_id=coord_id)
            await _move_linear_with_hold(_target(p3, raise_3, c5p3), label="p3_raise", coord_id=coord_id)
            await _move_linear_with_hold(_target(p3, z3, c5p3), label="p3_down", coord_id=coord_id)
            await _move_linear_with_hold(_target(p3, raise_3, c5p3), label="p3_raise_return", coord_id=coord_id)
            await _move_linear_with_hold(
                _target(p1, z1, c5p1), label="p1_home", hold_seconds=0.0, coord_id=coord_id
            )
            if coord_id != ROBOT_COORDINATE_ID:
                await api.set_current_coordinate_id(ROBOT_COORDINATE_ID)

        try:
            await _with_robot_api(robot_uri or DEFAULT_ROBOT_URI, _verification_motion)
        except OperationAbort as exc:
            message = f"Verification move stopped: {exc}"
            _set_robot_guidance(
                message,
                status="canceled",
                step=5,
                verification_ready=False,
            )
            _restore_calibration_gripper_width()
            if run_id is not None:
                _finish_operation(run_id, status="stopped", message=message)
            return {"status": "canceled", "message": message, "step": 5}
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("Verification move failed")
            message = f"Verification move failed: {exc}"
            _set_robot_guidance(
                message,
                status="error",
                step=5,
                verification_ready=False,
            )
            _restore_calibration_gripper_width()
            if run_id is not None:
                _finish_operation(run_id, status="idle", message=message)
            return {"status": "error", "message": message, "step": 5}

        _restore_calibration_gripper_width()
        _set_robot_guidance(
            "Calibration verified. Coordinate ID 5 is active.",
            status="complete",
            step=5,
            verification_ready=False,
        )
        if run_id is not None:
            _finish_operation(run_id, status="idle",
                              message="Verification complete.")
        return {"status": "complete", "message": "Calibration verified.", "step": 5}


def create_app() -> Dash:
    app = Dash(
        __name__,
        assets_folder=str(ASSETS_DIR),
        suppress_callback_exceptions=True,
        update_title=None,
    )
    app.title = "Estun Codroid Pick-and-Place"
    app._favicon = "favicon.ico"
    app.layout = build_layout()
    register_callbacks(app)

    @app.callback(
        Output("prerequisites-state", "data"),
        Input("camera-interval", "n_intervals"),
        Input("monitor-interval", "n_intervals"),
        Input("gripper-connection-state", "data"),
    )
    def _update_prerequisites(
        _n: int,
        _monitor: int,
        gripper_state: Optional[dict],
    ) -> dict:
        """Update prerequisites state based on current system status."""
        return {
            "camera_calibrated": SESSION.calibration_bundle is not None,
            "workspace_defined": SESSION.workspace_definition is not None,
            "robot_calibrated": SESSION.robot_calibration_min_z is not None,
            "model_loaded": SESSION.model_loaded,
            "gripper_ready": (gripper_state or {}).get("status") == "online",
            "drop_target_defined": SESSION.drop_target_mm is not None,
        }

    @app.callback(
        Output("auto-start", "disabled"),
        Output("auto-stop", "disabled"),
        Input("prerequisites-state", "data"),
        Input("auto-operation-state", "data"),
    )
    def _update_auto_buttons(
        prerequisites: Optional[dict],
        auto_state: Optional[dict],
    ) -> Tuple[bool, bool]:
        """Enable/disable auto operation buttons based on prerequisites and operation state."""
        prereqs = prerequisites or {}
        all_ready = all([
            prereqs.get("camera_calibrated", False),
            prereqs.get("workspace_defined", False),
            prereqs.get("robot_calibrated", False),
            prereqs.get("model_loaded", False),
            prereqs.get("gripper_ready", False),
            prereqs.get("drop_target_defined", False),
        ])

        running = (auto_state or {}).get("running", False)

        # Start button: enabled only when all prerequisites met and not running
        start_disabled = not all_ready or running
        # Stop button: enabled only when running
        stop_disabled = not running

        return start_disabled, stop_disabled

    @app.callback(
        Output("auto-operation-state", "data"),
        Output("app-notifications", "children", allow_duplicate=True),
        Input("auto-start", "n_clicks"),
        Input("auto-stop", "n_clicks"),
        State("auto-operation-state", "data"),
        State("robot-uri", "value"),
        prevent_initial_call=True,
    )
    async def _handle_auto_operation(
        start_clicks: int,
        stop_clicks: int,
        current_state: Optional[dict],
        robot_uri: Optional[str],
    ) -> Tuple[dict, dmc.Notification]:
        """Handle auto operation start/stop."""
        ctx = callback_context
        if not ctx.triggered:
            return no_update, no_update

        trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]

        if trigger_id == "auto-start":
            if SESSION.auto_task is not None and not SESSION.auto_task.done():
                return (
                    {"running": True, "message": "Auto operation already running"},
                    dmc.Notification(
                        title="Auto Operation",
                        message="Auto operation is already running.",
                        color="yellow",
                        action="show",
                    ),
                )
            try:
                run_id = _begin_operation(
                    "auto",
                    "Auto pick & place",
                    message="Auto pick & place running.",
                )
            except RuntimeError as exc:
                return (
                    no_update,
                    dmc.Notification(
                        title="Auto Operation",
                        message=str(exc),
                        color="red",
                        action="show",
                    ),
                )
            SESSION.auto_waiting_at_drop = False
            _set_auto_state("Auto pick & place running.", running=True)
            SESSION.auto_task = asyncio.create_task(
                _auto_operation_task(run_id, robot_uri)
            )
            return (
                {"running": True, "message": "Auto operation started"},
                dmc.Notification(
                    title="Auto Operation",
                    message="Auto pick & place operation started",
                    color="green",
                    action="show",
                ),
            )
        elif trigger_id == "auto-stop":
            _request_stop_operation("Auto stop requested.")
            _set_auto_state("Auto stop requested.", running=True)

            async def _stop_motion(api: CodroidAPI) -> None:
                await api.stop_command()

            try:
                await _with_robot_api(robot_uri or DEFAULT_ROBOT_URI, _stop_motion)
            except Exception as exc:  # noqa: BLE001
                LOGGER.debug("Auto stop command failed: %s", exc)
            return (
                {"running": True, "message": "Auto operation stopping"},
                dmc.Notification(
                    title="Auto Operation",
                    message="Auto pick & place operation stopping",
                    color="orange",
                    action="show",
                ),
            )

        return no_update, no_update

    @app.callback(
        Output("auto-operation-state", "data", allow_duplicate=True),
        Input("monitor-interval", "n_intervals"),
        State("auto-operation-state", "data"),
        prevent_initial_call=True,
    )
    def _sync_auto_operation_state(
        _ticks: int,
        current_state: Optional[dict],
    ) -> dict:
        snapshot = _auto_state_snapshot()
        if current_state == snapshot:
            return no_update
        return snapshot

    @app.callback(
        Output("robot-operation-state", "data"),
        Input("monitor-interval", "n_intervals"),
        State("robot-operation-state", "data"),
    )
    def _sync_operation_state(
        _ticks: int,
        current_state: Optional[dict],
    ) -> dict:
        snapshot = _operation_snapshot()
        if current_state == snapshot:
            return no_update
        return snapshot

    @app.callback(
        Output("program-pause", "children"),
        Output("program-pause", "disabled"),
        Output("program-stop", "disabled"),
        Output("program-reset", "disabled"),
        Output("program-status-text", "children"),
        Input("robot-operation-state", "data"),
    )
    def _update_program_controls(
        operation_state: Optional[dict],
    ) -> Tuple[str, bool, bool, bool, str]:
        state = operation_state or {}
        status = state.get("status", "idle")
        pause_reason = state.get("pause_reason")
        label = state.get("label") or "Program"
        message = state.get("message") or ""

        if status == "running":
            pause_label = "Pause"
            pause_disabled = False
            stop_disabled = False
            reset_disabled = True
            status_text = message or f"{label} running."
        elif status == "paused":
            pause_label = "Resume"
            pause_disabled = False
            stop_disabled = False
            reset_disabled = False
            if pause_reason:
                status_text = f"Paused ({pause_reason.replace('_', ' ')})."
            else:
                status_text = message or f"{label} paused."
        elif status in {"stopping", "stopped"}:
            pause_label = "Pause"
            pause_disabled = True
            stop_disabled = True
            reset_disabled = False
            status_text = message or "Program stopped."
        else:
            pause_label = "Pause"
            pause_disabled = True
            stop_disabled = True
            reset_disabled = True
            status_text = message or "Program idle."

        return pause_label, pause_disabled, stop_disabled, reset_disabled, status_text

    @app.callback(
        Output("app-notifications", "children", allow_duplicate=True),
        Input("program-pause", "n_clicks"),
        Input("program-stop", "n_clicks"),
        Input("program-reset", "n_clicks"),
        State("robot-operation-state", "data"),
        State("robot-uri", "value"),
        prevent_initial_call=True,
    )
    async def _handle_program_controls(
        _pause_clicks: int,
        _stop_clicks: int,
        _reset_clicks: int,
        operation_state: Optional[dict],
        robot_uri: Optional[str],
    ) -> dmc.Notification:
        trigger = callback_context.triggered_id
        status = (operation_state or {}).get("status", "idle")

        if trigger == "program-pause":
            if status == "running":
                _pause_operation("operator", "Program paused by operator.")

                async def _stop_motion(api: CodroidAPI) -> None:
                    await api.stop_command()

                try:
                    await _with_robot_api(robot_uri or DEFAULT_ROBOT_URI, _stop_motion)
                except Exception as exc:  # noqa: BLE001
                    LOGGER.debug("Pause stop command failed: %s", exc)
                return dmc.Notification(
                    action="show",
                    title="Program paused",
                    message="Robot program paused.",
                    color="yellow",
                    autoClose=4000,
                )
            if status == "paused":
                _resume_operation("Program resumed.")
                return dmc.Notification(
                    action="show",
                    title="Program resumed",
                    message="Robot program resumed.",
                    color="teal",
                    autoClose=4000,
                )
        elif trigger == "program-stop":
            _request_stop_operation("Program stop requested.")

            async def _stop_motion(api: CodroidAPI) -> None:
                await api.stop_command()

            try:
                await _with_robot_api(robot_uri or DEFAULT_ROBOT_URI, _stop_motion)
            except Exception as exc:  # noqa: BLE001
                LOGGER.debug("Stop command failed: %s", exc)
            return dmc.Notification(
                action="show",
                title="Program stopping",
                message="Stop command sent. Reset when safe.",
                color="red",
                autoClose=4500,
            )
        elif trigger == "program-reset":
            _reset_operation_state("Program reset.")
            return dmc.Notification(
                action="show",
                title="Program reset",
                message="Program state cleared.",
                color="gray",
                autoClose=3500,
            )

        return no_update

    @app.server.route("/video_feed")
    def video_feed():
        """Stream Motion JPEG video for fullscreen playback."""
        print("DEBUG: /video_feed endpoint accessed")

        def generate():
            boundary = b'--frame\r\n'
            frame_count = 0
            while True:
                frame_data = MJPEG_STREAM.get_frame()

                if not frame_data:
                    time.sleep(0.033)
                    continue

                frame_count += 1
                if frame_count == 1:
                    logging.debug(
                        "/video_feed serving first frame, size=%s bytes", len(frame_data))

                yield boundary
                yield b'Content-Type: image/jpeg\r\n'
                yield b'Content-Length: ' + str(len(frame_data)).encode() + b'\r\n\r\n'
                yield frame_data
                yield b'\r\n'
                time.sleep(0.01)  # Small delay to avoid hammering

        return app.server.response_class(
            generate(),
            mimetype='multipart/x-mixed-replace; boundary=frame'
        )

    @app.callback(
        Output("calibration-progress", "value"),
        Output("calibration-progress", "style"),
        Output("calibration-progress-text", "children"),
        Output("calibration-progress-text", "style"),
        Input("calibration-progress-interval", "n_intervals"),
    )
    def _update_calibration_progress(_):
        """Update progress bar and text from global calibration progress tracker."""
        progress = CALIBRATION_PROGRESS.get("value", 0)
        current_frame = CALIBRATION_PROGRESS.get("current_frame", 0)
        max_frames = CALIBRATION_PROGRESS.get("max_frames", 0)

        # Show progress bar and text only if capturing (1-99%)
        if progress > 0 and progress < 100:
            text = f"Capturing images: {current_frame}/{max_frames}"
            return progress, {"display": "block"}, text, {"display": "block", "marginBottom": "8px"}

        return 0, {"display": "none"}, "", {"display": "none"}

    @app.server.route("/camera_frame.jpg")
    def camera_frame():
        """Get the latest camera frame as a JPEG image."""
        frame_data = MJPEG_STREAM.get_frame()
        if not frame_data:
            # Return placeholder as JPEG bytes
            placeholder = _placeholder_frame("Camera stream offline")
            success, buffer = cv2.imencode(".jpg", placeholder)
            if success:
                frame_data = buffer.tobytes()
            else:
                return app.server.response_class(
                    b"",
                    mimetype='image/jpeg'
                )
        return app.server.response_class(
            frame_data,
            mimetype='image/jpeg'
        )

    @app.server.route("/raw_camera_frame.jpg")
    def raw_camera_frame():
        """Get the latest raw camera frame (no overlays) as a JPEG image."""
        frame_data = RAW_STREAM.get_frame()
        if not frame_data:
            placeholder = _placeholder_frame("Camera stream offline")
            success, buffer = cv2.imencode(".jpg", placeholder)
            if success:
                frame_data = buffer.tobytes()
            else:
                return app.server.response_class(
                    b"",
                    mimetype='image/jpeg'
                )
        return app.server.response_class(
            frame_data,
            mimetype='image/jpeg'
        )

    # Auto-load model on startup if one is configured
    if SETTINGS.active_model:
        try:
            model_path = Path(SETTINGS.active_model)
            if model_path.exists():
                activate_model(model_path)
                LOGGER.info(f"Auto-loaded model on startup: {model_path.name}")
            else:
                LOGGER.warning(f"Configured model not found: {model_path}")
        except Exception as exc:
            LOGGER.error(f"Failed to auto-load model: {exc}")

    return app


app = create_app()
server = app.server


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    app.run(debug=True)
