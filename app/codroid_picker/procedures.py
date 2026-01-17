"""Robot control procedures and coordinate transformations for the calibration UI.

This module contains robot-related functions extracted from main.py:
- Robot session management
- Coordinate transformations (pixel ↔ workspace ↔ robot)
- Async coordinate system control
- Robot posture and status utilities
- Calibration data persistence
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

from codroid_api import (
    CodroidAPI,
    CodroidSettings,
    RobotPosture,
    RobotSession,
    DEFAULT_FLANGE_BUTTON_PORT,
    DEFAULT_FLANGE_BUTTON_PORTS,
)
from vision import workspace
from app.codroid_picker.session import SESSION
from app.codroid_picker import state

LOGGER = logging.getLogger(__name__)


# ==============================================================================
# Constants and Settings Access
# ==============================================================================

def _get_settings():
    """Get settings module - imported locally to avoid circular deps."""
    from app import settings
    return settings


def _robot_calibration_path() -> Path:
    """Return the currently configured robot calibration data path."""
    settings = _get_settings()
    return settings.DEFAULT_ROBOT_CALIBRATION_PATH


def _get_robot_coordinate_id() -> int:
    """Get the robot coordinate ID for calibrated operations."""
    settings = _get_settings()
    return settings.ROBOT_COORDINATE_ID


# ==============================================================================
# Robot Session Management
# ==============================================================================

def _get_robot_session() -> RobotSession:
    """Get or create the robot session."""
    if SESSION.robot_session is None:
        SESSION.robot_session = RobotSession(
            CodroidSettings(),
            flange_button_port=DEFAULT_FLANGE_BUTTON_PORT,
            flange_button_ports=DEFAULT_FLANGE_BUTTON_PORTS,
        )
    return SESSION.robot_session


def _robot_posture_ready(
    max_age_s: Optional[float] = None,
) -> Tuple[bool, str]:
    """Check if robot posture data is available and recent."""
    if max_age_s is None:
        settings = _get_settings()
        max_age_s = settings.SETTINGS.robot_posture_max_age_s

    session = _get_robot_session()
    if not session.posture_seen():
        return False, "Robot posture stream not available yet."
    age = session.position_snapshot().age_seconds()
    if age > max_age_s:
        return False, f"Robot posture stale ({age:.1f}s old)."
    return True, "Robot posture stream healthy."


def _current_coordinate_id(session: RobotSession) -> Optional[int]:
    """Get the current coordinate ID from robot status."""
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


def _posture_to_dict(posture: RobotPosture) -> dict:
    """Convert RobotPosture to dictionary."""
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


def _ensure_posture_axes(posture: dict, fallback: RobotPosture) -> dict:
    """Ensure posture dict has all axes, using fallback for missing ones."""
    normalized = dict(posture)
    for axis in ("a", "b", "c"):
        if axis not in normalized:
            normalized[axis] = float(getattr(fallback, axis, 0.0))
    return normalized


# ==============================================================================
# Async Coordinate System Control
# ==============================================================================

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
    coord_id = _get_robot_coordinate_id()
    await _ensure_robot_coordinate(robot_api, coord_id, timeout_s=timeout_s)


# ==============================================================================
# Robot Calibration Data Persistence
# ==============================================================================

def _load_robot_calibration(path: Optional[Path] = None) -> None:
    """Load robot calibration data from file."""
    if path is None:
        path = _robot_calibration_path()
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
        except Exception as exc:
            LOGGER.warning(
                "Failed to load plate reference homography: %s", exc)

    SESSION.plate_to_robot_transform = None
    SESSION.plate_origin_mm = None
    LOGGER.info("Loaded robot calibration data from %s", path)


def _save_robot_calibration(points: list[dict], path: Optional[Path] = None) -> None:
    """Save robot calibration data to file."""
    if path is None:
        path = _robot_calibration_path()

    min_z = None
    if points:
        min_z = min(float(pt.get("z", 0.0)) for pt in points)

    coord_id = _get_robot_coordinate_id()
    payload = {
        "points": points,
        "min_z": min_z,
        "timestamp": time.time(),
        "coordinate_id": coord_id,
    }

    if SESSION.plate_reference_homography is not None:
        payload["plate_reference_homography"] = (
            np.asarray(SESSION.plate_reference_homography,
                       dtype=np.float64).tolist()
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))
    LOGGER.info("Saved robot calibration to %s", path)


# ==============================================================================
# Robot Calibration Guidance (State Management)
# ==============================================================================

def _set_robot_guidance(
    message: str,
    *,
    step: Optional[int] = None,
    status: Optional[str] = None,
    verification_ready: Optional[bool] = None,
    modal_suppressed: Optional[bool] = None,
) -> None:
    """Set robot calibration guidance message and state."""
    state.update_calibration_guidance(
        message,
        step=step,
        status=status,
        verification_ready=verification_ready,
        modal_suppressed=modal_suppressed
    )

    # Clear accuracy if status changed to non-accuracy state
    if status is not None and status not in {
        "accuracy_ready",
        "awaiting_verification",
        "verifying",
        "complete",
    }:
        state.clear_calibration_accuracy()


def _clear_robot_accuracy_guidance() -> None:
    """Reset the cached accuracy summary."""
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


# ==============================================================================
# Error Handling
# ==============================================================================

def _acknowledge_robot_error() -> dict:
    """Acknowledge the current robot error."""
    error_snapshot = state.get_error_snapshot()
    if error_snapshot.get("active"):
        state.set_error_state(
            error_type=error_snapshot.get("error_type", ""),
            message=error_snapshot.get("message", ""),
            physical_pressed=False,
            acknowledge_existing=True
        )
    return state.get_error_snapshot()


# ==============================================================================
# Coordinate Transformations: Pixel ↔ Workspace ↔ Robot
# ==============================================================================

def _workspace_pixel_to_mm(
    workspace_def: workspace.WorkspaceDefinition,
    pixel_xy: Tuple[float, float],
) -> Tuple[float, float]:
    """Convert pixel coordinates to workspace millimeters (simple method)."""
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
    """Convert pixel coordinates to workspace millimeters (with calibration bundle)."""
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


def _pixel_to_robot_base_xyz(
    pixel: Tuple[float, float],
) -> Optional[np.ndarray]:
    """Convert pixel coordinates to robot base XYZ coordinates."""
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

    # DepthAI plane fit can be in meters or millimeters depending on stream config
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
        [workspace_def.x_axis_camera, workspace_def.y_axis_camera]
    )
    try:
        coeffs, *_ = np.linalg.lstsq(basis, camera_xyz - origin, rcond=None)
    except Exception as exc:
        raise RuntimeError(
            f"Unable to project robot pose into workspace: {exc}"
        ) from exc
    return float(coeffs[0]), float(coeffs[1])


# ==============================================================================
# Rigid Transform Utilities
# ==============================================================================

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
    """Compute 2D rigid transform (rotation matrix, translation vector)."""
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
