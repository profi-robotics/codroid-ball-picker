"""Robot control procedures and coordinate transformations for the calibration UI.

This module contains robot-related functions extracted from main.py:
- Robot session management
- Coordinate transformations (pixel \u2194 workspace \u2194 robot)
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
from app.codroid_picker.app_instance import registry

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
    return registry().robot_calibration_path


def _get_robot_coordinate_id() -> int:
    """Get the robot coordinate ID for calibrated operations."""
    return registry().robot_coordinate_id


# ==============================================================================
# Robot Session Management
# ==============================================================================

def _get_robot_session() -> RobotSession:
    """Get or create the robot session."""
    return registry().get_or_create_robot_session()


def _robot_posture_ready(
    max_age_s: Optional[float] = None,
) -> Tuple[bool, str]:
    """Check if robot posture data is available and recent."""
    if max_age_s is None:
        settings = _get_settings()
        max_age_s = settings.SETTINGS.robot_posture_max_age_s

    session = _get_robot_session()
    snapshot = session.snapshot()
    if not snapshot.posture_seen:
        return False, "Robot posture stream not available yet."
    age = snapshot.posture_age_seconds()
    if age > max_age_s:
        return False, f"Robot posture stale ({age:.1f}s old)."
    return True, "Robot posture stream healthy."


def _current_coordinate_id(session: RobotSession) -> Optional[int]:
    """Get the current coordinate ID from robot status."""
    status = session.snapshot().robot_status
    if not isinstance(status, dict):
        return None
    coord_id = status.get("CoordinateId")
    if coord_id is None:
        return None
    try:
        return int(coord_id)
    except (TypeError, ValueError):
        return None



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

def _check_coordinate_id(expected_id: int) -> Tuple[bool, Optional[int]]:
    """Check if the current coordinate ID matches the expected ID.

    Returns:
        (matched, current_id): True if matched, and the current coordinate ID.
    """
    session = _get_robot_session()
    status = session.snapshot().robot_status
    coord_id = None
    if isinstance(status, dict):
        coord_id = status.get("CoordinateId")
    if coord_id is not None:
        try:
            if int(coord_id) == expected_id:
                return True, int(coord_id)
        except (TypeError, ValueError):
            pass
    return False, coord_id


async def _ensure_robot_coordinate(
    robot_api: CodroidAPI,
    expected_id: int,
    *,
    timeout_s: float = 5.0,
    require: bool = True,
) -> None:
    """Wait for the robot to report the desired coordinate frame."""
    await robot_api.set_current_coordinate_id(expected_id)
    deadline = time.monotonic() + timeout_s
    coord_id = None
    resent = False
    while time.monotonic() < deadline:
        matched, coord_id = _check_coordinate_id(expected_id)
        if matched:
            return
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

def _read_json(path: Path) -> Optional[dict]:
    """Read JSON from a file and return dict or None; logs on error."""
    if path is None or not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        LOGGER.warning("Failed to parse JSON from %s: %s", path, exc)
        return None


def _write_json_atomic(path: Path, payload: dict) -> None:
    """Write JSON payload to path, ensuring parent directory exists."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))
    LOGGER.info("Saved robot calibration to %s", path)


def _deserialize_homography(data: list) -> Optional[np.ndarray]:
    """Convert a list representation of homography into np.ndarray or None."""
    if not isinstance(data, list):
        return None
    try:
        return np.asarray(data, dtype=np.float64)
    except Exception as exc:
        LOGGER.warning("Failed to load plate reference homography: %s", exc)
        return None


def _serialize_homography(h: np.ndarray) -> list:
    """Convert homography ndarray to a JSON-serializable list."""
    if h is None:
        return []
    return np.asarray(h, dtype=np.float64).tolist()


def _compute_min_z(points: list[dict]) -> Optional[float]:
    """Compute minimum z from a list of point dicts or return None."""
    if not points:
        return None
    try:
        return min(float(pt.get("z", 0.0)) for pt in points)
    except Exception:
        return None


def _load_robot_calibration(path: Optional[Path] = None) -> None:
    """Load robot calibration data from file."""
    if path is None:
        path = _robot_calibration_path()
    data = _read_json(path)
    if not data:
        return

    points = data.get("points", [])
    if isinstance(points, list):
        SESSION.robot_calibration_points = points

    min_z = data.get("min_z")
    if isinstance(min_z, (int, float)):
        SESSION.robot_calibration_min_z = float(min_z)
    elif SESSION.robot_calibration_points:
        SESSION.robot_calibration_min_z = _compute_min_z(SESSION.robot_calibration_points)

    plate_reference = data.get("plate_reference_homography")
    homography = _deserialize_homography(plate_reference)
    if homography is not None:
        SESSION.plate_reference_homography = homography

    SESSION.plate_to_robot_transform = None
    SESSION.plate_origin_mm = None
    LOGGER.info("Loaded robot calibration data from %s", path)


def _save_robot_calibration(points: list[dict], path: Optional[Path] = None) -> None:
    """Save robot calibration data to file."""
    if path is None:
        path = _robot_calibration_path()

    min_z = _compute_min_z(points)

    coord_id = _get_robot_coordinate_id()
    payload = {
        "points": points,
        "min_z": min_z,
        "timestamp": time.time(),
        "coordinate_id": coord_id,
    }

    if SESSION.plate_reference_homography is not None:
        payload["plate_reference_homography"] = _serialize_homography(
            SESSION.plate_reference_homography
        )

    _write_json_atomic(path, payload)


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
# Coordinate Transformations: Pixel \u2194 Workspace \u2194 Robot
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


def _workspace_origin_and_basis(
    workspace_def: workspace.WorkspaceDefinition,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (origin, basis) arrays used to project camera points into workspace coords."""
    origin = workspace_def.origin_camera
    basis = np.column_stack([workspace_def.x_axis_camera, workspace_def.y_axis_camera])
    return origin, basis


def _project_camera_xyz_to_workspace(
    camera_xyz: np.ndarray,
    origin: np.ndarray,
    basis: np.ndarray,
) -> Tuple[float, float]:
    """Project a camera-space 3D point (xyz) into workspace mm coordinates (x,y).

    Returns the (x_mm, y_mm) coefficients solving basis * coeffs = camera_xyz - origin.
    """
    coeffs, *_ = np.linalg.lstsq(basis, camera_xyz - origin, rcond=None)
    return float(coeffs[0]), float(coeffs[1])


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
    origin, basis = _workspace_origin_and_basis(workspace_def)
    return _project_camera_xyz_to_workspace(camera_xyz, origin, basis)


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

    origin, basis = _workspace_origin_and_basis(workspace_def)
    try:
        return _project_camera_xyz_to_workspace(camera_xyz, origin, basis)
    except Exception as exc:
        raise RuntimeError(
            f"Unable to project robot pose into workspace: {exc}"
        ) from exc


# ==============================================================================
# Rigid Transform Utilities
# ==============================================================================


def _validate_point_clouds_shape(
    from_pts: np.ndarray,
    to_pts: np.ndarray,
    min_points: int = 3,
) -> None:
    """Validate that point clouds have matching shapes and sufficient points.

    Raises ValueError with messages matching previous behavior.
    """
    if from_pts.shape != to_pts.shape:
        raise ValueError("Point clouds must have the same shape.")
    if from_pts.shape[0] < min_points:
        if min_points == 3:
            raise ValueError("At least 3 points are required for a rigid transform")
        raise ValueError(
            f"At least {min_points} points are required for a 2D rigid transform.")


def _compute_rigid_transform_internal(
    from_pts: np.ndarray,
    to_pts: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute the rotation and translation matrices for rigid transform.

    Common logic shared by _rigid_transform and _rigid_transform_2d.
    Assumes validation has already been performed.

    Args:
        from_pts: Source point cloud (N x D).
        to_pts: Target point cloud (N x D).

    Returns:
        (rotation, translation): Rotation matrix (D x D) and translation vector (D,).
    """
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


def _rigid_transform(from_pts: np.ndarray, to_pts: np.ndarray) -> workspace.RigidTransform:
    """Compute best-fit rigid transform from from_pts -> to_pts."""
    _validate_point_clouds_shape(from_pts, to_pts, min_points=3)
    r, t = _compute_rigid_transform_internal(from_pts, to_pts)
    return workspace.RigidTransform(rotation=r, translation=t)


def _rigid_transform_2d(
    from_pts: np.ndarray,
    to_pts: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute 2D rigid transform (rotation matrix, translation vector)."""
    _validate_point_clouds_shape(from_pts, to_pts, min_points=2)
    return _compute_rigid_transform_internal(from_pts, to_pts)
