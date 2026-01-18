"""
Coordinate transform utilities for the calibration UI.

This module provides functions for converting between different coordinate systems:
- Workspace (mm) coordinates
- Robot base coordinates
- Coord5 (calibrated) coordinates
- Plate coordinates
- Pixel coordinates
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple, TYPE_CHECKING

import cv2
import numpy as np

import app.settings as app_settings
from app.codroid_picker.session import SESSION
from app.codroid_picker.robot_helpers import safe_float as _safe_float
from vision import plate, workspace

if TYPE_CHECKING:
    pass

LOGGER = logging.getLogger(__name__)

DEFAULT_PLATE_PATH = app_settings.DEFAULT_PLATE_PATH
SETTINGS = app_settings.SETTINGS


def _get_robot_session():
    """Get the robot session - imported locally to avoid circular deps."""
    from app.codroid_picker.app_instance import registry
    return registry().robot_session


def _rigid_transform(from_pts: np.ndarray, to_pts: np.ndarray) -> workspace.RigidTransform:
    """Compute rigid transform (rotation + translation) from one point set to another."""
    from app.codroid_picker.procedures import _rigid_transform as proc_rigid_transform
    return proc_rigid_transform(from_pts, to_pts)


def _robot_to_workspace_mm(
    workspace_def: workspace.WorkspaceDefinition,
    robot_xyz: np.ndarray,
) -> Optional[Tuple[float, float]]:
    """Convert robot base coordinates to workspace coordinates."""
    from app.codroid_picker.procedures import _robot_to_workspace_mm as proc_robot_to_workspace
    return proc_robot_to_workspace(workspace_def, robot_xyz)


# ---------------------------------------------------------------------------
# Coordinate clamping
# ---------------------------------------------------------------------------


def _clamp_z_for_coord5(z: float, *, use_base_coords: bool) -> float:
    """Clamp Z to safe maximum when using coordinate system 5.

    CRITICAL: Coord5 Z-axis is INVERTED:
    - Negative Z = UP (safe, away from table)
    - Positive Z = DOWN (dangerous, toward table)

    This prevents the robot from diving into the table by clamping
    Z values that are TOO POSITIVE (too far down).
    """
    if use_base_coords:
        # Base coordinates: positive Z = up, so we don't clamp
        return z
    max_z = SETTINGS.coord5_max_z_mm
    if z > max_z:
        LOGGER.warning(
            "Clamping coord5 Z from %.1f to max %.1f mm",
            z,
            max_z,
        )
        return max_z
    return z


def _apply_robot_z_floor(robot_xyz: np.ndarray) -> np.ndarray:
    """Apply minimum Z floor constraint from robot calibration."""
    min_z = SESSION.robot_calibration_min_z
    if min_z is None or robot_xyz[2] >= min_z:
        return robot_xyz
    adjusted = np.array(robot_xyz, dtype=np.float64, copy=True)
    adjusted[2] = min_z
    return adjusted


# ---------------------------------------------------------------------------
# Rotation utilities
# ---------------------------------------------------------------------------


def _rotation_from_abc(
    a: float,
    b: float,
    c: float,
    *,
    degrees: bool = True,
) -> np.ndarray:
    """Build a rotation matrix from Euler angles (XYZ convention)."""
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
    """Compute the inverse of a rigid transform."""
    inv_rot = transform.rotation.T
    inv_trans = -inv_rot @ transform.translation
    return workspace.RigidTransform(rotation=inv_rot, translation=inv_trans)


def _coord_frame_to_transform(frame: dict) -> workspace.RigidTransform:
    """Convert a robot coordinate frame dict to a RigidTransform."""
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


# ---------------------------------------------------------------------------
# Coord5 transform (base <-> coord5)
# ---------------------------------------------------------------------------


def _extract_plate_robot_points(
    min_points: int = 3,
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Extract plate and robot calibration points for transform computation.

    This consolidates the duplicate point extraction logic used by
    _coord5_transform() and _plate_to_robot_transform().

    Args:
        min_points: Minimum number of points required.

    Returns:
        Tuple of (plate_pts, robot_pts, origin_mm) or None if insufficient data.
        - plate_pts: Nx3 array of plate points relative to origin
        - robot_pts: Nx3 array of corresponding robot points
        - origin_mm: Origin point in plate mm coordinates
    """
    if not SESSION.robot_calibration_points or len(SESSION.robot_calibration_points) < min_points:
        return None
    try:
        plate_layout = plate.load_plate_layout(DEFAULT_PLATE_PATH)
        plate_points = plate_layout.robot_calibration_points()
        if len(plate_points) < min_points:
            return None
        origin_mm = np.asarray(plate_points[0].center_mm, dtype=np.float64)
        plate_pts = np.asarray(
            [
                [
                    float(pt.center_mm[0] - origin_mm[0]),
                    float(pt.center_mm[1] - origin_mm[1]),
                    0.0,
                ]
                for pt in plate_points[:min_points]
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
                for pt in SESSION.robot_calibration_points[:min_points]
            ],
            dtype=np.float64,
        )
        return plate_pts, robot_pts, origin_mm
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("Failed to extract calibration points: %s", exc)
        return None


def _coord5_transform() -> Optional[workspace.RigidTransform]:
    """Get or compute the coord5 transform from calibration data."""
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

    # Extract points using shared helper
    points = _extract_plate_robot_points(min_points=3)
    if points is None:
        return None

    plate_pts, robot_pts, _origin = points
    try:
        transform = _rigid_transform(plate_pts, robot_pts)
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("Failed to build coord5 transform: %s", exc)
        return None
    SESSION.coord5_transform = transform
    return transform


def _base_to_coord5(base_xyz: np.ndarray) -> Optional[np.ndarray]:
    """Convert base coordinates to coord5 coordinates."""
    transform = _coord5_transform()
    if transform is None:
        return None
    inv = _invert_rigid_transform(transform)
    return inv.rotation @ base_xyz + inv.translation


def _coord5_to_base(coord5_xyz: np.ndarray) -> Optional[np.ndarray]:
    """Convert coord5 coordinates to base coordinates."""
    transform = _coord5_transform()
    if transform is None:
        return None
    return transform.rotation @ coord5_xyz + transform.translation


# ---------------------------------------------------------------------------
# Workspace <-> Coord5 conversions
# ---------------------------------------------------------------------------


def _workspace_to_coord5_mm(
    workspace_def: workspace.WorkspaceDefinition,
    workspace_xy: Tuple[float, float],
    z_offset_mm: float = 0.0,
) -> Optional[Tuple[float, float]]:
    """Convert workspace (mm) coordinates to coord5 (mm) coordinates."""
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
    """Convert coord5 (mm) coordinates to workspace (mm) coordinates."""
    coord5_xyz = np.array(
        [coord5_xy[0], coord5_xy[1], 0.0], dtype=np.float64
    )
    base_xyz = _coord5_to_base(coord5_xyz)
    if base_xyz is None:
        return None
    return _robot_to_workspace_mm(workspace_def, base_xyz)


# ---------------------------------------------------------------------------
# Pose builders
# ---------------------------------------------------------------------------


def _pose_from_workspace_mm(
    workspace_def: workspace.WorkspaceDefinition,
    x_mm: float,
    y_mm: float,
    z_offset_mm: float,
) -> dict:
    """Build a robot pose dict from workspace (mm) coordinates."""
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


def _pose_from_workspace_coord5_mm(
    workspace_def: workspace.WorkspaceDefinition,
    x_mm: float,
    y_mm: float,
    z_offset_mm: float,
) -> dict:
    """Build a robot pose dict for coord5 from workspace (mm) coordinates."""
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
    """Build a robot pose dict directly from coord5 (mm) coordinates."""
    z_val = _clamp_z_for_coord5(float(z_offset_mm), use_base_coords=False)
    return {
        "x": float(x_mm),
        "y": float(y_mm),
        "z": float(z_val),
        "a": 0.0,
        "b": 0.0,
        "c": 0.0,
    }


# ---------------------------------------------------------------------------
# Plate transforms
# ---------------------------------------------------------------------------


def _plate_reference_homography() -> Optional[np.ndarray]:
    """Get the plate reference homography from session or calibration bundle."""
    if SESSION.plate_reference_homography is not None:
        return SESSION.plate_reference_homography
    bundle = SESSION.calibration_bundle
    if bundle is not None and bundle.plate_homography is not None:
        return bundle.plate_homography
    return None


def _pixel_to_plate_mm(
    pixel: Tuple[float, float],
) -> Optional[np.ndarray]:
    """Convert a pixel coordinate to plate (mm) coordinates using homography."""
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
    """Get or compute the plate-to-robot transform from calibration data."""
    if (
        SESSION.plate_to_robot_transform is not None
        and SESSION.plate_origin_mm is not None
    ):
        return SESSION.plate_to_robot_transform, SESSION.plate_origin_mm

    # Extract points using shared helper
    points = _extract_plate_robot_points(min_points=3)
    if points is None:
        return None

    plate_pts, robot_pts, origin_mm = points
    try:
        transform = _rigid_transform(plate_pts, robot_pts)
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("Failed to build plate-to-robot transform: %s", exc)
        return None
    SESSION.plate_to_robot_transform = transform
    SESSION.plate_origin_mm = origin_mm
    return transform, origin_mm


def _default_transform() -> workspace.RigidTransform:
    """Return a default identity transform."""
    return workspace.RigidTransform(rotation=np.eye(3), translation=np.zeros(3))
