"""Robot plate calibration procedure.

Handles the 3-point calibration process for setting up the robot coordinate
system (CoordinateId 5) using a calibration plate with edge markers.
"""

from __future__ import annotations

import logging
from typing import Tuple

import numpy as np

from vision import markers, plate, workspace

from app.codroid_picker.app_instance import registry
from app.codroid_picker.persistence import _plate_is_stable
from app.codroid_picker.procedures import (
    _workspace_pixel_to_mm,
    _record_robot_accuracy,
    _rigid_transform,
)
from app.codroid_picker.camera_calibration import save_workspace_definition

LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Drop marker detection
# ---------------------------------------------------------------------------


def _detect_drop_marker(
    frame: np.ndarray,
    marker_id: int,
) -> Tuple[markers.MarkerDetection, Tuple[float, float]]:
    """Detect a drop marker in the frame and return its workspace coordinates."""
    session = registry().session
    detections = markers.detect_all_markers(frame)
    if marker_id not in detections:
        raise RuntimeError(f"Drop marker {marker_id} not detected.")
    marker = detections[marker_id]
    workspace_mm = None
    if session.workspace_definition is not None:
        workspace_mm = _workspace_pixel_to_mm(
            session.workspace_definition, marker.center
        )
    if workspace_mm is None:
        raise RuntimeError(
            "Workspace is not defined; run workspace detection first."
        )
    return marker, workspace_mm


# ---------------------------------------------------------------------------
# Plate visibility/stability checks
# ---------------------------------------------------------------------------


def _ensure_plate_visible(
    max_age_s: float = None,
) -> None:
    """Ensure the calibration plate markers are visible."""
    settings = registry().settings
    if max_age_s is None:
        max_age_s = settings.plate_visibility_max_age_s
    ok, message = _plate_is_stable(
        max_age_s=max_age_s,
        require_baseline=False,
        allow_lost=False,
    )
    if not ok:
        raise RuntimeError(message)


def _ensure_plate_stable(
    max_age_s: float = None,
    *,
    allow_lost: bool = False,
) -> None:
    """Ensure the calibration plate is stable and baseline is set."""
    settings = registry().settings
    if max_age_s is None:
        max_age_s = settings.plate_visibility_max_age_s
    ok, message = _plate_is_stable(
        max_age_s=max_age_s,
        require_baseline=True,
        allow_lost=allow_lost,
    )
    if not ok:
        raise RuntimeError(message)


# ---------------------------------------------------------------------------
# Apply calibration results
# ---------------------------------------------------------------------------


def _apply_robot_calibration_results(
    robot_points: list[dict],
    plate_layout: plate.CalibrationPlate,
) -> Tuple[workspace.RigidTransform, dict[str, float]]:
    """Apply the robot calibration results and compute the transform.

    Returns the rigid transform and calibration accuracy metrics.
    """
    plate_pts = []
    for calib_point in plate_layout.robot_calibration_points()[:3]:
        plate_pts.append(
            [float(calib_point.center_mm[0]), float(
                calib_point.center_mm[1]), 0.0]
        )
    robot_pts = []
    for pt in robot_points[:3]:
        robot_pts.append([float(pt["x"]), float(pt["y"]), float(pt["z"])])
    plate_pts_np = np.asarray(plate_pts, dtype=np.float64)
    robot_pts_np = np.asarray(robot_pts, dtype=np.float64)
    transform = _rigid_transform(plate_pts_np, robot_pts_np)

    # Compute calibration error diagnostics
    # Transform plate points to robot coordinates using the computed transform
    transformed_plate_pts = (
        transform.rotation @ plate_pts_np.T
    ).T + transform.translation

    # Compute position errors for each calibration point
    errors = robot_pts_np - transformed_plate_pts
    LOGGER.info("=== Calibration Error Diagnostics ===")
    LOGGER.info(
        "This shows how far the gripper was positioned from the ideal location:"
    )
    for i, (error, plate_pt, robot_pt) in enumerate(
        zip(errors, plate_pts_np, robot_pts_np)
    ):
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
        (
            "Good calibration!"
            if max_xy_error < 2.0
            else "Consider recalibrating with more careful positioning."
        ),
    )
    LOGGER.info("=" * 50)

    session = registry().session
    workspace_path = registry().workspace_path
    if session.workspace_definition is not None:
        ws = session.workspace_definition
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

    session.workspace_definition = updated_ws
    session.coord5_transform = None
    session.plate_to_robot_transform = None
    session.plate_origin_mm = None
    save_workspace_definition(updated_ws, workspace_path)
    accuracy = _record_robot_accuracy(mean_xy_error, max_xy_error)
    return transform, accuracy
