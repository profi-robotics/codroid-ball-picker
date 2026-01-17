"""Robot plate calibration procedure.

Handles the 3-point calibration process for setting up the robot coordinate
system (CoordinateId 5) using a calibration plate with edge markers.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import TYPE_CHECKING, Callable, Optional, Tuple

import numpy as np

from codroid_api import CodroidAPI
from vision import markers, plate, workspace

from app.codroid_picker import state
from app.codroid_picker.app_instance import registry
from app.codroid_picker.utils import safe_float
from app.codroid_picker.overlays.base import _plate_center_px
from app.codroid_picker.persistence import _plate_is_stable
from app.codroid_picker.procedures import (
    _workspace_pixel_to_mm,
    _set_robot_guidance,
    _record_robot_accuracy,
    _clear_robot_accuracy_guidance,
    _robot_posture_ready,
    _ensure_posture_axes,
    _rigid_transform,
)
from app.codroid_picker.camera_calibration import get_latest_frame, save_workspace_definition

if TYPE_CHECKING:
    pass

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
# Robot plate calibration procedure
# ---------------------------------------------------------------------------


async def _run_robot_plate_calibration(
    robot_uri: str,
    plate_layout: plate.CalibrationPlate,
) -> Tuple[list[dict], dict]:
    """Capture three robot poses via flange button 1 and calibrate coord ID 5."""
    robot_session = registry().get_or_create_robot_session()
    app_session = registry().session
    settings = registry().settings
    robot_coordinate_id = registry().robot_coordinate_id

    def _ensure_plate_baseline() -> bool:
        if app_session.robot_calibration_baseline_px is not None:
            return True
        plate_status = state.get_plate_status_snapshot()
        center = plate_status.get("center_px")
        if center is not None:
            app_session.robot_calibration_baseline_px = np.asarray(
                center, dtype=np.float64
            )
            return True
        frame, _depth = get_latest_frame()
        if frame is None:
            return False
        try:
            calibration_result = plate.estimate_plate_calibration(
                frame, plate_layout
            )
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("Plate baseline detection failed: %s", exc)
            return False
        center_px = _plate_center_px(calibration_result.detections)
        if center_px is None:
            return False
        app_session.robot_calibration_baseline_px = np.asarray(
            center_px, dtype=np.float64
        )
        state.update_plate_status(
            ok=len(calibration_result.detections) >= settings.min_plate_markers,
            stable=len(
                calibration_result.detections) >= settings.min_plate_markers,
            message="Plate baseline refreshed from live detection.",
            count=len(calibration_result.detections),
            center_px=(float(center_px[0]), float(center_px[1])),
            movement_px=None,
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
        coord_id = None
        while time.monotonic() < deadline:
            status = robot_session.robot_status_snapshot()
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
            if app_session.robot_calibration_baseline_px is None:
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
                return await robot_session.next_flange_press(timeout=min(0.5, remaining))
            except asyncio.TimeoutError:
                continue

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
        await robot_session.reset_flange_queue()

        robot_points: list[dict] = []
        calib_points = plate_layout.robot_calibration_points()
        if len(calib_points) < 3:
            raise RuntimeError(
                "Calibration plate layout is missing calibration points."
            )

        point_labels = [
            "Point 1 (Origin)",
            "Point 2 (+X)",
            "Point 3 (+X/+Y)",
        ]
        for idx in range(3):
            _ensure_plate_stable(allow_lost=True)
            point_label = (
                point_labels[idx] if idx < len(
                    point_labels) else f"Point {idx + 1}"
            )
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
                posture, robot_session.position_snapshot())
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
            f"Switching to CoordinateId {robot_coordinate_id} for calibration...",
            status="running",
            verification_ready=False,
        )
        await _ensure_coordinate_id(
            robot_api, robot_coordinate_id, timeout_s=5.0, require=False
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

        calibration_request_time = time.time()
        payload_points = [
            robot_api.attach_coordinate_to_cpos(point, coordinate_id=0)
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
                robot_session.calibration_frame_snapshot,
                robot_session.calibration_frame_timestamp,
                calibration_request_time,
                settings.robot_coordinate_timeout_s,
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
                    return safe_float(coord_frame.get(key), 0.0)
            return 0.0

        coord_frame["a"] = _coord_axis(("a", "rx", "roll"))
        coord_frame["b"] = _coord_axis(("b", "ry", "pitch"))
        coord_frame["c"] = _coord_axis(("c", "rz", "yaw"))

        change_request_time = time.time()
        change_payload = {
            "id": robot_coordinate_id,
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

        publish_timeout = min(settings.robot_coordinate_timeout_s, 5.0)
        try:
            await _await_listener_frame(
                robot_session.coordinate_frame_snapshot,
                robot_session.coordinate_frame_timestamp,
                change_request_time,
                publish_timeout,
                "coordinate frame publish",
            )
        except asyncio.TimeoutError:
            LOGGER.warning(
                "No RobotCoordinate publish after calibration; reselecting CoordinateId."
            )
            await robot_api.set_current_coordinate_id(robot_coordinate_id)
            try:
                await _await_listener_frame(
                    robot_session.coordinate_frame_snapshot,
                    robot_session.coordinate_frame_timestamp,
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

    return await robot_session.run_with_robot(robot_uri, _runner)


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
