"""Calibration overlay drawing functions (ChArUco, workspace, plate)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np

from vision import aruco_board, markers, plate, workspace
from app.codroid_picker.session import SESSION
from app.codroid_picker import state
import app.settings as app_settings

from .drawing import _draw_label, _draw_dotted_polyline, _workspace_corners_px

# Get SETTINGS from app_settings
SETTINGS = app_settings.SETTINGS


def _draw_marker_detections(vis: np.ndarray, detections: dict, color: tuple[int, int, int]) -> None:
    """Draw marker polygons and labels for a detection result."""
    for marker_id, detection in detections.items():
        corners = detection.corners.astype(int)
        cv2.polylines(vis, [corners], True, color, 2)
        center = np.mean(corners, axis=0).astype(int)
        _draw_label(vis, str(marker_id), tuple(center), color)


def _evaluate_plate_status(calibration_result) -> dict:
    """Compute plate marker status metrics used by overlay and refresh helpers."""
    detections = calibration_result.detections
    count = len(detections)
    min_required = SETTINGS.min_plate_markers
    is_ok = count >= min_required

    current_center = None
    movement_px = None
    if count > 0:
        current_center = np.mean([
            np.mean(det.corners, axis=0) for det in detections.values()
        ], axis=0)
        if SESSION.robot_calibration_baseline_px is not None:
            movement_px = float(np.linalg.norm(
                current_center - SESSION.robot_calibration_baseline_px
            ))

    stable = movement_px is None or movement_px <= SETTINGS.plate_movement_threshold_px
    return {
        "count": count,
        "min_required": min_required,
        "is_ok": is_ok,
        "stable": stable,
        "movement_px": movement_px,
        "current_center": current_center,
    }


def draw_charuco_overlay(frame: np.ndarray, board_path: Path) -> Tuple[np.ndarray, str, str]:
    """Draw ChArUco board detection overlay."""
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
        detector = cv2.aruco.ArucoDetector(dictionary, detector_params)

        # Detect markers
        marker_corners, marker_ids, rejected = detector.detectMarkers(gray)

        # Draw detected markers
        cv2.aruco.drawDetectedMarkers(vis, marker_corners, marker_ids)

        if marker_corners and len(marker_corners) > 0:
            # Interpolate ChArUco corners
            charuco_retval, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
                marker_corners, marker_ids, gray, board
            )

            if charuco_retval > 0:
                cv2.aruco.drawDetectedCornersCharuco(
                    vis, charuco_corners, charuco_ids, (0, 255, 0)
                )
                status = f"ChArUco corners detected: {charuco_retval}"
                metric = f"ChArUco corners: {charuco_retval}"
            else:
                status = "No ChArUco corners detected."
                metric = "ChArUco corners: 0"
        else:
            status = "No ArUco markers detected."
            metric = "ChArUco corners: 0"

    except Exception as exc:
        status = f"ChArUco detection error: {exc}"
        metric = "ChArUco corners: --"

    return vis, status, metric


def draw_workspace_overlay(
    frame: np.ndarray,
    marker_ids: Tuple[int, int],
    marker_size_mm: float,
) -> Tuple[np.ndarray, str, str]:
    """Draw workspace definition overlay."""
    vis = frame.copy()

    try:
        detection_result = markers.detect_workspace_markers(
            frame, marker_ids, marker_size_mm
        )

        if len(detection_result.detections) == 0:
            status = "No workspace markers detected."
            return vis, status, "Workspace markers: 0/2"

        _draw_marker_detections(vis, detection_result.detections, (0, 255, 0))

        if len(detection_result.detections) == 2:
            # Both markers detected, try to compute workspace
            try:
                workspace_def = workspace.compute_workspace_definition(
                    detection_result, marker_ids, marker_size_mm
                )

                # Draw workspace boundary
                corners_px = _workspace_corners_px(workspace_def)
                _draw_dotted_polyline(vis, corners_px, (255, 255, 0), 2)

                status = f"Workspace defined: {workspace_def.width_mm:.1f}×{workspace_def.height_mm:.1f} mm"
                metric = "Workspace markers: 2/2"

            except Exception as exc:
                status = f"Workspace computation failed: {exc}"
                metric = "Workspace markers: 2/2 (error)"
        else:
            count = len(detection_result.detections)
            status = f"Partial workspace markers detected: {count}/2"
            metric = f"Workspace markers: {count}/2"

    except Exception as exc:
        status = f"Workspace detection error: {exc}"
        metric = "Workspace markers: error"

    return vis, status, metric


def draw_plate_overlay(frame: np.ndarray, layout_path: Path) -> Tuple[np.ndarray, str, str]:
    """Draw calibration plate detection overlay."""
    vis = frame.copy()

    if not layout_path.exists():
        status = "Calibration plate layout not found."
        return vis, status, "Plate markers: --"

    try:
        calibration_result = plate.detect_calibration_plate(frame, layout_path)

        if len(calibration_result.detections) == 0:
            status = "No calibration plate markers detected."
            state.update_plate_status(
                ok=False,
                stable=False,
                message=status,
                count=0
            )
            return vis, status, "Plate markers: 0"

        _draw_marker_detections(vis, calibration_result.detections, (255, 0, 255))

        status_data = _evaluate_plate_status(calibration_result)
        count = status_data["count"]
        metric = f"Plate markers: {count}"

        if status_data["is_ok"]:
            status = f"Plate markers detected: {count} (stable: {'yes' if status_data['stable'] else 'no'})"
        else:
            status = f"Insufficient plate markers: {count}/{status_data['min_required']}"

        state.update_plate_status(
            ok=status_data["is_ok"],
            stable=status_data["stable"],
            message=status,
            count=count,
            movement_px=status_data["movement_px"],
            center_px=tuple(status_data["current_center"]) if status_data["current_center"] is not None else None,
        )

    except Exception as exc:
        status = f"Plate detection error: {exc}"
        metric = "Plate markers: error"
        state.update_plate_status(
            ok=False,
            stable=False,
            message=status,
            count=0
        )

    return vis, status, metric


def _refresh_plate_status(frame: np.ndarray, layout_path: Path) -> None:
    """Update PLATE_STATUS without modifying the displayed overlay frame."""
    try:
        state.update_plate_status(
            ok=False,
            stable=False,
            message="Checking plate status...",
            count=0
        )

        calibration_result = plate.detect_calibration_plate(frame, layout_path)
        status_data = _evaluate_plate_status(calibration_result)
        count = status_data["count"]

        if status_data["is_ok"] and count > 0:
            if status_data["movement_px"] is not None:
                status = f"Plate markers: {count} (movement: {status_data['movement_px']:.1f}px)"
            else:
                status = f"Plate markers: {count}"
            state.update_plate_status(
                ok=True,
                stable=status_data["stable"],
                message=status,
                count=count,
                movement_px=status_data["movement_px"],
                center_px=tuple(status_data["current_center"]) if status_data["current_center"] is not None else None,
            )
        else:
            status = f"Insufficient plate markers: {count}/{status_data['min_required']}"
            state.update_plate_status(
                ok=False,
                stable=False,
                message=status,
                count=count
            )

    except Exception as exc:  # noqa: BLE001
        logging.debug("Plate status refresh skipped: %s", exc)
        state.update_plate_status(
            ok=False,
            stable=False,
            message=f"Plate status error: {exc}",
            count=0
        )


