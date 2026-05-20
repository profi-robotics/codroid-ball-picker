"""Calibration overlay drawing functions (ChArUco, workspace, plate)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Tuple

import cv2
import numpy as np

from vision import aruco_board, markers, plate, workspace
from app.codroid_picker.session import SESSION
from app.codroid_picker import state
import app.settings as app_settings

from .drawing import _draw_label, _draw_dotted_polyline, _workspace_corners_px

# Get SETTINGS from app_settings
SETTINGS = app_settings.SETTINGS

_CHARUCO_CORNER_COLOR = "rgb(0, 255, 0)"
_CHARUCO_MARKER_COLOR = "rgb(0, 128, 255)"
_PLATE_MARKER_COLOR = "rgb(255, 0, 255)"
_PLATE_POINT_COLOR = "rgb(255, 255, 0)"
_WORKSPACE_MARKER_COLOR = "rgb(0, 255, 0)"
_WORKSPACE_BOUNDARY_COLOR = "rgb(255, 215, 0)"


def _draw_marker_detections(
    vis: np.ndarray,
    detections: dict,
    color: tuple[int, int, int],
) -> None:
    """Draw marker polygons and labels for a detection result."""
    for marker_id, detection in detections.items():
        corners = detection.corners.astype(int)
        cv2.polylines(vis, [corners], True, color, 2)
        center = np.mean(corners, axis=0).astype(int)
        _draw_label(vis, str(marker_id), tuple(center), color)


def _point_payload(point: np.ndarray) -> dict[str, int]:
    """Return an integer point payload for browser overlay JSON."""
    x, y = np.ravel(point)[:2]
    return {"x": int(round(float(x))), "y": int(round(float(y)))}


def _marker_outline_primitives(
    marker_corners: Any,
    marker_ids: Any,
) -> list[dict[str, Any]]:
    """Build browser primitives for detected ArUco marker outlines."""
    if marker_corners is None or marker_ids is None:
        return []

    primitives: list[dict[str, Any]] = []
    ids = np.asarray(marker_ids).reshape(-1)
    for raw_corners, raw_id in zip(marker_corners, ids):
        corners = np.asarray(raw_corners, dtype=np.float64).reshape(-1, 2)
        if len(corners) != 4:
            continue
        points = [_point_payload(point) for point in corners]
        center = np.mean(corners, axis=0)
        primitives.append(
            {
                "type": "polygon",
                "points": points,
                "color": _CHARUCO_MARKER_COLOR,
                "lineWidth": 2,
            }
        )
        primitives.append(
            {
                "type": "text",
                "x": int(round(float(center[0]))),
                "y": int(round(float(center[1]))),
                "text": f"id={int(raw_id)}",
                "color": _CHARUCO_MARKER_COLOR,
            }
        )
    return primitives


def _detection_outline_primitives(
    detections: dict[int, Any],
    color: str,
    *,
    label_prefix: str = "",
) -> list[dict[str, Any]]:
    """Build browser primitives for marker detection outlines."""
    primitives: list[dict[str, Any]] = []
    for marker_id, detection in detections.items():
        corners = np.asarray(detection.corners, dtype=np.float64).reshape(-1, 2)
        if len(corners) != 4:
            continue
        center = np.mean(corners, axis=0)
        primitives.append(
            {
                "type": "polygon",
                "points": [_point_payload(point) for point in corners],
                "color": color,
                "lineWidth": 2,
            }
        )
        primitives.append(
            {
                "type": "text",
                "x": int(round(float(center[0]))),
                "y": int(round(float(center[1]))),
                "text": f"{label_prefix}{marker_id}",
                "color": color,
            }
        )
    return primitives


def _polyline_loop_primitives(
    points: np.ndarray,
    color: str,
    *,
    dashed: bool = False,
    line_width: int = 2,
) -> list[dict[str, Any]]:
    """Build line primitives for a closed polygon boundary."""
    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    if len(points) < 2 or not np.all(np.isfinite(points)):
        return []

    primitives: list[dict[str, Any]] = []
    for idx, start in enumerate(points):
        end = points[(idx + 1) % len(points)]
        primitives.append(
            {
                "type": "line",
                "from": _point_payload(start),
                "to": _point_payload(end),
                "color": color,
                "lineWidth": line_width,
                "dashed": dashed,
            }
        )
    return primitives


def _workspace_boundary_primitives(
    workspace_def: workspace.WorkspaceDefinition,
) -> list[dict[str, Any]]:
    """Build browser primitives for a workspace boundary."""
    try:
        corners_px = _workspace_corners_px(workspace_def)
    except Exception:
        return []
    return _polyline_loop_primitives(
        corners_px,
        _WORKSPACE_BOUNDARY_COLOR,
        dashed=True,
        line_width=2,
    )


def _charuco_corner_primitives(
    charuco_corners: Any,
    charuco_ids: Any,
) -> list[dict[str, Any]]:
    """Build browser primitives for detected ChArUco chessboard corners."""
    if charuco_corners is None or charuco_ids is None:
        return []

    primitives: list[dict[str, Any]] = []
    corners = np.asarray(charuco_corners, dtype=np.float64).reshape(-1, 2)
    ids = np.asarray(charuco_ids).reshape(-1)
    for corner, raw_id in zip(corners, ids):
        point = _point_payload(corner)
        primitives.append(
            {
                "type": "circle",
                **point,
                "radius": 4,
                "color": _CHARUCO_CORNER_COLOR,
                "fill": True,
            }
        )
        primitives.append(
            {
                "type": "text",
                **point,
                "text": str(int(raw_id)),
                "color": _CHARUCO_CORNER_COLOR,
            }
        )
    return primitives


def _draw_charuco_semantic_overlay(
    vis: np.ndarray,
    marker_corners: Any,
    marker_ids: Any,
    charuco_corners: Any,
    charuco_ids: Any,
) -> None:
    """Draw the same simple geometry used by browser primitives onto an image."""
    if marker_corners is not None and marker_ids is not None:
        ids = np.asarray(marker_ids).reshape(-1)
        for raw_corners, raw_id in zip(marker_corners, ids):
            corners = np.asarray(raw_corners, dtype=np.float64).reshape(-1, 2)
            if len(corners) != 4:
                continue
            int_corners = np.round(corners).astype(np.int32)
            cv2.polylines(vis, [int_corners], True, (255, 128, 0), 2)
            center = tuple(np.round(np.mean(corners, axis=0)).astype(int))
            _draw_label(vis, f"id={int(raw_id)}", center, (255, 128, 0))

    if charuco_corners is not None and charuco_ids is not None:
        corners = np.asarray(charuco_corners, dtype=np.float64).reshape(-1, 2)
        ids = np.asarray(charuco_ids).reshape(-1)
        for corner, raw_id in zip(corners, ids):
            center = tuple(np.round(corner).astype(int))
            cv2.circle(vis, center, 4, (0, 255, 0), -1)
            _draw_label(
                vis,
                str(int(raw_id)),
                (center[0] + 6, center[1] - 6),
                (0, 255, 0),
            )


def _draw_plate_calibration_points(vis: np.ndarray, calibration_result) -> None:
    """Draw robot calibration point locations on the plate overlay."""
    for center, label in _plate_calibration_point_pixels(calibration_result):
        cv2.circle(vis, center, 8, (0, 255, 255), 2)
        label_pos = (center[0] + 10, center[1] - 10)
        _draw_label(vis, label, label_pos, (0, 255, 255))


def _plate_calibration_point_pixels(
    calibration_result: plate.PlateCalibration,
) -> list[tuple[tuple[int, int], str]]:
    """Return robot calibration point image locations and labels."""
    points = calibration_result.plate.robot_calibration_points()
    if not points:
        return []
    try:
        inv_h = np.linalg.inv(calibration_result.homography)
    except np.linalg.LinAlgError:
        return []
    point_mm = np.array(
        [pt.center_mm for pt in points[:3]],
        dtype=np.float64,
    ).reshape(-1, 1, 2)
    point_px = cv2.perspectiveTransform(point_mm, inv_h).reshape(-1, 2)
    if not np.all(np.isfinite(point_px)):
        return []

    result: list[tuple[tuple[int, int], str]] = []
    for idx, pt in enumerate(point_px, start=1):
        rounded = np.round(pt).astype(int)
        center = (int(rounded[0]), int(rounded[1]))
        result.append((center, f"P{idx}"))
    return result


def _plate_calibration_point_primitives(
    calibration_result: plate.PlateCalibration,
) -> list[dict[str, Any]]:
    """Build browser primitives for robot calibration points on the plate."""
    primitives: list[dict[str, Any]] = []
    for center, label in _plate_calibration_point_pixels(calibration_result):
        primitives.append(
            {
                "type": "circle",
                "x": center[0],
                "y": center[1],
                "radius": 8,
                "color": _PLATE_POINT_COLOR,
            }
        )
        primitives.append(
            {
                "type": "text",
                "x": center[0] + 10,
                "y": center[1] - 10,
                "text": label,
                "color": _PLATE_POINT_COLOR,
            }
        )
    return primitives


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


def build_charuco_overlay(
    frame: np.ndarray,
    board_path: Path,
) -> Tuple[np.ndarray, str, str, list[dict[str, Any]]]:
    """Draw ChArUco board detection overlay and return browser primitives."""
    vis = frame.copy()
    if not board_path.exists():
        status = f"ChArUco layout not found: {board_path}"
        return vis, status, "ChArUco corners: --", []

    try:
        board_spec = aruco_board.load_charuco_board_spec(board_path)
        board = board_spec.create_board()
        gray = cv2.cvtColor(vis, cv2.COLOR_BGR2GRAY)

        # Set up detector parameters with inverted marker support
        detector_params = cv2.aruco.DetectorParameters()
        detector_params.detectInvertedMarker = True

        # Use CharucoDetector for combined marker + corner detection
        charuco_params = cv2.aruco.CharucoParameters()
        charuco_detector = cv2.aruco.CharucoDetector(
            board,
            charuco_params,
            detector_params,
        )
        (
            charuco_corners,
            charuco_ids,
            marker_corners,
            marker_ids,
        ) = charuco_detector.detectBoard(gray)

        primitives = _marker_outline_primitives(marker_corners, marker_ids)

        if charuco_corners is not None and len(charuco_corners) > 0:
            primitives.extend(
                _charuco_corner_primitives(charuco_corners, charuco_ids)
            )
            _draw_charuco_semantic_overlay(
                vis,
                marker_corners,
                marker_ids,
                charuco_corners,
                charuco_ids,
            )
            charuco_count = len(charuco_corners)
            status = f"ChArUco corners detected: {charuco_count}"
            metric = f"ChArUco corners: {charuco_count}"
        elif marker_corners is not None and len(marker_corners) > 0:
            _draw_charuco_semantic_overlay(
                vis,
                marker_corners,
                marker_ids,
                None,
                None,
            )
            status = "Markers detected but no ChArUco corners."
            metric = "ChArUco corners: 0"
        else:
            status = "No ArUco markers detected."
            metric = "ChArUco corners: 0"

    except Exception as exc:
        status = f"ChArUco detection error: {exc}"
        metric = "ChArUco corners: --"
        primitives = []

    return vis, status, metric, primitives


def draw_charuco_overlay(
    frame: np.ndarray,
    board_path: Path,
) -> Tuple[np.ndarray, str, str]:
    """Draw ChArUco board detection overlay."""
    vis, status, metric, _primitives = build_charuco_overlay(
        frame,
        board_path,
    )
    return vis, status, metric


def build_workspace_overlay(
    frame: np.ndarray,
    marker_ids: Tuple[int, int],
    marker_size_mm: float,
) -> Tuple[np.ndarray, str, str, list[dict[str, Any]]]:
    """Draw workspace marker overlay and return browser primitives."""
    vis = frame.copy()
    primitives: list[dict[str, Any]] = []

    try:
        detection_result = markers.detect_workspace_markers(
            frame, marker_ids, marker_size_mm
        )

        if len(detection_result.detections) == 0:
            status = "No workspace markers detected."
            return vis, status, "Workspace markers: 0/2", primitives

        _draw_marker_detections(vis, detection_result.detections, (0, 255, 0))
        primitives.extend(
            _detection_outline_primitives(
                detection_result.detections,
                _WORKSPACE_MARKER_COLOR,
            )
        )

        if len(detection_result.detections) == 2:
            # Both markers detected, try to compute workspace
            try:
                workspace_def = workspace.compute_workspace_definition(
                    detection_result, marker_ids, marker_size_mm
                )

                # Draw workspace boundary
                corners_px = _workspace_corners_px(workspace_def)
                _draw_dotted_polyline(vis, corners_px, (255, 255, 0), 2)
                primitives.extend(_workspace_boundary_primitives(workspace_def))

                status = (
                    f"Workspace defined: {workspace_def.width_mm:.1f}×"
                    f"{workspace_def.height_mm:.1f} mm"
                )
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

    return vis, status, metric, primitives


def draw_workspace_overlay(
    frame: np.ndarray,
    marker_ids: Tuple[int, int],
    marker_size_mm: float,
) -> Tuple[np.ndarray, str, str]:
    """Draw workspace definition overlay."""
    vis, status, metric, _primitives = build_workspace_overlay(
        frame,
        marker_ids,
        marker_size_mm,
    )
    return vis, status, metric


def build_plate_overlay(
    frame: np.ndarray,
    layout_path: Path,
) -> Tuple[np.ndarray, str, str, list[dict[str, Any]]]:
    """Draw calibration plate overlay and return browser primitives."""
    vis = frame.copy()
    primitives: list[dict[str, Any]] = []

    if not layout_path.exists():
        status = "Calibration plate layout not found."
        return vis, status, "Plate markers: --", primitives

    try:
        plate_layout = plate.load_plate_layout(layout_path)
        calibration_result = plate.estimate_plate_calibration(frame, plate_layout)

        if len(calibration_result.detections) == 0:
            status = "No calibration plate markers detected."
            state.update_plate_status(
                ok=False,
                stable=False,
                message=status,
                count=0
            )
            return vis, status, "Plate markers: 0", primitives

        _draw_marker_detections(vis, calibration_result.detections, (255, 0, 255))
        _draw_plate_calibration_points(vis, calibration_result)
        primitives.extend(
            _detection_outline_primitives(
                calibration_result.detections,
                _PLATE_MARKER_COLOR,
            )
        )
        primitives.extend(_plate_calibration_point_primitives(calibration_result))

        status_data = _evaluate_plate_status(calibration_result)
        count = status_data["count"]
        metric = f"Plate markers: {count}"

        if status_data["is_ok"]:
            stable_text = "yes" if status_data["stable"] else "no"
            status = f"Plate markers detected: {count} (stable: {stable_text})"
        else:
            status = f"Insufficient plate markers: {count}/{status_data['min_required']}"

        state.update_plate_status(
            ok=status_data["is_ok"],
            stable=status_data["stable"],
            message=status,
            count=count,
            movement_px=status_data["movement_px"],
            center_px=(
                tuple(status_data["current_center"])
                if status_data["current_center"] is not None
                else None
            ),
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

    return vis, status, metric, primitives


def draw_plate_overlay(frame: np.ndarray, layout_path: Path) -> Tuple[np.ndarray, str, str]:
    """Draw calibration plate detection overlay."""
    vis, status, metric, _primitives = build_plate_overlay(frame, layout_path)
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

        plate_layout = plate.load_plate_layout(layout_path)
        calibration_result = plate.estimate_plate_calibration(frame, plate_layout)
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
                center_px=(
                    tuple(status_data["current_center"])
                    if status_data["current_center"] is not None
                    else None
                ),
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
