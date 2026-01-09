"""Overlay rendering helpers for the calibration UI."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
from vision import aruco_board, plate, workspace

from .constants import (
    MIN_PLATE_MARKERS,
    PLATE_MOVEMENT_THRESHOLD_PX,
)
from .state import PLATE_STATUS, SESSION

_workspace_rect_buffer: list[np.ndarray] = []
_WORKSPACE_SMOOTH_FRAMES = 5


def _draw_label(frame: np.ndarray, text: str, origin: Tuple[int, int], color: Tuple[int, int, int]) -> None:
    cv2.putText(frame, text, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


def _workspace_corners_px(workspace_def: workspace.WorkspaceDefinition) -> np.ndarray:
    if SESSION.calibration_bundle is None:
        raise RuntimeError("Calibration bundle required to display workspace")

    scale = SESSION.calibration_bundle.pixel_to_mm
    if scale == 0:
        raise RuntimeError("Workspace scale is zero; recompute workspace markers.")

    origin_px = np.array([workspace_def.origin_camera[0], workspace_def.origin_camera[1]]) / scale
    width_px = workspace_def.width_mm / scale
    height_px = workspace_def.height_mm / scale

    x_dir = np.array([workspace_def.x_axis_camera[0], workspace_def.x_axis_camera[1]])
    y_dir = np.array([workspace_def.y_axis_camera[0], workspace_def.y_axis_camera[1]])

    x_len = np.linalg.norm(x_dir)
    y_len = np.linalg.norm(y_dir)
    if x_len > 0:
        x_dir = x_dir / x_len
    if y_len > 0:
        y_dir = y_dir / y_len

    points = np.asarray(
        [
            origin_px,
            origin_px + width_px * x_dir,
            origin_px + width_px * x_dir + height_px * y_dir,
            origin_px + height_px * y_dir,
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
        return
    _draw_dotted_polyline(frame, corners_px)


def _plate_center_px(detections: dict) -> Optional[np.ndarray]:
    if not detections:
        return None
    centers = [det.center for det in detections.values()]
    if not centers:
        return None
    return np.mean(np.asarray(centers, dtype=np.float64), axis=0)


def draw_charuco_overlay(frame: np.ndarray, board_path: Path) -> Tuple[np.ndarray, str, str]:
    vis = frame.copy()
    if not board_path.exists():
        return vis, "ChArUco layout not found.", "ChArUco corners: --"

    try:
        board_spec = aruco_board.load_charuco_board_spec(board_path)
        board = board_spec.create_board()
        gray = cv2.cvtColor(vis, cv2.COLOR_BGR2GRAY)

        dictionary_id = getattr(cv2.aruco, board_spec.dictionary)
        dictionary = cv2.aruco.getPredefinedDictionary(dictionary_id)
        detector_params = cv2.aruco.DetectorParameters()
        detector_params.detectInvertedMarker = True

        detector = cv2.aruco.ArucoDetector(dictionary, detector_params)
        marker_corners, marker_ids, _ = detector.detectMarkers(gray)
        marker_count = 0 if marker_ids is None else len(marker_ids)

        if marker_ids is not None:
            vis = cv2.aruco.drawDetectedMarkers(vis, marker_corners, marker_ids)

        corner_count = 0
        if marker_ids is not None and len(marker_ids) > 0:
            charuco_params = cv2.aruco.CharucoParameters()
            charuco_detector = cv2.aruco.CharucoDetector(
                board, charuco_params, detector_params)
            charuco_corners, charuco_ids, _, _ = charuco_detector.detectBoard(gray)
            corner_count = 0 if charuco_ids is None else len(charuco_ids)
            if charuco_ids is not None:
                vis = cv2.aruco.drawDetectedCornersCharuco(vis, charuco_corners, charuco_ids)

        _draw_label(vis, f"Markers: {marker_count}", (10, 30), (255, 255, 255))
        _draw_label(vis, f"Corners: {corner_count}", (10, 55), (255, 255, 255))
        status = f"Detected {marker_count} markers and {corner_count} corners."
        metric = f"ChArUco corners: {corner_count}"
        return vis, status, metric
    except Exception as exc:
        _draw_label(vis, f"Error: {str(exc)[:50]}", (10, 30), (255, 0, 0))
        return vis, f"Detection error: {exc}", "ChArUco corners: --"


def draw_workspace_overlay(
    frame: np.ndarray,
    marker_ids: Tuple[int, int],
    marker_size_mm: float,
) -> Tuple[np.ndarray, str, str]:
    vis = frame.copy()
    try:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        detector_params = cv2.aruco.DetectorParameters()
        detector_params.detectInvertedMarker = True
        detector = cv2.aruco.ArucoDetector(dictionary, detector_params)
        marker_corners, marker_ids_detected, _ = detector.detectMarkers(gray)

        if marker_ids_detected is None or len(marker_ids_detected) == 0:
            raise RuntimeError("No markers detected. Ensure markers are visible and high-contrast.")

        detected_map = {}
        for corners, mid in zip(marker_corners, marker_ids_detected.flatten()):
            pts = corners.reshape((4, 2))
            center = tuple(np.mean(pts, axis=0).tolist())
            detected_map[int(mid)] = {
                "corners": pts,
                "center": center,
                "id": int(mid),
            }

        if marker_ids[0] not in detected_map or marker_ids[1] not in detected_map:
            found_ids = sorted(detected_map.keys())
            raise RuntimeError(
                f"Missing required markers {marker_ids}. Found: {found_ids}")

        marker_a_center = None
        marker_b_center = None
        colors = {marker_ids[0]: (0, 255, 0), marker_ids[1]: (0, 140, 255)}
        for mid, color in colors.items():
            if mid in detected_map:
                det = detected_map[mid]
                pts = det["corners"].astype(int)
                cv2.polylines(vis, [pts], isClosed=True, color=color, thickness=2)
                center = tuple(np.round(det["center"]).astype(int))
                cv2.circle(vis, center, 4, color, -1)
                cv2.putText(vis, f"ID {mid}", center, cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                if mid == marker_ids[0]:
                    marker_a_center = center
                elif mid == marker_ids[1]:
                    marker_b_center = center

        if marker_a_center is not None and marker_b_center is not None:
            marker_a_corners = detected_map[marker_ids[0]]["corners"]
            x_axis_px = marker_a_corners[1] - marker_a_corners[0]
            x_axis_norm = x_axis_px / np.linalg.norm(x_axis_px) if np.linalg.norm(x_axis_px) > 0 else np.array([1.0, 0.0])
            y_axis_norm = np.array([-x_axis_norm[1], x_axis_norm[0]])
            diagonal = np.array(marker_b_center) - np.array(marker_a_center)
            width_px = np.dot(diagonal, x_axis_norm)
            height_px = np.dot(diagonal, y_axis_norm)
            origin = np.array(marker_a_center, dtype=float)
            corner_tl = origin
            corner_tr = origin + width_px * x_axis_norm
            corner_br = origin + width_px * x_axis_norm + height_px * y_axis_norm
            corner_bl = origin + height_px * y_axis_norm
            rect_corners_float = np.array([corner_tl, corner_tr, corner_br, corner_bl], dtype=np.float64)
            _workspace_rect_buffer.append(rect_corners_float)
            if len(_workspace_rect_buffer) > _WORKSPACE_SMOOTH_FRAMES:
                _workspace_rect_buffer.pop(0)
            if _workspace_rect_buffer:
                rect_corners = np.round(np.mean(_workspace_rect_buffer, axis=0)).astype(np.int32)
            else:
                rect_corners = np.round(rect_corners_float).astype(np.int32)
            cv2.polylines(vis, [rect_corners], isClosed=True, color=(0, 255, 255), thickness=2)
            cv2.line(vis, tuple(rect_corners[0]), tuple(rect_corners[2]), (0, 255, 255), 1)

        status = "Workspace markers detected."
        metric = "Workspace: --"
        if SESSION.calibration_bundle is not None:
            try:
                det_a = type("MarkerDetection", (), {
                    "marker_id": marker_ids[0],
                    "corners": detected_map[marker_ids[0]]["corners"],
                    "center": detected_map[marker_ids[0]]["center"],
                })()
                det_b = type("MarkerDetection", (), {
                    "marker_id": marker_ids[1],
                    "corners": detected_map[marker_ids[1]]["corners"],
                    "center": detected_map[marker_ids[1]]["center"],
                })()
                detections = type("WorkspaceMarkers", (), {
                    "marker_a": det_a,
                    "marker_b": det_b,
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


def _default_transform() -> workspace.RigidTransform:
    return workspace.RigidTransform(rotation=np.eye(3), translation=np.zeros(3))


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
        calibration_result = plate.estimate_plate_calibration(frame, plate_layout)
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
        cv2.polylines(vis, [pts], isClosed=True, color=(255, 204, 0), thickness=2)
        center = tuple(np.round(det.center).astype(int))
        cv2.circle(vis, center, 4, (255, 204, 0), -1)
        cv2.putText(vis, f"ID {det.marker_id}", center, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 204, 0), 2)

    metric = f"Reprojection error: {calibration_result.reprojection_error_mm:.2f} mm"
    _draw_label(vis, metric, (10, 30), (255, 255, 255))
    status = f"Plate markers detected: {len(calibration_result.detections)}"
    center_px = _plate_center_px(calibration_result.detections)
    center_list = None
    if center_px is not None:
        center_list = [float(center_px[0]), float(center_px[1])]
    movement_px = None
    stable = len(calibration_result.detections) >= MIN_PLATE_MARKERS
    baseline_px = SESSION.robot_calibration_baseline_px
    if baseline_px is not None and center_px is not None:
        movement_px = float(np.linalg.norm(center_px - baseline_px))
        if movement_px > PLATE_MOVEMENT_THRESHOLD_PX:
            stable = False
            status = f"Plate moved ({movement_px:.1f}px); hold it steady."
    PLATE_STATUS.update(
        {
            "count": len(calibration_result.detections),
            "ok": len(calibration_result.detections) >= MIN_PLATE_MARKERS,
            "stable": stable,
            "center_px": center_list,
            "movement_px": movement_px,
            "timestamp": time.time(),
            "message": status,
        }
    )
    return vis, status, metric


__all__ = [
    "_apply_workspace_overlay",
    "_draw_label",
    "_plate_center_px",
    "draw_charuco_overlay",
    "draw_plate_overlay",
    "draw_workspace_overlay",
]
