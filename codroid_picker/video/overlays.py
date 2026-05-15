"""Overlay mode definitions."""

from __future__ import annotations

from enum import StrEnum
from typing import Any

import cv2
import numpy as np


class OverlayMode(StrEnum):
    AUTO = "auto"
    CALIBRATION = "calibration"
    WORKSPACE = "workspace"
    PLATE = "plate"
    DETECTION = "detection"
    AUTO_PLAN = "auto_plan"
    NONE = "none"


def render_overlay(frame: np.ndarray, *, mode: str, state: dict[str, Any]) -> np.ndarray:
    """Draw lightweight operator overlays on a BGR frame."""
    normalized = mode.replace("-", "_")
    if normalized == OverlayMode.NONE:
        return frame
    output = frame.copy()
    height, width = output.shape[:2]
    _label(output, f"Overlay: {normalized}", 16, 28, (230, 230, 230))
    if normalized in {OverlayMode.AUTO, OverlayMode.DETECTION, OverlayMode.AUTO_PLAN}:
        _draw_detections(output, state.get("detection", {}).get("detections", []))
    if normalized in {OverlayMode.AUTO, OverlayMode.WORKSPACE, OverlayMode.CALIBRATION}:
        _draw_workspace_hint(output, width, height, state.get("calibration", {}))
    if normalized in {OverlayMode.AUTO, OverlayMode.PLATE, OverlayMode.CALIBRATION}:
        _draw_plate_hint(output, width, height, state.get("calibration", {}).get("plate", {}))
    if normalized in {OverlayMode.AUTO, OverlayMode.AUTO_PLAN}:
        _draw_auto_plan(output, state.get("auto_pick", {}).get("plan_snapshot", {}))
        _draw_drop_target(output, state.get("auto_pick", {}).get("drop_target"))
    if normalized in {OverlayMode.AUTO, OverlayMode.WORKSPACE, OverlayMode.AUTO_PLAN}:
        _draw_robot_status(output, state.get("robot", {}))
    if state.get("safety", {}).get("blocked"):
        _draw_safety_block(output, width, height, state.get("safety", {}))
    return output


def _label(frame: np.ndarray, text: str, x: int, y: int, color: tuple[int, int, int]) -> None:
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)


def _draw_detections(frame: np.ndarray, detections: Any) -> None:
    if not isinstance(detections, list):
        return
    for item in detections:
        if not isinstance(item, dict):
            continue
        bbox = item.get("bbox")
        if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
            x1, y1, x2, y2 = [int(float(value)) for value in bbox[:4]]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 180, 255), 2)
            _label(frame, f"{item.get('label', 'object')} {float(item.get('confidence', 0.0)):.2f}", x1, max(18, y1 - 6), (0, 220, 255))
            continue
        center = item.get("center")
        if isinstance(center, (list, tuple)) and len(center) >= 2:
            x_val, y_val = int(float(center[0])), int(float(center[1]))
            cv2.circle(frame, (x_val, y_val), 12, (0, 180, 255), 2)


def _draw_workspace_hint(frame: np.ndarray, width: int, height: int, calibration: dict[str, Any]) -> None:
    color = (80, 220, 120) if calibration.get("workspace_status") in {"loaded", "ready"} else (80, 160, 220)
    margin_x = max(16, width // 20)
    margin_y = max(16, height // 16)
    cv2.rectangle(frame, (margin_x, margin_y), (width - margin_x, height - margin_y), color, 2)
    _label(frame, f"Workspace: {calibration.get('workspace_status', 'missing')}", margin_x + 8, margin_y + 24, color)


def _draw_plate_hint(frame: np.ndarray, width: int, height: int, plate: dict[str, Any]) -> None:
    count = int(plate.get("count") or 0)
    color = (255, 210, 80) if plate.get("ok") else (160, 160, 160)
    center = (width - 92, height - 72)
    cv2.drawMarker(frame, center, color, markerType=cv2.MARKER_TILTED_CROSS, markerSize=34, thickness=2)
    _label(frame, f"Plate markers: {count}", max(16, width - 210), height - 28, color)


def _draw_auto_plan(frame: np.ndarray, plan: dict[str, Any]) -> None:
    if not plan.get("available"):
        _label(frame, "Auto plan unavailable", 16, 56, (80, 180, 255))
        return
    selected = plan.get("selected") if isinstance(plan.get("selected"), dict) else {}
    text = f"Plan: {plan.get('mode', '--')} candidates={plan.get('candidate_count', 0)}"
    _label(frame, text, 16, 56, (120, 220, 255))
    xy = selected.get("coord5_xy") if isinstance(selected, dict) else None
    if isinstance(xy, (list, tuple)) and len(xy) >= 2:
        _label(frame, f"Target Coord5: {float(xy[0]):.1f}, {float(xy[1]):.1f}", 16, 82, (120, 220, 255))
        _draw_coord_marker(frame, xy, (120, 220, 255), "T")


def _draw_drop_target(frame: np.ndarray, drop_target: Any) -> None:
    if not isinstance(drop_target, dict):
        return
    xy = None
    if "x" in drop_target and "y" in drop_target:
        xy = (drop_target.get("x"), drop_target.get("y"))
    elif "robot_x" in drop_target and "robot_y" in drop_target:
        xy = (drop_target.get("robot_x"), drop_target.get("robot_y"))
    if xy is None:
        return
    _draw_coord_marker(frame, xy, (80, 255, 160), "D")
    _label(frame, "Drop target", 16, 108, (80, 255, 160))


def _draw_robot_status(frame: np.ndarray, robot: dict[str, Any]) -> None:
    posture = robot.get("posture")
    if isinstance(posture, dict):
        text = (
            f"Robot Coord{robot.get('coordinate_id', '--')} "
            f"X={float(posture.get('x', 0.0)):.1f} Y={float(posture.get('y', 0.0)):.1f} "
            f"Z={float(posture.get('z', 0.0)):.1f}"
        )
    else:
        text = f"Robot: {robot.get('status', 'offline')}"
    if robot.get("stale"):
        text = f"{text} stale"
    _label(frame, text, 16, frame.shape[0] - 16, (190, 220, 255))


def _draw_safety_block(frame: np.ndarray, width: int, height: int, safety: dict[str, Any]) -> None:
    cv2.rectangle(frame, (4, 4), (width - 5, height - 5), (0, 0, 255), 4)
    match_count = len(safety.get("matches") or [])
    _label(frame, f"SAFETY BLOCKED matches={match_count}", 16, min(height - 44, 132), (80, 80, 255))


def _draw_coord_marker(frame: np.ndarray, xy: Any, color: tuple[int, int, int], label: str) -> None:
    if not isinstance(xy, (list, tuple)) or len(xy) < 2:
        return
    height, width = frame.shape[:2]
    try:
        x_val = int(float(xy[0]))
        y_val = int(float(xy[1]))
    except (TypeError, ValueError):
        return
    x_val = max(12, min(width - 12, x_val))
    y_val = max(12, min(height - 12, y_val))
    cv2.drawMarker(frame, (x_val, y_val), color, markerType=cv2.MARKER_CROSS, markerSize=28, thickness=2)
    cv2.circle(frame, (x_val, y_val), 12, color, 2)
    _label(frame, label, x_val + 10, max(18, y_val - 10), color)
