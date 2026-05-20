"""
Overlay task factory functions.

This module provides factory functions for creating overlay processing tasks
for different calibration and application modes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional, Tuple

import cv2
import numpy as np

from app.codroid_picker.session import SESSION
from app.codroid_picker.streaming import OverlayResult, OverlayTask
from app.codroid_picker.auto_plan_state import get_auto_plan_snapshot
from app.codroid_picker.overlays import (
    _apply_workspace_overlay,
    _workspace_corners_px,
    build_charuco_overlay,
    build_plate_overlay,
    build_workspace_overlay,
    draw_detection_overlay,
    draw_drop_marker_overlay,
    draw_auto_plan_overlay,
)
from app.codroid_picker.overlays.auto_plan import build_auto_plan_primitives


def _css_color(bgr: tuple[int, int, int]) -> str:
    """Return a CSS color from an OpenCV BGR tuple."""
    b, g, r = bgr
    return f"rgb({int(r)}, {int(g)}, {int(b)})"


def _diff_primitives(frame: np.ndarray, vis: np.ndarray) -> list[dict]:
    """Convert rendered overlay differences into browser-drawn polygons."""
    if frame.shape != vis.shape:
        return []
    mask = np.any(vis != frame, axis=2).astype(np.uint8) * 255
    contours, _hierarchy = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    primitives: list[dict] = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 4.0:
            continue
        epsilon = max(1.0, 0.01 * cv2.arcLength(contour, True))
        approx = cv2.approxPolyDP(contour, epsilon, True).reshape(-1, 2)
        if len(approx) < 3:
            x, y, w, h = cv2.boundingRect(contour)
            approx = np.array(
                [[x, y], [x + w, y], [x + w, y + h], [x, y + h]],
                dtype=np.int32,
            )
        sample = approx[0]
        color = tuple(int(v) for v in vis[int(sample[1]), int(sample[0])])
        primitives.append(
            {
                "type": "polygon",
                "points": [[int(x), int(y)] for x, y in approx.tolist()],
                "color": _css_color(color),
                "fill": _css_color(color),
                "alpha": 0.85,
            }
        )
    return primitives


def _detection_primitives() -> list[dict]:
    """Create browser primitives from the current detection list."""
    primitives: list[dict] = []
    for detection in SESSION.detections or []:
        box = detection.get("bbox") or detection.get("box")
        if box is None:
            continue
        x1, y1, x2, y2 = [int(round(float(v))) for v in box]
        label = detection.get("label") or detection.get("class_name") or "object"
        confidence = detection.get("confidence")
        text = str(label)
        if isinstance(confidence, (int, float)):
            text = f"{text}: {confidence:.2f}"
        primitives.append(
            {
                "type": "box",
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "label": text,
                "confidence": confidence,
                "color": "rgb(0, 255, 0)",
            }
        )
    return primitives


def _workspace_primitives() -> list[dict]:
    """Create browser primitives for the workspace outline."""
    if SESSION.workspace_definition is None or SESSION.calibration_bundle is None:
        return []
    try:
        corners = _workspace_corners_px(SESSION.workspace_definition)
    except Exception:
        return []
    primitives: list[dict] = []
    color = _css_color((0, 215, 255))
    for idx in range(len(corners)):
        start = corners[idx]
        end = corners[(idx + 1) % len(corners)]
        primitives.append(
            {
                "type": "line",
                "from": {
                    "x": int(round(float(start[0]))),
                    "y": int(round(float(start[1]))),
                },
                "to": {
                    "x": int(round(float(end[0]))),
                    "y": int(round(float(end[1]))),
                },
                "color": color,
                "lineWidth": 2,
                "dashed": True,
            }
        )
    return primitives


def _auto_plan_has_targets() -> bool:
    """Return whether the current auto-plan snapshot contains target geometry."""
    snapshot = get_auto_plan_snapshot()
    if not snapshot.get("available"):
        return False
    ranked = snapshot.get("ranked_candidates") or []
    selected = snapshot.get("selected_candidate") or {}
    return bool(ranked or selected)


def _result_from_render(
    frame: np.ndarray,
    vis: np.ndarray,
    status: str,
    metric: str,
    *,
    primitives: Optional[list[dict]] = None,
    has_targets: Optional[bool] = None,
) -> OverlayResult:
    """Build an OverlayResult with browser primitives."""
    if primitives is None:
        primitives = _diff_primitives(frame, vis)
    if has_targets is None:
        has_targets = bool(primitives)
    return OverlayResult(
        vis,
        status,
        metric,
        primitives=primitives,
        has_targets=has_targets,
    )


def _run_charuco_task(board_path: Path) -> Callable[[np.ndarray], OverlayResult]:
    """Create a task runner for ChArUco board detection overlay."""

    def _runner(frame: np.ndarray) -> OverlayResult:
        vis, status, metric, primitives = build_charuco_overlay(
            frame,
            board_path,
        )
        _apply_workspace_overlay(vis)
        primitives = primitives + _workspace_primitives()
        return _result_from_render(
            frame,
            vis,
            status,
            metric,
            primitives=primitives,
            has_targets=bool(primitives),
        )

    return _runner


def _run_workspace_task(
    marker_ids: Tuple[int, int], marker_size_mm: float
) -> Callable[[np.ndarray], OverlayResult]:
    """Create a task runner for workspace marker detection overlay."""

    def _runner(frame: np.ndarray) -> OverlayResult:
        vis, status, metric, primitives = build_workspace_overlay(
            frame,
            marker_ids,
            marker_size_mm,
        )
        _apply_workspace_overlay(vis)
        primitives = primitives + _workspace_primitives()
        return _result_from_render(
            frame,
            vis,
            status,
            metric,
            primitives=primitives,
            has_targets="detected" in status.lower() or "defined" in status.lower(),
        )

    return _runner


def _run_plate_task(layout_path: Path) -> Callable[[np.ndarray], OverlayResult]:
    """Create a task runner for calibration plate detection overlay."""

    def _runner(frame: np.ndarray) -> OverlayResult:
        vis, status, metric, primitives = build_plate_overlay(frame, layout_path)
        _apply_workspace_overlay(vis)
        primitives = primitives + _workspace_primitives()
        return _result_from_render(
            frame,
            vis,
            status,
            metric,
            primitives=primitives,
            has_targets="detected" in status.lower(),
        )

    return _runner


def _run_detection_task() -> Callable[[np.ndarray], OverlayResult]:
    """Create a task runner for YOLO detection overlay."""

    def _runner(frame: np.ndarray) -> OverlayResult:
        vis, status, metric = draw_detection_overlay(frame)
        _apply_workspace_overlay(vis)
        primitives = _detection_primitives()
        return _result_from_render(
            frame,
            vis,
            status,
            metric,
            primitives=primitives,
            has_targets=bool(primitives),
        )

    return _runner


def _run_drop_marker_task(marker_id: int) -> Callable[[np.ndarray], OverlayResult]:
    """Create a task runner for drop marker detection overlay."""

    def _runner(frame: np.ndarray) -> OverlayResult:
        vis, status, metric = draw_drop_marker_overlay(frame, marker_id)
        _apply_workspace_overlay(vis)
        return _result_from_render(
            frame,
            vis,
            status,
            metric,
            has_targets="detected" in status.lower(),
        )

    return _runner


def _run_auto_plan_task() -> Callable[[np.ndarray], OverlayResult]:
    """Create a task runner for auto planning intent overlay."""

    def _runner(frame: np.ndarray) -> OverlayResult:
        vis, status, metric = draw_auto_plan_overlay(frame)
        _apply_workspace_overlay(vis)
        primitives = _workspace_primitives() + build_auto_plan_primitives(frame.shape)
        return _result_from_render(
            frame,
            vis,
            status,
            metric,
            primitives=primitives,
            has_targets=_auto_plan_has_targets(),
        )

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
    """Select the appropriate overlay task based on current UI state."""
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
            # Avoid running YOLO overlay inference on every incoming camera frame
            # in default "auto" mode; this is too expensive on Jetson Nano.
            # Users can still explicitly enable detection overlay from the UI.
            return None
    return None
