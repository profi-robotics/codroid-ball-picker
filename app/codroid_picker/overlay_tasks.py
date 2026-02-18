"""
Overlay task factory functions.

This module provides factory functions for creating overlay processing tasks
for different calibration and application modes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional, Tuple

import numpy as np

from app.codroid_picker.streaming import OverlayResult, OverlayTask
from app.codroid_picker.overlays import (
    _apply_workspace_overlay,
    draw_charuco_overlay,
    draw_workspace_overlay,
    draw_plate_overlay,
    draw_detection_overlay,
    draw_drop_marker_overlay,
    draw_auto_plan_overlay,
)


def _run_charuco_task(board_path: Path) -> Callable[[np.ndarray], OverlayResult]:
    """Create a task runner for ChArUco board detection overlay."""

    def _runner(frame: np.ndarray) -> OverlayResult:
        vis, status, metric = draw_charuco_overlay(frame, board_path)
        _apply_workspace_overlay(vis)
        return OverlayResult(vis, status, metric)

    return _runner


def _run_workspace_task(
    marker_ids: Tuple[int, int], marker_size_mm: float
) -> Callable[[np.ndarray], OverlayResult]:
    """Create a task runner for workspace marker detection overlay."""

    def _runner(frame: np.ndarray) -> OverlayResult:
        vis, status, metric = draw_workspace_overlay(
            frame, marker_ids, marker_size_mm)
        _apply_workspace_overlay(vis)
        return OverlayResult(vis, status, metric)

    return _runner


def _run_plate_task(layout_path: Path) -> Callable[[np.ndarray], OverlayResult]:
    """Create a task runner for calibration plate detection overlay."""

    def _runner(frame: np.ndarray) -> OverlayResult:
        vis, status, metric = draw_plate_overlay(frame, layout_path)
        _apply_workspace_overlay(vis)
        return OverlayResult(vis, status, metric)

    return _runner


def _run_detection_task() -> Callable[[np.ndarray], OverlayResult]:
    """Create a task runner for YOLO detection overlay."""

    def _runner(frame: np.ndarray) -> OverlayResult:
        vis, status, metric = draw_detection_overlay(frame)
        _apply_workspace_overlay(vis)
        return OverlayResult(vis, status, metric)

    return _runner


def _run_drop_marker_task(marker_id: int) -> Callable[[np.ndarray], OverlayResult]:
    """Create a task runner for drop marker detection overlay."""

    def _runner(frame: np.ndarray) -> OverlayResult:
        vis, status, metric = draw_drop_marker_overlay(frame, marker_id)
        _apply_workspace_overlay(vis)
        return OverlayResult(vis, status, metric)

    return _runner


def _run_auto_plan_task() -> Callable[[np.ndarray], OverlayResult]:
    """Create a task runner for auto planning intent overlay."""

    def _runner(frame: np.ndarray) -> OverlayResult:
        vis, status, metric = draw_auto_plan_overlay(frame)
        _apply_workspace_overlay(vis)
        return OverlayResult(vis, status, metric)

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
