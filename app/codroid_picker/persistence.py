"""Persistence utilities for calibration UI state and captures.

Handles saving/loading drop targets, plate test captures, robot calibration
metadata, and persisted session state restoration.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Tuple

import numpy as np

from vision import calibration, workspace
from app.codroid_picker.app_instance import registry
from app.settings import _commit_settings
from app.codroid_picker.procedures import _load_robot_calibration, _save_robot_calibration
from app.codroid_picker.models import activate_model, ensure_model_available

if TYPE_CHECKING:
    pass

LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pick cycle tracking
# ---------------------------------------------------------------------------


def _record_pick_cycle(success: bool, start_time: Optional[float]) -> None:
    """Track pick/place performance metrics."""
    session = registry().session
    session.pick_attempts += 1
    if success:
        session.pick_successes += 1
    else:
        session.pick_failures += 1

    if start_time is None:
        return

    duration = max(time.monotonic() - start_time, 0.0)
    session.last_cycle_s = duration
    session.cycle_times.append(duration)


# ---------------------------------------------------------------------------
# Plate test capture utilities
# ---------------------------------------------------------------------------


def _plate_test_capture_path() -> Path:
    """Generate timestamped path for plate test capture."""
    settings = registry().settings
    settings.plate_test_capture_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    return settings.plate_test_capture_dir / f"plate_test_{stamp}.json"


def _save_plate_test_capture(payload: dict) -> Path:
    """Save plate test capture to timestamped JSON file."""
    path = _plate_test_capture_path()
    payload["capture_path"] = str(path)
    path.write_text(json.dumps(payload, indent=2))
    registry().session.last_plate_test_capture = str(path)
    return path


def _analyze_plate_test_capture(capture: dict) -> str:
    """Analyze plate test capture and return error summary."""
    moves = capture.get("moves", [])
    if not moves:
        return "No motion samples captured."
    distances = []
    lines = []
    for move in moves:
        cmd = move.get("command_coord5")
        act = move.get("posture_coord5")
        label = move.get("label", "move")
        if not cmd or not act:
            continue
        dx = act["x"] - cmd["x"]
        dy = act["y"] - cmd["y"]
        dz = act["z"] - cmd["z"]
        dist = float(np.linalg.norm([dx, dy, dz]))
        distances.append(dist)
        lines.append(
            f"{label}: d={dist:.2f} (dx={dx:.2f}, dy={dy:.2f}, dz={dz:.2f})"
        )
    if not distances:
        return "No usable move comparisons captured."
    mean_dist = float(np.mean(distances))
    max_dist = float(np.max(distances))
    summary = [
        f"Move error mean: {mean_dist:.2f} mm",
        f"Move error max: {max_dist:.2f} mm",
    ]
    capture_path = capture.get("capture_path")
    if capture_path:
        summary.append(f"Capture saved: {capture_path}")
    return "\n".join(summary + lines[:12])


# ---------------------------------------------------------------------------
# Drop target persistence
# ---------------------------------------------------------------------------


def _save_drop_target(
    drop_target_mm: Optional[Tuple[float, float]],
    orientation: Optional[Tuple[float, float, float]] = None,
    robot_xyz: Optional[Tuple[float, float, float]] = None,
    coordinate_id: Optional[int] = None,
) -> None:
    """Save drop target coordinates to JSON file."""
    drop_target_path = registry().drop_target_path
    drop_target_path.parent.mkdir(parents=True, exist_ok=True)
    if not drop_target_mm:
        if drop_target_path.exists():
            drop_target_path.unlink()
        _commit_settings(drop_target=None)
        return
    payload = {"x": float(drop_target_mm[0]), "y": float(drop_target_mm[1])}
    if orientation is not None:
        payload.update(
            {
                "a": float(orientation[0]),
                "b": float(orientation[1]),
                "c": float(orientation[2]),
            }
        )
    if robot_xyz is not None:
        payload.update(
            {
                "robot_x": float(robot_xyz[0]),
                "robot_y": float(robot_xyz[1]),
                "robot_z": float(robot_xyz[2]),
            }
        )
    if coordinate_id is not None:
        payload["coord_id"] = int(coordinate_id)
    drop_target_path.write_text(json.dumps(payload, indent=2))
    _commit_settings(
        drop_target=drop_target_mm,
        workspace_path=str(registry().workspace_path),
        calibration_bundle_path=str(registry().bundle_path),
    )


def _load_drop_target() -> None:
    """Load drop target from JSON file or settings fallback."""
    drop_target_path = registry().drop_target_path
    session = registry().session
    settings = registry().settings
    if drop_target_path.exists():
        try:
            data = json.loads(drop_target_path.read_text())
            x = float(data["x"])
            y = float(data["y"])
            session.drop_target_mm = (x, y)
            if all(key in data for key in ("a", "b", "c")):
                session.drop_target_orientation = (
                    float(data["a"]),
                    float(data["b"]),
                    float(data["c"]),
                )
            if all(key in data for key in ("robot_x", "robot_y", "robot_z")):
                session.drop_target_robot = (
                    float(data["robot_x"]),
                    float(data["robot_y"]),
                    float(data["robot_z"]),
                )
            if "coord_id" in data:
                try:
                    session.drop_target_coord_id = int(data["coord_id"])
                except (TypeError, ValueError):
                    session.drop_target_coord_id = None
            return
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("Failed to load drop target: %s", exc)
    if settings.drop_target:
        session.drop_target_mm = settings.drop_target


# ---------------------------------------------------------------------------
# Robot calibration metadata
# ---------------------------------------------------------------------------


def _update_robot_calibration_meta(robot_points: list[dict]) -> None:
    """Update robot calibration metadata in session and save to disk."""
    session = registry().session
    session.robot_calibration_points = list(robot_points)
    if robot_points:
        session.robot_calibration_min_z = min(
            float(pt.get("z", 0.0)) for pt in robot_points
        )
    else:
        session.robot_calibration_min_z = None
    session.plate_to_robot_transform = None
    session.plate_origin_mm = None
    _save_robot_calibration(robot_points)


# ---------------------------------------------------------------------------
# Plate test state management
# ---------------------------------------------------------------------------


def _set_plate_test_state(
    message: str,
    *,
    running: bool = True,
    label: Optional[str] = None,
) -> None:
    """Update plate test state in session."""
    session = registry().session
    state_dict = {"running": running,
                  "message": message, "timestamp": time.time()}
    if label is not None:
        state_dict["label"] = label
    session.plate_test_state = state_dict
    session.last_action = message


# ---------------------------------------------------------------------------
# Plate stability check
# ---------------------------------------------------------------------------


def _plate_is_stable(
    max_age_s: float = None,
    require_baseline: bool = False,
    allow_lost: bool = False,
) -> Tuple[bool, str]:
    """Check if calibration plate is stable and visible.

    Returns:
        Tuple of (is_stable, message).
    """
    from . import state

    session = registry().session
    settings = registry().settings

    if max_age_s is None:
        max_age_s = settings.plate_visibility_max_age_s

    if require_baseline and session.robot_calibration_baseline_px is None:
        return False, "Calibration plate baseline not set."

    # Get plate status from state module
    plate_status = state.get_plate_status_snapshot()

    if not plate_status.get("ok"):
        if allow_lost and session.robot_calibration_baseline_px is not None:
            return True, "Plate markers not visible; waiting for re-acquire."
        return False, "Calibration plate markers are not visible."
    movement_px = plate_status.get("movement_px")
    if (
        movement_px is not None
        and movement_px > settings.plate_movement_threshold_px
    ):
        return False, f"Calibration plate moved ({movement_px:.1f}px)."
    if plate_status.get("stable") is False:
        return False, plate_status.get("message", "Calibration plate not stable.")
    return True, plate_status.get("message", "Plate markers stable.")


# ---------------------------------------------------------------------------
# Session state restoration
# ---------------------------------------------------------------------------


def _restore_persisted_state() -> None:
    """Restore persisted session state on startup."""
    session = registry().session
    settings = registry().settings
    bundle_path = registry().bundle_path
    workspace_path = registry().workspace_path
    robot_calibration_path = registry().robot_calibration_path

    if session.calibration_bundle is None and bundle_path.exists():
        try:
            session.calibration_bundle = calibration.load_calibration(
                bundle_path
            )
            LOGGER.info("Loaded calibration bundle from %s", bundle_path)
        except Exception as exc:
            LOGGER.warning(
                "Failed to load calibration bundle from %s: %s",
                bundle_path,
                exc,
            )
    if session.workspace_definition is None and workspace_path.exists():
        try:
            session.workspace_definition = workspace.load_workspace(
                workspace_path
            )
            LOGGER.info("Loaded workspace definition from %s", workspace_path)
        except Exception as exc:
            LOGGER.warning(
                "Failed to load workspace definition from %s: %s",
                workspace_path,
                exc,
            )
    _load_drop_target()
    
    # Try to restore last model, or download default if none available
    model_to_load = None
    if settings.last_model:
        model_path = Path(settings.last_model)
        if model_path.exists():
            model_to_load = model_path
        else:
            LOGGER.warning("Last model not found: %s", settings.last_model)
    
    # If no valid model configured, ensure we have at least one available
    if model_to_load is None:
        model_to_load = ensure_model_available(registry().model_dir)
    
    if model_to_load:
        try:
            activate_model(model_to_load)
            LOGGER.info("Activated model: %s", model_to_load.name)
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("Failed to activate model: %s", exc)
    
    if robot_calibration_path.exists():
        _load_robot_calibration(robot_calibration_path)
