"""Shared helpers for session state, calibration persistence, and robot posture."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from codroid_api import CodroidSettings
from codroid_api.robot_session import RobotPosture, RobotSession

from .constants import (
    DEFAULT_BUNDLE_PATH,
    DEFAULT_ROBOT_CALIBRATION_PATH,
    DEFAULT_WORKSPACE_PATH,
    DROP_TARGET_PATH,
    FLANGE_BUTTON_PORT,
    FLANGE_BUTTON_PORTS,
    PLATE_MOVEMENT_THRESHOLD_PX,
    PLATE_TEST_CAPTURE_DIR,
    PLATE_VISIBILITY_MAX_AGE_S,
    ROBOT_COORDINATE_ID,
    ROBOT_POSTURE_MAX_AGE_S,
)
from .state import (
    ROBOT_CALIBRATION_GUIDANCE,
    SESSION,
    SETTINGS,
    _commit_settings,
    PLATE_STATUS,
)

LOGGER = logging.getLogger(__name__)


def _record_pick_cycle(success: bool, start_time: Optional[float]) -> None:
    """Track pick/place performance metrics."""
    SESSION.pick_attempts += 1
    if success:
        SESSION.pick_successes += 1
    else:
        SESSION.pick_failures += 1

    if start_time is None:
        return

    duration = max(time.monotonic() - start_time, 0.0)
    SESSION.last_cycle_s = duration
    SESSION.cycle_times.append(duration)


def _set_robot_guidance(
    message: str,
    *,
    step: Optional[int] = None,
    status: Optional[str] = None,
    verification_ready: Optional[bool] = None,
) -> None:
    ROBOT_CALIBRATION_GUIDANCE["message"] = message
    if step is not None:
        ROBOT_CALIBRATION_GUIDANCE["step"] = step
    if status is not None:
        ROBOT_CALIBRATION_GUIDANCE["status"] = status
    if verification_ready is not None:
        ROBOT_CALIBRATION_GUIDANCE["verification_ready"] = verification_ready
    ROBOT_CALIBRATION_GUIDANCE["timestamp"] = time.time()


def _get_robot_session() -> RobotSession:
    if SESSION.robot_session is None:
        SESSION.robot_session = RobotSession(
            CodroidSettings(),
            flange_button_port=FLANGE_BUTTON_PORT,
            flange_button_ports=FLANGE_BUTTON_PORTS,
        )
    return SESSION.robot_session


def _load_robot_calibration(path: Path = DEFAULT_ROBOT_CALIBRATION_PATH) -> None:
    if not path.exists():
        return
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        LOGGER.warning("Failed to parse robot calibration data: %s", exc)
        return
    points = data.get("points", [])
    if isinstance(points, list):
        SESSION.robot_calibration_points = points
    min_z = data.get("min_z")
    if isinstance(min_z, (int, float)):
        SESSION.robot_calibration_min_z = float(min_z)
    elif SESSION.robot_calibration_points:
        SESSION.robot_calibration_min_z = min(
            float(pt.get("z", 0.0)) for pt in SESSION.robot_calibration_points
        )
    plate_reference = data.get("plate_reference_homography")
    if isinstance(plate_reference, list):
        try:
            SESSION.plate_reference_homography = np.asarray(
                plate_reference, dtype=np.float64
            )
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("Failed to load plate reference homography: %s", exc)
    LOGGER.info("Loaded robot calibration data from %s", path)
    _commit_settings(robot_calibration_path=str(path))


def _save_robot_calibration(
    points: list[dict],
    path: Path = DEFAULT_ROBOT_CALIBRATION_PATH,
) -> None:
    min_z = None
    if points:
        min_z = min(float(pt.get("z", 0.0)) for pt in points)
    payload = {
        "points": points,
        "min_z": min_z,
        "timestamp": time.time(),
        "coordinate_id": ROBOT_COORDINATE_ID,
    }
    if SESSION.plate_reference_homography is not None:
        payload["plate_reference_homography"] = (
            np.asarray(SESSION.plate_reference_homography, dtype=np.float64).tolist()
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))
    _commit_settings(robot_calibration_path=str(path))


def _ensure_posture_axes(posture: dict, fallback: RobotPosture) -> dict:
    normalized = dict(posture)
    for axis in ("a", "b", "c"):
        if axis not in normalized:
            normalized[axis] = float(getattr(fallback, axis, 0.0))
    return normalized


def _plate_test_capture_path() -> Path:
    PLATE_TEST_CAPTURE_DIR.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    return PLATE_TEST_CAPTURE_DIR / f"plate_test_{stamp}.json"


def _save_plate_test_capture(payload: dict) -> Path:
    path = _plate_test_capture_path()
    payload["capture_path"] = str(path)
    path.write_text(json.dumps(payload, indent=2))
    SESSION.last_plate_test_capture = str(path)
    return path


def _posture_to_dict(posture: RobotPosture) -> dict:
    return {
        "x": float(posture.x),
        "y": float(posture.y),
        "z": float(posture.z),
        "a": float(posture.a),
        "b": float(posture.b),
        "c": float(posture.c),
        "timestamp": float(posture.timestamp),
        "age_s": float(posture.age_seconds()),
    }


def _analyze_plate_test_capture(capture: dict) -> str:
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


def _save_drop_target(
    drop_target_mm: Optional[Tuple[float, float]],
    orientation: Optional[Tuple[float, float, float]] = None,
    robot_xyz: Optional[Tuple[float, float, float]] = None,
) -> None:
    DROP_TARGET_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not drop_target_mm:
        if DROP_TARGET_PATH.exists():
            DROP_TARGET_PATH.unlink()
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
    DROP_TARGET_PATH.write_text(json.dumps(payload, indent=2))
    _commit_settings(
        drop_target=drop_target_mm,
        workspace_path=str(DEFAULT_WORKSPACE_PATH),
        calibration_bundle_path=str(DEFAULT_BUNDLE_PATH),
    )


def _load_drop_target() -> None:
    if DROP_TARGET_PATH.exists():
        try:
            data = json.loads(DROP_TARGET_PATH.read_text())
            x = float(data["x"])
            y = float(data["y"])
            SESSION.drop_target_mm = (x, y)
            if all(key in data for key in ("a", "b", "c")):
                SESSION.drop_target_orientation = (
                    float(data["a"]),
                    float(data["b"]),
                    float(data["c"]),
                )
            if all(key in data for key in ("robot_x", "robot_y", "robot_z")):
                SESSION.drop_target_robot = (
                    float(data["robot_x"]),
                    float(data["robot_y"]),
                    float(data["robot_z"]),
                )
            return
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("Failed to load drop target: %s", exc)
    if SETTINGS.drop_target:
        SESSION.drop_target_mm = SETTINGS.drop_target


def _update_robot_calibration_meta(robot_points: list[dict]) -> None:
    SESSION.robot_calibration_points = list(robot_points)
    if robot_points:
        SESSION.robot_calibration_min_z = min(
            float(pt.get("z", 0.0)) for pt in robot_points
        )
    else:
        SESSION.robot_calibration_min_z = None
    _save_robot_calibration(robot_points)


def _current_coordinate_id(session: RobotSession) -> Optional[int]:
    status = session.robot_status_snapshot()
    if not isinstance(status, dict):
        return None
    coord_id = status.get("CoordinateId")
    if coord_id is None:
        return None
    try:
        return int(coord_id)
    except (TypeError, ValueError):
        return None


def _set_plate_test_state(
    message: str,
    *,
    running: bool = True,
    label: Optional[str] = None,
) -> None:
    state = {"running": running, "message": message, "timestamp": time.time()}
    if label is not None:
        state["label"] = label
    SESSION.plate_test_state = state
    SESSION.last_action = message


def _robot_posture_ready(
    max_age_s: float = ROBOT_POSTURE_MAX_AGE_S,
) -> Tuple[bool, str]:
    session = _get_robot_session()
    if not session.posture_seen():
        return False, "Robot posture stream not available yet."
    age = session.position_snapshot().age_seconds()
    if age > max_age_s:
        return False, f"Robot posture stale ({age:.1f}s old)."
    return True, "Robot posture stream healthy."


def _plate_is_stable(
    max_age_s: float = PLATE_VISIBILITY_MAX_AGE_S,
    require_baseline: bool = False,
    allow_lost: bool = False,
) -> Tuple[bool, str]:
    if require_baseline and SESSION.robot_calibration_baseline_px is None:
        return False, "Calibration plate baseline not set."
    if not PLATE_STATUS.get("ok"):
        if allow_lost and SESSION.robot_calibration_baseline_px is not None:
            return True, "Plate markers not visible; waiting for re-acquire."
        return False, "Calibration plate markers are not visible."
    movement_px = PLATE_STATUS.get("movement_px")
    if movement_px is not None and movement_px > PLATE_MOVEMENT_THRESHOLD_PX:
        return False, (
            f"Calibration plate moved ({movement_px:.1f}px)."
        )
    if PLATE_STATUS.get("stable") is False:
        return False, PLATE_STATUS.get("message", "Calibration plate not stable.")
    return True, PLATE_STATUS.get("message", "Plate markers stable.")


def _safe_int(value: Optional[float], fallback: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return fallback


def _safe_float(value: Optional[float], fallback: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return fallback


def _safe_path(value: Optional[str], fallback: Path) -> Path:
    if value is None:
        return fallback
    candidate = str(value).strip()
    return Path(candidate) if candidate else fallback
