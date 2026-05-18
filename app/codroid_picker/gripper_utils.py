"""Gripper synchronization and status utilities.

Provides functions to track gripper sync state, poll grip status,
and refresh gripper constants from settings.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Optional

from app.codroid_picker.app_instance import registry
from app.codroid_picker.robot_helpers import (
    resolve_gripper_limits,
    record_gripper_sync,
    is_recently_synced,
    safe_int,
)

LOGGER = logging.getLogger(__name__)

# Gripper profile constants (refreshed via refresh_gripper_constants)
GRIPPER_PROFILE = None
GRIPPER_FORCE_MIN: float = 0.0
GRIPPER_FORCE_MAX: float = 100.0
GRIPPER_FORCE_DEFAULT: float = 50.0
GRIPPER_SPEED_MIN: float = 0.0
GRIPPER_SPEED_DEFAULT: float = 50.0
GRIPPER_CALIBRATION_WIDTH_MM: float = 0.0
GRIPPER_OPEN_WIDTH_DEFAULT: float = 100.0
GRIPPER_CLOSE_WIDTH_DEFAULT: float = 0.0


def refresh_gripper_constants() -> None:
    """Refresh cached gripper limits when settings change."""
    global GRIPPER_PROFILE
    global GRIPPER_FORCE_MIN, GRIPPER_FORCE_MAX, GRIPPER_FORCE_DEFAULT
    global GRIPPER_SPEED_MIN, GRIPPER_SPEED_DEFAULT
    global GRIPPER_CALIBRATION_WIDTH_MM, GRIPPER_OPEN_WIDTH_DEFAULT, GRIPPER_CLOSE_WIDTH_DEFAULT

    settings = registry().settings
    GRIPPER_PROFILE = settings.gripper_profile
    GRIPPER_FORCE_MIN = GRIPPER_PROFILE.force_min
    GRIPPER_FORCE_MAX = GRIPPER_PROFILE.force_max
    GRIPPER_FORCE_DEFAULT = GRIPPER_PROFILE.force_default
    GRIPPER_SPEED_MIN = GRIPPER_PROFILE.speed_min
    GRIPPER_SPEED_DEFAULT = GRIPPER_PROFILE.speed_default
    GRIPPER_CALIBRATION_WIDTH_MM = GRIPPER_PROFILE.calibration_width
    GRIPPER_OPEN_WIDTH_DEFAULT = GRIPPER_PROFILE.open_width_default
    GRIPPER_CLOSE_WIDTH_DEFAULT = GRIPPER_PROFILE.close_width_default


def _recent_gripper_sync(
    key: str,
    value: Optional[float],
    *,
    window_s: float = 0.75,
) -> bool:
    """Check if a gripper value was recently synced within the time window."""
    return is_recently_synced(key, value, window_s=window_s)


def _record_gripper_sync(width: Optional[float], force: Optional[float]) -> None:
    """Record a gripper sync event with timestamp."""
    record_gripper_sync(width=width, force=force)


async def _await_grip_status(timeout_s: float) -> bool:
    """Backward-compatible alias for await_grip_status."""
    return await await_grip_status(timeout_s)


async def await_grip_status(timeout_s: float) -> bool:
    """Poll the gripper until it reports a gripped state or timeout."""
    session = registry().session
    gripper_defaults = registry().gripper_defaults
    gripper = session.gripper
    if gripper is None:
        return False
    t_index = gripper_defaults["t_index"]
    deadline = time.monotonic() + max(timeout_s, 0.1)
    while time.monotonic() < deadline:
        try:
            if gripper.isGripped(t_index=t_index):
                return True
        except Exception as exc:  # noqa: BLE001
            LOGGER.debug("Failed to poll gripper grip state: %s", exc)
            return False
        await asyncio.sleep(0.2)
    return False


def _safe_int(value: Optional[float], default: int) -> int:
    """Convert value to int, returning default if None or invalid."""
    return safe_int(value, default)


def _wait_for_gripper_motion(t_index: int) -> bool:
    """Wait for gripper motion to finish without requiring grip detection."""
    session = registry().session
    gripper = session.gripper
    if gripper is None:
        return False
    policy = getattr(gripper, "_policy", None)
    timeout_s = float(getattr(policy, "busy_timeout_s", 5.0) or 5.0)
    deadline = time.monotonic() + max(timeout_s, 0.1)
    while time.monotonic() < deadline:
        try:
            busy = gripper.isBusy(t_index=t_index)
            if isinstance(busy, (int, float)) and busy < 0:
                return False
            if not bool(busy):
                return True
        except Exception as exc:  # noqa: BLE001
            LOGGER.debug("Failed to poll gripper busy state: %s", exc)
            return False
        time.sleep(0.05)
    LOGGER.warning("Gripper motion timeout after %.1fs", timeout_s)
    return False


def _command_gripper(
    target_width: float,
    action_label: str,
    *,
    force: Optional[int] = None,
    speed: Optional[int] = None,
    prevent_overclose: bool = False,
) -> str:
    """Move gripper to target width and return status text."""
    session = registry().session
    gripper_defaults = registry().gripper_defaults
    if (
        session.gripper is None
        or not session.gripper.isConnected(t_index=gripper_defaults["t_index"])
    ):
        return "Gripper is not connected. Enable gripper in Setup menu."

    try:
        safe_width = float(target_width)
        min_width, max_width = _resolve_gripper_limits()
        if min_width > max_width:
            min_width, max_width = max_width, min_width
        if prevent_overclose:
            current_width = session.gripper.get_ext_width(
                t_index=gripper_defaults["t_index"]
            )
            if isinstance(current_width, (int, float)) and current_width >= 0:
                if safe_width < current_width:
                    safe_width = current_width
        safe_width = min(max(safe_width, min_width), max_width)

        applied_force = _safe_int(force, GRIPPER_FORCE_DEFAULT)
        applied_force = min(
            max(applied_force, GRIPPER_FORCE_MIN), GRIPPER_FORCE_MAX
        )
        applied_speed = _safe_int(speed, 80)

        command_result = session.gripper.grip(
            t_index=gripper_defaults["t_index"],
            t_width=safe_width,
            n_force=applied_force,
            p_speed=applied_speed,
            f_wait=False,
        )
        if isinstance(command_result, (int, float)) and command_result < 0:
            return f"Gripper action failed: command returned {command_result}."
        _wait_for_gripper_motion(gripper_defaults["t_index"])
        session.last_action = (
            f"Gripper {action_label} to {safe_width:.1f} mm @ "
            f"{applied_force} N / {applied_speed}% speed."
        )
        return session.last_action
    except Exception as exc:  # noqa: BLE001
        LOGGER.exception("Gripper command failed")
        return f"Gripper action failed: {exc}"


def _resolve_gripper_limits() -> tuple[float, float]:
    """Get current gripper min/max width limits from session or device."""
    return resolve_gripper_limits()


def _prepare_gripper_for_calibration() -> None:
    """Set gripper to calibration defaults and remember prior width."""
    session = registry().session
    gripper_defaults = registry().gripper_defaults
    session.robot_calibration_gripper_width = None
    gripper = session.gripper
    if gripper is not None:
        try:
            width = gripper.get_ext_width(
                t_index=gripper_defaults["t_index"])
        except Exception as exc:  # noqa: BLE001
            LOGGER.debug(
                "Unable to read gripper width before calibration: %s", exc)
        else:
            if isinstance(width, (int, float)):
                session.robot_calibration_gripper_width = float(width)
    response = _command_gripper(
        GRIPPER_CALIBRATION_WIDTH_MM,
        "prepping for calibration",
        force=GRIPPER_FORCE_MIN,
        speed=GRIPPER_SPEED_MIN,
    )
    LOGGER.debug("Calibration gripper prep: %s", response)


def _restore_calibration_gripper_width() -> None:
    """Restore gripper width saved prior to calibration."""
    session = registry().session
    width = session.robot_calibration_gripper_width
    if width is None:
        return
    response = _command_gripper(
        width,
        "restored after calibration",
        force=GRIPPER_FORCE_DEFAULT,
        speed=GRIPPER_SPEED_DEFAULT,
        prevent_overclose=True,
    )
    LOGGER.debug("Calibration gripper restored: %s", response)
    session.robot_calibration_gripper_width = None
