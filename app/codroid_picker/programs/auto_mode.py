"""Shared helpers for controller Auto mode programs."""

from __future__ import annotations

import asyncio
import inspect
import logging
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from codroid_api import CodroidAPI

from app.codroid_picker.session import SESSION

AUTO_CONTROL_PROGRAMS = {"auto_pick_place", "auto_motion_test"}
MANUAL_MODE_RETRIES = 3
MANUAL_MODE_SETTLE_S = 0.4


def clamp_robot_speed_rate(rate: object, *, minimum: float = 0.0) -> float:
    """Clamp a robot speed rate to the controller's 0.0-1.0 range."""
    try:
        speed_rate = float(rate)
    except (TypeError, ValueError):
        speed_rate = float(SESSION.robot_speed_rate)
    return max(float(minimum), min(1.0, speed_rate))


def current_robot_speed_rate(*, minimum: float = 0.0) -> float:
    """Return the current UI-selected robot speed rate."""
    return clamp_robot_speed_rate(SESSION.robot_speed_rate, minimum=minimum)


def _mode_key(key: object) -> bool:
    lowered = str(key).lower()
    return "mode" in lowered and "model" not in lowered


def _normalize_controller_mode(value: object) -> str | None:
    if isinstance(value, dict) and "value" in value:
        return _normalize_controller_mode(value.get("value"))
    if isinstance(value, str):
        token = "".join(char for char in value.lower() if char.isalnum())
        if "auto" in token:
            return "auto"
        if "manual" in token or "teach" in token or "jog" in token:
            return "manual"
    if isinstance(value, (int, float)):
        command_value = int(value)
        if command_value == 5:
            return "auto"
        if command_value == 3:
            return "manual"
    return None


def find_controller_mode(payload: object) -> str | None:
    """Find the controller mode reported by a nested robot status payload."""
    if isinstance(payload, dict):
        for key, value in payload.items():
            if _mode_key(key):
                mode = _normalize_controller_mode(value)
                if mode is not None:
                    return mode

        for value in payload.values():
            nested = find_controller_mode(value)
            if nested is not None:
                return nested

    elif isinstance(payload, list):
        for item in payload:
            nested = find_controller_mode(item)
            if nested is not None:
                return nested

    return None


def controller_is_auto_mode(status_snapshot: object) -> bool:
    """Return True when the robot status snapshot reports controller Auto mode."""
    return find_controller_mode(status_snapshot or {}) == "auto"


def controller_is_manual_mode(status_snapshot: object) -> bool:
    """Return True when the robot status snapshot reports controller Manual mode."""
    return find_controller_mode(status_snapshot or {}) == "manual"


async def apply_selected_auto_speed(robot_api: "CodroidAPI") -> None:
    """Apply the UI-selected speed rate before auto-socket motion."""
    speed_rate = current_robot_speed_rate(minimum=0.01)
    await robot_api.set_speed_multiplier(speed_rate)
    SESSION.robot_speed_rate = speed_rate
    SESSION.robot_speed_last_set = time.time()


async def enter_controller_auto_mode(robot_api: "CodroidAPI") -> None:
    """Switch the controller to Auto mode and apply the selected robot speed."""
    await robot_api.set_auto_mode()
    await apply_selected_auto_speed(robot_api)


def _latest_controller_status() -> object:
    """Return the latest robot status snapshot, if the app session has one."""
    try:
        from app.codroid_picker.app_instance import registry

        session = registry().get_or_create_robot_session()
        status_snapshot = getattr(session, "robot_status_snapshot", None)
        if callable(status_snapshot):
            return status_snapshot()
    except Exception:  # noqa: BLE001
        return {}
    return {}


async def _set_manual_mode_with_retries(
    robot_api: "CodroidAPI",
    *,
    logger: logging.Logger,
    context: str,
) -> None:
    """Send Manual mode repeatedly until confirmed or attempts are exhausted."""
    last_error: Exception | None = None
    for attempt in range(1, MANUAL_MODE_RETRIES + 1):
        try:
            await robot_api.set_manual_mode()
            status = _latest_controller_status()
            mode = find_controller_mode(status)
            if mode == "manual":
                return
            if mode not in {None, "auto"}:
                return
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            logger.debug(
                "%s: set_manual_mode attempt %d/%d failed: %s",
                context,
                attempt,
                MANUAL_MODE_RETRIES,
                exc,
            )
        if attempt < MANUAL_MODE_RETRIES:
            await asyncio.sleep(MANUAL_MODE_SETTLE_S)

    if last_error is not None:
        logger.warning(
            "%s: failed to switch controller to Manual mode: %s",
            context,
            last_error,
        )
    else:
        logger.warning(
            "%s: Manual mode was not confirmed after %d attempts.",
            context,
            MANUAL_MODE_RETRIES,
        )


async def pause_auto_program_manual_mode(
    active_program: object | None,
    *,
    logger: logging.Logger,
    context: str,
) -> bool:
    """Switch a paused auto-control program to Manual without closing its socket."""
    if active_program is None:
        return False
    if getattr(active_program, "name", None) not in AUTO_CONTROL_PROGRAMS:
        return False
    robot_api = getattr(active_program, "robot_api", None)
    if robot_api is None:
        return False
    try:
        backend = getattr(robot_api, "_auto_socket_backend", None)
        if backend is not None:
            try:
                backend.clear_target_position()
            except Exception as exc:  # noqa: BLE001
                logger.debug(
                    "%s: socket clear_target_position before pause failed: %s",
                    context,
                    exc,
                )
        await _set_manual_mode_with_retries(
            robot_api,
            logger=logger,
            context=context,
        )
        return True
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "%s: failed to switch paused auto program to Manual: %s",
            context,
            exc,
        )
        return False


async def resume_auto_program_auto_mode(
    active_program: object | None,
    *,
    logger: logging.Logger,
    context: str,
) -> bool:
    """Switch a paused auto-control program back to Auto before resuming."""
    if active_program is None:
        return False
    if getattr(active_program, "name", None) not in AUTO_CONTROL_PROGRAMS:
        return False
    robot_api = getattr(active_program, "robot_api", None)
    if robot_api is None:
        return False
    try:
        await enter_controller_auto_mode(robot_api)
        return True
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "%s: failed to restore Auto mode before resume: %s",
            context,
            exc,
        )
        return False


async def restore_manual_mode(
    robot_api: "CodroidAPI",
    *,
    logger: logging.Logger,
    context: str,
) -> None:
    """Best-effort return from controller Auto mode to Manual mode."""
    auto_socket_backend = getattr(robot_api, "_auto_socket_backend", None)
    if auto_socket_backend is not None:
        try:
            auto_socket_backend.clear_target_position()
        except Exception as exc:  # noqa: BLE001
            logger.debug(
                "%s: socket clear_target_position before Manual mode failed: %s",
                context,
                exc,
            )
        robot_api._auto_socket_backend = None

    try:
        await robot_api.clear_target_position()
    except Exception as exc:  # noqa: BLE001
        logger.debug(
            "%s: clear_target_position before Manual mode failed: %s",
            context,
            exc,
        )
    try:
        await robot_api.stop_command()
    except Exception as exc:  # noqa: BLE001
        logger.debug(
            "%s: stop_command before Manual mode failed: %s",
            context,
            exc,
        )
    try:
        await robot_api.stop_auto_socket_project()
    except Exception as exc:  # noqa: BLE001
        logger.debug(
            "%s: stop_auto_socket_project before Manual mode failed: %s",
            context,
            exc,
        )
    if auto_socket_backend is not None:
        close = getattr(auto_socket_backend, "close", None)
        if callable(close):
            try:
                close_result = close()
                if inspect.isawaitable(close_result):
                    await close_result
            except Exception as exc:  # noqa: BLE001
                logger.debug("%s: auto socket close failed: %s", context, exc)
    await asyncio.sleep(MANUAL_MODE_SETTLE_S)
    await _set_manual_mode_with_retries(robot_api, logger=logger, context=context)


async def restore_active_auto_program_manual_mode(
    active_program: object | None,
    *,
    logger: logging.Logger,
    context: str,
) -> bool:
    """Restore Manual mode for the currently active auto-control program."""
    if active_program is None:
        return False
    if getattr(active_program, "name", None) not in AUTO_CONTROL_PROGRAMS:
        return False
    robot_api = getattr(active_program, "robot_api", None)
    if robot_api is None:
        return False
    await restore_manual_mode(robot_api, logger=logger, context=context)
    return True
