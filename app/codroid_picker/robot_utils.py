"""Robot API and device utilities.

Provides helper functions for probing robots, running actions via the API,
and formatting device lists for the UI.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Awaitable, Callable

from app.codroid_picker.app_instance import registry

if TYPE_CHECKING:
    from codroid_api import CodroidAPI

LOGGER = logging.getLogger(__name__)


def _robot_control_mode(control_state: dict | None = None) -> str:
    """Return the active robot control mode."""
    if isinstance(control_state, dict):
        mode = str(control_state.get("mode", "")).strip().lower()
        if mode in {"acquired", "released"}:
            return mode
        if mode:
            return "acquired"
    session = registry().session
    mode = str(getattr(session, "robot_control_mode", "acquired")).strip().lower()
    if mode not in {"acquired", "released"}:
        return "acquired"
    return mode


def _robot_control_released(control_state: dict | None = None) -> bool:
    """Return True when robot control has been explicitly released."""
    return _robot_control_mode(control_state) == "released"

async def _probe_robot(uri: str, *, require_acquired: bool = True) -> dict:
    """Probe a robot at the given URI and return status info."""
    robot_session = registry().get_or_create_robot_session()
    default_uri = registry().default_robot_uri
    return await robot_session.probe(
        uri or default_uri,
        require_acquired=require_acquired,
    )


async def _with_robot_api(
    uri: str,
    action: Callable[["CodroidAPI"], Awaitable[None]],
    *,
    require_acquired: bool = True,
) -> None:
    """Run an async action with a robot API connection."""
    robot_session = registry().get_or_create_robot_session()
    default_uri = registry().default_robot_uri
    await robot_session.run_with_robot(
        uri or default_uri,
        action,
        require_acquired=require_acquired,
    )


async def _get_robot_api(uri: str | None = None) -> "CodroidAPI":
    """Get a robot API instance for the given URI.

    This provides direct access to the robot API for use cases where
    a callback-based pattern is not suitable (e.g., long-running programs).

    Args:
        uri: Robot WebSocket URI, or None to use default.

    Returns:
        CodroidAPI instance connected to the robot.
    """
    robot_session = registry().get_or_create_robot_session()
    default_uri = registry().default_robot_uri

    # Use run_with_robot to get the API instance via callback
    api_instance = None
    async def _capture_api(api: "CodroidAPI") -> None:
        nonlocal api_instance
        api_instance = api

    await robot_session.run_with_robot(
        uri or default_uri,
        _capture_api,
        require_acquired=True,
    )

    if api_instance is None:
        raise RuntimeError("Failed to get robot API instance")

    return api_instance


async def _ensure_robot_coord5_for_uri(uri: str | None) -> None:
    """Switch the robot to CoordinateId 5 via the provided URI."""
    from app.codroid_picker.procedures import _ensure_coordinate5_active

    async def _activate(api: "CodroidAPI") -> None:
        await _ensure_coordinate5_active(api)

    default_uri = registry().default_robot_uri
    await _with_robot_api(uri or default_uri, _activate)


def _log_move_target(
    label: str,
    target_pose: dict,
    *,
    coordinate_id: int | None = None,
) -> None:
    """Log a robot move target for debugging purposes."""
    coord_info = f" (coord={coordinate_id})" if coordinate_id is not None else ""
    pos_str = ", ".join(f"{k}={v:.2f}" for k, v in target_pose.items() if isinstance(v, (int, float)))
    LOGGER.debug("Move target [%s]%s: %s", label, coord_info, pos_str)
