"""Shared robot and gripper helper utilities.

This module consolidates common patterns used across robot_motion.py,
gripper_utils.py, and robot callbacks to reduce code duplication:
- Gripper limit resolution
- Robot API wrapper patterns
- Session access helpers
"""

from __future__ import annotations

import logging
import time
from typing import (
    Any,
    Awaitable,
    Callable,
    Optional,
    Tuple,
    TypeVar,
    TYPE_CHECKING,
)

if TYPE_CHECKING:
    from codroid_api import CodroidAPI, RobotSession
    from onrobot.twofg import TWOFG

from app.codroid_picker.app_instance import registry

LOGGER = logging.getLogger(__name__)

T = TypeVar("T")


# =============================================================================
# Session Access Helpers
# =============================================================================


def get_session():
    """Get the current session from registry."""
    return registry().session


def get_robot_session() -> "RobotSession":
    """Get the RobotSession instance from registry."""
    return registry().robot_session


def get_gripper() -> Optional["TWOFG"]:
    """Get the gripper instance from session."""
    return get_session().gripper


# =============================================================================
# Gripper Limit Resolution (consolidated from gripper_utils and robot_motion)
# =============================================================================


def resolve_gripper_limits(
    gripper: Optional["TWOFG"] = None,
) -> Tuple[float, float]:
    """Get gripper min/max width limits from device or defaults.

    Attempts to read limits from:
    1. Gripper status snapshot (cached)
    2. Direct gripper query
    3. Session cached values
    4. Gripper profile defaults

    Args:
        gripper: Optional gripper instance. If None, uses session gripper.

    Returns:
        Tuple of (min_width, max_width) in mm.
    """
    session = get_session()
    gripper_defaults = registry().gripper_defaults
    settings = registry().settings
    gripper_profile = settings.gripper_profile

    if gripper is None:
        gripper = session.gripper

    t_index = gripper_defaults["t_index"]

    min_width = session.gripper_min_width
    max_width = session.gripper_max_width

    if gripper is not None:
        # Try to get from gripper status snapshot (cached)
        status_snapshot = gripper.get_status_snapshot(t_index=t_index)
        if status_snapshot:
            snapshot_min = status_snapshot.get("min_external_width")
            snapshot_max = status_snapshot.get("max_external_width")
            if isinstance(snapshot_min, (int, float)) and snapshot_min >= 0:
                min_width = float(snapshot_min)
                session.gripper_min_width = min_width
            if isinstance(snapshot_max, (int, float)) and snapshot_max >= 0:
                max_width = float(snapshot_max)
                session.gripper_max_width = max_width

        # Fallback to querying gripper directly
        if min_width is None:
            try:
                value = gripper.get_min_ext_width(t_index=t_index)
                if isinstance(value, (int, float)) and value >= 0:
                    min_width = float(value)
            except Exception as exc:
                LOGGER.debug("Failed to read gripper min width: %s", exc)

        if max_width is None:
            try:
                value = gripper.get_max_ext_width(t_index=t_index)
                if isinstance(value, (int, float)) and value >= 0:
                    max_width = float(value)
            except Exception as exc:
                LOGGER.debug("Failed to read gripper max width: %s", exc)

    # Final fallbacks from profile
    if min_width is None:
        min_width = gripper_profile.close_width_default
    if max_width is None:
        max_width = gripper_profile.open_width_default

    # Ensure correct ordering
    if min_width > max_width:
        LOGGER.warning(
            "Gripper width limits inverted (min=%.1f, max=%.1f); swapping.",
            min_width,
            max_width,
        )
        min_width, max_width = max_width, min_width

    return min_width, max_width


def is_gripper_connected(gripper: Optional["TWOFG"] = None) -> bool:
    """Check if gripper is connected.

    Args:
        gripper: Optional gripper instance. If None, uses session gripper.

    Returns:
        True if gripper is connected, False otherwise.
    """
    session = get_session()
    gripper_defaults = registry().gripper_defaults

    if gripper is None:
        gripper = session.gripper

    if gripper is None:
        return False

    try:
        return gripper.isConnected(t_index=gripper_defaults["t_index"])
    except Exception:
        return False


def clamp_gripper_width(
    width: float,
    min_width: Optional[float] = None,
    max_width: Optional[float] = None,
) -> float:
    """Clamp width to valid gripper range.

    Args:
        width: Target width in mm.
        min_width: Minimum width. If None, resolves from gripper.
        max_width: Maximum width. If None, resolves from gripper.

    Returns:
        Clamped width value.
    """
    if min_width is None or max_width is None:
        resolved_min, resolved_max = resolve_gripper_limits()
        if min_width is None:
            min_width = resolved_min
        if max_width is None:
            max_width = resolved_max

    return min(max(width, min_width), max_width)


# =============================================================================
# Robot API Wrapper Patterns
# =============================================================================


async def with_robot_api(
    robot_uri: str,
    action: Callable[["CodroidAPI"], Awaitable[T]],
    *,
    timeout_s: float = 10.0,
) -> T:
    """Execute an async action with a robot API connection.

    This is the canonical pattern for one-shot robot API calls,
    used by callbacks that need to execute a single command.

    Args:
        robot_uri: Robot WebSocket URI.
        action: Async function that takes CodroidAPI and returns result.
        timeout_s: Connection timeout in seconds (unused, kept for API compat).

    Returns:
        Result of the action function.

    Raises:
        Exception: If connection fails or action raises.
    """
    robot_session = registry().get_or_create_robot_session()
    default_uri = registry().default_robot_uri
    return await robot_session.run_with_robot(robot_uri or default_uri, action)


async def with_session_api(
    action: Callable[["CodroidAPI"], Awaitable[T]],
    *,
    require_running: bool = False,
) -> Optional[T]:
    """Execute an action using the session's robot API.

    Uses the existing robot session rather than creating a new connection.

    Args:
        action: Async function that takes CodroidAPI and returns result.
        require_running: If True, returns None if no active session.

    Returns:
        Result of the action function, or None if session unavailable.
    """
    robot_session = get_robot_session()

    if not robot_session.is_connected():
        if require_running:
            LOGGER.warning("No active robot session for API call")
            return None
        raise RuntimeError("No active robot session")

    return await robot_session.run_with_api(action)


# =============================================================================
# Gripper Sync Tracking
# =============================================================================


def record_gripper_sync(
    width: Optional[float] = None,
    force: Optional[float] = None,
) -> None:
    """Record a gripper sync event with timestamp.

    Args:
        width: Current gripper width in mm.
        force: Current gripper force in N.
    """
    session = get_session()
    payload = dict(session.gripper_sync)
    payload["updated"] = time.time()
    if width is not None:
        payload["width"] = width
    if force is not None:
        payload["force"] = force
    session.gripper_sync = payload


def is_recently_synced(
    key: str,
    value: Optional[float],
    *,
    window_s: float = 0.75,
) -> bool:
    """Check if a gripper value was recently synced within the time window.

    Args:
        key: The sync key to check (e.g., 'width', 'force').
        value: The expected value.
        window_s: Time window in seconds.

    Returns:
        True if the value matches and was synced recently.
    """
    session = get_session()
    sync_data = session.gripper_sync

    if not sync_data or value is None:
        return False

    try:
        updated = float(sync_data.get("updated", 0.0))
        if time.time() - updated > window_s:
            return False
        synced_value = sync_data.get(key)
        if synced_value is None:
            return False
        return abs(float(value) - float(synced_value)) < 0.01
    except (TypeError, ValueError):
        return False


# =============================================================================
# Utility Functions
# =============================================================================


def safe_int(value: Optional[float], default: int) -> int:
    """Convert value to int, returning default if None or invalid.

    Args:
        value: Value to convert.
        default: Default value if conversion fails.

    Returns:
        Integer value or default.
    """
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def safe_float(value: Any, default: float) -> float:
    """Convert value to float, returning default if None or invalid.

    Args:
        value: Value to convert.
        default: Default value if conversion fails.

    Returns:
        Float value or default.
    """
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default
