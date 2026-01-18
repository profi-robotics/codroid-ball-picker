"""
Robot error detection utilities.

This module provides functions for detecting various robot error states
from status dictionaries received from the robot API.
"""

from __future__ import annotations

from typing import Any, Tuple


def _coerce_truthy(value: Any) -> bool:
    """Coerce a value to a boolean, treating various string/numeric values as truthy."""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "on", "yes", "enabled"}
    return False


def _status_flag(status: dict, keys: Tuple[str, ...]) -> bool:
    """Check if any of the given keys are truthy in the status dict."""
    for key in keys:
        if key in status and _coerce_truthy(status.get(key)):
            return True
    return False


def _alarm_contains(status: dict, keywords: Tuple[str, ...]) -> bool:
    """Check if any alarm text contains any of the given keywords."""
    alarms = status.get("alarms") or status.get("alarm") or []
    if isinstance(alarms, dict):
        alarms = [alarms]
    if not isinstance(alarms, list):
        alarms = [alarms]
    for alarm in alarms:
        text = str(alarm).lower()
        if any(keyword in text for keyword in keywords):
            return True
    return False


def _detect_status_pattern(
    status: dict,
    flag_keys: Tuple[str, ...],
    alarm_keywords: Tuple[str, ...],
) -> bool:
    """Generic pattern for detecting robot error states.

    This consolidates the common logic used by all error detection functions:
    1. Check for truthy flag keys in status
    2. Check for truthy flag keys in nested robot_status
    3. Check for alarm text containing keywords

    Args:
        status: Robot status dictionary.
        flag_keys: Tuple of status flag keys to check.
        alarm_keywords: Tuple of alarm text keywords to search for.

    Returns:
        True if any detection pattern matches.
    """
    if _status_flag(status, flag_keys):
        return True
    nested = status.get("robot_status")
    if isinstance(nested, dict) and _status_flag(nested, flag_keys):
        return True
    return _alarm_contains(status, alarm_keywords)


def _detect_emergency_stop_status(status: dict) -> bool:
    """Detect if the robot is in emergency stop state."""
    emergency_keys = (
        "emergency_stop",
        "emergencyStop",
        "estop",
        "e_stop",
        "EmergencyStop",
        "EStop",
    )
    if _detect_status_pattern(status, emergency_keys, ("emergency", "e-stop", "estop")):
        return True
    # Additional check for emergency + stop in payload text
    payload_text = str(status).lower()
    return "emergency" in payload_text and "stop" in payload_text


def _detect_overspeed_status(status: dict) -> bool:
    """Detect if the robot is in overspeed state."""
    overspeed_keys = (
        "overspeed",
        "over_speed",
        "overSpeed",
        "speed_alarm",
        "overspeed_alarm",
    )
    return _detect_status_pattern(status, overspeed_keys, ("overspeed", "over speed"))


def _detect_joint_protection_status(status: dict) -> bool:
    """Detect if the robot is in joint protection/collision state."""
    joint_keys = (
        "joint_protection",
        "jointProtection",
        "joint_collision",
        "jointCollision",
        "collision",
    )
    return _detect_status_pattern(status, joint_keys, ("joint", "collision"))


def _detect_wrong_tool_status(status: dict) -> bool:
    """Detect if the robot has a wrong tool error."""
    tool_keys = (
        "wrong_tool",
        "wrongTool",
        "tool_error",
        "payload_error",
        "payloadError",
    )
    return _detect_status_pattern(status, tool_keys, ("tool", "payload"))


def _detect_rescue_mode_status(status: dict) -> bool:
    """Detect if the robot is in rescue mode (state == 3)."""
    state = status.get("state")
    if isinstance(state, (int, float)) and int(state) == 3:
        return True
    return False


def _detect_drag_not_allowed_status(status: dict) -> bool:
    """Detect if the robot has a 'drag not allowed' warning."""
    drag_keys = (
        "drag_not_allowed",
        "dragNotAllowed",
        "drag_not_allowed_warning",
    )
    return _detect_status_pattern(status, drag_keys, ("drag not allowed",))


# Error state management wrappers
def _activate_robot_error(
    error_type: str,
    *,
    message: str,
    physical_pressed: bool = False,
) -> dict:
    """Activate a robot error state and return the snapshot."""
    from app.codroid_picker import state
    state.set_error_state(error_type, message, physical_pressed=physical_pressed)
    return state.get_error_snapshot()


def _error_snapshot() -> dict:
    """Get the current error state snapshot."""
    from app.codroid_picker import state
    return state.get_error_snapshot()


def _update_robot_error_state(*, physical_pressed: bool) -> dict:
    """Update the physical_pressed flag on an existing error and return snapshot."""
    from app.codroid_picker import state
    state.set_error_state(
        state.error_state.error_type or "unknown",
        state.error_state.message or "",
        physical_pressed=physical_pressed,
        acknowledge_existing=True,
    )
    return state.get_error_snapshot()

