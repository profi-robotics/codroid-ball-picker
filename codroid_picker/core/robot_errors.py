"""Robot status error detection helpers."""

from __future__ import annotations

from typing import Any


def coerce_truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "on", "yes", "enabled"}
    return False


def status_flag(status: dict[str, Any], *keys: str) -> bool:
    for key in keys:
        if key in status and coerce_truthy(status.get(key)):
            return True
    nested = status.get("robot_status")
    if isinstance(nested, dict):
        return status_flag(nested, *keys)
    return False


def alarm_contains(status: dict[str, Any], *keywords: str) -> bool:
    alarms = status.get("alarms") or status.get("alarm") or []
    if isinstance(alarms, dict):
        alarms = [alarms]
    if not isinstance(alarms, list):
        alarms = [alarms]
    return any(any(keyword in str(alarm).lower() for keyword in keywords) for alarm in alarms)


def detect_robot_error(status: dict[str, Any]) -> tuple[str | None, str | None, bool]:
    """Return active error type, message, and rescue-mode flag from robot status."""
    rescue_mode = False
    raw_state = status.get("state")
    if isinstance(raw_state, (int, float)) and int(raw_state) == 3:
        rescue_mode = True

    payload_text = str(status).lower()
    if (
        status_flag(status, "emergency_stop", "emergencyStop", "estop", "e_stop", "EmergencyStop", "EStop")
        or alarm_contains(status, "emergency", "e-stop", "estop")
        or ("emergency" in payload_text and "stop" in payload_text)
    ):
        return "emergency", "Emergency stop detected.", rescue_mode
    if status_flag(status, "overspeed", "over_speed", "overSpeed", "speed_alarm", "overspeed_alarm") or alarm_contains(status, "overspeed", "over speed"):
        return "overspeed", "Overspeed detected.", rescue_mode
    if status_flag(status, "joint_protection", "jointProtection", "joint_collision", "jointCollision", "collision") or alarm_contains(status, "joint", "collision"):
        return "joint_protection", "Joint protection detected.", rescue_mode
    if status_flag(status, "wrong_tool", "wrongTool", "tool_error", "payload_error", "payloadError") or alarm_contains(status, "tool", "payload"):
        return "wrong_tool", "Tool or payload mismatch detected.", rescue_mode
    if status_flag(status, "drag_not_allowed", "dragNotAllowed", "drag_not_allowed_warning") or alarm_contains(status, "drag not allowed"):
        return "drag", "Drag not allowed.", rescue_mode
    return None, None, rescue_mode
