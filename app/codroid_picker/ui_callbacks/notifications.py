"""Connection notification callbacks.

This module contains Dash callbacks for:
- Camera/robot connection status notifications
"""

from typing import Optional, Tuple

from dash import Dash, Input, Output, State, no_update


def register_notification_callbacks(app: Dash) -> None:
    """Register notification callbacks with the Dash app."""

    @app.callback(
        Output("app-notifications", "sendNotifications"),
        Output("notification-state", "data"),
        Input("camera-connection-state", "data"),
        Input("robot-connection-state", "data"),
        State("notification-state", "data"),
    )
    def _notify_connections(
        camera_state: Optional[dict],
        robot_state: Optional[dict],
        previous_state: Optional[dict],
    ) -> Tuple[list, dict]:
        """Send notifications when camera/robot connection status changes."""
        prev = previous_state or {"camera": None, "robot": None}
        current = {"camera": prev.get("camera"), "robot": prev.get("robot")}
        notifications = []

        def _append_notification(name: str, state: Optional[dict]) -> None:
            if not state:
                return
            status = state.get("status")
            if prev.get(name) == status:
                return
            current[name] = status
            color = "teal"
            title = f"{name.capitalize()} connected"
            if status in ("error", "off"):
                color = "red" if status == "error" else "gray"
                title = f"{name.capitalize()} disconnected"
            notifications.append(
                {
                    "action": "show",
                    "id": f"{name}-connection",
                    "title": title,
                    "message": state.get("message", ""),
                    "color": color,
                    "autoClose": 4500,
                }
            )

        _append_notification("camera", camera_state)
        _append_notification("robot", robot_state)

        if not notifications:
            return no_update, prev
        return notifications, current
