"""Robot state display and synchronization callbacks.

This module handles real-time robot state synchronization:
- Robot position display
- Error state detection and synchronization
- Rescue mode state detection
- Rescue mode UI control display
"""

import time
from typing import Optional, Tuple

from dash import Dash, Input, no_update, Output, State
from app.codroid_picker.i18n import t


def register_state_sync_callbacks(app: Dash) -> None:
    """Register robot state display and sync callbacks."""

    @app.callback(
        Output("current-robot-position", "children"),
        Input("calibration-ui-interval", "n_intervals"),
        State("robot-connection-state", "data"),
        prevent_initial_call=True,
    )
    def _update_robot_position_display(
        _intervals: int,
        robot_conn_state: Optional[dict],
    ) -> str:
        import app.codroid_picker as codroid_picker_module

        """Display current robot position with real-time updates."""
        session = codroid_picker_module._get_robot_session()
        pos = session.position_snapshot()
        age_s = pos.age_seconds()

        # Show position if we have recent data (from any source)
        if session.posture_seen() and age_s < 2.0:
            return (
                t("Robot position: X={x:.1f}, Y={y:.1f}, Z={z:.1f}", x=pos.x, y=pos.y, z=pos.z)
            )

        # Check if robot is connected via test connection
        if (robot_conn_state or {}).get("status") == "online":
            if not session.posture_seen():
                return t("Robot: Awaiting posture stream...")
            return t("Robot: Position stale (last seen {age:.1f}s ago)", age=age_s)

        return t("Robot: --")

    @app.callback(
        Output("robot-error-state", "data", allow_duplicate=True),
        Input("robot-status-interval", "n_intervals"),
        State("robot-error-state", "data"),
        prevent_initial_call=True,
    )
    def _sync_robot_error_state(
        _intervals: int,
        current_state: Optional[dict],
    ) -> dict:
        import app.codroid_picker as codroid_picker_module

        session = codroid_picker_module._get_robot_session()
        status = session.robot_status_snapshot()
        status_age = session.robot_status_age_seconds()
        if not status or status_age is None or status_age > 2.0:
            return no_update

        warning = session.robot_warning_snapshot()
        warning_age = session.robot_warning_age_seconds()
        error_msg = session.robot_error_snapshot()
        error_age = session.robot_error_age_seconds()

        emergency_active = codroid_picker_module._detect_emergency_stop_status(
            status)
        overspeed_active = codroid_picker_module._detect_overspeed_status(
            status)
        joint_active = codroid_picker_module._detect_joint_protection_status(
            status)
        wrong_tool_active = codroid_picker_module._detect_wrong_tool_status(
            status)
        drag_active = codroid_picker_module._detect_drag_not_allowed_status(
            status)

        if warning and warning_age is not None and warning_age <= 2.0:
            emergency_active = emergency_active or codroid_picker_module.CodroidAPI.is_emergency_button_warning(
                warning
            )
            overspeed_active = overspeed_active or codroid_picker_module.CodroidAPI.is_overspeed_warning(
                warning
            )
            joint_active = joint_active or codroid_picker_module.CodroidAPI.is_joint_protection_warning(
                warning
            )
            drag_active = drag_active or codroid_picker_module.CodroidAPI.is_drag_not_allowed_warning(
                warning
            )

        if error_msg and error_age is not None and error_age <= 5.0:
            wrong_tool_active = wrong_tool_active or codroid_picker_module.CodroidAPI.is_wrong_tool_error(
                error_msg
            )

        physical_pressed = session.emergency_button_active()
        if not emergency_active:
            physical_pressed = False

        if emergency_active:
            snapshot = codroid_picker_module._activate_robot_error(
                "emergency_stop",
                message=t("Emergency stop reported by robot controller."),
                physical_pressed=physical_pressed,
            )
        elif overspeed_active:
            snapshot = codroid_picker_module._activate_robot_error(
                "overspeed",
                message=t("Overspeed alarm reported by robot controller."),
                physical_pressed=False,
            )
        elif joint_active:
            snapshot = codroid_picker_module._activate_robot_error(
                "joint_protection",
                message=t("Joint protection alarm reported by robot controller."),
                physical_pressed=False,
            )
        elif wrong_tool_active:
            snapshot = codroid_picker_module._activate_robot_error(
                "wrong_tool",
                message=t("Tool/payload mismatch reported by robot controller."),
                physical_pressed=False,
            )
        elif drag_active:
            snapshot = codroid_picker_module._activate_robot_error(
                "drag_not_allowed",
                message=t("Drag not allowed warning reported by robot controller."),
                physical_pressed=False,
            )
        else:
            if (current_state or {}).get("error_type") == "emergency_stop":
                snapshot = codroid_picker_module._update_robot_error_state(
                    physical_pressed=physical_pressed
                )
            else:
                snapshot = codroid_picker_module._error_snapshot()

        if current_state == snapshot:
            return no_update
        return snapshot

    @app.callback(
        Output("robot-rescue-state", "data"),
        Input("robot-status-interval", "n_intervals"),
        State("robot-rescue-state", "data"),
        prevent_initial_call=True,
    )
    def _sync_robot_rescue_state(
        _intervals: int,
        current_state: Optional[dict],
    ) -> dict:
        import app.codroid_picker as codroid_picker_module

        session = codroid_picker_module._get_robot_session()
        status = session.robot_status_snapshot()
        status_age = session.robot_status_age_seconds()
        if not status or status_age is None or status_age > 2.0:
            return no_update
        active = codroid_picker_module._detect_rescue_mode_status(status)
        if current_state and current_state.get("active") == active:
            return no_update
        return {"active": active, "timestamp": time.time()}

    @app.callback(
        Output("robot-controls-standard", "style"),
        Output("robot-controls-rescue", "style"),
        Output("rescue-mode-status", "children"),
        Output("rescue-mode-enable", "disabled"),
        Output("rescue-mode-exit", "style"),
        Input("robot-rescue-state", "data"),
        Input("robot-connection-state", "data"),
    )
    def _toggle_rescue_controls(
        rescue_state: Optional[dict],
        robot_state: Optional[dict],
    ) -> Tuple[dict, dict, str, bool, dict]:
        active = bool((rescue_state or {}).get("active"))
        robot_connected = (robot_state or {}).get("status") == "online"

        if active:
            return (
                {"display": "none"},
                {"display": "block"},
                t("Rescue mode active."),
                True,
                {"display": "inline-flex"},
            )
        return (
            {"display": "flex"},
            {"display": "none"},
            t("Rescue mode idle.") if robot_connected else t("Rescue mode unavailable (robot not connected)."),
            not robot_connected,  # Disabled when robot not connected
            {"display": "none"},
        )

    @app.callback(
        Output("footer-robot-position", "children"),
        Input("robot-connection-state", "data"),
        Input("robot-status-interval", "n_intervals"),
        prevent_initial_call=True,
    )
    def _update_footer_robot_position(
        robot_conn_state: Optional[dict],
        _intervals: int,
    ) -> str:
        import app.codroid_picker as codroid_picker_module

        session = codroid_picker_module._get_robot_session()
        if not session.posture_seen():
            if (robot_conn_state or {}).get("status") == "online":
                return t("X=--.--- Y=--.--- Z=--.---  A=--.--- B=--.--- C=--.---")
            return t("X=--.--- Y=--.--- Z=--.---  A=--.--- B=--.--- C=--.---")

        pos = session.position_snapshot()
        text = (
            f"X={pos.x:.3f} Y={pos.y:.3f} Z={pos.z:.3f}  "
            f"A={pos.a:.3f} B={pos.b:.3f} C={pos.c:.3f}"
        )
        age_s = pos.age_seconds()
        if age_s > 2.0:
            text = f"{text} · stale {age_s:.1f}s"
        return text

    @app.callback(
        Output("footer-flange-indicator-1", "className"),
        Output("footer-flange-indicator-2", "className"),
        Output("footer-flange-indicator-3", "className"),
        Output("footer-flange-indicator-4", "className"),
        Input("robot-status-interval", "n_intervals"),
        prevent_initial_call=True,
    )
    def _update_footer_flange_indicators(_intervals: int) -> Tuple[str, str, str, str]:
        import app.codroid_picker as codroid_picker_module

        session = codroid_picker_module._get_robot_session()
        states = session.flange_button_states()
        classes = []
        for port in codroid_picker_module.DEFAULT_FLANGE_BUTTON_PORTS:
            active = states.get(port, 0) == 1
            classes.append(
                "footer-flange-indicator active" if active else "footer-flange-indicator"
            )
        return tuple(classes)
