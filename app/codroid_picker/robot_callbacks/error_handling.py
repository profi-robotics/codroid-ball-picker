"""Error handling modal updates and response handlers for robot errors."""

from typing import Callable, Optional, Tuple

import dash_mantine_components as dmc
from dash import Dash, Input, Output, State


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _notify(title: str, message: str, color: str, auto_close: int = 5000) -> dmc.Notification:
    """Build a Notification with consistent defaults."""
    import app.codroid_picker as codroid_picker_module
    return codroid_picker_module.dmc.Notification(
        action="show",
        title=title,
        message=message,
        color=color,
        autoClose=auto_close,
    )


def _run_robot_action_with_notification(
    robot_uri: Optional[str],
    action_fn,
    *,
    success_title: str,
    success_message: str,
    success_color: str,
    failure_title: str,
) -> Tuple[dict, dmc.Notification]:
    """Execute a robot API action and return state/no_update plus notification."""
    import app.codroid_picker as codroid_picker_module

    async def _runner() -> Tuple[dict, dmc.Notification]:
        try:
            await codroid_picker_module._with_robot_api(
                robot_uri or codroid_picker_module.DEFAULT_ROBOT_URI,
                action_fn,
            )
            from app.codroid_picker import state

            state.clear_error_state()
            snapshot = state.get_error_snapshot()
            return snapshot, _notify(success_title, success_message, success_color)
        except Exception as exc:  # noqa: BLE001
            codroid_picker_module.LOGGER.exception(failure_title)
            return (
                codroid_picker_module.no_update,
                _notify(failure_title, str(exc), "red", auto_close=7000),
            )

    return _runner()


def _create_modal_update_callback(
    error_type: str,
    default_message: str,
) -> Callable[[Optional[dict]], Tuple[bool, str]]:
    """Create a reusable modal update callback for error modals.

    Args:
        error_type: The error type identifier (e.g., "overspeed")
        default_message: Default message when none is provided

    Returns:
        A callback handler function that returns (opened, message).
    """
    def _handler(error_state: Optional[dict]) -> Tuple[bool, str]:
        state = error_state or {}
        active = bool(state.get("active")) and state.get("error_type") == error_type
        message = state.get("message") or default_message
        return active, message

    return _handler


def _create_simple_error_handler(
    error_type: str,
    clear_action_id: str,
    stay_paused_action_id: str,
    clear_method_name: str,
    error_name: str,
) -> Callable:
    """Create a reusable error handler for simple clear/stay-paused errors.

    Args:
        error_type: The error type identifier (e.g., "joint_protection")
        clear_action_id: The button ID for clear action (e.g., "joint-protection-clear")
        stay_paused_action_id: The button ID for stay-paused action
        clear_method_name: The API method name to call (e.g., "clear_joint_protection")
        error_name: Human-readable error name for messages (e.g., "Joint protection")

    Returns:
        An async callback handler function.
    """
    async def _handler(
        _clear_clicks: int,
        _stay_clicks: int,
        error_state: Optional[dict],
        robot_uri: Optional[str],
    ) -> Tuple[dict, dmc.Notification]:
        import app.codroid_picker as codroid_picker_module

        state = error_state or {}
        if state.get("error_type") != error_type:
            return codroid_picker_module.no_update, codroid_picker_module.no_update

        trigger = codroid_picker_module.callback_context.triggered_id

        async def _clear(api) -> None:
            method = getattr(api, clear_method_name)
            await method()

        if trigger == clear_action_id:
            snapshot, note = await _run_robot_action_with_notification(
                robot_uri,
                _clear,
                success_title=f"{error_name} cleared",
                success_message="Recovery sequence sent. Program remains paused.",
                success_color="orange",
                failure_title=f"Failed to clear {error_name.lower()}",
            )
            if snapshot is not codroid_picker_module.no_update:
                codroid_picker_module._pause_operation(
                    error_type, f"Paused after {error_name.lower()} clear."
                )
            return snapshot, note

        if trigger == stay_paused_action_id:
            snapshot = codroid_picker_module._acknowledge_robot_error()
            codroid_picker_module._pause_operation(
                error_type,
                f"Paused after {error_name.lower()} acknowledgment.",
            )
            return snapshot, _notify(
                f"{error_name} acknowledged",
                "Program remains paused.",
                "yellow",
                auto_close=4500,
            )

        return codroid_picker_module.no_update, codroid_picker_module.no_update

    return _handler


def register_error_handling_callbacks(app: Dash) -> None:
    """Register error handling modal and response callbacks."""

    # === Modal Updates ===
    # Note: emergency_stop modal needs special handling for physical_pressed flag

    @app.callback(
        Output("emergency-stop-modal", "opened"),
        Output("emergency-stop-modal-text", "children"),
        Output("emergency-ack", "disabled"),
        Input("robot-error-state", "data"),
    )
    def _update_emergency_modal(
        error_state: Optional[dict],
    ) -> Tuple[bool, str, bool]:
        state = error_state or {}
        active = bool(state.get("active")) and state.get(
            "error_type") == "emergency_stop"
        message = state.get("message") or "Emergency stop detected."
        physical_pressed = bool(state.get("physical_pressed"))
        if physical_pressed:
            message = f"{message} Release the physical E-stop to clear."
        return active, message, not active

    # Use factory for standard 2-output modal callbacks
    _overspeed_handler = _create_modal_update_callback(
        "overspeed", "Overspeed alarm detected."
    )
    _joint_protection_handler = _create_modal_update_callback(
        "joint_protection", "Joint protection warning detected."
    )
    _wrong_tool_handler = _create_modal_update_callback(
        "wrong_tool", "Tool/payload mismatch detected."
    )
    _drag_not_allowed_handler = _create_modal_update_callback(
        "drag_not_allowed", "Drag not allowed warning detected."
    )

    app.callback(
        Output("overspeed-modal", "opened"),
        Output("overspeed-modal-text", "children"),
        Input("robot-error-state", "data"),
    )(_overspeed_handler)

    app.callback(
        Output("joint-protection-modal", "opened"),
        Output("joint-protection-modal-text", "children"),
        Input("robot-error-state", "data"),
    )(_joint_protection_handler)

    app.callback(
        Output("wrong-tool-modal", "opened"),
        Output("wrong-tool-modal-text", "children"),
        Input("robot-error-state", "data"),
    )(_wrong_tool_handler)

    app.callback(
        Output("drag-not-allowed-modal", "opened"),
        Output("drag-not-allowed-modal-text", "children"),
        Input("robot-error-state", "data"),
    )(_drag_not_allowed_handler)

    # === Response Handlers ===

    @app.callback(
        Output("robot-error-state", "data", allow_duplicate=True),
        Output("app-notifications", "children", allow_duplicate=True),
        Input("emergency-ack", "n_clicks"),
        State("robot-error-state", "data"),
        State("robot-uri", "value"),
        prevent_initial_call=True,
    )
    async def _ack_emergency_stop(
        _clicks: int,
        error_state: Optional[dict],
        robot_uri: Optional[str],
    ) -> Tuple[dict, dmc.Notification]:
        import app.codroid_picker as codroid_picker_module

        state = error_state or {}
        if state.get("error_type") != "emergency_stop":
            return codroid_picker_module.no_update, codroid_picker_module.no_update

        async def _clear(api) -> None:
            await api.clear_emergency_stop()

        return await _run_robot_action_with_notification(
            robot_uri,
            _clear,
            success_title="Emergency stop acknowledged",
            success_message="Recovery sequence sent. Program remains paused.",
            success_color="red",
            failure_title="Failed to clear emergency stop",
        )

    @app.callback(
        Output("robot-error-state", "data", allow_duplicate=True),
        Output("app-notifications", "children", allow_duplicate=True),
        Input("overspeed-clear", "n_clicks"),
        Input("overspeed-continue", "n_clicks"),
        Input("overspeed-stay-paused", "n_clicks"),
        Input("overspeed-reset", "n_clicks"),
        State("robot-error-state", "data"),
        State("robot-uri", "value"),
        prevent_initial_call=True,
    )
    async def _handle_overspeed_response(
        _clear_clicks: int,
        _continue_clicks: int,
        _stay_clicks: int,
        _reset_clicks: int,
        error_state: Optional[dict],
        robot_uri: Optional[str],
    ) -> Tuple[dict, dmc.Notification]:
        import app.codroid_picker as codroid_picker_module

        state = error_state or {}
        if state.get("error_type") != "overspeed":
            return codroid_picker_module.no_update, codroid_picker_module.no_update

        trigger = codroid_picker_module.callback_context.triggered_id

        async def _clear(api) -> None:
            await api.clear_overspeed()

        async def _clear_and_power_on(api) -> None:
            await api.clear_overspeed()
            await api.power_on()

        if trigger == "overspeed-clear":
            snapshot, note = await _run_robot_action_with_notification(
                robot_uri,
                _clear,
                success_title="Overspeed cleared",
                success_message="Recovery sequence sent. Program remains paused.",
                success_color="orange",
                failure_title="Failed to clear overspeed",
            )
            codroid_picker_module._pause_operation(
                "overspeed", "Paused after overspeed clear.")
            return snapshot, note

        if trigger == "overspeed-continue":
            snapshot, note = await _run_robot_action_with_notification(
                robot_uri,
                _clear_and_power_on,
                success_title="Overspeed cleared",
                success_message="Recovery sequence sent. Robot powered on and program resumed.",
                success_color="teal",
                failure_title="Failed to clear overspeed",
            )
            if snapshot is not codroid_picker_module.no_update:
                codroid_picker_module._resume_operation(
                    "Resuming after overspeed.")
            return snapshot, note

        if trigger == "overspeed-stay-paused":
            snapshot = codroid_picker_module._acknowledge_robot_error()
            codroid_picker_module._pause_operation(
                "overspeed", "Paused after overspeed acknowledgment.")
            return snapshot, _notify(
                "Overspeed acknowledged",
                "Program remains paused.",
                "yellow",
                auto_close=4500,
            )

        if trigger == "overspeed-reset":
            codroid_picker_module._request_stop_operation(
                "Program canceled after overspeed.")

            async def _stop_motion(api) -> None:
                await api.stop_command()

            try:
                await codroid_picker_module._with_robot_api(
                    robot_uri or codroid_picker_module.DEFAULT_ROBOT_URI, _stop_motion
                )
            except Exception as exc:  # noqa: BLE001
                codroid_picker_module.LOGGER.debug(
                    "Stop command failed: %s", exc)
            codroid_picker_module._reset_operation_state(
                "Program reset after overspeed.")
            snapshot = codroid_picker_module._acknowledge_robot_error()
            return snapshot, _notify(
                "Program reset",
                "Overspeed acknowledged and program reset.",
                "red",
            )

        return codroid_picker_module.no_update, codroid_picker_module.no_update

    @app.callback(
        Output("robot-error-state", "data", allow_duplicate=True),
        Output("app-notifications", "children", allow_duplicate=True),
        Input("joint-protection-clear", "n_clicks"),
        Input("joint-protection-stay-paused", "n_clicks"),
        State("robot-error-state", "data"),
        State("robot-uri", "value"),
        prevent_initial_call=True,
    )
    async def _handle_joint_protection_response(
        _clear_clicks: int,
        _stay_clicks: int,
        error_state: Optional[dict],
        robot_uri: Optional[str],
    ) -> Tuple[dict, dmc.Notification]:
        handler = _create_simple_error_handler(
            error_type="joint_protection",
            clear_action_id="joint-protection-clear",
            stay_paused_action_id="joint-protection-stay-paused",
            clear_method_name="clear_joint_protection",
            error_name="Joint protection",
        )
        return await handler(_clear_clicks, _stay_clicks, error_state, robot_uri)

    @app.callback(
        Output("robot-error-state", "data", allow_duplicate=True),
        Output("app-notifications", "children", allow_duplicate=True),
        Input("wrong-tool-clear", "n_clicks"),
        Input("wrong-tool-rescue", "n_clicks"),
        State("robot-error-state", "data"),
        State("robot-uri", "value"),
        prevent_initial_call=True,
    )
    async def _handle_wrong_tool_response(
        _clear_clicks: int,
        _rescue_clicks: int,
        error_state: Optional[dict],
        robot_uri: Optional[str],
    ) -> Tuple[dict, dmc.Notification]:
        import app.codroid_picker as codroid_picker_module

        state = error_state or {}
        if state.get("error_type") != "wrong_tool":
            return codroid_picker_module.no_update, codroid_picker_module.no_update

        trigger = codroid_picker_module.callback_context.triggered_id

        async def _clear(api) -> None:
            await api.clear_tool_error()

        async def _enter_rescue(api) -> None:
            await api.enter_rescue_mode()

        if trigger == "wrong-tool-clear":
            snapshot, note = await _run_robot_action_with_notification(
                robot_uri,
                _clear,
                success_title="Tool error cleared",
                success_message="Recovery sequence sent. Program remains paused.",
                success_color="red",
                failure_title="Failed to clear tool error",
            )
            if snapshot is not codroid_picker_module.no_update:
                codroid_picker_module._pause_operation(
                    "wrong_tool", "Paused after tool error clear."
                )
            return snapshot, note

        if trigger == "wrong-tool-rescue":
            try:
                await codroid_picker_module._with_robot_api(
                    robot_uri or codroid_picker_module.DEFAULT_ROBOT_URI, _enter_rescue
                )
                return (
                    codroid_picker_module.no_update,
                    _notify(
                        "Rescue mode requested",
                        "Rescue mode command sent.",
                        "orange",
                    ),
                )
            except Exception as exc:  # noqa: BLE001
                codroid_picker_module.LOGGER.exception(
                    "Failed to enter rescue mode")
                return (
                    codroid_picker_module.no_update,
                    _notify(
                        "Rescue mode failed",
                        str(exc),
                        "red",
                        auto_close=7000,
                    ),
                )

        return codroid_picker_module.no_update, codroid_picker_module.no_update

    @app.callback(
        Output("robot-error-state", "data", allow_duplicate=True),
        Output("app-notifications", "children", allow_duplicate=True),
        Input("drag-not-allowed-clear", "n_clicks"),
        Input("drag-not-allowed-stay-paused", "n_clicks"),
        State("robot-error-state", "data"),
        State("robot-uri", "value"),
        prevent_initial_call=True,
    )
    async def _handle_drag_not_allowed_response(
        _clear_clicks: int,
        _stay_clicks: int,
        error_state: Optional[dict],
        robot_uri: Optional[str],
    ) -> Tuple[dict, dmc.Notification]:
        handler = _create_simple_error_handler(
            error_type="drag_not_allowed",
            clear_action_id="drag-not-allowed-clear",
            stay_paused_action_id="drag-not-allowed-stay-paused",
            clear_method_name="clear_drag_not_allowed",
            error_name="Drag warning",
        )
        return await handler(_clear_clicks, _stay_clicks, error_state, robot_uri)
