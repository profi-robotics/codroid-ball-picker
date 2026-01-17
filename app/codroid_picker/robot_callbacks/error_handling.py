"""Error handling modal updates and response handlers for robot errors."""

from typing import Callable, Optional, Tuple

import dash_mantine_components as dmc
from dash import Dash, Input, Output, State


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
            try:
                await codroid_picker_module._with_robot_api(
                    robot_uri or codroid_picker_module.DEFAULT_ROBOT_URI, _clear
                )
                snapshot = codroid_picker_module._acknowledge_robot_error()
                codroid_picker_module._pause_operation(
                    error_type, f"Paused after {error_name.lower()} clear."
                )
                return (
                    snapshot,
                    codroid_picker_module.dmc.Notification(
                        action="show",
                        title=f"{error_name} cleared",
                        message="Recovery sequence sent. Program remains paused.",
                        color="orange",
                        autoClose=5000,
                    ),
                )
            except Exception as exc:  # noqa: BLE001
                codroid_picker_module.LOGGER.exception(
                    f"Failed to clear {error_name.lower()}")
                return (
                    codroid_picker_module.no_update,
                    codroid_picker_module.dmc.Notification(
                        action="show",
                        title=f"{error_name} clear failed",
                        message=str(exc),
                        color="red",
                        autoClose=7000,
                    ),
                )
        if trigger == stay_paused_action_id:
            snapshot = codroid_picker_module._acknowledge_robot_error()
            codroid_picker_module._pause_operation(
                error_type,
                f"Paused after {error_name.lower()} acknowledgment.",
            )
            return (
                snapshot,
                codroid_picker_module.dmc.Notification(
                    action="show",
                    title=f"{error_name} acknowledged",
                    message="Program remains paused.",
                    color="yellow",
                    autoClose=4500,
                ),
            )

        return codroid_picker_module.no_update, codroid_picker_module.no_update

    return _handler


def register_error_handling_callbacks(app: Dash) -> None:
    """Register error handling modal and response callbacks."""

    # === Modal Updates ===

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
        return active, message, physical_pressed

    @app.callback(
        Output("overspeed-modal", "opened"),
        Output("overspeed-modal-text", "children"),
        Input("robot-error-state", "data"),
    )
    def _update_overspeed_modal(
        error_state: Optional[dict],
    ) -> Tuple[bool, str]:
        state = error_state or {}
        active = bool(state.get("active")) and state.get(
            "error_type") == "overspeed"
        message = state.get("message") or "Overspeed alarm detected."
        return active, message

    @app.callback(
        Output("joint-protection-modal", "opened"),
        Output("joint-protection-modal-text", "children"),
        Input("robot-error-state", "data"),
    )
    def _update_joint_protection_modal(
        error_state: Optional[dict],
    ) -> Tuple[bool, str]:
        state = error_state or {}
        active = bool(state.get("active")) and state.get(
            "error_type") == "joint_protection"
        message = state.get("message") or "Joint protection warning detected."
        return active, message

    @app.callback(
        Output("wrong-tool-modal", "opened"),
        Output("wrong-tool-modal-text", "children"),
        Input("robot-error-state", "data"),
    )
    def _update_wrong_tool_modal(
        error_state: Optional[dict],
    ) -> Tuple[bool, str]:
        state = error_state or {}
        active = bool(state.get("active")) and state.get(
            "error_type") == "wrong_tool"
        message = state.get("message") or "Tool/payload mismatch detected."
        return active, message

    @app.callback(
        Output("drag-not-allowed-modal", "opened"),
        Output("drag-not-allowed-modal-text", "children"),
        Input("robot-error-state", "data"),
    )
    def _update_drag_not_allowed_modal(
        error_state: Optional[dict],
    ) -> Tuple[bool, str]:
        state = error_state or {}
        active = bool(state.get("active")) and state.get(
            "error_type") == "drag_not_allowed"
        message = state.get("message") or "Drag not allowed warning detected."
        return active, message

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
        if state.get("physical_pressed"):
            return codroid_picker_module.no_update, codroid_picker_module.no_update

        async def _clear(api) -> None:
            await api.clear_emergency_stop()

        try:
            await codroid_picker_module._with_robot_api(
                robot_uri or codroid_picker_module.DEFAULT_ROBOT_URI, _clear
            )
            snapshot = codroid_picker_module._acknowledge_robot_error()
            return (
                snapshot,
                codroid_picker_module.dmc.Notification(
                    action="show",
                    title="Emergency stop acknowledged",
                    message="Recovery sequence sent. Program remains paused.",
                    color="red",
                    autoClose=5000,
                ),
            )
        except Exception as exc:  # noqa: BLE001
            codroid_picker_module.LOGGER.exception(
                "Failed to clear emergency stop")
            return (
                codroid_picker_module.no_update,
                codroid_picker_module.dmc.Notification(
                    action="show",
                    title="Emergency stop clear failed",
                    message=str(exc),
                    color="red",
                    autoClose=7000,
                ),
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

        if trigger == "overspeed-clear":
            try:
                await codroid_picker_module._with_robot_api(
                    robot_uri or codroid_picker_module.DEFAULT_ROBOT_URI, _clear
                )
                snapshot = codroid_picker_module._acknowledge_robot_error()
                codroid_picker_module._pause_operation(
                    "overspeed", "Paused after overspeed clear.")
                return (
                    snapshot,
                    codroid_picker_module.dmc.Notification(
                        action="show",
                        title="Overspeed cleared",
                        message="Recovery sequence sent. Program remains paused.",
                        color="orange",
                        autoClose=5000,
                    ),
                )
            except Exception as exc:  # noqa: BLE001
                codroid_picker_module.LOGGER.exception(
                    "Failed to clear overspeed")
                return (
                    codroid_picker_module.no_update,
                    codroid_picker_module.dmc.Notification(
                        action="show",
                        title="Overspeed clear failed",
                        message=str(exc),
                        color="red",
                        autoClose=7000,
                    ),
                )
        if trigger == "overspeed-continue":
            try:
                await codroid_picker_module._with_robot_api(
                    robot_uri or codroid_picker_module.DEFAULT_ROBOT_URI, _clear
                )
                codroid_picker_module._resume_operation(
                    "Resuming after overspeed.")
                snapshot = codroid_picker_module._acknowledge_robot_error()
                return (
                    snapshot,
                    codroid_picker_module.dmc.Notification(
                        action="show",
                        title="Overspeed cleared",
                        message="Recovery sequence sent. Program resumed.",
                        color="teal",
                        autoClose=5000,
                    ),
                )
            except Exception as exc:  # noqa: BLE001
                codroid_picker_module.LOGGER.exception(
                    "Failed to clear overspeed")
                return (
                    codroid_picker_module.no_update,
                    codroid_picker_module.dmc.Notification(
                        action="show",
                        title="Overspeed clear failed",
                        message=str(exc),
                        color="red",
                        autoClose=7000,
                    ),
                )
        if trigger == "overspeed-stay-paused":
            snapshot = codroid_picker_module._acknowledge_robot_error()
            codroid_picker_module._pause_operation(
                "overspeed", "Paused after overspeed acknowledgment.")
            return (
                snapshot,
                codroid_picker_module.dmc.Notification(
                    action="show",
                    title="Overspeed acknowledged",
                    message="Program remains paused.",
                    color="yellow",
                    autoClose=4500,
                ),
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
            return (
                snapshot,
                codroid_picker_module.dmc.Notification(
                    action="show",
                    title="Program reset",
                    message="Overspeed acknowledged and program reset.",
                    color="red",
                    autoClose=5000,
                ),
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
            try:
                await codroid_picker_module._with_robot_api(
                    robot_uri or codroid_picker_module.DEFAULT_ROBOT_URI, _clear
                )
                snapshot = codroid_picker_module._acknowledge_robot_error()
                codroid_picker_module._pause_operation(
                    "wrong_tool", "Paused after tool error clear.")
                return (
                    snapshot,
                    codroid_picker_module.dmc.Notification(
                        action="show",
                        title="Tool error cleared",
                        message="Recovery sequence sent. Program remains paused.",
                        color="red",
                        autoClose=5000,
                    ),
                )
            except Exception as exc:  # noqa: BLE001
                codroid_picker_module.LOGGER.exception(
                    "Failed to clear tool error")
                return (
                    codroid_picker_module.no_update,
                    codroid_picker_module.dmc.Notification(
                        action="show",
                        title="Tool error clear failed",
                        message=str(exc),
                        color="red",
                        autoClose=7000,
                    ),
                )
        if trigger == "wrong-tool-rescue":
            try:
                await codroid_picker_module._with_robot_api(
                    robot_uri or codroid_picker_module.DEFAULT_ROBOT_URI, _enter_rescue
                )
                return (
                    codroid_picker_module.no_update,
                    codroid_picker_module.dmc.Notification(
                        action="show",
                        title="Rescue mode requested",
                        message="Rescue mode command sent.",
                        color="orange",
                        autoClose=5000,
                    ),
                )
            except Exception as exc:  # noqa: BLE001
                codroid_picker_module.LOGGER.exception(
                    "Failed to enter rescue mode")
                return (
                    codroid_picker_module.no_update,
                    codroid_picker_module.dmc.Notification(
                        action="show",
                        title="Rescue mode failed",
                        message=str(exc),
                        color="red",
                        autoClose=7000,
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
