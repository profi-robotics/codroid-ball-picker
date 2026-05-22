"""Robot control callbacks (speed, safety stop, UI enablement)."""

import time
from typing import Optional, Tuple

import dash_mantine_components as dmc
from dash import Dash, Input, Output, State, callback_context, no_update
from app.codroid_picker.i18n import t


def _active_auto_control_api(active_program: object) -> object | None:
    """Return the active auto-control robot API, if one is running."""
    from app.codroid_picker.programs.auto_mode import AUTO_CONTROL_PROGRAMS

    if active_program is None:
        return None
    if getattr(active_program, "name", None) not in AUTO_CONTROL_PROGRAMS:
        return None
    if not getattr(active_program, "is_running", False):
        return None
    return getattr(active_program, "robot_api", None)


def register_controls_callbacks(app: Dash) -> None:
    """Register robot control callbacks."""

    @app.callback(
        Output("robot-speed-slider", "value"),
        Input("robot-speed-decrease", "n_clicks"),
        Input("robot-speed-increase", "n_clicks"),
        State("robot-speed-slider", "value"),
        prevent_initial_call=True,
    )
    def _nudge_robot_speed(
        dec_clicks: Optional[int],
        inc_clicks: Optional[int],
        current: Optional[float],
    ) -> float:
        current_value = float(current or 0.0)
        trigger = callback_context.triggered_id
        step = 5.0
        if trigger == "robot-speed-decrease":
            return max(0.0, current_value - step)
        if trigger == "robot-speed-increase":
            return min(100.0, current_value + step)
        return current_value

    @app.callback(
        Output("robot-speed-state", "data"),
        Output("robot-speed-slider", "value", allow_duplicate=True),
        Input("robot-speed-slider", "value"),
        State("robot-connection-state", "data"),
        State("robot-uri", "value"),
        State("robot-control-state", "data"),
        prevent_initial_call=True,
    )
    async def _apply_robot_speed(
        value: Optional[float],
        robot_state: Optional[dict],
        robot_uri: Optional[str],
        control_state: Optional[dict],
    ) -> Tuple[dict, float]:
        import app.codroid_picker as codroid_picker_module
        from app.codroid_picker.programs import get_registry
        from app.codroid_picker.programs.auto_mode import apply_selected_auto_speed

        if value is None:
            return no_update, no_update
        if codroid_picker_module._robot_control_released(control_state):
            return {
                "status": "released",
                "value": value,
                "error": t("Robot control is released. Acquire control first."),
                "updated": time.time(),
            }, no_update
        rate = max(0.0, min(1.0, float(value) / 100.0))
        now = time.time()
        codroid_picker_module.SESSION.robot_speed_rate = rate

        registry = get_registry()
        active_program = registry.active_program
        auto_api = _active_auto_control_api(active_program)
        if auto_api is not None:
            await apply_selected_auto_speed(auto_api)
            return {
                "status": "active_auto",
                "value": value,
                "rate": rate,
                "updated": time.time(),
            }, no_update

        active_api = (
            active_program.robot_api
            if active_program and active_program.robot_api
            else None
        )

        if (robot_state or {}).get("status") != "online":
            codroid_picker_module.LOGGER.info(
                "Robot speed requested: %.0f%% (robot offline)", value
            )
            return {
                "status": "offline",
                "value": value,
                "rate": rate,
                "updated": now,
            }, no_update

        async def _set_speed(api: codroid_picker_module.CodroidAPI) -> None:
            await api.set_manual_move_rate(rate)

        try:
            if active_api is not None:
                await active_api.set_manual_move_rate(rate)
            else:
                await codroid_picker_module._with_robot_api(
                    robot_uri or codroid_picker_module.DEFAULT_ROBOT_URI, _set_speed
                )
            codroid_picker_module.SESSION.robot_speed_last_set = now
            codroid_picker_module.SESSION.last_action = t("Robot speed set to {rate:.2f}", rate=rate)
            codroid_picker_module.LOGGER.info(
                "Robot speed set to %.0f%%", value)
            return {
                "status": "ok",
                "value": value,
                "rate": rate,
                "updated": now,
            }, no_update
        except Exception as exc:  # noqa: BLE001
            codroid_picker_module.LOGGER.exception("Failed to set robot speed")
            return {
                "status": "error",
                "value": value,
                "rate": rate,
                "error": str(exc),
                "updated": now,
            }, no_update

    @app.callback(
        Output("app-notifications", "children", allow_duplicate=True),
        Output("robot-error-state", "data", allow_duplicate=True),
        Input("safety-stop", "n_clicks"),
        State("robot-uri", "value"),
        State("robot-control-state", "data"),
        prevent_initial_call=True,
    )
    async def _handle_safety_stop(
        _clicks: int,
        robot_uri: Optional[str],
        control_state: Optional[dict],
    ) -> Tuple[dmc.Notification, dict]:
        import app.codroid_picker as codroid_picker_module

        if codroid_picker_module._robot_control_released(control_state):
            return (
                dmc.Notification(
                    action="show",
                    title=t("Robot control released"),
                    message=t("Robot control is released. Acquire control first."),
                    color="yellow",
                    autoClose=5000,
                ),
                no_update,
            )
        try:
            async def _power_off(api: codroid_picker_module.CodroidAPI) -> None:
                await api.stop_command()
                await api.power_off()

            await codroid_picker_module._with_robot_api(
                robot_uri or codroid_picker_module.DEFAULT_ROBOT_URI, _power_off
            )
            codroid_picker_module.SESSION.last_action = t("Emergency stop: power off command sent.")
            error_state = codroid_picker_module._activate_robot_error(
                "emergency_stop",
                message=t("Emergency stop triggered from UI."),
                physical_pressed=False,
            )
            return (
                dmc.Notification(
                    action="show",
                    title=t("Emergency stop"),
                    message=t("Stop command + power off sent to robot controller."),
                    color="red",
                    autoClose=5000,
                ),
                error_state,
            )
        except Exception as exc:  # noqa: BLE001
            codroid_picker_module.LOGGER.exception("Emergency stop failed")
            error_state = codroid_picker_module._activate_robot_error(
                "emergency_stop",
                message=t("Emergency stop triggered from UI."),
                physical_pressed=False,
            )
            return (
                dmc.Notification(
                    action="show",
                    title=t("Emergency stop failed"),
                    message=str(exc),
                    color="red",
                    autoClose=7000,
                ),
                error_state,
            )

    @app.callback(
        Output("safety-stop", "disabled"),
        Output("robot-power-toggle", "disabled"),
        Output("robot-speed-slider", "disabled"),
        Output("robot-speed-decrease", "disabled"),
        Output("robot-speed-increase", "disabled"),
        Input("robot-connection-state", "data"),
        Input("robot-control-state", "data"),
    )
    def _sync_robot_controls(
        robot_state: Optional[dict],
        control_state: Optional[dict],
    ) -> Tuple[bool, bool, bool, bool, bool]:
        import app.codroid_picker as codroid_picker_module

        status = (robot_state or {}).get("status") or "offline"
        disabled = status != "online" or codroid_picker_module._robot_control_released(control_state)
        return disabled, disabled, disabled, disabled, disabled
