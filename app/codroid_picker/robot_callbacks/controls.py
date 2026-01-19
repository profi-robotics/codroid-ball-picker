"""Robot control callbacks (speed, safety stop, UI enablement)."""

import time
from typing import Optional, Tuple

import dash_mantine_components as dmc
from dash import Dash, Input, Output, State, callback_context, no_update


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
        Input("robot-speed-slider", "value"),
        State("robot-connection-state", "data"),
        State("robot-uri", "value"),
        prevent_initial_call=True,
    )
    async def _apply_robot_speed(
        value: Optional[float],
        robot_state: Optional[dict],
        robot_uri: Optional[str],
    ) -> dict:
        import app.codroid_picker as codroid_picker_module
        from app.codroid_picker.programs import get_registry

        if value is None:
            return no_update
        rate = max(0.0, min(1.0, float(value) / 100.0))
        now = time.time()

        codroid_picker_module.SESSION.robot_speed_rate = rate

        registry = get_registry()
        active_program = registry.active_program
        active_api = active_program.robot_api if active_program and active_program.robot_api else None

        if (robot_state or {}).get("status") != "online":
            codroid_picker_module.LOGGER.info(
                "Robot speed requested: %.0f%% (robot offline)", value
            )
            return {"status": "offline", "value": value, "rate": rate, "updated": now}

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
            codroid_picker_module.SESSION.last_action = f"Robot speed set to {rate:.2f}"
            codroid_picker_module.LOGGER.info(
                "Robot speed set to %.0f%%", value)
            return {"status": "ok", "value": value, "rate": rate, "updated": now}
        except Exception as exc:  # noqa: BLE001
            codroid_picker_module.LOGGER.exception("Failed to set robot speed")
            return {
                "status": "error",
                "value": value,
                "rate": rate,
                "error": str(exc),
                "updated": now,
            }

    @app.callback(
        Output("app-notifications", "children", allow_duplicate=True),
        Output("robot-error-state", "data", allow_duplicate=True),
        Input("safety-stop", "n_clicks"),
        State("robot-uri", "value"),
        prevent_initial_call=True,
    )
    async def _handle_safety_stop(
        _clicks: int,
        robot_uri: Optional[str],
    ) -> Tuple[dmc.Notification, dict]:
        import app.codroid_picker as codroid_picker_module

        try:
            async def _power_off(api: codroid_picker_module.CodroidAPI) -> None:
                await api.stop_command()
                await api.power_off()

            await codroid_picker_module._with_robot_api(
                robot_uri or codroid_picker_module.DEFAULT_ROBOT_URI, _power_off
            )
            codroid_picker_module.SESSION.last_action = "Emergency stop: power off command sent."
            error_state = codroid_picker_module._activate_robot_error(
                "emergency_stop",
                message="Emergency stop triggered from UI.",
                physical_pressed=False,
            )
            return (
                dmc.Notification(
                    action="show",
                    title="Emergency stop",
                    message="Stop command + power off sent to robot controller.",
                    color="red",
                    autoClose=5000,
                ),
                error_state,
            )
        except Exception as exc:  # noqa: BLE001
            codroid_picker_module.LOGGER.exception("Emergency stop failed")
            error_state = codroid_picker_module._activate_robot_error(
                "emergency_stop",
                message="Emergency stop triggered from UI.",
                physical_pressed=False,
            )
            return (
                dmc.Notification(
                    action="show",
                    title="Emergency stop failed",
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
    )
    def _sync_robot_controls(robot_state: Optional[dict]) -> Tuple[bool, bool, bool, bool, bool]:
        status = (robot_state or {}).get("status") or "offline"
        disabled = status != "online"
        return disabled, disabled, disabled, disabled, disabled
