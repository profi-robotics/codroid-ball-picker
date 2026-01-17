"""Rescue mode control callbacks for manual robot recovery."""

from typing import Optional

import dash_mantine_components as dmc
from dash import Dash, Input, Output, State


def register_rescue_mode_callbacks(app: Dash) -> None:
    """Register rescue mode control callbacks."""

    @app.callback(
        Output("app-notifications", "children", allow_duplicate=True),
        Input("rescue-mode-enable", "n_clicks"),
        State("robot-uri", "value"),
        State("robot-connection-state", "data"),
        State("robot-rescue-state", "data"),
        prevent_initial_call=True,
    )
    async def _enable_rescue_mode(
        _clicks: int,
        robot_uri: Optional[str],
        robot_state: Optional[dict],
        rescue_state: Optional[dict],
    ) -> dmc.Notification:
        import app.codroid_picker as codroid_picker_module

        if (robot_state or {}).get("status") != "online":
            return codroid_picker_module.dmc.Notification(
                action="show",
                title="Rescue mode unavailable",
                message="Robot is not connected.",
                color="red",
                autoClose=5000,
            )
        if (rescue_state or {}).get("active"):
            return codroid_picker_module.dmc.Notification(
                action="show",
                title="Rescue mode",
                message="Rescue mode is already active.",
                color="orange",
                autoClose=4000,
            )

        async def _enter(api) -> None:
            await api.enter_rescue_mode()

        try:
            await codroid_picker_module._with_robot_api(
                robot_uri or codroid_picker_module.DEFAULT_ROBOT_URI, _enter
            )
            return codroid_picker_module.dmc.Notification(
                action="show",
                title="Rescue mode requested",
                message="Rescue mode command sent.",
                color="orange",
                autoClose=5000,
            )
        except Exception as exc:  # noqa: BLE001
            codroid_picker_module.LOGGER.exception(
                "Failed to enter rescue mode")
            return codroid_picker_module.dmc.Notification(
                action="show",
                title="Rescue mode failed",
                message=str(exc),
                color="red",
                autoClose=7000,
            )

    @app.callback(
        Output("app-notifications", "children", allow_duplicate=True),
        Input("rescue-mode-exit", "n_clicks"),
        State("robot-uri", "value"),
        State("robot-rescue-state", "data"),
        prevent_initial_call=True,
    )
    async def _exit_rescue_mode(
        _clicks: int,
        robot_uri: Optional[str],
        rescue_state: Optional[dict],
    ) -> dmc.Notification:
        import app.codroid_picker as codroid_picker_module

        if not (rescue_state or {}).get("active"):
            return codroid_picker_module.dmc.Notification(
                action="show",
                title="Rescue mode",
                message="Rescue mode is not active.",
                color="gray",
                autoClose=4000,
            )

        async def _exit(api) -> None:
            await api.exit_rescue_mode()

        try:
            await codroid_picker_module._with_robot_api(
                robot_uri or codroid_picker_module.DEFAULT_ROBOT_URI, _exit
            )
            return codroid_picker_module.dmc.Notification(
                action="show",
                title="Rescue mode exit requested",
                message="Exit command sent.",
                color="orange",
                autoClose=5000,
            )
        except Exception as exc:  # noqa: BLE001
            codroid_picker_module.LOGGER.exception(
                "Failed to exit rescue mode")
            return codroid_picker_module.dmc.Notification(
                action="show",
                title="Exit rescue mode failed",
                message=str(exc),
                color="red",
                autoClose=7000,
            )

    @app.callback(
        Output("rescue-jog-status", "children"),
        Input("rescue-joint-1-minus", "n_clicks"),
        Input("rescue-joint-1-plus", "n_clicks"),
        Input("rescue-joint-2-minus", "n_clicks"),
        Input("rescue-joint-2-plus", "n_clicks"),
        Input("rescue-joint-3-minus", "n_clicks"),
        Input("rescue-joint-3-plus", "n_clicks"),
        Input("rescue-joint-4-minus", "n_clicks"),
        Input("rescue-joint-4-plus", "n_clicks"),
        Input("rescue-joint-5-minus", "n_clicks"),
        Input("rescue-joint-5-plus", "n_clicks"),
        Input("rescue-joint-6-minus", "n_clicks"),
        Input("rescue-joint-6-plus", "n_clicks"),
        State("robot-uri", "value"),
        State("robot-rescue-state", "data"),
        prevent_initial_call=True,
    )
    async def _handle_rescue_joint_jog(
        _j1_minus: int,
        _j1_plus: int,
        _j2_minus: int,
        _j2_plus: int,
        _j3_minus: int,
        _j3_plus: int,
        _j4_minus: int,
        _j4_plus: int,
        _j5_minus: int,
        _j5_plus: int,
        _j6_minus: int,
        _j6_plus: int,
        robot_uri: Optional[str],
        rescue_state: Optional[dict],
    ) -> str:
        import app.codroid_picker as codroid_picker_module

        if not (rescue_state or {}).get("active"):
            return "Rescue mode is not active."

        trigger = codroid_picker_module.callback_context.triggered_id
        if not trigger:
            return codroid_picker_module.no_update

        mapping = {
            "rescue-joint-1-minus": (1, -1),
            "rescue-joint-1-plus": (1, 1),
            "rescue-joint-2-minus": (2, -1),
            "rescue-joint-2-plus": (2, 1),
            "rescue-joint-3-minus": (3, -1),
            "rescue-joint-3-plus": (3, 1),
            "rescue-joint-4-minus": (4, -1),
            "rescue-joint-4-plus": (4, 1),
            "rescue-joint-5-minus": (5, -1),
            "rescue-joint-5-plus": (5, 1),
            "rescue-joint-6-minus": (6, -1),
            "rescue-joint-6-plus": (6, 1),
        }
        if trigger not in mapping:
            return codroid_picker_module.no_update
        axis, direction = mapping[trigger]

        async def _jog(api) -> None:
            await api.set_robot_command(api.config.commands.manual_mode)
            await api.hold_jog(
                api.config.jog_modes.joint,
                axis=axis,
                direction=direction,
                hold_seconds=codroid_picker_module.SETTINGS.rescue_jog_hold_s,
                heartbeat_interval=0.25,
            )

        try:
            await codroid_picker_module._with_robot_api(
                robot_uri or codroid_picker_module.DEFAULT_ROBOT_URI, _jog
            )
        except Exception as exc:  # noqa: BLE001
            codroid_picker_module.LOGGER.exception("Rescue joint jog failed")
            return f"Rescue jog failed: {exc}"
        sign = "+" if direction > 0 else "-"
        return f"Jogged J{axis}{sign} for {codroid_picker_module.SETTINGS.rescue_jog_hold_s:.2f}s."
