"""Robot connection, gripper, and power management callbacks."""

import time
from typing import Optional, Tuple

import dash_mantine_components as dmc
from dash import Dash, Input, Output, State, callback_context, html, no_update


def register_connection_callbacks(app: Dash) -> None:
    """Register robot connection, gripper, and power management callbacks."""

    @app.callback(
        Output("robot-status-text", "children"),
        Output("header-robot-badge", "children"),
        Output("header-robot-badge", "color"),
        Output("robot-connection-state", "data"),
        Input("robot-test", "n_clicks"),
        Input("robot-disconnect", "n_clicks"),
        Input("auto-init", "n_intervals"),
        State("robot-uri", "value"),
        prevent_initial_call=True,
    )
    async def _test_robot(
        _clicks: int,
        _disconnect_clicks: int,
        _init: int,
        uri: str,
    ) -> Tuple[str, str, str, dict]:
        import app.codroid_picker as codroid_picker_module

        trigger = callback_context.triggered_id
        session = codroid_picker_module._get_robot_session()
        if trigger == "robot-disconnect":
            await session.close()
            codroid_picker_module.SESSION.robot_connection_attempted = False
            codroid_picker_module.SESSION.robot_last_connection_error = None
            session.clear_position()
            codroid_picker_module._set_robot_guidance(
                "Robot calibration idle. Click 'Begin calibrating robot' to start.",
                status="idle",
                step=0,
                verification_ready=False,
            )
            status = "Robot disconnected."
            return status, "Robot: offline", "gray", {"status": "offline", "message": status}
        if trigger == "auto-init":
            if session.is_connected(uri or codroid_picker_module.DEFAULT_ROBOT_URI):
                target = session.resolve_robot_target(
                    uri or codroid_picker_module.DEFAULT_ROBOT_URI)
                status = f"Robot connected to {target['host']}:{target['port']}."
                codroid_picker_module.SESSION.robot_connection_attempted = True
                codroid_picker_module.SESSION.robot_last_connection_error = None
                return status, "Robot: online", "brand", {"status": "online", "message": status}
            if codroid_picker_module.SESSION.robot_connection_attempted:
                if codroid_picker_module.SESSION.robot_last_connection_error:
                    error = codroid_picker_module.SESSION.robot_last_connection_error
                    return error, "Robot: error", "yellow", {"status": "error", "message": error}
                return no_update, no_update, no_update, no_update
        try:
            response = await codroid_picker_module._probe_robot(uri)
            if not response:
                raise RuntimeError("Robot probe returned no response.")
            host = response.get("host", "unknown")
            port = response.get("port", "unknown")
            status = f"Robot connected to {host}:{port}."
            codroid_picker_module.SESSION.robot_connection_attempted = True
            codroid_picker_module.SESSION.robot_last_connection_error = None
            return status, "Robot: online", "brand", {"status": "online", "message": status}
        except Exception as exc:
            error = f"Robot connection failed: {exc}"
            codroid_picker_module.SESSION.robot_connection_attempted = True
            codroid_picker_module.SESSION.robot_last_connection_error = error
            return error, "Robot: error", "yellow", {"status": "error", "message": error}

    @app.callback(
        Output("gripper-status-text", "children"),
        Output("gripper-device-info", "children"),
        Output("header-gripper-badge", "children"),
        Output("header-gripper-badge", "color"),
        Output("gripper-connection-state", "data"),
        Input("gripper-test", "n_clicks"),
        Input("gripper-enable", "checked"),
        Input("auto-init", "n_intervals"),
        State("gripper-cb-ip", "value"),
    )
    def _handle_gripper(
        _test_clicks: int,
        enabled: bool,
        _init: int,
        cb_ip: str,
    ) -> Tuple[str, html.Div, str, str, dict]:
        """Handle gripper connection and testing."""
        import app.codroid_picker as codroid_picker_module

        trigger = callback_context.triggered_id

        if not enabled and trigger == "gripper-enable":
            # Disable gripper
            if codroid_picker_module.SESSION.gripper is not None:
                codroid_picker_module.SESSION.gripper = None
                codroid_picker_module.SESSION.gripper_device = None
            return (
                "Gripper disabled",
                html.Div(),
                "Gripper: off",
                "gray",
                {"status": "disabled", "message": "Gripper disabled"},
            )

        # Auto-connect once on load
        if trigger == "auto-init":
            enabled = True
        if not enabled:
            return no_update, no_update, no_update, no_update, no_update

        # Enable or test gripper
        try:
            if codroid_picker_module.SESSION.gripper_device is None or codroid_picker_module.SESSION.gripper_device.Global_cbip != cb_ip:
                # Create new connection
                device = codroid_picker_module.GripperDevice(Global_cbip=cb_ip)
                cb = device.getCB()

                if cb is None:
                    return (
                        f"Failed to connect to Compute Box at {cb_ip}",
                        html.Div(
                            dmc.Alert(
                                "Cannot reach the gripper control box. Check power and network.",
                                title="Gripper Unavailable",
                                color="yellow",
                            )
                        ),
                        "Gripper: error",
                        "yellow",
                        {
                            "status": "error",
                            "message": f"Failed to connect to Compute Box at {cb_ip}",
                        },
                    )

                codroid_picker_module.SESSION.gripper_device = device
                codroid_picker_module.SESSION.gripper = codroid_picker_module.TwoFG7Gripper(
                    device)

            # Check gripper connection
            try:
                is_connected = codroid_picker_module.SESSION.gripper.isConnected(
                    t_index=codroid_picker_module.GRIPPER_DEFAULTS["t_index"]
                )
            except Exception as exc:  # noqa: BLE001
                codroid_picker_module.SESSION.gripper = None
                codroid_picker_module.SESSION.gripper_device = None
                return (
                    f"Failed to query gripper at {cb_ip}: {exc}",
                    html.Div(
                        dmc.Alert(
                            "Could not reach the gripper. Verify the Compute Box IP and network connection.",
                            title="Gripper Unavailable",
                            color="yellow",
                        )
                    ),
                    "Gripper: error",
                    "yellow",
                    {
                        "status": "error",
                        "message": f"Failed to query gripper at {cb_ip}: {exc}",
                    },
                )

            if not is_connected:
                return (
                    f"No 2FG7 gripper detected at Compute Box {cb_ip}",
                    html.Div(
                        dmc.Alert(
                            "Ensure the 2FG7 gripper is connected to the Compute Box "
                            "and powered on. Check the tool index setting if using "
                            "a dual gripper setup.",
                            title="Gripper Not Detected",
                            color="yellow",
                        )
                    ),
                    "Gripper: error",
                    "yellow",
                    {
                        "status": "error",
                        "message": f"No 2FG7 gripper detected at Compute Box {cb_ip}",
                    },
                )

            # Get gripper status and information
            status_text = f"2FG7 connected at {cb_ip}"
            t_index = codroid_picker_module.GRIPPER_DEFAULTS["t_index"]
            codroid_picker_module.SESSION.gripper.start_status_stream()

            # Query gripper state
            width = codroid_picker_module.SESSION.gripper.get_ext_width(
                t_index=t_index)
            force = codroid_picker_module.SESSION.gripper.get_force(
                t_index=t_index)
            is_gripped = codroid_picker_module.SESSION.gripper.isGripped(
                t_index=t_index)
            codroid_picker_module.SESSION.robot_has_object = bool(is_gripped)

            info_card = html.Div(
                dmc.Card(
                    [
                        dmc.Group(
                            [
                                dmc.Stack(
                                    [
                                        dmc.Text(
                                            "Width", size="sm", c="dimmed"),
                                        dmc.Text(
                                            f"{width:.1f} mm" if isinstance(
                                                width, (int, float)) and width > -2 else "N/A",
                                            fw=500,
                                        ),
                                    ],
                                    gap=0,
                                ),
                                dmc.Stack(
                                    [
                                        dmc.Text(
                                            "Force", size="sm", c="dimmed"),
                                        dmc.Text(
                                            f"{force:.1f} N" if isinstance(
                                                force, (int, float)) and force > -2 else "N/A",
                                            fw=500,
                                        ),
                                    ],
                                    gap=0,
                                ),
                                dmc.Stack(
                                    [
                                        dmc.Text(
                                            "Status", size="sm", c="dimmed"),
                                        dmc.Badge(
                                            "Gripped" if is_gripped else "Open",
                                            color="teal" if is_gripped else "gray",
                                            variant="light",
                                        ),
                                    ],
                                    gap=0,
                                ),
                            ],
                            grow=True,
                        ),
                    ],
                    withBorder=True,
                    padding="sm",
                    mt="xs",
                )
            )

            return (
                status_text,
                info_card,
                "Gripper: online",
                "brand",
                {"status": "online", "message": status_text},
            )

        except Exception as exc:
            codroid_picker_module.LOGGER.exception(
                "Gripper connection failed: %s", exc)
            return (
                f"Gripper error: {exc}",
                html.Div(
                    dmc.Alert(
                        "Failed to connect to gripper. Check network configuration and hardware.",
                        title="Connection Error",
                        color="red",
                    )
                ),
                "Gripper: error",
                "red",
                {"status": "error", "message": f"Gripper error: {exc}"},
            )

    @app.callback(
        Output("app-notifications", "children", allow_duplicate=True),
        Input("robot-power-toggle", "checked"),
        State("robot-uri", "value"),
        State("robot-power-state", "data"),
        prevent_initial_call=True,
    )
    async def _handle_power_toggle(
        checked: bool,
        robot_uri: Optional[str],
        power_state: Optional[dict],
    ) -> dmc.Notification:
        import app.codroid_picker as codroid_picker_module

        if power_state and power_state.get("value") == checked:
            return no_update
        try:
            async def _apply_power(api) -> None:
                if checked:
                    await api.power_on()
                else:
                    await api.power_off()

            await codroid_picker_module._with_robot_api(robot_uri or codroid_picker_module.DEFAULT_ROBOT_URI, _apply_power)
            action = "Power on" if checked else "Power off"
            codroid_picker_module.SESSION.last_action = f"{action} command sent."
            return dmc.Notification(
                action="show",
                title="Robot power",
                message=f"{action} command sent to robot controller.",
                color="red" if not checked else "teal",
                autoClose=4000,
            )
        except Exception as exc:  # noqa: BLE001
            codroid_picker_module.LOGGER.exception("Power toggle failed")
            return dmc.Notification(
                action="show",
                title="Robot power failed",
                message=str(exc),
                color="red",
                autoClose=7000,
            )

    @app.callback(
        Output("robot-power-state", "data"),
        Input("robot-status-interval", "n_intervals"),
        State("robot-connection-state", "data"),
        State("robot-uri", "value"),
        State("robot-power-state", "data"),
        prevent_initial_call=True,
    )
    async def _sync_robot_power_state(
        _intervals: int,
        robot_state: Optional[dict],
        robot_uri: Optional[str],
        current_state: Optional[dict],
    ) -> dict:
        import app.codroid_picker as codroid_picker_module

        if (robot_state or {}).get("status") != "online":
            return no_update
        now = time.time()
        if now - codroid_picker_module.SESSION.robot_status_last_poll < 1.0:
            return no_update

        async def _poll(api) -> None:
            await api.read_system_data()
            await api.read_global_data()

        try:
            await codroid_picker_module._with_robot_api(robot_uri or codroid_picker_module.DEFAULT_ROBOT_URI, _poll)
            codroid_picker_module.SESSION.robot_status_last_poll = now
        except Exception as exc:  # noqa: BLE001
            codroid_picker_module.LOGGER.debug(
                "Robot power poll failed: %s", exc)
            return no_update

        session = codroid_picker_module._get_robot_session()
        power_on = session.robot_power_on()
        if power_on is None:
            return no_update
        desired = {"value": bool(power_on), "timestamp": time.time()}
        if current_state and current_state.get("value") == desired["value"]:
            return no_update
        return desired

    @app.callback(
        Output("robot-power-toggle", "checked"),
        Input("robot-power-state", "data"),
        State("robot-power-toggle", "checked"),
        prevent_initial_call=True,
    )
    def _sync_power_toggle(
        power_state: Optional[dict],
        current: Optional[bool],
    ) -> bool:
        if not power_state or power_state.get("value") is None:
            return no_update
        desired = bool(power_state.get("value"))
        if current is None or current != desired:
            return desired
        return no_update
