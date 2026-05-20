"""Robot connection, gripper, and power management callbacks."""

import asyncio
import time
from typing import Optional, Tuple

import dash_mantine_components as dmc
from dash import Dash, Input, Output, State, callback_context, html, no_update
from app.codroid_picker.i18n import t
from app.settings import SETTINGS


ROBOT_CONNECT_TIMEOUT_S = 10.0
ROBOT_STALE_PROBE_RESET_S = ROBOT_CONNECT_TIMEOUT_S + 5.0
ROBOT_STARTUP_COORDINATE_ID = 5
ROBOT_STARTUP_INIT_TIMEOUT_S = 8.0


def _control_state_payload(
    *,
    mode: str,
    message: str = "",
    released_at: float = 0.0,
) -> dict:
    """Build a normalized control state payload."""
    normalized_mode = "released" if mode == "released" else "acquired"
    return {
        "mode": normalized_mode,
        "released_at": float(released_at if normalized_mode == "released" else 0.0),
        "message": message,
    }


async def _release_robot_ownership(
    codroid_picker_module,
    resolved_uri: str,
) -> dict:
    """Stop active operations, power off, and close the robot session."""
    from app.codroid_picker.programs import get_registry

    registry = get_registry()
    if registry.active_program is not None:
        registry.stop()
    session = codroid_picker_module._get_robot_session()
    return await session.release_control(resolved_uri, power_off=True, wait_closed_s=2.0)


async def _initialize_robot_after_connect(
    codroid_picker_module,
    resolved_uri: str,
) -> None:
    """Power on the acquired robot and leave it ready for manual operation."""

    async def _initialize(api) -> None:
        await api.power_on()
        await api.set_manual_mode()
        await api.set_current_coordinate_id(ROBOT_STARTUP_COORDINATE_ID)

    await asyncio.wait_for(
        codroid_picker_module._with_robot_api(resolved_uri, _initialize),
        timeout=ROBOT_STARTUP_INIT_TIMEOUT_S,
    )


def _resolve_trigger_id() -> str:
    """Return a stable trigger identifier for this callback invocation."""
    trigger = callback_context.triggered_id
    if isinstance(trigger, str):
        return trigger.split(".")[0]

    if isinstance(trigger, dict):
        maybe_id = trigger.get("id")
        if isinstance(maybe_id, str):
            return maybe_id
        if isinstance(maybe_id, dict):
            return str(maybe_id)

    if callback_context.triggered:
        prop_id = callback_context.triggered[0].get("prop_id", "")
        if isinstance(prop_id, str):
            return prop_id.split(".")[0]

    return "initial-load"


def _probe_stale(started_at: float) -> bool:
    """Return True when a robot probe marker is too old to trust."""
    if started_at <= 0.0:
        return True
    return time.time() - started_at > ROBOT_STALE_PROBE_RESET_S


def register_connection_callbacks(app: Dash) -> None:
    """Register robot connection, gripper, and power management callbacks."""

    @app.callback(
        Output("robot-status-text", "children"),
        Output("header-robot-badge", "children"),
        Output("header-robot-badge", "color"),
        Output("robot-connection-state", "data"),
        Output("robot-control-state", "data"),
        Input("robot-test", "n_clicks"),
        Input("robot-acquire-control", "n_clicks"),
        Input("robot-release-control", "n_clicks"),
        Input("hardware-init", "n_intervals"),
        Input("monitor-interval", "n_intervals"),
        State("robot-uri", "value"),
        State("robot-control-state", "data"),
        running=[
            (Output("robot-test", "loading"), True, False),
            (Output("robot-acquire-control", "loading"), True, False),
            (Output("robot-release-control", "loading"), True, False),
        ],
        prevent_initial_call=False,
    )
    async def _test_robot(
        _clicks: int,
        _acquire_clicks: int,
        _release_clicks: int,
        _init: int,
        _monitor: int,
        uri: str,
        control_state: Optional[dict],
    ) -> Tuple[str, str, str, dict, dict]:
        import app.codroid_picker as codroid_picker_module

        trigger = _resolve_trigger_id()
        session = codroid_picker_module._get_robot_session()
        ui_session = codroid_picker_module.SESSION
        resolved_uri = uri or codroid_picker_module.DEFAULT_ROBOT_URI
        control_released = codroid_picker_module._robot_control_released(control_state)

        if trigger is None:
            trigger = "initial-load"

        startup_triggers = {"hardware-init"}
        if trigger in startup_triggers and control_released:
            return no_update, no_update, no_update, no_update, no_update
        if (
            trigger in startup_triggers
            and not SETTINGS.robot_auto_connect_after_first_paint
        ):
            return no_update, no_update, no_update, no_update, no_update

        if trigger == "robot-release-control":
            try:
                release_state = await _release_robot_ownership(
                    codroid_picker_module, resolved_uri
                )
            except Exception as exc:  # noqa: BLE001
                state = session.control_state() if hasattr(session, "control_state") else {}
                error_text = state.get("last_error") or str(exc)
                error = t("Robot control release failed: {error}", error=error_text)
                codroid_picker_module.LOGGER.warning(
                    "Robot release sequence encountered errors: %s", exc
                )
                ui_session.robot_control_mode = "acquired"
                ui_session.robot_control_released_at = 0.0
                ui_session.robot_connection_attempted = True
                ui_session.robot_last_connection_error = error
                ui_session.robot_probe_inflight = False
                ui_session.robot_last_probe_started_at = 0.0
                return (
                    error,
                    t("Robot: error"),
                    "yellow",
                    {"status": "error", "message": error},
                    _control_state_payload(mode="acquired", message=error),
                )
            now = float(release_state.get("released_at") or time.time())
            ui_session.robot_control_mode = "released"
            ui_session.robot_control_released_at = now
            ui_session.robot_connection_attempted = False
            ui_session.robot_last_connection_error = None
            ui_session.robot_probe_inflight = False
            ui_session.robot_last_probe_started_at = 0.0
            session.clear_position()
            codroid_picker_module._set_robot_guidance(
                t("Robot calibration idle. Click 'Begin calibrating robot' to start."),
                status="idle",
                step=0,
                verification_ready=False,
            )
            status = t("Robot control released. Acquire control to reconnect.")
            return (
                status,
                t("Robot: released"),
                "yellow",
                {"status": "offline", "message": status},
                _control_state_payload(mode="released", message=status, released_at=now),
            )
        if trigger == "monitor-interval" and control_released:
            return no_update, no_update, no_update, no_update, no_update
        if trigger == "monitor-interval":
            if session.is_connected(resolved_uri):
                target = session.resolve_robot_target(resolved_uri)
                status = t("Robot connected to {host}:{port}.", host=target["host"], port=target["port"])
                ui_session.robot_connection_attempted = True
                ui_session.robot_last_connection_error = None
                return (
                    status,
                    t("Robot: online"),
                    "brand",
                    {"status": "online", "message": status},
                    _control_state_payload(mode="acquired"),
                )
            if not ui_session.robot_connection_attempted:
                return no_update, no_update, no_update, no_update, no_update
            status = (
                ui_session.robot_last_connection_error
                or t("Robot disconnected. Acquire control to reconnect.")
            )
            return (
                status,
                t("Robot: error"),
                "yellow",
                {"status": "error", "message": status},
                _control_state_payload(mode="acquired"),
            )
        if trigger not in {"robot-acquire-control", "robot-test", *startup_triggers}:
            return no_update, no_update, no_update, no_update, no_update
        if ui_session.robot_probe_inflight:
            if _probe_stale(ui_session.robot_last_probe_started_at):
                codroid_picker_module.LOGGER.warning(
                    "Resetting stale robot connection probe marker after %.1fs.",
                    time.time() - ui_session.robot_last_probe_started_at,
                )
                ui_session.robot_probe_inflight = False
                ui_session.robot_last_probe_started_at = 0.0
            else:
                status = t("Robot connection check already in progress.")
                return (
                    status,
                    t("Robot: probing"),
                    "yellow",
                    {"status": "probing", "message": status},
                    _control_state_payload(mode="acquired"),
                )
        ui_session.robot_probe_inflight = True
        ui_session.robot_last_probe_started_at = time.time()
        try:
            await asyncio.wait_for(
                session.acquire_control(resolved_uri),
                timeout=ROBOT_CONNECT_TIMEOUT_S,
            )
            await _initialize_robot_after_connect(
                codroid_picker_module,
                resolved_uri,
            )
            target = session.resolve_robot_target(resolved_uri)
            host = target.get("host", t("unknown"))
            port = target.get("port", t("unknown"))
            status = t("Robot connected to {host}:{port}.", host=host, port=port)
            ui_session.robot_control_mode = "acquired"
            ui_session.robot_control_released_at = 0.0
            ui_session.robot_connection_attempted = True
            ui_session.robot_last_connection_error = None
            return (
                status,
                t("Robot: online"),
                "brand",
                {"status": "online", "message": status},
                _control_state_payload(mode="acquired"),
            )
        except asyncio.TimeoutError:
            error = t(
                "Robot connection timed out after {seconds} seconds.",
                seconds=int(ROBOT_CONNECT_TIMEOUT_S),
            )
            ui_session.robot_connection_attempted = True
            ui_session.robot_last_connection_error = error
            return (
                error,
                t("Robot: error"),
                "yellow",
                {"status": "error", "message": error},
                _control_state_payload(mode="acquired"),
            )
        except Exception as exc:
            error = t("Robot connection failed: {error}", error=exc)
            ui_session.robot_connection_attempted = True
            ui_session.robot_last_connection_error = error
            return (
                error,
                t("Robot: error"),
                "yellow",
                {"status": "error", "message": error},
                _control_state_payload(mode="acquired"),
            )
        finally:
            ui_session.robot_probe_inflight = False

    @app.callback(
        Output("gripper-status-text", "children"),
        Output("gripper-device-info", "children"),
        Output("header-gripper-badge", "children"),
        Output("header-gripper-badge", "color"),
        Output("gripper-connection-state", "data"),
        Input("gripper-test", "n_clicks"),
        Input("gripper-enable", "checked"),
        Input("hardware-init", "n_intervals"),
        State("gripper-cb-ip", "value"),
        prevent_initial_call=False,
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

        if trigger is None:
            return no_update, no_update, no_update, no_update, no_update

        if trigger == "hardware-init":
            if not SETTINGS.gripper_auto_connect_after_first_paint:
                return no_update, no_update, no_update, no_update, no_update
            enabled = True

        if not enabled and trigger == "gripper-enable":
            # Disable gripper
            if codroid_picker_module.SESSION.gripper is not None:
                codroid_picker_module.SESSION.gripper = None
                codroid_picker_module.SESSION.gripper_device = None
            return (
                t("Gripper disabled"),
                html.Div(),
                t("Gripper: off"),
                "gray",
                {"status": "disabled", "message": t("Gripper disabled")},
            )

        if not enabled:
            return no_update, no_update, no_update, no_update, no_update

        # Enable or test gripper
        try:
            if codroid_picker_module.SESSION.gripper_device is None or codroid_picker_module.SESSION.gripper_device.Global_cbip != cb_ip:
                # Create new connection
                # Note: GripperDevice uses XML-RPC with default socket timeout (likely 60s)
                # Timeout cannot be configured via constructor parameters
                device = codroid_picker_module.GripperDevice(Global_cbip=cb_ip)
                cb = device.getCB()

                if cb is None:
                    return (
                        t("Failed to connect to Compute Box at {ip}", ip=cb_ip),
                        html.Div(
                            dmc.Alert(
                                t("Cannot reach the gripper control box. Check power and network."),
                                title=t("Gripper Unavailable"),
                                color="yellow",
                            )
                        ),
                        t("Gripper: error"),
                        "yellow",
                        {
                            "status": "error",
                            "message": t("Failed to connect to Compute Box at {ip}", ip=cb_ip),
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
                    t("Failed to query gripper at {ip}: {error}", ip=cb_ip, error=exc),
                    html.Div(
                        dmc.Alert(
                            t("Could not reach the gripper. Verify the Compute Box IP and network connection."),
                            title=t("Gripper Unavailable"),
                            color="yellow",
                        )
                    ),
                    t("Gripper: error"),
                    "yellow",
                    {
                        "status": "error",
                        "message": t("Failed to query gripper at {ip}: {error}", ip=cb_ip, error=exc),
                    },
                )

            if not is_connected:
                return (
                    t("No 2FG7 gripper detected at Compute Box {ip}", ip=cb_ip),
                    html.Div(
                        dmc.Alert(
                            t("Ensure the 2FG7 gripper is connected to the Compute Box and powered on. Check the tool index setting if using a dual gripper setup."),
                            title=t("Gripper Not Detected"),
                            color="yellow",
                        )
                    ),
                    t("Gripper: error"),
                    "yellow",
                    {
                        "status": "error",
                        "message": t("No 2FG7 gripper detected at Compute Box {ip}", ip=cb_ip),
                    },
                )

            # Get gripper status and information
            status_text = t("2FG7 connected at {ip}", ip=cb_ip)
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
                                            t("Width"), size="sm", c="dimmed"),
                                        dmc.Text(
                                            f"{width:.1f} mm" if isinstance(
                                                width, (int, float)) and width > -2 else t("N/A"),
                                            fw=500,
                                        ),
                                    ],
                                    gap=0,
                                ),
                                dmc.Stack(
                                    [
                                        dmc.Text(
                                            t("Force"), size="sm", c="dimmed"),
                                        dmc.Text(
                                            f"{force:.1f} N" if isinstance(
                                                force, (int, float)) and force > -2 else t("N/A"),
                                            fw=500,
                                        ),
                                    ],
                                    gap=0,
                                ),
                                dmc.Stack(
                                    [
                                        dmc.Text(
                                            t("Status"), size="sm", c="dimmed"),
                                        dmc.Badge(
                                            t("Gripped") if is_gripped else t("Open"),
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
                t("Gripper: online"),
                "brand",
                {"status": "online", "message": status_text},
            )

        except Exception as exc:
            codroid_picker_module.LOGGER.exception(
                "Gripper connection failed: %s", exc)
            return (
                t("Gripper error: {error}", error=exc),
                html.Div(
                    dmc.Alert(
                        t("Failed to connect to gripper. Check network configuration and hardware."),
                        title=t("Connection Error"),
                        color="red",
                    )
                ),
                t("Gripper: error"),
                "red",
                {"status": "error", "message": t("Gripper error: {error}", error=exc)},
            )

    @app.callback(
        Output("app-notifications", "children", allow_duplicate=True),
        Input("robot-power-toggle", "checked"),
        State("robot-uri", "value"),
        State("robot-power-state", "data"),
        State("robot-control-state", "data"),
        prevent_initial_call=True,
    )
    async def _handle_power_toggle(
        checked: bool,
        robot_uri: Optional[str],
        power_state: Optional[dict],
        control_state: Optional[dict],
    ) -> dmc.Notification:
        import app.codroid_picker as codroid_picker_module

        if codroid_picker_module._robot_control_released(control_state):
            return dmc.Notification(
                action="show",
                title=t("Robot control released"),
                message=t("Robot control is released. Acquire control first."),
                color="yellow",
                autoClose=5000,
            )
        if power_state and power_state.get("value") == checked:
            return no_update
        try:
            async def _apply_power(api) -> None:
                if checked:
                    await api.power_on()
                else:
                    await api.power_off()

            await codroid_picker_module._with_robot_api(robot_uri or codroid_picker_module.DEFAULT_ROBOT_URI, _apply_power)
            action = t("Power on") if checked else t("Power off")
            codroid_picker_module.SESSION.last_action = t("{action} command sent.", action=action)
            # Set a debounce timestamp to prevent sync from overriding user action immediately
            codroid_picker_module.SESSION.robot_power_toggle_time = time.time()
            return dmc.Notification(
                action="show",
                title=t("Robot power"),
                message=t("{action} command sent to robot controller.", action=action),
                color="red" if not checked else "teal",
                autoClose=4000,
            )
        except Exception as exc:  # noqa: BLE001
            codroid_picker_module.LOGGER.exception("Power toggle failed")
            return dmc.Notification(
                action="show",
                title=t("Robot power failed"),
                message=str(exc),
                color="red",
                autoClose=7000,
            )

    @app.callback(
        Output("robot-power-state", "data"),
        Input("robot-status-interval", "n_intervals"),
        Input("robot-connection-state", "data"),
        Input("robot-control-state", "data"),
        State("robot-uri", "value"),
        State("robot-power-state", "data"),
        prevent_initial_call=False,
    )
    async def _sync_robot_power_state(
        _intervals: int,
        robot_state: Optional[dict],
        control_state: Optional[dict],
        robot_uri: Optional[str],
        current_state: Optional[dict],
    ) -> dict:
        import asyncio
        import app.codroid_picker as codroid_picker_module

        if codroid_picker_module._robot_control_released(control_state):
            return no_update
        if (robot_state or {}).get("status") != "online":
            return no_update
        now = time.time()

        # Don't sync for 3 seconds after user manually toggles power
        # to give robot time to respond to the command
        last_toggle = getattr(codroid_picker_module.SESSION, 'robot_power_toggle_time', 0)
        if now - last_toggle < 3.0:
            return no_update

        session = codroid_picker_module._get_robot_session()
        status_snapshot = session.robot_status_snapshot()
        status_age = session.robot_status_age_seconds()
        status_is_fresh = (
            bool(status_snapshot)
            and status_age is not None
            and status_age <= 2.0
        )

        # Skip throttle on first poll (current_state is None or empty).
        # Back off polling when recent power polls are failing.
        if current_state and current_state.get("value") is not None:
            min_poll_interval_s = 5.0 if (
                now - codroid_picker_module.SESSION.robot_power_poll_error_logged_at < 5.0
            ) else 1.0
            if now - codroid_picker_module.SESSION.robot_status_last_poll < min_poll_interval_s:
                return no_update

        async def _poll(api) -> None:
            await api.read_system_data()
            await api.read_global_data()
            # Small delay to allow listener to process response
            await asyncio.sleep(0.1)

        if not status_is_fresh:
            try:
                await codroid_picker_module._with_robot_api(
                    robot_uri or codroid_picker_module.DEFAULT_ROBOT_URI,
                    _poll,
                )
                codroid_picker_module.SESSION.robot_status_last_poll = now
            except Exception as exc:  # noqa: BLE001
                if (
                    now
                    - codroid_picker_module.SESSION.robot_power_poll_error_logged_at
                    >= 5.0
                ):
                    codroid_picker_module.LOGGER.debug(
                        "Robot power poll failed: %s", exc
                    )
                codroid_picker_module.SESSION.robot_power_poll_error_logged_at = now
                codroid_picker_module.SESSION.robot_status_last_poll = now
                return no_update

        power_on = session.robot_power_on()
        status_snapshot = session.robot_status_snapshot()
        
        # Fallback: if robot_power_on returns None but we have a state field,
        # interpret state >= 2 as powered on (common for Dobot/Codroid robots)
        if power_on is None and isinstance(status_snapshot, dict):
            state_value = status_snapshot.get("state")
            if isinstance(state_value, (int, float)):
                # State 0/1: typically off or idle
                # State 2+: typically powered on / servo enabled
                power_on = state_value >= 2

        if power_on is None:
            # Status not yet available, try to use status age to determine if we should wait
            status_age = session.robot_status_age_seconds()
            if status_age is None or status_age > 5.0:
                # No recent status, don't update toggle
                return no_update
        desired = {"value": bool(power_on), "timestamp": time.time()}
        # Always update on first poll (when current value is None) to sync toggle state
        if current_state and current_state.get("value") is not None:
            if current_state.get("value") == desired["value"]:
                return no_update
        return desired

    @app.callback(
        Output("robot-power-toggle", "checked"),
        Input("robot-power-state", "data"),
        State("robot-power-toggle", "checked"),
        prevent_initial_call=False,
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
