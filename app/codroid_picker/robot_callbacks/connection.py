"""Robot connection, gripper, and power management callbacks."""

import contextlib
import time
from typing import Optional, Tuple

import dash_mantine_components as dmc
from dash import Dash, Input, Output, State, callback_context, html, no_update
from onrobot.errors import (
    OnRobotConnectionError,
    OnRobotTimeoutError,
    OnRobotValidationError,
)

from app.codroid_picker.i18n import t
from app.codroid_picker.gripper_adapters import detect_or_create_gripper
from app.codroid_picker.app_instance import registry


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
) -> None:
    """Stop active operations, power off, and close the robot session."""
    from app.codroid_picker.programs import get_registry

    registry = get_registry()
    if registry.active_program is not None:
        registry.stop()

    async def _stop_and_power_off(api) -> None:
        with contextlib.suppress(Exception):
            await api.stop_command()
        with contextlib.suppress(Exception):
            await api.power_off()

    with contextlib.suppress(Exception):
        await codroid_picker_module._with_robot_api(resolved_uri, _stop_and_power_off)

    await codroid_picker_module._get_robot_session().close()


def _connection_no_update() -> Tuple[object, object, object, object, object]:
    """Return no-update payload for connection callback outputs."""
    return no_update, no_update, no_update, no_update, no_update


async def _handle_robot_connection_event(
    codroid_picker_module,
    *,
    trigger: Optional[str],
    uri: str,
    control_state: Optional[dict],
) -> Tuple[str, str, str, dict, dict]:
    """Handle robot connect/release/probe actions."""
    if trigger is None:
        return _connection_no_update()

    session = codroid_picker_module._get_robot_session()
    ui_session = codroid_picker_module.SESSION
    resolved_uri = uri or codroid_picker_module.DEFAULT_ROBOT_URI
    probe_retry_interval_s = 5.0
    control_released = codroid_picker_module._robot_control_released(control_state)

    if trigger == "auto-init":
        return _connection_no_update()

    if trigger == "robot-release-control":
        try:
            await _release_robot_ownership(codroid_picker_module, resolved_uri)
        except Exception as exc:  # noqa: BLE001
            codroid_picker_module.LOGGER.warning(
                "Robot release sequence encountered errors: %s",
                exc,
            )

        now = time.time()
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

    if trigger in {"robot-acquire-control", "robot-test"}:
        control_released = False
        ui_session.robot_control_mode = "acquired"
        ui_session.robot_control_released_at = 0.0

    if trigger == "monitor-interval" and control_released:
        return _connection_no_update()

    if trigger == "monitor-interval":
        if session.is_connected(resolved_uri):
            target = session.resolve_robot_target(resolved_uri)
            status = t(
                "Robot connected to {host}:{port}.",
                host=target["host"],
                port=target["port"],
            )
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
            return _connection_no_update()
        if ui_session.robot_probe_inflight:
            return _connection_no_update()

        now = time.time()
        if now - ui_session.robot_last_probe_started_at < probe_retry_interval_s:
            return _connection_no_update()

    if ui_session.robot_probe_inflight:
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
        response = await codroid_picker_module._probe_robot(resolved_uri)
        if not response:
            raise RuntimeError(t("Robot probe returned no response."))

        host = response.get("host", t("unknown"))
        port = response.get("port", t("unknown"))
        status = t("Robot connected to {host}:{port}.", host=host, port=port)
        ui_session.robot_connection_attempted = True
        ui_session.robot_last_connection_error = None
        return (
            status,
            t("Robot: online"),
            "brand",
            {"status": "online", "message": status},
            _control_state_payload(mode="acquired"),
        )
    except Exception as exc:  # noqa: BLE001
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


def _build_gripper_info_card(
    display_name: str,
    telemetry: dict,
    product_code: Optional[int] = None,
) -> html.Div:
    """Build compact gripper telemetry card for the setup panel."""
    def _metric(label: str, value: str) -> dmc.Stack:
        return dmc.Stack(
            [
                dmc.Text(t(label), size="sm", c="dimmed"),
                dmc.Text(value, fw=500),
            ],
            gap=0,
        )

    width = telemetry.get("width")
    force = telemetry.get("force")
    depth = telemetry.get("depth")
    vac_a = telemetry.get("vacuum_a")
    vac_b = telemetry.get("vacuum_b")
    is_gripped = bool(telemetry.get("gripped"))
    metrics = []
    if isinstance(width, (int, float)) and width > -2:
        metrics.append(_metric("Width", f"{width:.1f} mm"))
    if isinstance(force, (int, float)) and force > -2:
        metrics.append(_metric("Force", f"{force:.1f} N"))
    if isinstance(depth, (int, float)) and depth > -2:
        metrics.append(_metric("Depth", f"{depth:.1f} mm"))
    if isinstance(vac_a, (int, float)):
        metrics.append(_metric("Vacuum A", f"{vac_a:.1f} kPa"))
    if isinstance(vac_b, (int, float)):
        metrics.append(_metric("Vacuum B", f"{vac_b:.1f} kPa"))
    if not metrics:
        metrics.append(_metric("Telemetry", t("N/A")))
    metrics.append(
        dmc.Stack(
            [
                dmc.Text(t("Status"), size="sm", c="dimmed"),
                dmc.Badge(
                    t("Gripped") if is_gripped else t("Open"),
                    color="teal" if is_gripped else "gray",
                    variant="light",
                ),
            ],
            gap=0,
        )
    )
    subtitle = display_name
    if product_code:
        subtitle = f"{display_name} · 0x{int(product_code):02X}"
    return html.Div(
        dmc.Card(
            [
                dmc.Text(subtitle, fw=600, mb="xs"),
                dmc.Group(
                    metrics,
                    grow=True,
                )
            ],
            withBorder=True,
            padding="sm",
            mt="xs",
        )
    )


def _handle_gripper_event(
    codroid_picker_module,
    *,
    trigger: Optional[str],
    enabled: bool,
    cb_ip: str,
    requested_type: Optional[str] = None,
) -> Tuple[str, html.Div, str, str, dict]:
    """Handle gripper enable/test flow."""
    if trigger is None:
        return no_update, no_update, no_update, no_update, no_update

    if not enabled and trigger == "gripper-enable":
        codroid_picker_module.SESSION.gripper = None
        codroid_picker_module.SESSION.gripper_device = None
        codroid_picker_module.SESSION.gripper_adapter = None
        codroid_picker_module.SESSION.robot.gripper_key = None
        codroid_picker_module.SESSION.robot.gripper_display_name = None
        codroid_picker_module.SESSION.robot.gripper_product_code = None
        codroid_picker_module.SESSION.robot.gripper_dimensions = None
        with contextlib.suppress(Exception):
            registry().set_gripper(None)
        return (
            t("Gripper disabled"),
            html.Div(),
            t("Gripper: off"),
            "gray",
            {"status": "disabled", "message": t("Gripper disabled")},
        )

    if not enabled:
        return no_update, no_update, no_update, no_update, no_update

    try:
        device = codroid_picker_module.GripperDevice(Global_cbip=cb_ip)
        cb = device.getCB()
        if cb is None:
            message = t("Failed to connect to Compute Box at {ip}", ip=cb_ip)
            return (
                message,
                html.Div(
                    dmc.Alert(
                        t("Cannot reach the gripper control box. Check power and network."),
                        title=t("Gripper Unavailable"),
                        color="yellow",
                    )
                ),
                t("Gripper: error"),
                "yellow",
                {"status": "error", "message": message},
            )

        t_index = codroid_picker_module.GRIPPER_DEFAULTS["t_index"]
        detection = detect_or_create_gripper(
            device,
            requested_key=requested_type or codroid_picker_module.SETTINGS.gripper_type,
            t_index=t_index,
        )

        codroid_picker_module.SESSION.gripper_device = device
        codroid_picker_module.SESSION.gripper = detection.adapter.client
        codroid_picker_module.SESSION.gripper_adapter = detection.adapter
        codroid_picker_module.SESSION.robot.gripper_key = detection.key
        codroid_picker_module.SESSION.robot.gripper_display_name = detection.display_name
        codroid_picker_module.SESSION.robot.gripper_product_code = detection.product_code
        codroid_picker_module.SESSION.robot.gripper_dimensions = detection.dimensions
        with contextlib.suppress(Exception):
            registry().set_gripper(detection.adapter.client)

        detection.adapter.start_status_stream(t_index=t_index)
        telemetry = detection.adapter.read_telemetry(t_index=t_index)
        codroid_picker_module.SESSION.robot_has_object = bool(telemetry.get("gripped"))
        if isinstance(telemetry.get("min_width"), (int, float)):
            codroid_picker_module.SESSION.gripper_min_width = float(telemetry["min_width"])
        if isinstance(telemetry.get("max_width"), (int, float)):
            codroid_picker_module.SESSION.gripper_max_width = float(telemetry["max_width"])

        status_text = t(
            "{gripper} connected at {ip}",
            gripper=detection.display_name,
            ip=cb_ip,
        )
        state = {
            "status": "online",
            "message": status_text,
            **detection.adapter.to_state(),
        }

        return (
            status_text,
            _build_gripper_info_card(
                detection.display_name,
                telemetry,
                detection.product_code,
            ),
            t("Gripper: online"),
            "brand",
            state,
        )

    except (
        OnRobotConnectionError,
        OnRobotTimeoutError,
        OnRobotValidationError,
        RuntimeError,
        ValueError,
        TypeError,
    ) as exc:
        codroid_picker_module.LOGGER.warning("Gripper connection failed: %s", exc)
        codroid_picker_module.SESSION.gripper = None
        codroid_picker_module.SESSION.gripper_adapter = None
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


def _register_connection_status_callback(app: Dash) -> None:
    """Register robot connect/release callback."""

    @app.callback(
        Output("robot-status-text", "children"),
        Output("header-robot-badge", "children"),
        Output("header-robot-badge", "color"),
        Output("robot-connection-state", "data"),
        Output("robot-control-state", "data"),
        Input("robot-test", "n_clicks"),
        Input("robot-acquire-control", "n_clicks"),
        Input("robot-release-control", "n_clicks"),
        Input("auto-init", "n_intervals"),
        Input("monitor-interval", "n_intervals"),
        State("robot-uri", "value"),
        State("robot-control-state", "data"),
        prevent_initial_call=True,
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

        return await _handle_robot_connection_event(
            codroid_picker_module,
            trigger=callback_context.triggered_id,
            uri=uri,
            control_state=control_state,
        )


def _register_gripper_callback(app: Dash) -> None:
    """Register gripper test/enable callback."""

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
        State("gripper-type", "value"),
        prevent_initial_call=True,
    )
    def _handle_gripper(
        _test_clicks: int,
        enabled: bool,
        _init: int,
        cb_ip: str,
        requested_type: Optional[str],
    ) -> Tuple[str, html.Div, str, str, dict]:
        import app.codroid_picker as codroid_picker_module

        return _handle_gripper_event(
            codroid_picker_module,
            trigger=callback_context.triggered_id,
            enabled=enabled,
            cb_ip=cb_ip,
            requested_type=requested_type,
        )


def _register_power_callbacks(app: Dash) -> None:
    """Register robot power toggle and sync callbacks."""

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

            await codroid_picker_module._with_robot_api(
                robot_uri or codroid_picker_module.DEFAULT_ROBOT_URI,
                _apply_power,
            )
            action = t("Power on") if checked else t("Power off")
            codroid_picker_module.SESSION.last_action = t(
                "{action} command sent.",
                action=action,
            )
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
        import app.codroid_picker as codroid_picker_module

        if codroid_picker_module._robot_control_released(control_state):
            return no_update
        if (robot_state or {}).get("status") != "online":
            return no_update

        now = time.time()
        last_toggle = getattr(
            codroid_picker_module.SESSION,
            "robot_power_toggle_time",
            0,
        )
        if now - last_toggle < 3.0:
            return no_update

        session = getattr(codroid_picker_module.SESSION, "robot_session", None)
        if session is None:
            return no_update
        snapshot = session.snapshot()
        power_on = session.robot_power_on()
        status_snapshot = snapshot.robot_status

        if power_on is None and isinstance(status_snapshot, dict):
            state_value = status_snapshot.get("state")
            if isinstance(state_value, (int, float)):
                power_on = state_value >= 2

        if power_on is None:
            status_age = snapshot.robot_status_age_seconds()
            if status_age is None or status_age > 5.0:
                return no_update

        desired = {"value": bool(power_on), "timestamp": time.time()}
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


def register_connection_callbacks(app: Dash) -> None:
    """Register robot connection, gripper, and power management callbacks."""
    _register_connection_status_callback(app)
    _register_gripper_callback(app)
    _register_power_callbacks(app)
