"""Gripper synchronization and orientation callbacks."""

from typing import Optional, Tuple

from dash import Dash, Input, no_update, Output, State
from app.codroid_picker.i18n import t


def register_gripper_callbacks(app: Dash) -> None:
    """Register gripper-related callbacks."""


    @app.callback(
        Output("footer-gripper-width", "value", allow_duplicate=True),
        Output("footer-gripper-force", "value", allow_duplicate=True),
        Output("footer-gripper-width", "min", allow_duplicate=True),
        Output("footer-gripper-width", "max", allow_duplicate=True),
        Output("footer-gripper-meta", "children"),
        Output("gripper-orientation", "value"),
        Input("robot-status-interval", "n_intervals"),
        State("gripper-connection-state", "data"),
        State("footer-gripper-width", "value"),
        State("footer-gripper-force", "value"),
        prevent_initial_call=True,
    )
    def _sync_footer_gripper_values(
        _intervals: int,
        gripper_state: Optional[dict],
        current_width: Optional[float],
        current_force: Optional[float],
    ) -> Tuple[
        Optional[float],
        Optional[float],
        Optional[float],
        Optional[float],
        str,
        Optional[str],
    ]:
        import app.codroid_picker as codroid_picker_module

        if (gripper_state or {}).get("status") != "online":
            return no_update, no_update, no_update, no_update, no_update, no_update
        if codroid_picker_module.SESSION.gripper is None:
            return no_update, no_update, no_update, no_update, no_update, no_update

        status_snapshot = None
        try:
            t_index = codroid_picker_module.GRIPPER_DEFAULTS["t_index"]
            if codroid_picker_module.SESSION.gripper.get_status_snapshot(t_index=t_index) is None:
                codroid_picker_module.SESSION.gripper.start_status_stream()
            status_snapshot = codroid_picker_module.SESSION.gripper.get_status_snapshot(
                t_index=t_index
            )
            if status_snapshot:
                width = status_snapshot.get("external_width")
                force = status_snapshot.get("force")
                min_width = status_snapshot.get("min_external_width")
                max_width = status_snapshot.get("max_external_width")
                busy = status_snapshot.get("busy")
                gripped = status_snapshot.get("grip_detected")
                status = status_snapshot.get("status")
                outward = status_snapshot.get("finger_orientation_outward")
                orientation = None
                if isinstance(outward, bool):
                    orientation = "outward" if outward else "inward"
                elif isinstance(outward, str):
                    cleaned = outward.strip().lower()
                    if cleaned in {"true", "1", "yes", "on"}:
                        orientation = "outward"
                    elif cleaned in {"false", "0", "no", "off"}:
                        orientation = "inward"
            else:
                width = codroid_picker_module.SESSION.gripper.get_ext_width(
                    t_index=t_index)
                force = codroid_picker_module.SESSION.gripper.get_force(
                    t_index=t_index)
                min_width = codroid_picker_module.SESSION.gripper.get_min_ext_width(
                    t_index=t_index)
                max_width = codroid_picker_module.SESSION.gripper.get_max_ext_width(
                    t_index=t_index)
                busy = codroid_picker_module.SESSION.gripper.isBusy(
                    t_index=t_index)
                gripped = codroid_picker_module.SESSION.gripper.isGripped(
                    t_index=t_index)
                status = codroid_picker_module.SESSION.gripper.getStatus(
                    t_index=t_index)
                orientation = codroid_picker_module.SESSION.gripper.get_finger_orientation_label(
                    t_index=t_index
                )
        except Exception as exc:  # noqa: BLE001
            codroid_picker_module.LOGGER.debug(
                "Failed to poll gripper values: %s", exc)
            return no_update, no_update, no_update, no_update, no_update, no_update

        width_update: Optional[float] = no_update
        force_update: Optional[float] = no_update
        if isinstance(width, (int, float)) and width > -2:
            if current_width is None or abs(width - current_width) > 0.05:
                width_update = float(width)
        if (
            isinstance(force, (int, float))
            and codroid_picker_module.GRIPPER_FORCE_MIN <= force <= codroid_picker_module.GRIPPER_FORCE_MAX
        ):
            if current_force is None or abs(force - current_force) > 0.1:
                force_update = float(force)

        if (
            isinstance(force, (int, float))
            and force < 0
            and status_snapshot
            and isinstance(status_snapshot.get("max_force"), (int, float))
        ):
            force = None

        min_update: Optional[float] = no_update
        max_update: Optional[float] = no_update
        if isinstance(min_width, (int, float)) and min_width >= 0:
            if (
                codroid_picker_module.SESSION.gripper_min_width is None
                or abs(min_width - codroid_picker_module.SESSION.gripper_min_width) > 0.1
            ):
                min_update = float(min_width)
                codroid_picker_module.SESSION.gripper_min_width = float(
                    min_width)
        if isinstance(max_width, (int, float)) and max_width >= 0:
            if (
                codroid_picker_module.SESSION.gripper_max_width is None
                or abs(max_width - codroid_picker_module.SESSION.gripper_max_width) > 0.1
            ):
                max_update = float(max_width)
                codroid_picker_module.SESSION.gripper_max_width = float(
                    max_width)

        width_value = None if width_update is no_update else float(
            width_update)
        force_value = None if force_update is no_update else float(
            force_update)
        codroid_picker_module._record_gripper_sync(width_value, force_value)

        if isinstance(force, (int, float)) and force >= 0:
            codroid_picker_module.SESSION.gripper_force = float(force)
        if isinstance(busy, (bool, int, float)) and busy != -2:
            codroid_picker_module.SESSION.gripper_busy = bool(busy)
        if isinstance(gripped, (bool, int, float)) and gripped != -2:
            codroid_picker_module.SESSION.gripper_grip_detected = bool(gripped)
        if isinstance(status, (int, float)) and status != -2:
            codroid_picker_module.SESSION.gripper_status_code = int(status)
        previous_orientation = codroid_picker_module.SESSION.gripper_orientation
        if orientation in {"inward", "outward"}:
            codroid_picker_module.SESSION.gripper_orientation = orientation

        busy_text = (
            "unknown"
            if codroid_picker_module.SESSION.gripper_busy is None
            else ("yes" if codroid_picker_module.SESSION.gripper_busy else "no")
        )
        grip_text = (
            "unknown"
            if codroid_picker_module.SESSION.gripper_grip_detected is None
            else ("yes" if codroid_picker_module.SESSION.gripper_grip_detected else "no")
        )
        meta_text = (
            t(
                "Orientation: {orientation} · Busy: {busy} · Grip: {grip}",
                orientation=codroid_picker_module.SESSION.gripper_orientation or t("unknown"),
                busy=busy_text,
                grip=grip_text,
            )
        )
        if isinstance(force, (int, float)) and force >= 0:
            meta_text = t("{meta} · Force: {force:.0f} N", meta=meta_text, force=float(force))

        orientation_update: Optional[str] = no_update
        if orientation in {"inward", "outward"} and orientation != previous_orientation:
            orientation_update = orientation

        width_min = (
            codroid_picker_module.SESSION.gripper_min_width
            if codroid_picker_module.SESSION.gripper_min_width is not None
            else None
        )
        width_max = (
            codroid_picker_module.SESSION.gripper_max_width
            if codroid_picker_module.SESSION.gripper_max_width is not None
            else None
        )
        if width_update is no_update and width_min is not None and width_max is not None:
            if isinstance(current_width, (int, float)):
                clamped = min(max(current_width, width_min), width_max)
                if abs(clamped - current_width) > 0.05:
                    width_update = float(clamped)

        return (
            width_update,
            force_update,
            min_update,
            max_update,
            meta_text,
            orientation_update,
        )

    @app.callback(
        Output("gripper-orientation-status", "children"),
        Output("footer-gripper-width", "min", allow_duplicate=True),
        Output("footer-gripper-width", "max", allow_duplicate=True),
        Input("gripper-orientation", "value"),
        State("gripper-connection-state", "data"),
        prevent_initial_call=True,
    )
    def _apply_gripper_orientation(
        orientation: Optional[str],
        gripper_state: Optional[dict],
    ) -> Tuple[str, Optional[float], Optional[float]]:
        import app.codroid_picker as codroid_picker_module

        if (gripper_state or {}).get("status") != "online":
            return t("Gripper is offline."), no_update, no_update
        if codroid_picker_module.SESSION.gripper is None:
            return t("Gripper is unavailable."), no_update, no_update
        if orientation not in {"inward", "outward"}:
            return no_update, no_update, no_update
        if codroid_picker_module.SESSION.gripper_orientation == orientation:
            return no_update, no_update, no_update
        try:
            t_index = codroid_picker_module.GRIPPER_DEFAULTS["t_index"]
            result = codroid_picker_module.SESSION.gripper.set_finger_orientation(
                t_index=t_index,
                orientation=orientation,
            )
            if isinstance(result, (int, float)) and result < 0:
                return t("Failed to update gripper orientation."), no_update, no_update
            codroid_picker_module.SESSION.gripper_orientation = orientation
            codroid_picker_module.SESSION.gripper_min_width = None
            codroid_picker_module.SESSION.gripper_max_width = None
            min_width = codroid_picker_module.SESSION.gripper.get_min_ext_width(
                t_index=t_index)
            max_width = codroid_picker_module.SESSION.gripper.get_max_ext_width(
                t_index=t_index)
            min_update = (
                float(min_width)
                if isinstance(min_width, (int, float)) and min_width >= 0
                else no_update
            )
            max_update = (
                float(max_width)
                if isinstance(max_width, (int, float)) and max_width >= 0
                else no_update
            )
            if min_update is not no_update:
                codroid_picker_module.SESSION.gripper_min_width = min_update
            if max_update is not no_update:
                codroid_picker_module.SESSION.gripper_max_width = max_update
            return t("Orientation set to {orientation}.", orientation=orientation), min_update, max_update
        except Exception as exc:  # noqa: BLE001
            codroid_picker_module.LOGGER.exception(
                "Failed to set gripper orientation")
            return t("Failed to set orientation: {error}", error=exc), no_update, no_update

    @app.callback(
        Output("holding-status", "children"),
        Input("robot-status-interval", "n_intervals"),
        State("gripper-connection-state", "data"),
        prevent_initial_call=True,
    )
    def _update_holding_badge(
        _intervals: int, gripper_state: Optional[dict]
    ) -> str:
        import app.codroid_picker as codroid_picker_module

        status = (gripper_state or {}).get("status", "offline")
        if status != "online" or codroid_picker_module.SESSION.gripper is None:
            return t("Holding object: unknown (gripper offline)")
        try:
            t_index = codroid_picker_module.GRIPPER_DEFAULTS["t_index"]
            is_gripped = bool(
                codroid_picker_module.SESSION.gripper.isGripped(t_index=t_index))
            codroid_picker_module.SESSION.robot_has_object = is_gripped
            return t("Holding object: {state}", state=(t("yes") if is_gripped else t("no")))
        except Exception as exc:  # noqa: BLE001
            codroid_picker_module.LOGGER.debug(
                "Unable to query gripper grip state: %s", exc)
            return t("Holding object: unknown")

    # --- Footer gripper control callbacks ---

    @app.callback(
        Output("footer-gripper-width", "disabled"),
        Output("footer-gripper-force", "disabled"),
        Output("footer-gripper-speed", "disabled"),
        Output("footer-gripper-width-decrease", "disabled"),
        Output("footer-gripper-width-increase", "disabled"),
        Output("footer-gripper-force-decrease", "disabled"),
        Output("footer-gripper-force-increase", "disabled"),
        Output("footer-gripper-speed-decrease", "disabled"),
        Output("footer-gripper-speed-increase", "disabled"),
        Input("gripper-connection-state", "data"),
    )
    def _sync_footer_gripper_disabled(
        gripper_state: Optional[dict],
    ) -> Tuple[bool, bool, bool, bool, bool, bool, bool, bool, bool]:
        import app.codroid_picker as codroid_picker_module

        status = (gripper_state or {}).get("status", "offline")
        disabled = status != "online" or bool(
            codroid_picker_module.SESSION.gripper_busy)
        return (
            disabled,
            disabled,
            disabled,
            disabled,
            disabled,
            disabled,
            disabled,
            disabled,
            disabled,
        )

    @app.callback(
        Output("footer-gripper-width", "min", allow_duplicate=True),
        Output("footer-gripper-width", "max", allow_duplicate=True),
        Input("gripper-connection-state", "data"),
        prevent_initial_call=True,
    )
    def _sync_gripper_width_limits(gripper_state: Optional[dict]) -> Tuple[float, float]:
        import app.codroid_picker as codroid_picker_module

        if (gripper_state or {}).get("status") != "online":
            return no_update, no_update
        if codroid_picker_module.SESSION.gripper is None:
            return no_update, no_update
        min_width, max_width = codroid_picker_module._resolve_gripper_limits()
        return min_width, max_width

    @app.callback(
        Output("footer-gripper-status", "children"),
        Input("footer-gripper-width", "value"),
        Input("footer-gripper-force", "value"),
        Input("footer-gripper-speed", "value"),
        prevent_initial_call=True,
    )
    def _footer_gripper_actions(
        footer_width: Optional[float],
        footer_force: Optional[float],
        footer_speed: Optional[float],
    ) -> str:
        import app.codroid_picker as codroid_picker_module
        from dash import callback_context

        try:
            if codroid_picker_module.SESSION.gripper_busy:
                return t("Gripper is busy; wait for motion to finish.")
            trigger = callback_context.triggered_id
            if trigger == "footer-gripper-width":
                if codroid_picker_module._recent_gripper_sync("width", footer_width):
                    return no_update
            if trigger == "footer-gripper-force":
                if codroid_picker_module._recent_gripper_sync("force", footer_force):
                    return no_update

            applied_force = codroid_picker_module._safe_int(
                footer_force, codroid_picker_module.GRIPPER_FORCE_DEFAULT
            )
            applied_force = min(
                max(applied_force, codroid_picker_module.GRIPPER_FORCE_MIN),
                codroid_picker_module.GRIPPER_FORCE_MAX,
            )
            applied_speed = codroid_picker_module._safe_int(footer_speed, 80)
            if trigger in {"footer-gripper-width", "footer-gripper-force", "footer-gripper-speed"}:
                width = codroid_picker_module._safe_float(
                    footer_width, codroid_picker_module.GRIPPER_CLOSE_WIDTH_DEFAULT
                )
                return codroid_picker_module._command_gripper(
                    width,
                    "moved",
                    force=applied_force,
                    speed=applied_speed,
                )
            return no_update
        except Exception as exc:  # noqa: BLE001
            codroid_picker_module.LOGGER.exception(
                "Footer gripper action failed")
            return t("Robot action failed: {error}", error=exc)

    @app.callback(
        Output("footer-gripper-width", "value", allow_duplicate=True),
        Output("footer-gripper-force", "value", allow_duplicate=True),
        Output("footer-gripper-speed", "value", allow_duplicate=True),
        Input("footer-gripper-width-decrease", "n_clicks"),
        Input("footer-gripper-width-increase", "n_clicks"),
        Input("footer-gripper-force-decrease", "n_clicks"),
        Input("footer-gripper-force-increase", "n_clicks"),
        Input("footer-gripper-speed-decrease", "n_clicks"),
        Input("footer-gripper-speed-increase", "n_clicks"),
        State("footer-gripper-width", "value"),
        State("footer-gripper-force", "value"),
        State("footer-gripper-speed", "value"),
        State("footer-gripper-width", "min"),
        State("footer-gripper-width", "max"),
        prevent_initial_call=True,
    )
    def _step_footer_gripper_controls(
        _width_dec: int,
        _width_inc: int,
        _force_dec: int,
        _force_inc: int,
        _speed_dec: int,
        _speed_inc: int,
        width: Optional[float],
        force: Optional[float],
        speed: Optional[float],
        width_min: Optional[float],
        width_max: Optional[float],
    ) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        import app.codroid_picker as codroid_picker_module
        from dash import callback_context

        trigger = callback_context.triggered_id
        if trigger is None:
            return no_update, no_update, no_update
        if codroid_picker_module.SESSION.gripper_busy:
            return no_update, no_update, no_update

        width_value = codroid_picker_module._safe_float(
            width, codroid_picker_module.GRIPPER_CLOSE_WIDTH_DEFAULT
        )
        force_value = codroid_picker_module._safe_float(
            force, codroid_picker_module.GRIPPER_FORCE_DEFAULT
        )
        speed_value = codroid_picker_module._safe_float(
            speed, codroid_picker_module.GRIPPER_SPEED_DEFAULT
        )

        if trigger == "footer-gripper-width-decrease":
            width_value -= 0.5
        elif trigger == "footer-gripper-width-increase":
            width_value += 0.5
        elif trigger == "footer-gripper-force-decrease":
            force_value -= 5
        elif trigger == "footer-gripper-force-increase":
            force_value += 5
        elif trigger == "footer-gripper-speed-decrease":
            speed_value -= 5
        elif trigger == "footer-gripper-speed-increase":
            speed_value += 5

        if width_min is not None:
            width_value = max(width_value, float(width_min))
        if width_max is not None:
            width_value = min(width_value, float(width_max))
        force_value = min(
            max(force_value, codroid_picker_module.GRIPPER_FORCE_MIN),
            codroid_picker_module.GRIPPER_FORCE_MAX,
        )
        speed_value = min(max(speed_value, 0), 100)

        return width_value, force_value, speed_value
