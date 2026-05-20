"""Robot operation monitoring callbacks.

This module handles robot operation state synchronization and control:
- Operation state monitoring and snapshots
- Program controls (pause/resume/stop/reset)
- Operation status updates

Uses the new programs registry when available, falls back to legacy
state module for backwards compatibility.
"""

from typing import Optional, Tuple

import dash_mantine_components as dmc
from dash import callback_context, Dash, Input, no_update, Output, State
from app.codroid_picker.i18n import t

# Import from new programs module for registry access
from app.codroid_picker.programs import (
    get_registry,
    pause_program,
    resume_program,
    stop_program,
)
from app.codroid_picker.programs.auto_mode import (
    AUTO_CONTROL_PROGRAMS,
    pause_auto_program_manual_mode,
    restore_active_auto_program_manual_mode,
    restore_manual_mode,
    resume_auto_program_auto_mode,
)


def register_monitoring_callbacks(app: Dash) -> None:
    """Register robot operation monitoring callbacks."""

    @app.callback(
        Output("robot-operation-state", "data"),
        Input("monitor-interval", "n_intervals"),
        State("robot-operation-state", "data"),
        prevent_initial_call=True,
    )
    def _sync_operation_state(
        _ticks: int,
        current_state: Optional[dict],
    ) -> dict:
        import app.codroid_picker as codroid_picker_module

        snapshot = codroid_picker_module._operation_snapshot()
        if current_state == snapshot:
            return no_update
        return snapshot

    @app.callback(
        Output("program-pause", "children"),
        Output("program-pause", "disabled"),
        Output("program-stop", "disabled"),
        Output("program-reset", "disabled"),
        Output("program-status-text", "children"),
        Input("robot-operation-state", "data"),
    )
    def _update_program_controls(
        operation_state: Optional[dict],
    ) -> Tuple[str, bool, bool, bool, str]:
        state = operation_state or {}
        status = state.get("status", "idle")
        pause_reason = state.get("pause_reason")
        label = state.get("label") or t("Program")
        message = state.get("message") or ""
        translated_message = t(message) if message else ""

        if status == "running":
            pause_label = t("Pause")
            pause_disabled = False
            stop_disabled = False
            reset_disabled = True
            status_text = translated_message or t("{label} running.", label=label)
        elif status == "paused":
            pause_label = t("Resume")
            pause_disabled = False
            stop_disabled = False
            reset_disabled = False
            if pause_reason:
                status_text = t("Paused ({reason}).", reason=pause_reason.replace("_", " "))
            else:
                status_text = translated_message or t("{label} paused.", label=label)
        elif status in {"stopping", "stopped"}:
            pause_label = t("Pause")
            pause_disabled = True
            stop_disabled = True
            reset_disabled = False
            status_text = translated_message or t("Program stopped.")
        else:
            pause_label = t("Pause")
            pause_disabled = True
            stop_disabled = True
            reset_disabled = True
            status_text = translated_message or t("Program idle.")

        return pause_label, pause_disabled, stop_disabled, reset_disabled, status_text

    @app.callback(
        Output("app-notifications", "children", allow_duplicate=True),
        Input("program-pause", "n_clicks"),
        Input("program-stop", "n_clicks"),
        Input("program-reset", "n_clicks"),
        State("robot-operation-state", "data"),
        State("robot-uri", "value"),
        State("robot-control-state", "data"),
        prevent_initial_call=True,
    )
    async def _handle_program_controls(
        _pause_clicks: int,
        _stop_clicks: int,
        _reset_clicks: int,
        operation_state: Optional[dict],
        robot_uri: Optional[str],
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
        trigger = callback_context.triggered_id
        status = (operation_state or {}).get("status", "idle")

        # Try to use the new registry first, fall back to legacy state
        registry = get_registry()

        if trigger == "program-pause":
            if status == "running":
                # Use registry if active, otherwise legacy
                active_program = registry.active_program
                if registry.active_program is not None:
                    pause_program("operator")
                else:
                    codroid_picker_module._pause_operation(
                        "operator", t("Program paused by operator."))

                async def _stop_motion(api: codroid_picker_module.CodroidAPI) -> None:
                    await api.stop_command()

                try:
                    auto_paused = await pause_auto_program_manual_mode(
                        active_program,
                        logger=codroid_picker_module.LOGGER,
                        context="Program pause",
                    )
                    if not auto_paused:
                        await codroid_picker_module._with_robot_api(
                            robot_uri or codroid_picker_module.DEFAULT_ROBOT_URI,
                            _stop_motion,
                        )
                except Exception as exc:  # noqa: BLE001
                    codroid_picker_module.LOGGER.debug(
                        "Pause stop command failed: %s", exc)
                return dmc.Notification(
                    action="show",
                    title=t("Program paused"),
                    message=t("Robot program paused."),
                    color="yellow",
                    autoClose=4000,
                )
            if status == "paused":
                # Use registry if active, otherwise legacy
                active_program = registry.active_program
                if active_program is not None:
                    await resume_auto_program_auto_mode(
                        active_program,
                        logger=codroid_picker_module.LOGGER,
                        context="Program resume",
                    )
                    resume_program()
                else:
                    codroid_picker_module._resume_operation(t("Program resumed."))
                return dmc.Notification(
                    action="show",
                    title=t("Program resumed"),
                    message=t("Robot program resumed."),
                    color="teal",
                    autoClose=4000,
                )
        elif trigger == "program-stop":
            # Use registry if active, otherwise legacy
            active_program = registry.active_program
            source = (operation_state or {}).get("source", "")
            if active_program is not None:
                stop_program()
            else:
                codroid_picker_module._request_stop_operation(
                    t("Program stop requested."))

            async def _stop_motion(api: codroid_picker_module.CodroidAPI) -> None:
                if source in AUTO_CONTROL_PROGRAMS:
                    await restore_manual_mode(
                        api,
                        logger=codroid_picker_module.LOGGER,
                        context="Program stop",
                    )
                else:
                    await api.stop_command()

            try:
                auto_restored = await restore_active_auto_program_manual_mode(
                    active_program,
                    logger=codroid_picker_module.LOGGER,
                    context="Program stop",
                )
                if not auto_restored:
                    await codroid_picker_module._with_robot_api(
                        robot_uri or codroid_picker_module.DEFAULT_ROBOT_URI,
                        _stop_motion,
                    )
            except Exception as exc:  # noqa: BLE001
                codroid_picker_module.LOGGER.debug(
                    "Stop command failed: %s", exc)
            return dmc.Notification(
                action="show",
                title=t("Program stopping"),
                message=t("Stop command sent. Reset when safe."),
                color="red",
                autoClose=4500,
            )
        elif trigger == "program-reset":
            # Clear registry if active
            if registry.active_program is not None:
                registry.clear()
            codroid_picker_module._reset_operation_state(t("Program reset."))
            return dmc.Notification(
                action="show",
                title=t("Program reset"),
                message=t("Program state cleared."),
                color="gray",
                autoClose=3500,
            )

        return no_update
