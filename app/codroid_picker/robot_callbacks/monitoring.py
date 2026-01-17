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

# Import from new programs module for registry access
from app.codroid_picker.programs import (
    get_registry,
    pause_program,
    resume_program,
    stop_program,
)


def register_monitoring_callbacks(app: Dash) -> None:
    """Register robot operation monitoring callbacks."""

    @app.callback(
        Output("robot-operation-state", "data"),
        Input("monitor-interval", "n_intervals"),
        State("robot-operation-state", "data"),
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
        label = state.get("label") or "Program"
        message = state.get("message") or ""

        if status == "running":
            pause_label = "Pause"
            pause_disabled = False
            stop_disabled = False
            reset_disabled = True
            status_text = message or f"{label} running."
        elif status == "paused":
            pause_label = "Resume"
            pause_disabled = False
            stop_disabled = False
            reset_disabled = False
            if pause_reason:
                status_text = f"Paused ({pause_reason.replace('_', ' ')})."
            else:
                status_text = message or f"{label} paused."
        elif status in {"stopping", "stopped"}:
            pause_label = "Pause"
            pause_disabled = True
            stop_disabled = True
            reset_disabled = False
            status_text = message or "Program stopped."
        else:
            pause_label = "Pause"
            pause_disabled = True
            stop_disabled = True
            reset_disabled = True
            status_text = message or "Program idle."

        return pause_label, pause_disabled, stop_disabled, reset_disabled, status_text

    @app.callback(
        Output("app-notifications", "children", allow_duplicate=True),
        Input("program-pause", "n_clicks"),
        Input("program-stop", "n_clicks"),
        Input("program-reset", "n_clicks"),
        State("robot-operation-state", "data"),
        State("robot-uri", "value"),
        prevent_initial_call=True,
    )
    async def _handle_program_controls(
        _pause_clicks: int,
        _stop_clicks: int,
        _reset_clicks: int,
        operation_state: Optional[dict],
        robot_uri: Optional[str],
    ) -> dmc.Notification:
        import app.codroid_picker as codroid_picker_module

        trigger = callback_context.triggered_id
        status = (operation_state or {}).get("status", "idle")

        # Try to use the new registry first, fall back to legacy state
        registry = get_registry()

        if trigger == "program-pause":
            if status == "running":
                # Use registry if active, otherwise legacy
                if registry.active_program is not None:
                    pause_program("operator")
                else:
                    codroid_picker_module._pause_operation(
                        "operator", "Program paused by operator.")

                async def _stop_motion(api: codroid_picker_module.CodroidAPI) -> None:
                    await api.stop_command()

                try:
                    await codroid_picker_module._with_robot_api(
                        robot_uri or codroid_picker_module.DEFAULT_ROBOT_URI, _stop_motion
                    )
                except Exception as exc:  # noqa: BLE001
                    codroid_picker_module.LOGGER.debug(
                        "Pause stop command failed: %s", exc)
                return dmc.Notification(
                    action="show",
                    title="Program paused",
                    message="Robot program paused.",
                    color="yellow",
                    autoClose=4000,
                )
            if status == "paused":
                # Use registry if active, otherwise legacy
                if registry.active_program is not None:
                    resume_program()
                else:
                    codroid_picker_module._resume_operation("Program resumed.")
                return dmc.Notification(
                    action="show",
                    title="Program resumed",
                    message="Robot program resumed.",
                    color="teal",
                    autoClose=4000,
                )
        elif trigger == "program-stop":
            # Use registry if active, otherwise legacy
            if registry.active_program is not None:
                stop_program()
            else:
                codroid_picker_module._request_stop_operation(
                    "Program stop requested.")

            async def _stop_motion(api: codroid_picker_module.CodroidAPI) -> None:
                await api.stop_command()

            try:
                await codroid_picker_module._with_robot_api(
                    robot_uri or codroid_picker_module.DEFAULT_ROBOT_URI, _stop_motion
                )
            except Exception as exc:  # noqa: BLE001
                codroid_picker_module.LOGGER.debug(
                    "Stop command failed: %s", exc)
            return dmc.Notification(
                action="show",
                title="Program stopping",
                message="Stop command sent. Reset when safe.",
                color="red",
                autoClose=4500,
            )
        elif trigger == "program-reset":
            # Clear registry if active
            if registry.active_program is not None:
                registry.clear()
            codroid_picker_module._reset_operation_state("Program reset.")
            return dmc.Notification(
                action="show",
                title="Program reset",
                message="Program state cleared.",
                color="gray",
                autoClose=3500,
            )

        return no_update
