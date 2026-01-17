"""Auto operation callbacks for pick & place automation.

This module contains Dash callbacks for:
- Auto operation start/stop buttons
- Prerequisites checking
- Auto operation state synchronization
"""

import asyncio
from typing import Optional, Tuple

import dash_mantine_components as dmc
from dash import Dash, Input, Output, State, callback_context, no_update


def register_auto_operation_callbacks(app: Dash) -> None:
    """Register auto operation callbacks with the Dash app."""

    @app.callback(
        Output("auto-start", "disabled"),
        Output("auto-stop", "disabled"),
        Input("prerequisites-state", "data"),
        Input("auto-operation-state", "data"),
    )
    def _update_auto_buttons(
        prerequisites: Optional[dict],
        auto_state: Optional[dict],
    ) -> Tuple[bool, bool]:
        """Enable/disable auto operation buttons based on prerequisites and operation state."""
        prereqs = prerequisites or {}
        all_ready = all([
            prereqs.get("camera_calibrated", False),
            prereqs.get("workspace_defined", False),
            prereqs.get("robot_calibrated", False),
            prereqs.get("model_loaded", False),
            prereqs.get("gripper_ready", False),
            prereqs.get("drop_target_defined", False),
        ])

        running = (auto_state or {}).get("running", False)

        # Start button: enabled only when all prerequisites met and not running
        start_disabled = not all_ready or running
        # Stop button: enabled only when running
        stop_disabled = not running

        return start_disabled, stop_disabled

    @app.callback(
        Output("auto-operation-state", "data"),
        Output("app-notifications", "children", allow_duplicate=True),
        Input("auto-start", "n_clicks"),
        Input("auto-stop", "n_clicks"),
        State("auto-operation-state", "data"),
        State("robot-uri", "value"),
        prevent_initial_call=True,
    )
    async def _handle_auto_operation(
        start_clicks: int,
        stop_clicks: int,
        current_state: Optional[dict],
        robot_uri: Optional[str],
    ) -> Tuple[dict, dmc.Notification]:
        """Handle auto operation start/stop."""
        import app.codroid_picker as codroid_picker_module

        ctx = callback_context
        if not ctx.triggered:
            return no_update, no_update

        trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]

        if trigger_id == "auto-start":
            if codroid_picker_module.SESSION.auto_task is not None and not codroid_picker_module.SESSION.auto_task.done():
                return (
                    {"running": True, "message": "Auto operation already running"},
                    dmc.Notification(
                        title="Auto Operation",
                        message="Auto operation is already running.",
                        color="yellow",
                        action="show",
                    ),
                )
            try:
                run_id = codroid_picker_module._begin_operation(
                    "auto",
                    "Auto pick & place",
                    message="Auto pick & place running.",
                )
            except RuntimeError as exc:
                return (
                    no_update,
                    dmc.Notification(
                        title="Auto Operation",
                        message=str(exc),
                        color="red",
                        action="show",
                    ),
                )
            codroid_picker_module.SESSION.auto_waiting_at_drop = False
            codroid_picker_module._set_auto_state(
                "Auto pick & place running.", running=True)
            codroid_picker_module.SESSION.auto_task = asyncio.create_task(
                codroid_picker_module._auto_operation_task(run_id, robot_uri)
            )
            return (
                {"running": True, "message": "Auto operation started"},
                dmc.Notification(
                    title="Auto Operation",
                    message="Auto pick & place operation started",
                    color="green",
                    action="show",
                ),
            )
        elif trigger_id == "auto-stop":
            codroid_picker_module._request_stop_operation(
                "Auto stop requested.")
            codroid_picker_module._set_auto_state(
                "Auto stop requested.", running=True)

            async def _stop_motion(api) -> None:
                await api.stop_command()

            try:
                await codroid_picker_module._with_robot_api(
                    robot_uri or codroid_picker_module.DEFAULT_ROBOT_URI,
                    _stop_motion,
                )
            except Exception as exc:  # noqa: BLE001
                codroid_picker_module.LOGGER.debug(
                    "Auto stop command failed: %s", exc)
            return (
                {"running": True, "message": "Auto operation stopping"},
                dmc.Notification(
                    title="Auto Operation",
                    message="Auto pick & place operation stopping",
                    color="orange",
                    action="show",
                ),
            )

        return no_update, no_update

    @app.callback(
        Output("auto-operation-state", "data", allow_duplicate=True),
        Input("monitor-interval", "n_intervals"),
        State("auto-operation-state", "data"),
        prevent_initial_call=True,
    )
    def _sync_auto_operation_state(
        _ticks: int,
        current_state: Optional[dict],
    ) -> dict:
        """Synchronize auto operation state from session snapshot."""
        import app.codroid_picker as codroid_picker_module

        snapshot = codroid_picker_module._auto_state_snapshot()
        if current_state == snapshot:
            return no_update
        return snapshot
