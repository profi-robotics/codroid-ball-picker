"""Auto operation callbacks for pick & place automation.

This module contains Dash callbacks for:
- Auto operation start/stop/pause/resume buttons
- Prerequisites checking
- Auto operation state synchronization

Uses the AutoPickPlaceProgram via the program registry.
"""

import asyncio
from typing import Optional, Tuple

import dash_mantine_components as dmc
from dash import Dash, Input, Output, State, callback_context, no_update

from app.codroid_picker.programs import (
    get_registry,
    AutoPickPlaceProgram,
)
from app.codroid_picker.programs.base import ProgramStatus


def register_auto_operation_callbacks(app: Dash) -> None:
    """Register auto operation callbacks with the Dash app."""

    @app.callback(
        Output("auto-start", "disabled"),
        Output("auto-stop", "disabled"),
        Output("auto-pause", "disabled"),
        Output("auto-resume", "disabled"),
        Input("prerequisites-state", "data"),
        Input("auto-operation-state", "data"),
    )
    def _update_auto_buttons(
        prerequisites: Optional[dict],
        auto_state: Optional[dict],
    ) -> Tuple[bool, bool, bool, bool]:
        """Enable/disable auto operation buttons based on prerequisites and operation state."""
        prereqs = prerequisites or {}
        all_ready = all([
            prereqs.get("camera_calibrated", False),
            prereqs.get("workspace_defined", False),
            prereqs.get("robot_calibrated", False),
            prereqs.get("robot_connected", False),
            prereqs.get("model_loaded", False),
            prereqs.get("gripper_ready", False),
            prereqs.get("drop_target_defined", False),
        ])

        state_data = auto_state or {}
        running = state_data.get("running", False)
        paused = state_data.get("paused", False)

        # Start button: enabled only when all prerequisites met and not running
        start_disabled = not all_ready or running
        # Stop button: enabled only when running
        stop_disabled = not running
        # Pause button: enabled when running and not paused
        pause_disabled = not running or paused
        # Resume button: enabled when paused
        resume_disabled = not paused

        return start_disabled, stop_disabled, pause_disabled, resume_disabled

    @app.callback(
        Output("auto-operation-state", "data"),
        Output("app-notifications", "children", allow_duplicate=True),
        Input("auto-start", "n_clicks"),
        Input("auto-stop", "n_clicks"),
        Input("auto-pause", "n_clicks"),
        Input("auto-resume", "n_clicks"),
        State("auto-operation-state", "data"),
        State("robot-uri", "value"),
        prevent_initial_call=True,
    )
    async def _handle_auto_operation(
        start_clicks: int,
        stop_clicks: int,
        pause_clicks: int,
        resume_clicks: int,
        current_state: Optional[dict],
        robot_uri: Optional[str],
    ) -> Tuple[dict, dmc.Notification]:
        """Handle auto operation start/stop/pause/resume."""
        import app.codroid_picker as codroid_picker_module

        ctx = callback_context
        if not ctx.triggered:
            return no_update, no_update

        trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]
        registry = get_registry()
        SESSION = codroid_picker_module.SESSION
        LOGGER = codroid_picker_module.LOGGER
        DEFAULT_ROBOT_URI = codroid_picker_module.DEFAULT_ROBOT_URI

        if trigger_id == "auto-start":
            # Check if already running
            if registry.is_program_running:
                return (
                    {"running": True, "paused": False, "message": "Program already running"},
                    dmc.Notification(
                        title="Auto Operation",
                        message=f"A program is already running: {registry.active_program_name}",
                        color="yellow",
                        action="show",
                    ),
                )

            # Create program instance
            program = AutoPickPlaceProgram()

            # Check prerequisites
            ok, message = program.check_prerequisites()
            if not ok:
                return (
                    no_update,
                    dmc.Notification(
                        title="Auto Operation",
                        message=message,
                        color="red",
                        action="show",
                    ),
                )

            # Get robot API and start program
            try:
                robot_api = await codroid_picker_module._get_robot_api(
                    robot_uri or DEFAULT_ROBOT_URI
                )

                # Start program as background task
                async def _run_auto():
                    try:
                        result = await registry.start_instance(program, robot_api)
                        LOGGER.info(f"Auto operation completed: {result.message}")
                    except Exception as exc:
                        LOGGER.exception("Auto operation failed")

                SESSION.auto_task = asyncio.create_task(_run_auto())

                return (
                    {"running": True, "paused": False, "message": "Auto operation started"},
                    dmc.Notification(
                        title="Auto Operation",
                        message="Auto pick & place operation started",
                        color="green",
                        action="show",
                    ),
                )
            except Exception as exc:
                LOGGER.exception("Failed to start auto operation")
                return (
                    no_update,
                    dmc.Notification(
                        title="Auto Operation",
                        message=f"Failed to start: {exc}",
                        color="red",
                        action="show",
                    ),
                )

        elif trigger_id == "auto-stop":
            if registry.stop():
                return (
                    {"running": False, "paused": False, "message": "Auto operation stopped"},
                    dmc.Notification(
                        title="Auto Operation",
                        message="Auto pick & place operation stopped",
                        color="orange",
                        action="show",
                    ),
                )
            return no_update, no_update

        elif trigger_id == "auto-pause":
            if registry.pause("user_request"):
                return (
                    {"running": True, "paused": True, "message": "Auto operation paused"},
                    dmc.Notification(
                        title="Auto Operation",
                        message="Auto pick & place operation paused",
                        color="blue",
                        action="show",
                    ),
                )
            return no_update, no_update

        elif trigger_id == "auto-resume":
            if registry.resume():
                return (
                    {"running": True, "paused": False, "message": "Auto operation resumed"},
                    dmc.Notification(
                        title="Auto Operation",
                        message="Auto pick & place operation resumed",
                        color="green",
                        action="show",
                    ),
                )
            return no_update, no_update

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
        """Synchronize auto operation state from program registry."""
        registry = get_registry()

        # Build state from registry
        active = registry.active_program
        if active is None or active.name != "auto_pick_place":
            snapshot = {"running": False, "paused": False, "message": ""}
        else:
            snapshot = {
                "running": active.is_running,
                "paused": active.status == ProgramStatus.PAUSED,
                "message": active.message or "",
            }

        if current_state == snapshot:
            return no_update
        return snapshot
