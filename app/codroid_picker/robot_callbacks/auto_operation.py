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
from app.codroid_picker.i18n import t

from app.codroid_picker.programs import (
    get_registry,
    AutoPickPlaceProgram,
)
from app.codroid_picker.programs.base import ProgramStatus
from app.codroid_picker.app_instance import registry


def register_auto_operation_callbacks(app: Dash) -> None:
    """Register auto operation callbacks with the Dash app."""
    import logging
    LOGGER = logging.getLogger(__name__)

    # Guard against duplicate registration
    if hasattr(app, '_auto_operation_callbacks_registered'):
        LOGGER.info("Auto operation callbacks already registered, skipping")
        return
    app._auto_operation_callbacks_registered = True

    LOGGER.info("Registering auto operation callbacks")

    prereq_labels = [
        ("camera_live", "Camera live"),
        ("camera_calibrated", "Camera calib"),
        ("workspace_defined", "Workspace"),
        ("robot_calibrated", "Robot calib"),
        ("robot_connected", "Robot online"),
        ("model_loaded", "Model"),
        ("gripper_ready", "Gripper"),
        ("drop_target_defined", "Drop target"),
    ]

    @app.callback(
        Output("auto-start", "disabled"),
        Output("auto-stop", "disabled"),
        Output("auto-prereq-badges", "children"),
        Input("prerequisites-state", "data"),
        Input("auto-operation-state", "data"),
        Input("model-status", "children"),
        Input("workspace-action-status", "children"),
        Input("camera-connection-state", "data"),
        prevent_initial_call=False,
    )
    def _update_auto_buttons(
        prerequisites: Optional[dict],
        auto_state: Optional[dict],
        _model_status: Optional[str],
        _workspace_status: Optional[str],
        camera_state: Optional[dict],
    ) -> Tuple[bool, bool, list]:
        """Enable/disable auto operation buttons and surface unmet prerequisites."""
        session = registry().session

        prereqs = dict(prerequisites or {})
        prereqs["camera_live"] = (camera_state or {}).get("status") == "live"
        prereqs["camera_calibrated"] = session.calibration_bundle is not None
        prereqs["workspace_defined"] = session.workspace_definition is not None
        prereqs["robot_calibrated"] = session.robot_calibration_min_z is not None
        prereqs["model_loaded"] = session.model_loaded
        prereqs["drop_target_defined"] = session.drop_target_mm is not None

        # Build badge items
        translated_labels = [(key, t(label)) for key, label in prereq_labels]

        badge_items = [
            dmc.Badge(
                label,
                color="green" if prereqs.get(key, False) else "red",
                variant="light",
                size="xs",
                radius="sm",
            )
            for key, label in translated_labels
        ]

        # Check if all prerequisites are met
        all_ready = all([
            prereqs.get("camera_live", False),
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

        # Start button: enabled only when all prerequisites met and not running
        start_disabled = not all_ready or running
        # Stop button: enabled only when running
        stop_disabled = not running

        prereq_children = [
            dmc.Group(
                badge_items,
                gap=6,
                align="center",
                wrap="wrap",
                justify="center",
            ),
        ]

        return start_disabled, stop_disabled, prereq_children

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
        """Handle auto operation start/stop.

        Note: Pause/resume is now handled by the program control buttons.
        """
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
                    {"running": True, "paused": False,
                        "message": t("Program already running")},
                    dmc.Notification(
                        title=t("Auto Operation"),
                        message=t("A program is already running: {name}", name=registry.active_program_name),
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
                        title=t("Auto Operation"),
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

                # Start program using direct await like plate test does
                # The callback will block until the program completes or is stopped
                # This is the pattern that works reliably
                result = await registry.start_instance(program, robot_api)
                LOGGER.info("Auto operation completed: %s", result.message)

                return (
                    {"running": False, "paused": False,
                        "message": result.message},
                    dmc.Notification(
                        title=t("Auto Operation"),
                        message=result.message,
                        color="green" if result.status == ProgramStatus.COMPLETED else "orange",
                        action="show",
                    ),
                )
            except Exception as exc:
                LOGGER.exception("Failed to start auto operation")
                return (
                    no_update,
                    dmc.Notification(
                        title=t("Auto Operation"),
                        message=t("Failed to start: {error}", error=exc),
                        color="red",
                        action="show",
                    ),
                )

        elif trigger_id == "auto-stop":
            if registry.stop():
                return (
                    {"running": False, "paused": False,
                        "message": t("Auto operation stopped")},
                    dmc.Notification(
                        title=t("Auto Operation"),
                        message=t("Auto pick & place operation stopped"),
                        color="orange",
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
            snapshot = {"running": False, "paused": False}
        else:
            snapshot = {
                "running": active.is_running,
                "paused": active.status == ProgramStatus.PAUSED,
            }

        if current_state == snapshot:
            return no_update
        return snapshot
