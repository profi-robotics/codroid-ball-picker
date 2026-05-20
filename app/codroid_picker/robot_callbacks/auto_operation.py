"""Auto operation callbacks for pick & place automation.

This module contains Dash callbacks for:
- Auto operation start/stop/pause/resume buttons
- Prerequisites checking
- Auto operation state synchronization

Uses the AutoPickPlaceProgram via the program registry.
"""

from typing import Optional, Tuple

import dash_mantine_components as dmc
from dash import Dash, Input, Output, State, callback_context, no_update
from app.codroid_picker.i18n import t

from app.codroid_picker.programs import (
    get_registry,
    AutoMotionTestProgram,
    AutoPickPlaceProgram,
)
from app.codroid_picker.programs.auto_mode import (
    AUTO_CONTROL_PROGRAMS,
    controller_is_auto_mode,
    restore_active_auto_program_manual_mode,
    restore_manual_mode,
)
from app.codroid_picker.programs.base import ProgramStatus
from app.codroid_picker.app_instance import registry


def _auto_button_disabled_state(
    *,
    all_ready: bool,
    motion_test_ready: bool,
    running: bool,
    registry_running: bool,
    auto_control_running: bool,
    controller_auto_mode: bool,
    trigger_id: Optional[str],
    last_auto_program: bool = False,
) -> Tuple[bool, bool, bool]:
    """Return disabled states for start, motion-test, and stop buttons."""
    any_running = registry_running

    if trigger_id == "auto-start" and all_ready and not any_running:
        return True, True, False
    if (
        trigger_id == "auto-motion-test-start"
        and motion_test_ready
        and not any_running
    ):
        return True, True, False
    if (
        trigger_id == "auto-stop"
        and (auto_control_running or controller_auto_mode or any_running)
    ):
        return True, True, True

    start_disabled = not all_ready or any_running
    motion_test_disabled = not motion_test_ready or any_running
    stop_disabled = not (
        running
        or auto_control_running
        or controller_auto_mode
        or registry_running
        or last_auto_program
    )
    return start_disabled, motion_test_disabled, stop_disabled


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
        Output("auto-motion-test-start", "disabled"),
        Output("auto-stop", "disabled"),
        Output("auto-prereq-badges", "children"),
        Input("prerequisites-state", "data"),
        Input("auto-operation-state", "data"),
        Input("camera-connection-state", "data"),
        Input("auto-start", "n_clicks"),
        Input("auto-motion-test-start", "n_clicks"),
        Input("auto-stop", "n_clicks"),
        Input("robot-status-interval", "n_intervals"),
        prevent_initial_call=False,
    )
    def _update_auto_buttons(
        prerequisites: Optional[dict],
        auto_state: Optional[dict],
        camera_state: Optional[dict],
        _start_clicks: Optional[int],
        _motion_test_clicks: Optional[int],
        _stop_clicks: Optional[int],
        _robot_status_ticks: Optional[int],
    ) -> Tuple[bool, bool, bool, list]:
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
        last_auto_program = state_data.get("program") in AUTO_CONTROL_PROGRAMS
        program_registry = get_registry()
        active_program = program_registry.active_program
        robot_status_source = (
            registry().robot_session
            or getattr(session, "robot_session", None)
            or session
        )
        status_snapshot_fn = getattr(
            robot_status_source, "robot_status_snapshot", None
        )
        status_age_fn = getattr(
            robot_status_source, "robot_status_age_seconds", None
        )
        status_snapshot = status_snapshot_fn() if callable(status_snapshot_fn) else {}
        status_age = status_age_fn() if callable(status_age_fn) else None
        controller_auto_mode = (
            status_age is not None
            and status_age <= 2.0
            and controller_is_auto_mode(status_snapshot)
        )
        auto_control_running = bool(
            running
            or (
                active_program is not None
                and active_program.name in AUTO_CONTROL_PROGRAMS
                and active_program.is_running
            )
        )
        registry_running = program_registry.is_program_running

        motion_test_ready = prereqs.get("robot_connected", False)
        trigger_id = callback_context.triggered_id
        start_disabled, motion_test_disabled, stop_disabled = (
            _auto_button_disabled_state(
                all_ready=all_ready,
                motion_test_ready=motion_test_ready,
                running=running,
                registry_running=registry_running,
                auto_control_running=auto_control_running,
                controller_auto_mode=controller_auto_mode,
                last_auto_program=last_auto_program,
                trigger_id=trigger_id,
            )
        )

        prereq_children = [
            dmc.Group(
                badge_items,
                gap=6,
                align="center",
                wrap="wrap",
                justify="center",
            ),
        ]

        return start_disabled, motion_test_disabled, stop_disabled, prereq_children

    @app.callback(
        Output("auto-operation-state", "data"),
        Output("app-notifications", "children", allow_duplicate=True),
        Input("auto-start", "n_clicks"),
        Input("auto-motion-test-start", "n_clicks"),
        Input("auto-stop", "n_clicks"),
        State("auto-operation-state", "data"),
        State("robot-uri", "value"),
        State("robot-control-state", "data"),
        prevent_initial_call=True,
    )
    async def _handle_auto_operation(
        start_clicks: int,
        motion_test_clicks: int,
        stop_clicks: int,
        current_state: Optional[dict],
        robot_uri: Optional[str],
        control_state: Optional[dict],
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
        LOGGER = codroid_picker_module.LOGGER
        DEFAULT_ROBOT_URI = codroid_picker_module.DEFAULT_ROBOT_URI

        if codroid_picker_module._robot_control_released(control_state):
            return (
                no_update,
                dmc.Notification(
                    title=t("Robot control released"),
                    message=t("Robot control is released. Acquire control first."),
                    color="yellow",
                    action="show",
                ),
            )

        if trigger_id in {"auto-start", "auto-motion-test-start"}:
            # Check if already running
            if registry.is_program_running:
                return (
                    {"running": True, "paused": False,
                        "message": t("Program already running")},
                    dmc.Notification(
                        title=t("Auto Operation"),
                        message=t(
                            "A program is already running: {name}",
                            name=registry.active_program_name,
                        ),
                        color="yellow",
                        action="show",
                    ),
                )

            # Create program instance
            if trigger_id == "auto-motion-test-start":
                program = AutoMotionTestProgram()
            else:
                program = AutoPickPlaceProgram()

            # Check prerequisites
            ok, message = program.check_prerequisites()
            if not ok:
                return (
                    {"running": False, "paused": False, "message": message},
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
                if program.name in AUTO_CONTROL_PROGRAMS:
                    await restore_manual_mode(
                        robot_api,
                        logger=LOGGER,
                        context=f"{program.label} completion",
                    )
                LOGGER.info("%s completed: %s", program.label, result.message)

                return (
                    {
                        "running": False,
                        "paused": False,
                        "program": program.name,
                        "message": result.message,
                    },
                    dmc.Notification(
                        title=t("Auto Operation"),
                        message=result.message,
                        color=(
                            "green"
                            if result.status == ProgramStatus.COMPLETED
                            else "orange"
                        ),
                        action="show",
                    ),
                )
            except Exception as exc:
                LOGGER.exception("Failed to start %s", program.label)
                return (
                    {
                        "running": False,
                        "paused": False,
                        "program": program.name,
                        "message": t("Failed to start: {error}", error=exc),
                    },
                    dmc.Notification(
                        title=t("Auto Operation"),
                        message=t("Failed to start: {error}", error=exc),
                        color="red",
                        action="show",
                    ),
                )

        elif trigger_id == "auto-stop":
            active_program = registry.active_program
            active_name = (
                registry.active_program_name
                or (current_state or {}).get("program")
                or "auto"
            )
            stopped_program = registry.stop()
            restored = await restore_active_auto_program_manual_mode(
                active_program,
                logger=LOGGER,
                context="Auto stop",
            )
            if not restored:
                async def _restore(api: codroid_picker_module.CodroidAPI) -> None:
                    await restore_manual_mode(
                        api,
                        logger=LOGGER,
                        context="Auto stop",
                    )

                try:
                    await codroid_picker_module._with_robot_api(
                        robot_uri or DEFAULT_ROBOT_URI,
                        _restore,
                    )
                    restored = True
                except Exception as exc:  # noqa: BLE001
                    LOGGER.exception("Auto stop failed to restore Manual mode")
                    return (
                        {
                            "running": False,
                            "paused": False,
                            "program": active_name,
                            "message": t(
                                "Auto stop failed: {error}",
                                error=exc,
                            ),
                        },
                        dmc.Notification(
                            title=t("Auto Operation"),
                            message=t("Auto stop failed: {error}", error=exc),
                            color="red",
                            action="show",
                        ),
                    )
            if stopped_program or restored:
                return (
                    {
                        "running": False,
                        "paused": False,
                        "program": active_name,
                        "message": t("Auto operation stopped"),
                    },
                    dmc.Notification(
                        title=t("Auto Operation"),
                        message=t("Auto operation stopped"),
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
        if active is None or active.name not in AUTO_CONTROL_PROGRAMS:
            snapshot = {"running": False, "paused": False}
        else:
            snapshot = {
                "running": active.is_running,
                "paused": active.status == ProgramStatus.PAUSED,
                "program": active.name,
            }

        if current_state == snapshot:
            return no_update
        return snapshot
