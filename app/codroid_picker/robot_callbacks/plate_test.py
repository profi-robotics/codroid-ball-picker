"""Plate test callback.

This module provides the UI callback wrapper for the plate pick & place test.
It delegates actual robot control logic to the PlateTestProgram class.
"""

from typing import Optional, Tuple

from dash import Dash, Input, Output, State

from app.codroid_picker.programs import (get_registry, PlateTestConfig, PlateTestProgram)
from app.codroid_picker.programs.base import ProgramStatus


def register_plate_test_callbacks(app: Dash) -> None:
    """Register plate test callbacks."""

    @app.callback(
        Output("plate-test-status", "children"),
        Output("plate-test-analysis", "value"),
        Input("plate-test-run", "n_clicks"),
        State("plate-test-tracking-mode", "value"),
        State("plate-test-segment-min", "value"),
        State("plate-test-segment-max", "value"),
        State("plate-test-segment-fraction", "value"),
        State("plate-test-approach-height", "value"),
        State("plate-test-pick-height", "value"),
        State("plate-test-drop-height", "value"),
        State("plate-test-grip-timeout", "value"),
        State("robot-uri", "value"),
        State("robot-connection-state", "data"),
        State("gripper-connection-state", "data"),
        prevent_initial_call=True,
    )
    async def _run_plate_pick_place_test(
        _clicks: int,
        tracking_mode: Optional[str],
        segment_min: Optional[float],
        segment_max: Optional[float],
        segment_fraction: Optional[float],
        approach_height: Optional[float],
        pick_height: Optional[float],
        drop_height: Optional[float],
        grip_timeout_s: Optional[float],
        robot_uri: Optional[str],
        robot_state: Optional[dict],
        gripper_state: Optional[dict],
    ) -> Tuple[str, str]:
        """Run plate pick & place test using the PlateTestProgram.

        This callback:
        1. Validates prerequisites (robot, gripper, camera, workspace)
        2. Creates a PlateTestConfig from UI parameters
        3. Starts the PlateTestProgram via the registry
        4. Returns status and analysis results
        """
        import app.codroid_picker as codroid_picker_module

        LOGGER = codroid_picker_module.LOGGER
        SETTINGS = codroid_picker_module.SETTINGS
        DEFAULT_ROBOT_URI = codroid_picker_module.DEFAULT_ROBOT_URI
        _safe_float = codroid_picker_module._safe_float
        _set_plate_test_state = codroid_picker_module._set_plate_test_state

        def _abort(message: str) -> Tuple[str, str]:
            _set_plate_test_state(message, running=False)
            return message, ""

        # Validate connection states
        if (robot_state or {}).get("status") != "online":
            return _abort("Robot is not connected. Click 'Test robot connection' first.")
        if (gripper_state or {}).get("status") != "online":
            return _abort("Gripper is not connected. Enable gripper in Setup menu.")

        # Build configuration from UI parameters
        config = PlateTestConfig(
            tracking_mode=(tracking_mode or "move").strip().lower(),
            segment_min_mm=_safe_float(segment_min, SETTINGS.segment_min),
            segment_max_mm=_safe_float(segment_max, SETTINGS.segment_max),
            segment_fraction=_safe_float(
                segment_fraction, SETTINGS.segment_fraction),
            approach_height_mm=_safe_float(approach_height, -30.0),
            pick_height_mm=_safe_float(pick_height, 0.0),
            drop_height_mm=_safe_float(drop_height, 0.0),
            grip_timeout_s=_safe_float(
                grip_timeout_s, SETTINGS.plate_test_grip_timeout_s),
            use_base_coordinates=SETTINGS.plate_test_use_base_coordinates,
        )

        # Create program instance
        program = PlateTestProgram(config=config)

        # Check prerequisites
        ok, message = program.check_prerequisites()
        if not ok:
            return _abort(message)

        # Run the program
        registry = get_registry()
        try:
            robot_api = await codroid_picker_module._get_robot_api(
                robot_uri or DEFAULT_ROBOT_URI
            )
            result = await registry.start_instance(program, robot_api)

            if result.status == ProgramStatus.COMPLETED:
                # Program completed successfully, analysis is in result data
                analysis = result.data.get(
                    "analysis", "") if result.data else ""
                return result.message, analysis
            elif result.status == ProgramStatus.STOPPED:
                return result.message, ""
            else:
                return f"Plate test failed: {result.message}", ""

        except RuntimeError as exc:
            # Another program is running
            return _abort(str(exc))
        except Exception as exc:
            LOGGER.exception("Plate pick & place test failed")
            return _abort(f"Plate test failed: {exc}")
