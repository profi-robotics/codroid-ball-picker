"""Manual pick and place callbacks.

This module provides UI callback wrappers for manual pick and place operations.
It delegates actual robot control logic to the ManualPickPlaceProgram class.
"""

from typing import Optional, Tuple

from dash import Dash, Input, Output, State, callback_context, no_update

from app.codroid_picker.programs import (
    get_registry,
    ManualPickPlaceProgram,
)
from app.codroid_picker.programs.base import ProgramStatus


def register_pick_place_callbacks(app: Dash) -> None:
    """Register manual pick and place callbacks."""

    @app.callback(
        Output("pick-place-status", "children"),
        Output("drop-position-metrics", "children", allow_duplicate=True),
        Input("teach-drop-position", "n_clicks"),
        Input("return-to-drop-position", "n_clicks"),
        Input("execute-drop", "n_clicks"),
        Input("manual-pick-closest", "n_clicks"),
        Input("manual-pick-place-target", "n_clicks"),
        State("pick-detection-select", "value"),
        State("plate-test-approach-height", "value"),
        State("plate-test-pick-height", "value"),
        State("plate-test-drop-height", "value"),
        State("robot-uri", "value"),
        prevent_initial_call=True,
    )
    async def _handle_pick_place(
        _teach_drop: int,
        _return_to_drop: int,
        _drop_action: int,
        _manual_pick: int,
        _manual_pick_place: int,
        selected_detection: Optional[str],
        approach_height: Optional[float],
        pick_height: Optional[float],
        drop_height: Optional[float],
        robot_uri: Optional[str],
    ) -> Tuple[str, str]:
        """Handle manual pick and place operations.

        This callback handles several trigger types:
        - teach-drop-position: Record current robot position as drop target
        - return-to-drop-position: Move robot back to saved drop position
        - execute-drop: Drop the currently held object
        - manual-pick-closest: Pick up the closest detected object
        - manual-pick-place-target: Full pick and place cycle
        """
        import app.codroid_picker as codroid_picker_module

        trigger = callback_context.triggered_id
        if trigger is None:
            return no_update, no_update

        SESSION = codroid_picker_module.SESSION
        LOGGER = codroid_picker_module.LOGGER
        ROBOT_COORDINATE_ID = codroid_picker_module.ROBOT_COORDINATE_ID
        DEFAULT_ROBOT_URI = codroid_picker_module.DEFAULT_ROBOT_URI
        GRIPPER_FORCE_DEFAULT = codroid_picker_module.GRIPPER_FORCE_DEFAULT
        GRIPPER_FORCE_MAX = codroid_picker_module.GRIPPER_FORCE_MAX
        GRIPPER_SPEED_DEFAULT = codroid_picker_module.GRIPPER_SPEED_DEFAULT
        GRIPPER_OPEN_WIDTH_DEFAULT = codroid_picker_module.GRIPPER_OPEN_WIDTH_DEFAULT
        GRIPPER_CLOSE_WIDTH_DEFAULT = codroid_picker_module.GRIPPER_CLOSE_WIDTH_DEFAULT

        _safe_float = codroid_picker_module._safe_float
        _with_robot_api = codroid_picker_module._with_robot_api
        _get_robot_session = codroid_picker_module._get_robot_session
        _current_coordinate_id = codroid_picker_module._current_coordinate_id
        _ensure_robot_coord5_for_uri = codroid_picker_module._ensure_robot_coord5_for_uri
        _coord5_to_workspace_mm = codroid_picker_module._coord5_to_workspace_mm
        _robot_to_workspace_mm = codroid_picker_module._robot_to_workspace_mm
        _save_drop_target = codroid_picker_module._save_drop_target
        _save_settings = codroid_picker_module._save_settings
        _begin_operation = codroid_picker_module._begin_operation
        _finish_operation = codroid_picker_module._finish_operation
        get_latest_frame = codroid_picker_module.get_latest_frame
        SETTINGS = codroid_picker_module.SETTINGS

        LOGGER.info(f"Pick/place callback triggered: {trigger}")
        drop_text = no_update

        # Prevent multiple rapid operations
        if SESSION.operation_in_progress:
            return "Operation already in progress, please wait...", no_update

        workspace_def = SESSION.workspace_definition
        approach_mm = _safe_float(approach_height, -30.0)
        pick_mm = _safe_float(pick_height, 0.0)
        drop_mm = _safe_float(drop_height, 0.0)

        # Handle teach-drop-position
        if trigger == "teach-drop-position":
            if workspace_def is None:
                return "Workspace is not defined; compute it first.", drop_text
            try:
                await _ensure_robot_coord5_for_uri(robot_uri)
            except Exception as exc:
                return f"Failed to set CoordinateId 5: {exc}", drop_text

            session = _get_robot_session()
            if not session.posture_seen():
                return "Robot posture stream not available yet.", drop_text

            pos = session.position_snapshot()
            coord_id = _current_coordinate_id(session)
            try:
                if coord_id == ROBOT_COORDINATE_ID:
                    drop_mm_xy = _coord5_to_workspace_mm(
                        workspace_def, (float(pos.x), float(pos.y))
                    )
                    if drop_mm_xy is None:
                        raise RuntimeError(
                            "Coord5 transform unavailable; check calibration.")
                else:
                    import numpy as np
                    robot_xyz = np.array([pos.x, pos.y, pos.z], dtype=float)
                    drop_mm_xy = _robot_to_workspace_mm(
                        workspace_def, robot_xyz)
            except Exception as exc:
                return f"Failed to record drop position: {exc}", drop_text

            SESSION.drop_target_mm = drop_mm_xy
            SESSION.drop_target_orientation = (pos.a, pos.b, pos.c)
            SESSION.drop_target_robot = (pos.x, pos.y, pos.z)
            SESSION.drop_target_coord_id = coord_id
            _save_drop_target(
                drop_mm_xy,
                SESSION.drop_target_orientation,
                SESSION.drop_target_robot,
                coordinate_id=coord_id,
            )
            _save_settings(SETTINGS.model_copy(
                update={"drop_target": drop_mm_xy}))
            SESSION.last_action = "Drop position taught."
            drop_text = (
                f"Drop at {drop_mm_xy[0]:.1f}, {drop_mm_xy[1]:.1f} mm (z={pos.z:.1f}) · "
                f"a={pos.a:.1f}, b={pos.b:.1f}, c={pos.c:.1f}"
            )
            return "Drop position recorded from robot pose.", drop_text

        # Handle return-to-drop-position
        if trigger == "return-to-drop-position":
            if SESSION.drop_target_robot is None:
                return "No drop position has been taught yet.", drop_text
            try:
                await _ensure_robot_coord5_for_uri(robot_uri)
            except Exception as exc:
                return f"Failed to set CoordinateId 5: {exc}", drop_text

            session = _get_robot_session()
            target_coord_id = SESSION.drop_target_coord_id or 5
            if target_coord_id != _current_coordinate_id(session):
                return (
                    f"Robot is not in coordinate system {target_coord_id}; "
                    "switch coordinate systems first."
                ), drop_text

            drop_robot_xyz = SESSION.drop_target_robot
            drop_orient = SESSION.drop_target_orientation or (0.0, 0.0, 0.0)

            async def _return_to_drop(robot_api) -> None:
                await robot_api.power_on()
                await codroid_picker_module._ensure_coordinate5_active(robot_api)
                target = {
                    "x": float(drop_robot_xyz[0]),
                    "y": float(drop_robot_xyz[1]),
                    "z": float(drop_robot_xyz[2]),
                    "a": float(drop_orient[0]),
                    "b": float(drop_orient[1]),
                    "c": float(drop_orient[2]),
                }
                codroid_picker_module._log_move_target(
                    "return_to_drop", target, coordinate_id=ROBOT_COORDINATE_ID
                )
                await robot_api.set_target_pos_type(
                    robot_api.config.target_pos_types.cpos
                )
                await robot_api.set_target_cpos(
                    robot_api.attach_coordinate_to_cpos(
                        robot_api.build_target_cpos(**target),
                        coordinate_id=ROBOT_COORDINATE_ID,
                    )
                )
                await robot_api.set_robot_command(
                    robot_api.config.commands.move_target_linear
                )
                import asyncio
                import time
                deadline = time.monotonic() + 5.0
                while time.monotonic() < deadline:
                    await robot_api.send_command_heartbeat()
                    await asyncio.sleep(0.2)
                await robot_api.clear_target_position()
                await robot_api.stop_command()

            try:
                await _with_robot_api(robot_uri or DEFAULT_ROBOT_URI, _return_to_drop)
                SESSION.last_action = "Returned to drop position."
                return "Returned to drop position.", drop_text
            except Exception as exc:
                LOGGER.exception("Return to drop position failed")
                return f"Return to drop failed: {exc}", drop_text

        # For pick operations, check camera is running
        frame, _depth = get_latest_frame()
        if frame is None and trigger not in {"teach-drop-position", "return-to-drop-position"}:
            return "Camera stream is not running.", drop_text

        # Use the programs registry for pick/drop operations
        registry = get_registry()

        # Resolve detection target for pick operations
        if trigger in {"manual-pick-closest", "manual-pick-place-target"}:
            if not SESSION.detections:
                return "No objects detected.", drop_text

            # Find closest detection
            from app.codroid_picker.pick_place_helpers import select_closest_detection
            detection = select_closest_detection(
                SESSION.detections,
                reference_xy=(0.0, 0.0),  # Pick closest to origin
                coord5_fn=lambda d: codroid_picker_module._detection_coord5_xy(
                    d, workspace_def
                ),
            )

            coord5_xy = codroid_picker_module._detection_coord5_xy(
                detection, workspace_def)
            if coord5_xy is None:
                return "Could not map detection to robot coordinates.", drop_text

            # Build pose fetchers for tracking
            tracker_coord5 = coord5_xy

            def make_pick_approach_fn():
                return codroid_picker_module._pose_from_coord5_mm(
                    tracker_coord5[0], tracker_coord5[1], approach_mm
                )

            def make_pick_pose_fn():
                return codroid_picker_module._pose_from_coord5_mm(
                    tracker_coord5[0], tracker_coord5[1], pick_mm
                )

            # Build drop poses
            drop_approach_pose = None
            drop_pose = None
            if SESSION.drop_target_robot is not None:
                base = SESSION.drop_target_robot
                drop_pose = {
                    "x": float(base[0]),
                    "y": float(base[1]),
                    "z": float(base[2]) + drop_mm,
                    "a": 0.0, "b": 0.0, "c": 0.0,
                }
                drop_approach_pose = {
                    "x": float(base[0]),
                    "y": float(base[1]),
                    "z": float(base[2]) + approach_mm,
                    "a": 0.0, "b": 0.0, "c": 0.0,
                }
                if SESSION.drop_target_orientation is not None:
                    for pose in (drop_pose, drop_approach_pose):
                        pose.update({
                            "a": float(SESSION.drop_target_orientation[0]),
                            "b": float(SESSION.drop_target_orientation[1]),
                            "c": float(SESSION.drop_target_orientation[2]),
                        })

            # Determine action
            if trigger == "manual-pick-closest":
                action = "pick"
            else:  # manual-pick-place-target
                if SESSION.drop_target_mm is None:
                    return "Teach a drop position first.", drop_text
                action = "pick_drop"

            # Create and run program
            program = ManualPickPlaceProgram(
                action=action,
                pick_approach_pose_fn=make_pick_approach_fn,
                pick_pose_fn=make_pick_pose_fn,
                drop_approach_pose=drop_approach_pose,
                drop_pose=drop_pose,
                approach_mm=approach_mm,
                pick_mm=pick_mm,
                drop_mm=drop_mm,
            )

            try:
                SESSION.operation_in_progress = True
                result = await registry.start_instance(
                    program,
                    await codroid_picker_module._get_robot_api(robot_uri or DEFAULT_ROBOT_URI),
                )
                if result.status == ProgramStatus.COMPLETED:
                    SESSION.last_action = f"{action.replace('_', ' ').title()} completed."
                    if action == "pick_drop" and SESSION.drop_target_mm:
                        drop_text = f"Drop at {SESSION.drop_target_mm[0]:.1f}, {SESSION.drop_target_mm[1]:.1f} mm"
                    return f"{action.replace('_', ' ').title()} command sent.", drop_text
                else:
                    return f"{action.replace('_', ' ').title()} failed: {result.message}", drop_text
            except Exception as exc:
                LOGGER.exception("Pick/place action failed")
                return f"Pick/place action failed: {exc}", drop_text
            finally:
                SESSION.operation_in_progress = False

        # Handle execute-drop
        if trigger == "execute-drop":
            if SESSION.drop_target_mm is None:
                return "Teach a drop position first.", drop_text

            drop_approach_pose = None
            drop_pose = None
            if SESSION.drop_target_robot is not None:
                base = SESSION.drop_target_robot
                drop_pose = {
                    "x": float(base[0]),
                    "y": float(base[1]),
                    "z": float(base[2]) + drop_mm,
                    "a": 0.0, "b": 0.0, "c": 0.0,
                }
                drop_approach_pose = {
                    "x": float(base[0]),
                    "y": float(base[1]),
                    "z": float(base[2]) + approach_mm,
                    "a": 0.0, "b": 0.0, "c": 0.0,
                }
                if SESSION.drop_target_orientation is not None:
                    for pose in (drop_pose, drop_approach_pose):
                        pose.update({
                            "a": float(SESSION.drop_target_orientation[0]),
                            "b": float(SESSION.drop_target_orientation[1]),
                            "c": float(SESSION.drop_target_orientation[2]),
                        })

            program = ManualPickPlaceProgram(
                action="drop",
                drop_approach_pose=drop_approach_pose,
                drop_pose=drop_pose,
                approach_mm=approach_mm,
                drop_mm=drop_mm,
            )

            try:
                SESSION.operation_in_progress = True
                result = await registry.start_instance(
                    program,
                    await codroid_picker_module._get_robot_api(robot_uri or DEFAULT_ROBOT_URI),
                )
                if result.status == ProgramStatus.COMPLETED:
                    SESSION.last_action = "Drop completed."
                    drop_text = f"Drop at {SESSION.drop_target_mm[0]:.1f}, {SESSION.drop_target_mm[1]:.1f} mm"
                    return "Drop command sent.", drop_text
                else:
                    return f"Drop failed: {result.message}", drop_text
            except Exception as exc:
                LOGGER.exception("Drop action failed")
                return f"Drop action failed: {exc}", drop_text
            finally:
                SESSION.operation_in_progress = False

        return no_update, drop_text
