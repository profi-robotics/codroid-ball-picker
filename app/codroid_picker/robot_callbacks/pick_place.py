"""Manual pick and place callbacks.

This module provides UI callback wrappers for manual pick and place operations.
It delegates actual robot control logic to the ManualPickPlaceProgram class.
"""

import time
from typing import Optional, Tuple

from dash import Dash, Input, Output, State, callback_context, no_update
from app.codroid_picker.i18n import t

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
        State("shared-tracking-mode", "value"),
        State("shared-segment-min", "value"),
        State("shared-segment-max", "value"),
        State("shared-segment-fraction", "value"),
        State("robot-uri", "value"),
        State("robot-control-state", "data"),
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
        tracking_mode: Optional[str],
        segment_min: Optional[float],
        segment_max: Optional[float],
        segment_fraction: Optional[float],
        robot_uri: Optional[str],
        control_state: Optional[dict],
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
        if codroid_picker_module._robot_control_released(control_state):
            return t("Robot control is released. Acquire control first."), no_update

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
            return t("Operation already in progress, please wait..."), no_update

        workspace_def = SESSION.workspace_definition
        approach_mm = _safe_float(approach_height, -30.0)
        pick_mm = _safe_float(pick_height, 0.0)
        drop_mm = _safe_float(drop_height, 0.0)

        # Handle teach-drop-position
        if trigger == "teach-drop-position":
            if workspace_def is None:
                return t("Workspace is not defined; compute it first."), drop_text
            try:
                await _ensure_robot_coord5_for_uri(robot_uri)
            except Exception as exc:
                return t("Failed to set CoordinateId 5: {error}", error=exc), drop_text

            session = _get_robot_session()
            if not session.posture_seen():
                return t("Robot posture stream not available yet."), drop_text

            pos = session.position_snapshot()
            coord_id = _current_coordinate_id(session)
            try:
                if coord_id == ROBOT_COORDINATE_ID:
                    drop_mm_xy = _coord5_to_workspace_mm(
                        workspace_def, (float(pos.x), float(pos.y))
                    )
                    if drop_mm_xy is None:
                        raise RuntimeError(
                            t("Coord5 transform unavailable; check calibration."))
                else:
                    import numpy as np
                    robot_xyz = np.array([pos.x, pos.y, pos.z], dtype=float)
                    drop_mm_xy = _robot_to_workspace_mm(
                        workspace_def, robot_xyz)
            except Exception as exc:
                return t("Failed to record drop position: {error}", error=exc), drop_text

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
            SESSION.last_action = t("Drop position taught.")
            drop_text = (
                t(
                    "Drop at {x:.1f}, {y:.1f} mm (z={z:.1f}) · a={a:.1f}, b={b:.1f}, c={c:.1f}",
                    x=drop_mm_xy[0],
                    y=drop_mm_xy[1],
                    z=pos.z,
                    a=pos.a,
                    b=pos.b,
                    c=pos.c,
                )
            )
            return t("Drop position recorded from robot pose."), drop_text

        # Handle return-to-drop-position
        if trigger == "return-to-drop-position":
            if SESSION.drop_target_robot is None:
                return t("No drop position has been taught yet."), drop_text
            try:
                await _ensure_robot_coord5_for_uri(robot_uri)
            except Exception as exc:
                return t("Failed to set CoordinateId 5: {error}", error=exc), drop_text

            session = _get_robot_session()
            target_coord_id = SESSION.drop_target_coord_id or 5
            if target_coord_id != _current_coordinate_id(session):
                return (
                    t(
                        "Robot is not in coordinate system {coord_id}; switch coordinate systems first.",
                        coord_id=target_coord_id,
                    )
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
                SESSION.last_action = t("Returned to drop position.")
                return t("Returned to drop position."), drop_text
            except Exception as exc:
                LOGGER.exception("Return to drop position failed")
                return t("Return to drop failed: {error}", error=exc), drop_text

        # For pick operations, check camera is running
        frame, _depth = get_latest_frame()
        if frame is None and trigger not in {"teach-drop-position", "return-to-drop-position"}:
            return t("Camera stream is not running."), drop_text

        # Use the programs registry for pick/drop operations
        registry = get_registry()

        # Resolve detection target for pick operations
        if trigger in {"manual-pick-closest", "manual-pick-place-target"}:
            if not SESSION.detections:
                return t("No objects detected."), drop_text

            # Keep pick moves in the current wrist orientation to avoid
            # unnecessary re-orientation during manual mode.
            pick_orientation = (0.0, 0.0, 0.0)
            try:
                robot_session = _get_robot_session()
                if robot_session.posture_seen():
                    pos = robot_session.position_snapshot()
                    pick_orientation = (
                        float(pos.a),
                        float(pos.b),
                        float(pos.c),
                    )
            except Exception as exc:  # noqa: BLE001
                LOGGER.debug("Falling back to default pick orientation: %s", exc)

            def build_pick_pose(x_mm: float, y_mm: float, z_mm: float) -> dict:
                pose = codroid_picker_module._pose_from_coord5_mm(x_mm, y_mm, z_mm)
                pose.update(
                    {
                        "a": pick_orientation[0],
                        "b": pick_orientation[1],
                        "c": pick_orientation[2],
                    }
                )
                return pose

            target_labels = SESSION.auto_config.get(
                "target_labels",
                [SETTINGS.auto_target_label],
            )
            if not target_labels:
                target_labels = [SETTINGS.auto_target_label]

            # Filter detections by target label
            def filter_by_label(detections):
                """Filter detections to only include the target label."""
                return [d for d in detections if d.get("label") in target_labels]

            filtered_detections = filter_by_label(SESSION.detections)
            if not filtered_detections:
                labels_text = ", ".join(target_labels)
                return t("No objects with label '{label}' detected.", label=labels_text), drop_text

            from app.codroid_picker.target_planner import (
                plan_pick_strategy,
                rank_targets,
            )

            robot_xy = (0.0, 0.0)
            if "pos" in locals():
                robot_xy = (float(pos.x), float(pos.y))
            ranked = rank_targets(
                filtered_detections,
                robot_xy=robot_xy,
                coord5_fn=lambda d: codroid_picker_module._detection_coord5_xy(
                    d, workspace_def
                ),
                neighbor_radius_mm=float(SETTINGS.auto_cluster_neighbor_mm),
            )
            if not ranked:
                return t("Could not map detection to robot coordinates."), drop_text
            selected_candidate = ranked[0]
            pick_plan = plan_pick_strategy(
                selected_candidate,
                ranked,
                current_c_deg=float(pick_orientation[2]),
                rotation_step_deg=float(SETTINGS.auto_rotation_step_deg),
                rotation_max_delta_deg=float(SETTINGS.auto_rotation_max_delta_deg),
                body_radius_mm=float(SETTINGS.auto_collision_body_radius_mm),
                finger_reach_mm=float(SETTINGS.auto_collision_finger_reach_mm),
                clearance_mm=float(SETTINGS.auto_rotation_clearance_mm),
                separation_entry_mm=float(SETTINGS.auto_separation_entry_mm),
                separation_push_mm=float(SETTINGS.auto_separation_push_mm),
            )
            if pick_plan.mode != "rotate_pick":
                return t("Target is too clustered for manual pick."), drop_text
            coord5_xy = selected_candidate.coord5_xy
            pick_final_c = float(pick_plan.final_c)

            # Determine tracking mode
            use_segment_tracking = (tracking_mode or "move").strip().lower() == "segment"

            # Capture static coordinates for "move" mode
            # These won't change during the operation
            static_coord5_xy = (coord5_xy[0], coord5_xy[1])

            if use_segment_tracking:
                # For segment tracking, we track during approach but then lock in
                # the final position for pick_pose (robot obscures target when close)
                # Use a mutable container to store the last tracked position
                tracking_state = {
                    "xy": (coord5_xy[0], coord5_xy[1]),
                    "lost_count": 0,  # Count consecutive frames where target is lost
                }

                # Build dynamic pose fetcher for approach phase
                # This re-queries SESSION.detections to track moving objects
                # and updates tracking_state with each successful detection
                # When target is not visible (robot obscures it), returns last known position
                def make_pick_approach_fn():
                    # Filter detections by target label
                    filtered = filter_by_label(SESSION.detections) if SESSION.detections else []

                    if not filtered:
                        # No valid detections - use last known position
                        # This is expected when robot obscures the target
                        tracking_state["lost_count"] += 1
                        if tracking_state["lost_count"] == 5:
                            LOGGER.info(
                                f"Target not visible for {tracking_state['lost_count']} frames, "
                                f"using last known position (robot likely obscuring target)"
                            )
                        # Always return last known position - don't return None
                        return build_pick_pose(
                            tracking_state["xy"][0], tracking_state["xy"][1], approach_mm
                        )

                    current_detection = select_closest_detection(
                        filtered,
                        reference_xy=(0.0, 0.0),
                        coord5_fn=lambda d: codroid_picker_module._detection_coord5_xy(
                            d, workspace_def
                        ),
                    )
                    current_coord5 = codroid_picker_module._detection_coord5_xy(
                        current_detection, workspace_def
                    )
                    if current_coord5 is None:
                        # Detection exists but can't map to coordinates - use last known
                        tracking_state["lost_count"] += 1
                        return build_pick_pose(
                            tracking_state["xy"][0], tracking_state["xy"][1], approach_mm
                        )

                    # Successfully tracked - reset lost counter and update position
                    tracking_state["lost_count"] = 0
                    tracking_state["xy"] = (current_coord5[0], current_coord5[1])
                    return build_pick_pose(
                        current_coord5[0], current_coord5[1], approach_mm
                    )

                # Build static pose fetcher for pick phase
                # Uses the last tracked position from approach (robot obscures target)
                def make_pick_pose_fn():
                    return build_pick_pose(
                        tracking_state["xy"][0], tracking_state["xy"][1], pick_mm
                    )
            else:
                # Build static pose fetchers for "move" mode
                # These use the captured coordinates and don't change
                def make_pick_approach_fn():
                    return build_pick_pose(
                        static_coord5_xy[0], static_coord5_xy[1], approach_mm
                    )

                def make_pick_pose_fn():
                    return build_pick_pose(
                        static_coord5_xy[0], static_coord5_xy[1], pick_mm
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
                    return t("Teach a drop position first."), drop_text
                action = "pick_drop"

            # Create and run program with tracking configuration
            program = ManualPickPlaceProgram(
                action=action,
                pick_approach_pose_fn=make_pick_approach_fn,
                pick_pose_fn=make_pick_pose_fn,
                drop_approach_pose=drop_approach_pose,
                drop_pose=drop_pose,
                pick_final_c=pick_final_c,
                approach_mm=approach_mm,
                pick_mm=pick_mm,
                drop_mm=drop_mm,
                tracking_mode=(tracking_mode or "move").strip().lower(),
                segment_min_mm=_safe_float(segment_min, SETTINGS.segment_min),
                segment_max_mm=_safe_float(segment_max, SETTINGS.segment_max),
                segment_fraction=_safe_float(segment_fraction, SETTINGS.segment_fraction),
            )

            try:
                SESSION.operation_in_progress = True
                SESSION.operation_started_at = time.monotonic()
                result = await registry.start_instance(
                    program,
                    await codroid_picker_module._get_robot_api(robot_uri or DEFAULT_ROBOT_URI),
                )
                if result.status == ProgramStatus.COMPLETED:
                    SESSION.last_action = t("{action} completed.", action=action.replace("_", " ").title())
                    if action == "pick_drop" and SESSION.drop_target_mm:
                        drop_text = t("Drop at {x:.1f}, {y:.1f} mm", x=SESSION.drop_target_mm[0], y=SESSION.drop_target_mm[1])
                    return t("{action} command sent.", action=action.replace("_", " ").title()), drop_text
                else:
                    return t("{action} failed: {error}", action=action.replace("_", " ").title(), error=result.message), drop_text
            except Exception as exc:
                LOGGER.exception("Pick/place action failed")
                return t("Pick/place action failed: {error}", error=exc), drop_text
            finally:
                SESSION.operation_in_progress = False
                SESSION.operation_started_at = None

        # Handle execute-drop
        if trigger == "execute-drop":
            if SESSION.drop_target_mm is None:
                return t("Teach a drop position first."), drop_text

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
                SESSION.operation_started_at = time.monotonic()
                result = await registry.start_instance(
                    program,
                    await codroid_picker_module._get_robot_api(robot_uri or DEFAULT_ROBOT_URI),
                )
                if result.status == ProgramStatus.COMPLETED:
                    SESSION.last_action = t("Drop completed.")
                    drop_text = t("Drop at {x:.1f}, {y:.1f} mm", x=SESSION.drop_target_mm[0], y=SESSION.drop_target_mm[1])
                    return t("Drop command sent."), drop_text
                else:
                    return t("Drop failed: {error}", error=result.message), drop_text
            except Exception as exc:
                LOGGER.exception("Drop action failed")
                return t("Drop action failed: {error}", error=exc), drop_text
            finally:
                SESSION.operation_in_progress = False
                SESSION.operation_started_at = None

        return no_update, drop_text
