"""Manual pick and place callbacks.

This module provides UI callback wrappers for manual pick and place operations.
It delegates robot control logic to ManualPickPlaceProgram helpers.
"""

import asyncio
import time
from typing import Optional, Tuple

from dash import Dash, Input, Output, State, callback_context, no_update

from app.codroid_picker.i18n import t
from app.codroid_picker.pick_place_helpers import select_closest_detection
from app.codroid_picker.programs import ManualPickPlaceProgram, get_registry
from app.codroid_picker.programs.base import ProgramStatus


def _build_drop_poses(
    session,
    *,
    approach_mm: float,
    drop_mm: float,
) -> tuple[Optional[dict], Optional[dict]]:
    """Build optional drop and drop-approach poses from taught target."""
    if session.drop_target_robot is None:
        return None, None

    base = session.drop_target_robot
    drop_pose = {
        "x": float(base[0]),
        "y": float(base[1]),
        "z": float(base[2]) + drop_mm,
        "a": 0.0,
        "b": 0.0,
        "c": 0.0,
    }
    drop_approach_pose = {
        "x": float(base[0]),
        "y": float(base[1]),
        "z": float(base[2]) + approach_mm,
        "a": 0.0,
        "b": 0.0,
        "c": 0.0,
    }
    if session.drop_target_orientation is not None:
        orientation = {
            "a": float(session.drop_target_orientation[0]),
            "b": float(session.drop_target_orientation[1]),
            "c": float(session.drop_target_orientation[2]),
        }
        drop_pose.update(orientation)
        drop_approach_pose.update(orientation)

    return drop_approach_pose, drop_pose


def _drop_metrics_text(session) -> str:
    """Return UI text for taught drop target."""
    if session.drop_target_mm is None:
        return no_update
    return t(
        "Drop at {x:.1f}, {y:.1f} mm",
        x=session.drop_target_mm[0],
        y=session.drop_target_mm[1],
    )


async def _handle_teach_drop_position(
    codroid_picker_module,
    *,
    robot_uri: Optional[str],
) -> Tuple[str, str]:
    """Record current robot pose as drop target."""
    session = codroid_picker_module.SESSION
    workspace_def = session.workspace_definition
    if workspace_def is None:
        return t("Workspace is not defined; compute it first."), no_update

    try:
        await codroid_picker_module._ensure_robot_coord5_for_uri(robot_uri)
    except Exception as exc:  # noqa: BLE001
        return t("Failed to set CoordinateId 5: {error}", error=exc), no_update

    robot_session = codroid_picker_module._get_robot_session()
    if not robot_session.posture_seen():
        return t("Robot posture stream not available yet."), no_update

    pos = robot_session.position_snapshot()
    coord_id = codroid_picker_module._current_coordinate_id(robot_session)
    try:
        if coord_id == codroid_picker_module.ROBOT_COORDINATE_ID:
            drop_mm_xy = codroid_picker_module._coord5_to_workspace_mm(
                workspace_def,
                (float(pos.x), float(pos.y)),
            )
            if drop_mm_xy is None:
                raise RuntimeError(t("Coord5 transform unavailable; check calibration."))
        else:
            import numpy as np

            robot_xyz = np.array([pos.x, pos.y, pos.z], dtype=float)
            drop_mm_xy = codroid_picker_module._robot_to_workspace_mm(
                workspace_def,
                robot_xyz,
            )
    except Exception as exc:  # noqa: BLE001
        return t("Failed to record drop position: {error}", error=exc), no_update

    session.drop_target_mm = drop_mm_xy
    session.drop_target_orientation = (pos.a, pos.b, pos.c)
    session.drop_target_robot = (pos.x, pos.y, pos.z)
    session.drop_target_coord_id = coord_id
    codroid_picker_module._save_drop_target(
        drop_mm_xy,
        session.drop_target_orientation,
        session.drop_target_robot,
        coordinate_id=coord_id,
    )
    codroid_picker_module._save_settings(
        codroid_picker_module.SETTINGS.model_copy(update={"drop_target": drop_mm_xy})
    )
    session.last_action = t("Drop position taught.")
    drop_text = t(
        "Drop at {x:.1f}, {y:.1f} mm (z={z:.1f}) · a={a:.1f}, b={b:.1f}, c={c:.1f}",
        x=drop_mm_xy[0],
        y=drop_mm_xy[1],
        z=pos.z,
        a=pos.a,
        b=pos.b,
        c=pos.c,
    )
    return t("Drop position recorded from robot pose."), drop_text


async def _handle_return_to_drop_position(
    codroid_picker_module,
    *,
    robot_uri: Optional[str],
) -> Tuple[str, str]:
    """Move robot to taught drop target."""
    session = codroid_picker_module.SESSION
    if session.drop_target_robot is None:
        return t("No drop position has been taught yet."), no_update

    try:
        await codroid_picker_module._ensure_robot_coord5_for_uri(robot_uri)
    except Exception as exc:  # noqa: BLE001
        return t("Failed to set CoordinateId 5: {error}", error=exc), no_update

    robot_session = codroid_picker_module._get_robot_session()
    target_coord_id = session.drop_target_coord_id or 5
    if target_coord_id != codroid_picker_module._current_coordinate_id(robot_session):
        return (
            t(
                "Robot is not in coordinate system {coord_id}; switch coordinate systems first.",
                coord_id=target_coord_id,
            ),
            no_update,
        )

    drop_robot_xyz = session.drop_target_robot
    drop_orient = session.drop_target_orientation or (0.0, 0.0, 0.0)

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
            "return_to_drop",
            target,
            coordinate_id=codroid_picker_module.ROBOT_COORDINATE_ID,
        )
        await robot_api.set_target_pos_type(robot_api.config.target_pos_types.cpos)
        await robot_api.set_target_cpos(
            robot_api.attach_coordinate_to_cpos(
                robot_api.build_target_cpos(**target),
                coordinate_id=codroid_picker_module.ROBOT_COORDINATE_ID,
            )
        )
        await robot_api.set_robot_command(robot_api.config.commands.move_target_linear)

        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            await robot_api.send_command_heartbeat()
            await asyncio.sleep(0.2)
        await robot_api.clear_target_position()
        await robot_api.stop_command()

    try:
        await codroid_picker_module._with_robot_api(
            robot_uri or codroid_picker_module.DEFAULT_ROBOT_URI,
            _return_to_drop,
        )
        session.last_action = t("Returned to drop position.")
        return t("Returned to drop position."), no_update
    except Exception as exc:  # noqa: BLE001
        codroid_picker_module.LOGGER.exception("Return to drop position failed")
        return t("Return to drop failed: {error}", error=exc), no_update


def _resolve_pick_orientation(codroid_picker_module) -> tuple[tuple[float, float, float], tuple[float, float]]:
    """Resolve wrist orientation and XY robot reference for ranking."""
    orientation = (0.0, 0.0, 0.0)
    robot_xy = (0.0, 0.0)
    try:
        robot_session = codroid_picker_module._get_robot_session()
        if robot_session.posture_seen():
            pos = robot_session.position_snapshot()
            orientation = (float(pos.a), float(pos.b), float(pos.c))
            robot_xy = (float(pos.x), float(pos.y))
    except Exception as exc:  # noqa: BLE001
        codroid_picker_module.LOGGER.debug(
            "Falling back to default pick orientation: %s",
            exc,
        )
    return orientation, robot_xy


def _prepare_pick_program(
    codroid_picker_module,
    *,
    trigger: str,
    approach_mm: float,
    pick_mm: float,
    drop_mm: float,
    tracking_mode: Optional[str],
    segment_min: Optional[float],
    segment_max: Optional[float],
    segment_fraction: Optional[float],
) -> tuple[Optional[ManualPickPlaceProgram], Optional[str], Optional[str]]:
    """Build manual pick or pick+drop program and return (program, action, error)."""
    from app.codroid_picker.target_planner import plan_pick_strategy, rank_targets

    session = codroid_picker_module.SESSION
    settings = codroid_picker_module.SETTINGS
    workspace_def = session.workspace_definition

    if workspace_def is None:
        return None, None, t("Workspace is not defined; compute it first.")
    if not session.detections:
        return None, None, t("No objects detected.")

    target_label = settings.auto_target_label
    filtered = [d for d in session.detections if d.get("label") == target_label]
    if not filtered:
        return None, None, t("No objects with label '{label}' detected.", label=target_label)

    pick_orientation, robot_xy = _resolve_pick_orientation(codroid_picker_module)

    ranked = rank_targets(
        filtered,
        robot_xy=robot_xy,
        coord5_fn=lambda d: codroid_picker_module._detection_coord5_xy(d, workspace_def),
        neighbor_radius_mm=float(settings.auto_cluster_neighbor_mm),
    )
    if not ranked:
        return None, None, t("Could not map detection to robot coordinates.")

    selected_candidate = ranked[0]
    pick_plan = plan_pick_strategy(
        selected_candidate,
        ranked,
        current_c_deg=float(pick_orientation[2]),
        rotation_step_deg=float(settings.auto_rotation_step_deg),
        rotation_max_delta_deg=float(settings.auto_rotation_max_delta_deg),
        body_radius_mm=float(settings.auto_collision_body_radius_mm),
        finger_reach_mm=float(settings.auto_collision_finger_reach_mm),
        clearance_mm=float(settings.auto_rotation_clearance_mm),
        separation_entry_mm=float(settings.auto_separation_entry_mm),
        separation_push_mm=float(settings.auto_separation_push_mm),
    )
    if pick_plan.mode != "rotate_pick":
        return None, None, t("Target is too clustered for manual pick.")

    coord5_xy = selected_candidate.coord5_xy
    pick_final_c = float(pick_plan.final_c)
    tracking_mode_value = (tracking_mode or "move").strip().lower()

    def build_pick_pose(x_mm: float, y_mm: float, z_mm: float) -> dict:
        pose = codroid_picker_module._pose_from_coord5_mm(x_mm, y_mm, z_mm)
        pose.update({"a": pick_orientation[0], "b": pick_orientation[1], "c": pick_orientation[2]})
        return pose

    if tracking_mode_value == "segment":
        tracking_state = {"xy": (coord5_xy[0], coord5_xy[1]), "lost_count": 0}

        def make_pick_approach_fn() -> dict:
            filtered_live = [
                d for d in (session.detections or []) if d.get("label") == target_label
            ]
            if not filtered_live:
                tracking_state["lost_count"] += 1
                return build_pick_pose(
                    tracking_state["xy"][0],
                    tracking_state["xy"][1],
                    approach_mm,
                )

            current_detection = select_closest_detection(
                filtered_live,
                reference_xy=(0.0, 0.0),
                coord5_fn=lambda d: codroid_picker_module._detection_coord5_xy(d, workspace_def),
            )
            if current_detection is None:
                tracking_state["lost_count"] += 1
                return build_pick_pose(
                    tracking_state["xy"][0],
                    tracking_state["xy"][1],
                    approach_mm,
                )

            current_coord5 = codroid_picker_module._detection_coord5_xy(
                current_detection,
                workspace_def,
            )
            if current_coord5 is None:
                tracking_state["lost_count"] += 1
                return build_pick_pose(
                    tracking_state["xy"][0],
                    tracking_state["xy"][1],
                    approach_mm,
                )

            tracking_state["lost_count"] = 0
            tracking_state["xy"] = (current_coord5[0], current_coord5[1])
            return build_pick_pose(current_coord5[0], current_coord5[1], approach_mm)

        def make_pick_pose_fn() -> dict:
            return build_pick_pose(tracking_state["xy"][0], tracking_state["xy"][1], pick_mm)

    else:
        static_xy = (coord5_xy[0], coord5_xy[1])

        def make_pick_approach_fn() -> dict:
            return build_pick_pose(static_xy[0], static_xy[1], approach_mm)

        def make_pick_pose_fn() -> dict:
            return build_pick_pose(static_xy[0], static_xy[1], pick_mm)

    if trigger == "manual-pick-closest":
        action = "pick"
    else:
        if session.drop_target_mm is None:
            return None, None, t("Teach a drop position first.")
        action = "pick_drop"

    drop_approach_pose, drop_pose = _build_drop_poses(
        session,
        approach_mm=approach_mm,
        drop_mm=drop_mm,
    )

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
        tracking_mode=tracking_mode_value,
        segment_min_mm=codroid_picker_module._safe_float(segment_min, settings.segment_min),
        segment_max_mm=codroid_picker_module._safe_float(segment_max, settings.segment_max),
        segment_fraction=codroid_picker_module._safe_float(
            segment_fraction,
            settings.segment_fraction,
        ),
    )
    return program, action, None


async def _run_manual_program(
    codroid_picker_module,
    *,
    program: ManualPickPlaceProgram,
    action: str,
    robot_uri: Optional[str],
) -> Tuple[str, str]:
    """Run ManualPickPlaceProgram and normalize response text."""
    session = codroid_picker_module.SESSION
    registry = get_registry()
    action_label = action.replace("_", " ").title()

    try:
        session.operation_in_progress = True
        session.operation_started_at = time.monotonic()
        result = await registry.start_instance(
            program,
            await codroid_picker_module._get_robot_api(
                robot_uri or codroid_picker_module.DEFAULT_ROBOT_URI
            ),
        )
        if result.status == ProgramStatus.COMPLETED:
            session.last_action = t("{action} completed.", action=action_label)
            drop_text = _drop_metrics_text(session) if action == "pick_drop" else no_update
            return t("{action} command sent.", action=action_label), drop_text
        return t("{action} failed: {error}", action=action_label, error=result.message), no_update
    except Exception as exc:  # noqa: BLE001
        codroid_picker_module.LOGGER.exception("Pick/place action failed")
        return t("Pick/place action failed: {error}", error=exc), no_update
    finally:
        session.operation_in_progress = False
        session.operation_started_at = None


async def _handle_execute_drop(
    codroid_picker_module,
    *,
    robot_uri: Optional[str],
    approach_mm: float,
    drop_mm: float,
) -> Tuple[str, str]:
    """Run drop-only manual program."""
    session = codroid_picker_module.SESSION
    if session.drop_target_mm is None:
        return t("Teach a drop position first."), no_update

    drop_approach_pose, drop_pose = _build_drop_poses(
        session,
        approach_mm=approach_mm,
        drop_mm=drop_mm,
    )
    program = ManualPickPlaceProgram(
        action="drop",
        drop_approach_pose=drop_approach_pose,
        drop_pose=drop_pose,
        approach_mm=approach_mm,
        drop_mm=drop_mm,
    )

    try:
        session.operation_in_progress = True
        session.operation_started_at = time.monotonic()
        result = await get_registry().start_instance(
            program,
            await codroid_picker_module._get_robot_api(
                robot_uri or codroid_picker_module.DEFAULT_ROBOT_URI
            ),
        )
        if result.status == ProgramStatus.COMPLETED:
            session.last_action = t("Drop completed.")
            return t("Drop command sent."), _drop_metrics_text(session)
        return t("Drop failed: {error}", error=result.message), no_update
    except Exception as exc:  # noqa: BLE001
        codroid_picker_module.LOGGER.exception("Drop action failed")
        return t("Drop action failed: {error}", error=exc), no_update
    finally:
        session.operation_in_progress = False
        session.operation_started_at = None


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
        _selected_detection: Optional[str],
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
        import app.codroid_picker as codroid_picker_module

        trigger = callback_context.triggered_id
        if trigger is None:
            return no_update, no_update
        if codroid_picker_module._robot_control_released(control_state):
            return t("Robot control is released. Acquire control first."), no_update
        if codroid_picker_module.SESSION.operation_in_progress:
            return t("Operation already in progress, please wait..."), no_update

        codroid_picker_module.LOGGER.info("Pick/place callback triggered: %s", trigger)

        if trigger == "teach-drop-position":
            return await _handle_teach_drop_position(codroid_picker_module, robot_uri=robot_uri)

        if trigger == "return-to-drop-position":
            return await _handle_return_to_drop_position(codroid_picker_module, robot_uri=robot_uri)

        frame, _depth = codroid_picker_module.get_latest_frame()
        if frame is None:
            return t("Camera stream is not running."), no_update

        approach_mm = codroid_picker_module._safe_float(approach_height, -30.0)
        pick_mm = codroid_picker_module._safe_float(pick_height, 0.0)
        drop_mm = codroid_picker_module._safe_float(drop_height, 0.0)

        if trigger in {"manual-pick-closest", "manual-pick-place-target"}:
            program, action, error_message = _prepare_pick_program(
                codroid_picker_module,
                trigger=trigger,
                approach_mm=approach_mm,
                pick_mm=pick_mm,
                drop_mm=drop_mm,
                tracking_mode=tracking_mode,
                segment_min=segment_min,
                segment_max=segment_max,
                segment_fraction=segment_fraction,
            )
            if error_message is not None or program is None or action is None:
                return error_message or t("Could not prepare pick program."), no_update
            return await _run_manual_program(
                codroid_picker_module,
                program=program,
                action=action,
                robot_uri=robot_uri,
            )

        if trigger == "execute-drop":
            return await _handle_execute_drop(
                codroid_picker_module,
                robot_uri=robot_uri,
                approach_mm=approach_mm,
                drop_mm=drop_mm,
            )

        return no_update, no_update
