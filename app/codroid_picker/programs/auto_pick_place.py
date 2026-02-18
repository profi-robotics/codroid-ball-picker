"""Automated pick and place robot program.

This program runs a continuous loop picking detected objects and
dropping them at the designated drop location.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Optional, Tuple, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from codroid_api import CodroidAPI

import time

from app.codroid_picker.programs.base import (
    RobotProgram,
    ProgramStatus,
    ProgramResult,
    OperationPaused,
    OperationAbort,
)
from app.codroid_picker.app_instance import registry
from app.codroid_picker.robot_motion import (
    ensure_operation_active,
    wait_for_operation_resume,
    move_linear_follow_target,
    command_gripper,
    ensure_coordinate_active,
    log_move_target,
    resolve_gripper_limits,
)
from app.codroid_picker.pick_place_helpers import build_drop_poses, select_closest_detection
from app.codroid_picker.target_planner import (
    PickPlan,
    TargetCandidate,
    plan_pick_strategy,
    rank_targets,
)

LOGGER = logging.getLogger(__name__)


class AutoPickPlaceProgram(RobotProgram):
    """Automated pick and place robot program.

    Runs a continuous loop:
    1. Detect objects in workspace
    2. Pick the nearest valid target
    3. Move to drop location
    4. Drop the object
    5. Repeat until stopped

    The loop respects pause/stop requests and handles workspace
    intrusion detection (objects entering the work area).
    """

    @property
    def name(self) -> str:
        return "auto_pick_place"

    @property
    def label(self) -> str:
        return "Auto Pick & Place"

    def check_prerequisites(self) -> Tuple[bool, str]:
        """Check if auto operation can start."""
        from app.codroid_picker.session import SESSION

        # Check robot posture stream
        session = registry().get_or_create_robot_session()
        if not session.posture_seen():
            return False, "Robot posture stream not available."

        # Check gripper
        gripper = registry().gripper or SESSION.gripper
        gripper_defaults = registry().gripper_defaults
        if gripper is None or not gripper.isConnected(t_index=gripper_defaults["t_index"]):
            return False, "Gripper not connected."

        # Check workspace
        if SESSION.workspace_definition is None:
            return False, "Workspace not defined."

        # Check robot calibration
        if SESSION.robot_calibration_min_z is None:
            return False, "Robot calibration not complete."

        # Check drop target
        if SESSION.drop_target_mm is None:
            return False, "Drop target not defined."

        # Check model
        if not SESSION.model_loaded:
            return False, "Detection model not loaded."

        # Check camera stream
        camera_stream = SESSION.camera_stream
        if camera_stream is None or not camera_stream.is_running():
            return False, "Camera stream is not running."

        # Check camera calibration
        if SESSION.calibration_bundle is None:
            return False, "Camera not calibrated."

        return True, "Ready for auto operation."

    @staticmethod
    def _resolve_run_orientation() -> Tuple[float, float, float]:
        """Resolve a stable orientation for auto mode to avoid wrist re-rotation."""
        from app.codroid_picker.session import SESSION

        try:
            robot_session = registry().get_or_create_robot_session()
            if robot_session.posture_seen():
                posture = robot_session.position_snapshot()
                return (
                    float(posture.a),
                    float(posture.b),
                    float(posture.c),
                )
        except Exception as exc:  # noqa: BLE001
            LOGGER.debug("Auto: failed to read posture orientation: %s", exc)

        if SESSION.drop_target_orientation is not None:
            return (
                float(SESSION.drop_target_orientation[0]),
                float(SESSION.drop_target_orientation[1]),
                float(SESSION.drop_target_orientation[2]),
            )
        return (0.0, 0.0, 0.0)

    async def execute(self, robot_api: "CodroidAPI") -> ProgramResult:
        """Execute the auto pick and place loop."""
        from app.codroid_picker.session import SESSION

        settings = registry().settings
        coordinate_id = registry().robot_coordinate_id
        workspace_def = SESSION.workspace_definition

        if workspace_def is None:
            return ProgramResult(
                status=ProgramStatus.ERROR,
                message="Workspace not defined.",
            )

        if SESSION.drop_target_mm is None:
            return ProgramResult(
                status=ProgramStatus.ERROR,
                message="Drop target not defined.",
            )

        # Power on and ensure coordinate system
        self.update_message("Powering on robot...")
        LOGGER.info("Auto: Starting power on sequence")
        import sys
        sys.stdout.flush()
        sys.stderr.flush()

        try:
            await asyncio.wait_for(robot_api.power_on(), timeout=2.0)
            LOGGER.info("Auto: Power on command sent successfully")
            sys.stdout.flush()
            sys.stderr.flush()
        except asyncio.TimeoutError:
            LOGGER.warning("Auto: Power on timed out, continuing anyway")
            sys.stdout.flush()
        except Exception as exc:
            LOGGER.warning(f"Auto: Power on failed: {exc}, continuing anyway")
            sys.stdout.flush()

        # Skip the sleep and data refresh - go straight to coordinate system
        LOGGER.info(
            "Auto: Skipping sleep and session refresh, proceeding to coordinate setup")
        sys.stdout.flush()

        # Ensure coordinate system is active
        self.update_message("Setting up coordinate system...")
        LOGGER.info(f"Auto: Setting up coordinate system {coordinate_id}")
        sys.stdout.flush()

        try:
            await asyncio.wait_for(ensure_coordinate_active(robot_api, coordinate_id), timeout=2.0)
            LOGGER.info("Auto: Coordinate system setup complete")
            sys.stdout.flush()
        except asyncio.TimeoutError:
            LOGGER.warning(
                "Auto: Coordinate system setup timed out, continuing anyway")
            sys.stdout.flush()
        except Exception as exc:
            LOGGER.warning(
                f"Auto: Coordinate system setup failed: {exc}, continuing anyway")
            sys.stdout.flush()

        SESSION.auto_waiting_at_drop = False
        cycles_completed = 0
        cycles_failed = 0
        run_orientation = self._resolve_run_orientation()
        LOGGER.info(
            "Auto: using fixed orientation a=%.2f b=%.2f c=%.2f",
            run_orientation[0],
            run_orientation[1],
            run_orientation[2],
        )

        self.update_message("Starting auto loop...")
        LOGGER.info("Auto: Entering main loop")

        try:
            while True:
                # Check for pause/stop
                try:
                    ensure_operation_active(self._run_id)
                except OperationPaused:
                    self.update_message("Paused, waiting to resume...")
                    await wait_for_operation_resume(self._run_id)
                    continue

                # Get configuration
                config = SESSION.auto_config or {}
                target_labels = config.get(
                    "target_labels", settings.auto_target_labels
                )
                # Support legacy single target_label config
                if "target_label" in config and config["target_label"] not in target_labels:
                    target_labels = [config["target_label"]]
                drop_ignore_radius_mm = float(
                    config.get("drop_ignore_radius_mm",
                               settings.auto_drop_ignore_radius_mm)
                )
                approach_mm = float(config.get(
                    "approach_mm", settings.auto_approach_mm))
                pick_mm = float(config.get("pick_mm", settings.auto_pick_mm))
                drop_mm = float(config.get("drop_mm", settings.auto_drop_mm))
                estimate_width = bool(config.get(
                    "estimate_width", settings.auto_estimate_width))
                tracking_mode = str(
                    config.get("tracking_mode", settings.tracking_mode)
                ).strip().lower()
                cluster_neighbor_mm = float(
                    config.get("cluster_neighbor_mm", settings.auto_cluster_neighbor_mm)
                )
                rotation_clearance_mm = float(
                    config.get("rotation_clearance_mm", settings.auto_rotation_clearance_mm)
                )
                rotation_step_deg = float(
                    config.get("rotation_step_deg", settings.auto_rotation_step_deg)
                )
                rotation_max_delta_deg = float(
                    config.get("rotation_max_delta_deg", settings.auto_rotation_max_delta_deg)
                )
                collision_body_radius_mm = float(
                    config.get("collision_body_radius_mm", settings.auto_collision_body_radius_mm)
                )
                collision_finger_reach_mm = float(
                    config.get("collision_finger_reach_mm", settings.auto_collision_finger_reach_mm)
                )
                separation_push_mm = float(
                    config.get("separation_push_mm", settings.auto_separation_push_mm)
                )
                separation_entry_mm = float(
                    config.get("separation_entry_mm", settings.auto_separation_entry_mm)
                )
                separation_max_attempts = int(
                    config.get("separation_max_attempts", settings.auto_separation_max_attempts)
                )
                rescan_delay_s = float(
                    config.get("rescan_delay_s", settings.auto_rescan_delay_s)
                )

                # Refresh detections from current frame
                from app.codroid_picker.auto_operation import _refresh_follow_detections, _detection_coord5_mm
                detections = _refresh_follow_detections()

                # Populate coord5_mm for all detections
                for det in detections:
                    if det.get("coord5_mm") is None:
                        coord5 = _detection_coord5_mm(det, workspace_def)
                        if coord5 is not None:
                            det["coord5_mm"] = coord5

                LOGGER.debug(f"Auto loop: {len(detections)} detections")

                # Filter targets
                targets = self._filter_targets(
                    detections,
                    workspace_def,
                    target_labels=target_labels,
                    drop_target_mm=SESSION.drop_target_mm,
                    drop_ignore_radius_mm=drop_ignore_radius_mm,
                )

                if not targets:
                    # No targets - wait at drop position
                    if not SESSION.auto_waiting_at_drop:
                        self.update_message(
                            "No targets, moving to drop position...")
                        await self._move_to_drop(
                            robot_api,
                            approach_mm=approach_mm,
                            drop_mm=drop_mm,
                            open_gripper=not SESSION.robot_has_object,
                            orientation=run_orientation,
                        )
                        SESSION.auto_waiting_at_drop = True

                    await asyncio.sleep(settings.auto_idle_sleep_s)
                    continue

                ranked_candidates = self._rank_target_candidates(
                    targets,
                    neighbor_radius_mm=cluster_neighbor_mm,
                )
                if not ranked_candidates:
                    await asyncio.sleep(settings.auto_idle_sleep_s)
                    continue

                SESSION.auto_waiting_at_drop = False
                picked_this_cycle = False
                for candidate in ranked_candidates:
                    pick_plan = plan_pick_strategy(
                        candidate,
                        ranked_candidates,
                        current_c_deg=float(run_orientation[2]),
                        rotation_step_deg=rotation_step_deg,
                        rotation_max_delta_deg=rotation_max_delta_deg,
                        body_radius_mm=collision_body_radius_mm,
                        finger_reach_mm=collision_finger_reach_mm,
                        clearance_mm=rotation_clearance_mm,
                        separation_entry_mm=separation_entry_mm,
                        separation_push_mm=separation_push_mm,
                    )

                    target_coord5 = candidate.coord5_xy
                    self.update_message(
                        f"Picking target at ({target_coord5[0]:.1f}, {target_coord5[1]:.1f})..."
                    )
                    if pick_plan.mode == "separate_then_rescan":
                        LOGGER.info(
                            "Auto: target crowded, trying side-separation before pick."
                        )
                        separated = False
                        for attempt in range(1, max(1, separation_max_attempts) + 1):
                            self.update_message(
                                f"Separating cluster attempt {attempt}/{max(1, separation_max_attempts)}..."
                            )
                            await self._execute_separation_nudge(
                                robot_api,
                                pick_plan=pick_plan,
                                approach_mm=approach_mm,
                                orientation=run_orientation,
                            )
                            await asyncio.sleep(max(0.0, rescan_delay_s))

                            detections = _refresh_follow_detections()
                            for det in detections:
                                if det.get("coord5_mm") is None:
                                    coord5 = _detection_coord5_mm(det, workspace_def)
                                    if coord5 is not None:
                                        det["coord5_mm"] = coord5
                            targets = self._filter_targets(
                                detections,
                                workspace_def,
                                target_labels=target_labels,
                                drop_target_mm=SESSION.drop_target_mm,
                                drop_ignore_radius_mm=drop_ignore_radius_mm,
                            )
                            ranked_candidates = self._rank_target_candidates(
                                targets,
                                neighbor_radius_mm=cluster_neighbor_mm,
                            )
                            if not ranked_candidates:
                                break
                            candidate = ranked_candidates[0]
                            pick_plan = plan_pick_strategy(
                                candidate,
                                ranked_candidates,
                                current_c_deg=float(run_orientation[2]),
                                rotation_step_deg=rotation_step_deg,
                                rotation_max_delta_deg=rotation_max_delta_deg,
                                body_radius_mm=collision_body_radius_mm,
                                finger_reach_mm=collision_finger_reach_mm,
                                clearance_mm=rotation_clearance_mm,
                                separation_entry_mm=separation_entry_mm,
                                separation_push_mm=separation_push_mm,
                            )
                            LOGGER.info(
                                "Auto: re-plan after separation attempt %d -> %s",
                                attempt,
                                pick_plan.mode,
                            )
                            if pick_plan.mode == "rotate_pick":
                                separated = True
                                break
                        if not separated:
                            continue

                    target = candidate.detection
                    target_coord5 = candidate.coord5_xy
                    LOGGER.info(
                        "Auto: selected C-axis %.1f for pick at (%.1f, %.1f)",
                        pick_plan.final_c,
                        target_coord5[0],
                        target_coord5[1],
                    )

                    try:
                        await self._execute_pick(
                            robot_api,
                            target_coord5=target_coord5,
                            approach_mm=approach_mm,
                            pick_mm=pick_mm,
                            estimate_width=estimate_width,
                            target=target,
                            target_labels=target_labels,
                            tracking_mode=tracking_mode,
                            orientation=run_orientation,
                            final_c=pick_plan.final_c,
                        )
                        await self._execute_drop(
                            robot_api,
                            approach_mm=approach_mm,
                            drop_mm=drop_mm,
                            orientation=run_orientation,
                        )
                        cycles_completed += 1
                        picked_this_cycle = True
                        self.update_message(
                            f"Cycle complete. Total: {cycles_completed} picked, {cycles_failed} failed."
                        )
                        break
                    except Exception as exc:
                        LOGGER.warning(f"Pick/place cycle failed for candidate: {exc}")
                        continue

                if not picked_this_cycle:
                    cycles_failed += 1
                    try:
                        await self._move_to_drop(
                            robot_api,
                            approach_mm=approach_mm,
                            drop_mm=drop_mm,
                            open_gripper=True,
                            orientation=run_orientation,
                        )
                    except Exception:
                        pass

        except OperationAbort:
            pass

        message = f"Auto operation stopped. {cycles_completed} cycles completed, {cycles_failed} failed."
        self.finish(status="idle", message=message)
        return ProgramResult(
            status=ProgramStatus.COMPLETED,
            message=message,
            data={"cycles_completed": cycles_completed,
                  "cycles_failed": cycles_failed},
        )

    async def _execute_pick(
        self,
        robot_api: "CodroidAPI",
        *,
        target_coord5: Tuple[float, float],
        approach_mm: float,
        pick_mm: float,
        estimate_width: bool,
        target: dict,
        target_labels: list[str],
        tracking_mode: str,
        orientation: Tuple[float, float, float],
        final_c: float,
    ) -> None:
        """Execute a pick operation."""
        from app.codroid_picker.session import SESSION
        from app.codroid_picker.auto_operation import _detection_coord5_mm

        settings = registry().settings
        coordinate_id = registry().robot_coordinate_id
        use_segment_tracking = tracking_mode == "segment"

        # Determine gripper width
        gripper_close = settings.gripper_profile.close_width_default
        if estimate_width and "bbox" in target:
            # Estimate width from bounding box
            bbox = target["bbox"]
            width_px = bbox[2] - bbox[0]
            # Simple estimation - would need calibration for accuracy
            width_mm = width_px * 0.2  # rough estimate
            gripper_close = max(gripper_close, width_mm * 0.9)

        posture = registry().get_or_create_robot_session().position_snapshot()
        transit_z = float(posture.z)
        base_orientation = (
            float(orientation[0]),
            float(orientation[1]),
            float(orientation[2]),
        )
        final_orientation = (
            float(orientation[0]),
            float(orientation[1]),
            float(final_c),
        )

        def _pose_from_coord5(
            coord5_xy: Tuple[float, float],
            z: float,
            orient: Tuple[float, float, float],
        ) -> dict:
            return {
                "x": float(coord5_xy[0]),
                "y": float(coord5_xy[1]),
                "z": float(z),
                "a": float(orient[0]),
                "b": float(orient[1]),
                "c": float(orient[2]),
            }

        static_coord5_xy = (float(target_coord5[0]), float(target_coord5[1]))
        tracking_state = {
            "xy": static_coord5_xy,
            "lost_count": 0,
        }

        def _select_tracking_coord5() -> Optional[Tuple[float, float]]:
            detections = SESSION.detections or []
            if target_labels:
                detections = [
                    det for det in detections
                    if str(det.get("label", "")).strip() in target_labels
                ]
            if not detections:
                tracking_state["lost_count"] += 1
                if tracking_state["lost_count"] == 5:
                    LOGGER.info(
                        "Auto: target not visible for %d frames; using last known position",
                        tracking_state["lost_count"],
                    )
                return None

            coord5 = select_closest_detection(
                detections,
                reference_xy=tracking_state["xy"],
                coord5_fn=lambda d: _detection_coord5_mm(
                    d, SESSION.workspace_definition),
            )
            coord5_xy = _detection_coord5_mm(
                coord5, SESSION.workspace_definition)
            if coord5_xy is None:
                tracking_state["lost_count"] += 1
                return None

            tracking_state["lost_count"] = 0
            tracking_state["xy"] = (float(coord5_xy[0]), float(coord5_xy[1]))
            return tracking_state["xy"]

        def _pick_transit_pose_fn() -> Optional[dict]:
            if use_segment_tracking:
                coord5_xy = _select_tracking_coord5()
                if coord5_xy is None:
                    coord5_xy = tracking_state["xy"]
            else:
                coord5_xy = static_coord5_xy
            return _pose_from_coord5(coord5_xy, transit_z, base_orientation)

        def _pick_approach_pose_fn() -> Optional[dict]:
            if use_segment_tracking:
                coord5_xy = _select_tracking_coord5()
                if coord5_xy is None:
                    coord5_xy = tracking_state["xy"]
            else:
                coord5_xy = static_coord5_xy
            return _pose_from_coord5(coord5_xy, approach_mm, base_orientation)

        def _pick_pose_fn() -> Optional[dict]:
            if use_segment_tracking:
                coord5_xy = tracking_state["xy"]
            else:
                coord5_xy = static_coord5_xy
            return _pose_from_coord5(coord5_xy, pick_mm, final_orientation)

        # Move XY transit first at current Z and with current orientation.
        await wait_for_operation_resume(self._run_id)
        transit_pose = _pick_transit_pose_fn()
        if transit_pose is None:
            raise RuntimeError("Pick transit pose unavailable.")
        await self._move_linear_with_hold(
            robot_api,
            transit_pose,
            coordinate_id,
            label="auto_pick_transit_xy",
            hold_s=settings.auto_hold_s,
        )

        # Move to approach height.
        await wait_for_operation_resume(self._run_id)
        approach_pose = _pick_approach_pose_fn()
        if approach_pose is None:
            raise RuntimeError("Pick approach pose unavailable.")

        log_move_target("pick_approach", approach_pose,
                        coordinate_id=coordinate_id)

        if use_segment_tracking:
            try:
                LOGGER.debug("Auto: Starting segment tracking approach move")
                await asyncio.wait_for(
                    move_linear_follow_target(
                        robot_api,
                        _pick_approach_pose_fn,
                        label="auto_pick_approach",
                        coordinate_id=coordinate_id,
                        settle_on_stall=True,
                        run_id=self._run_id,
                    ),
                    timeout=30.0,
                )
                LOGGER.debug("Auto: Segment tracking approach move complete")
            except asyncio.TimeoutError:
                LOGGER.error("Auto: Pick approach move timed out after 30s")
                raise RuntimeError("Pick approach move timed out")
        else:
            LOGGER.debug(
                "Auto: Starting _move_linear_with_hold approach move (hold=2.0s)")
            await self._move_linear_with_hold(
                robot_api,
                approach_pose,
                coordinate_id,
                label="auto_pick_approach",
                hold_s=settings.auto_hold_s,
            )
            LOGGER.debug("Auto: _move_linear_with_hold approach move complete")

        # Align orientation only above the target, then descend.
        await wait_for_operation_resume(self._run_id)
        orientation_pose = dict(approach_pose)
        orientation_pose["a"] = float(final_orientation[0])
        orientation_pose["b"] = float(final_orientation[1])
        orientation_pose["c"] = float(final_orientation[2])
        await self._move_linear_with_hold(
            robot_api,
            orientation_pose,
            coordinate_id,
            label="auto_pick_align_orientation",
            hold_s=settings.auto_hold_s,
        )

        # Move to pick
        LOGGER.debug("Auto: Approach done, preparing pick move")
        await wait_for_operation_resume(self._run_id)
        pick_pose = _pick_pose_fn()
        if pick_pose is None:
            raise RuntimeError("Pick pose unavailable.")

        log_move_target("pick_pose", pick_pose, coordinate_id=coordinate_id)

        if use_segment_tracking:
            await move_linear_follow_target(
                robot_api,
                _pick_pose_fn,
                label="auto_pick_pose",
                coordinate_id=coordinate_id,
                settle_on_stall=True,
                run_id=self._run_id,
            )
        else:
            await self._move_linear_with_hold(
                robot_api,
                pick_pose,
                coordinate_id,
                label="auto_pick_pose",
                hold_s=settings.auto_hold_s,
            )

        # Close gripper
        await wait_for_operation_resume(self._run_id)
        grip_force = min(
            settings.gripper_profile.force_default * 2,
            settings.gripper_profile.force_max,
        )
        result = command_gripper(gripper_close, "closed", force=grip_force)
        LOGGER.info(f"Gripper close result: {result}")
        SESSION.robot_has_object = True

        # Retreat
        await wait_for_operation_resume(self._run_id)
        retreat_pose = dict(orientation_pose)
        if retreat_pose is None:
            retreat_pose = approach_pose
        await self._move_linear_with_hold(
            robot_api,
            retreat_pose,
            coordinate_id,
            label="auto_pick_retreat",
            hold_s=settings.auto_hold_s,
        )

    def _rank_target_candidates(
        self,
        targets: list[dict],
        *,
        neighbor_radius_mm: float,
    ) -> list[TargetCandidate]:
        """Rank valid targets for collision-aware picking."""
        from app.codroid_picker.session import SESSION
        from app.codroid_picker.auto_operation import _detection_coord5_mm

        robot_xy = (0.0, 0.0)
        try:
            robot_session = registry().get_or_create_robot_session()
            if robot_session.posture_seen():
                pos = robot_session.position_snapshot()
                robot_xy = (float(pos.x), float(pos.y))
        except Exception as exc:  # noqa: BLE001
            LOGGER.debug("Auto: failed to read robot XY for ranking: %s", exc)

        return rank_targets(
            targets,
            robot_xy=robot_xy,
            coord5_fn=lambda det: _detection_coord5_mm(det, SESSION.workspace_definition),
            neighbor_radius_mm=neighbor_radius_mm,
        )

    async def _execute_separation_nudge(
        self,
        robot_api: "CodroidAPI",
        *,
        pick_plan: PickPlan,
        approach_mm: float,
        orientation: Tuple[float, float, float],
    ) -> None:
        """Push a clustered target sideways, then retreat for camera re-scan."""
        settings = registry().settings
        coordinate_id = registry().robot_coordinate_id
        separation = pick_plan.separation
        if separation is None:
            raise RuntimeError("Separation plan missing.")

        command_gripper(settings.gripper_profile.open_width_default, "opened")

        entry_pose = {
            "x": float(separation.entry_xy[0]),
            "y": float(separation.entry_xy[1]),
            "z": float(approach_mm),
            "a": float(orientation[0]),
            "b": float(orientation[1]),
            "c": float(orientation[2]),
        }
        push_pose = dict(entry_pose)
        push_pose["x"] = float(separation.push_end_xy[0])
        push_pose["y"] = float(separation.push_end_xy[1])

        await self._move_linear_with_hold(
            robot_api,
            entry_pose,
            coordinate_id,
            label="auto_cluster_nudge_entry",
            hold_s=settings.auto_hold_s,
        )
        await self._move_linear_with_hold(
            robot_api,
            push_pose,
            coordinate_id,
            label="auto_cluster_nudge_push",
            hold_s=settings.auto_hold_s,
        )
        await self._move_linear_with_hold(
            robot_api,
            entry_pose,
            coordinate_id,
            label="auto_cluster_nudge_retreat",
            hold_s=settings.auto_hold_s,
        )

    async def _execute_drop(
        self,
        robot_api: "CodroidAPI",
        *,
        approach_mm: float,
        drop_mm: float,
        orientation: Tuple[float, float, float],
    ) -> None:
        """Execute a drop operation."""
        from app.codroid_picker.session import SESSION

        settings = registry().settings
        coordinate_id = registry().robot_coordinate_id

        drop_target = SESSION.drop_target_robot
        if drop_target is None:
            raise RuntimeError("Drop target not defined")

        # Use shared helper to build poses
        approach_pose, drop_pose = build_drop_poses(
            drop_robot_xyz=drop_target,
            orientation=orientation,
            approach_mm=approach_mm,
            drop_mm=drop_mm,
        )

        # Move to approach
        await wait_for_operation_resume(self._run_id)
        await self._move_linear_with_hold(
            robot_api,
            approach_pose,
            coordinate_id,
            label="drop_approach",
            hold_s=settings.auto_hold_s,
        )

        # Move to drop position
        await wait_for_operation_resume(self._run_id)
        await self._move_linear_with_hold(
            robot_api,
            drop_pose,
            coordinate_id,
            label="drop_pose",
            hold_s=settings.auto_hold_s,
        )

        # Open gripper
        await wait_for_operation_resume(self._run_id)
        open_width = settings.gripper_profile.open_width_default
        gripper_close = settings.gripper_profile.close_width_default
        if open_width <= gripper_close:
            gripper = registry().gripper
            if gripper is not None:
                _min_width, max_width = resolve_gripper_limits(gripper)
                if max_width > open_width:
                    LOGGER.warning(
                        "Open width %.1f <= close width %.1f; using max width %.1f",
                        open_width,
                        gripper_close,
                        max_width,
                    )
                    open_width = max_width
        result = command_gripper(open_width, "opened")
        LOGGER.info(f"Gripper open result: {result}")

        # Wait for gripper to finish
        gripper = registry().gripper
        gripper_defaults = registry().gripper_defaults
        if gripper is not None and gripper.isConnected(t_index=gripper_defaults["t_index"]):
            max_polls = 20  # 2 seconds max
            for _ in range(max_polls):
                try:
                    if not gripper.isBusy(t_index=gripper_defaults["t_index"]):
                        break
                except Exception:
                    pass
                await asyncio.sleep(0.1)

        SESSION.robot_has_object = False

        # Retreat
        await wait_for_operation_resume(self._run_id)
        await self._move_linear_with_hold(
            robot_api,
            approach_pose,
            coordinate_id,
            label="drop_retreat",
            hold_s=settings.auto_hold_s,
        )

    async def _move_to_drop(
        self,
        robot_api: "CodroidAPI",
        *,
        approach_mm: float,
        drop_mm: float,
        open_gripper: bool,
        orientation: Tuple[float, float, float],
    ) -> None:
        """Move to drop position and optionally open gripper."""
        from app.codroid_picker.session import SESSION

        settings = registry().settings
        coordinate_id = registry().robot_coordinate_id

        drop_target = SESSION.drop_target_robot
        if drop_target is None:
            return

        # Use shared helper to build poses
        approach_pose, drop_pose = build_drop_poses(
            drop_robot_xyz=drop_target,
            orientation=orientation,
            approach_mm=approach_mm,
            drop_mm=drop_mm,
        )

        await wait_for_operation_resume(self._run_id)
        await self._move_linear_with_hold(
            robot_api,
            approach_pose,
            coordinate_id,
            label="move_to_drop_approach",
            hold_s=settings.auto_hold_s,
        )

        await wait_for_operation_resume(self._run_id)
        await self._move_linear_with_hold(
            robot_api,
            drop_pose,
            coordinate_id,
            label="move_to_drop_pose",
            hold_s=settings.auto_hold_s,
        )

        if open_gripper:
            gripper_open = settings.gripper_profile.open_width_default
            command_gripper(gripper_open, "opened")
            SESSION.robot_has_object = False

    async def _move_linear_with_hold(
        self,
        robot_api: "CodroidAPI",
        target_pose: dict,
        coordinate_id: int,
        *,
        label: str = "move",
        hold_s: float = 2.0,
    ) -> None:
        """Execute a linear move with optional hold.

        This is a self-contained movement loop similar to PlateTestProgram's
        implementation. Unlike the base class move_with_hold which delegates
        to execute_linear_move_with_hold, this method handles the heartbeat
        loop directly to avoid asyncio task cancellation issues in Dash callbacks.
        """
        settings = registry().settings
        session = registry().get_or_create_robot_session()

        await self.wait_for_resume()

        log_move_target(label, target_pose, coordinate_id=coordinate_id)

        await robot_api.set_target_pos_type(robot_api.config.target_pos_types.cpos)
        await robot_api.set_target_cpos(
            robot_api.attach_coordinate_to_cpos(
                robot_api.build_target_cpos(**target_pose),
                coordinate_id=coordinate_id,
            )
        )
        await robot_api.set_robot_command(robot_api.config.commands.move_target_linear)

        # Wait for motion to settle - use auto-specific faster settings
        timeout_s = getattr(settings, 'auto_move_timeout_s', 15.0)
        settle_s = getattr(settings, 'auto_settle_s', 0.1)
        tolerance_mm = getattr(settings, 'auto_tolerance_mm', 2.0)
        z_tolerance_mm = getattr(settings, 'auto_z_tolerance_mm', 3.0)

        deadline = time.monotonic() + timeout_s
        settle_since: float | None = None

        while time.monotonic() < deadline:
            self.ensure_active()
            await robot_api.send_command_heartbeat()
            await asyncio.sleep(0.1)

            posture = session.position_snapshot()
            curr_xyz = (posture.x, posture.y, posture.z)
            target_xyz = (target_pose["x"], target_pose["y"], target_pose["z"])

            if (
                abs(curr_xyz[0] - target_xyz[0]) <= tolerance_mm
                and abs(curr_xyz[1] - target_xyz[1]) <= tolerance_mm
                and abs(curr_xyz[2] - target_xyz[2]) <= z_tolerance_mm
            ):
                if settle_s <= 0:
                    break
                if settle_since is None:
                    settle_since = time.monotonic()
                elif time.monotonic() - settle_since >= settle_s:
                    break
            else:
                settle_since = None

        # Hold if requested
        if hold_s > 0:
            hold_deadline = time.monotonic() + hold_s
            while time.monotonic() < hold_deadline:
                self.ensure_active()
                await robot_api.send_command_heartbeat()
                await asyncio.sleep(0.1)

        await robot_api.clear_target_position()
        await robot_api.stop_command()

    def _filter_targets(
        self,
        detections: list,
        workspace_def,
        *,
        target_labels: list[str],
        drop_target_mm: Optional[Tuple[float, float]],
        drop_ignore_radius_mm: float,
    ) -> list:
        """Filter detections to valid pick targets."""
        targets = []

        for det in detections:
            # Check label - accept any of the target labels
            det_label = str(det.get("label", "")).strip()
            if det_label not in target_labels:
                continue

            # Check for workspace coordinates
            ws_mm = det.get("workspace_mm")
            if ws_mm is None:
                continue

            # Check if too close to drop zone
            if drop_target_mm is not None:
                dist = np.hypot(
                    ws_mm[0] - drop_target_mm[0],
                    ws_mm[1] - drop_target_mm[1],
                )
                if dist < drop_ignore_radius_mm:
                    continue

            targets.append(det)

        return targets
