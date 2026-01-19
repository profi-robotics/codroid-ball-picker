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
    drop_with_verify,
    ensure_coordinate_active,
    log_move_target,
    resolve_gripper_limits,
)
from app.codroid_picker.pick_place_helpers import build_drop_poses, select_closest_detection

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

        # Check camera calibration
        if SESSION.calibration_bundle is None:
            return False, "Camera not calibrated."

        return True, "Ready for auto operation."

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
        LOGGER.info("Auto: Skipping sleep and session refresh, proceeding to coordinate setup")
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
            LOGGER.warning("Auto: Coordinate system setup timed out, continuing anyway")
            sys.stdout.flush()
        except Exception as exc:
            LOGGER.warning(f"Auto: Coordinate system setup failed: {exc}, continuing anyway")
            sys.stdout.flush()

        SESSION.auto_waiting_at_drop = False
        cycles_completed = 0
        cycles_failed = 0

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
                target_label = str(config.get(
                    "target_label", settings.auto_target_label))
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
                    target_label=target_label,
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
                        )
                        SESSION.auto_waiting_at_drop = True

                    await asyncio.sleep(settings.auto_idle_sleep_s)
                    continue

                # Select nearest target
                target = self._select_nearest_target(targets, workspace_def)
                if target is None:
                    await asyncio.sleep(settings.auto_idle_sleep_s)
                    continue

                # Get target coordinates
                target_coord5 = target.get("coord5_mm")
                if target_coord5 is None:
                    LOGGER.warning("Target missing coord5 coordinates")
                    await asyncio.sleep(settings.auto_idle_sleep_s)
                    continue

                SESSION.auto_waiting_at_drop = False
                self.update_message(
                    f"Picking target at ({target_coord5[0]:.1f}, {target_coord5[1]:.1f})...")

                try:
                    # Execute pick
                    await self._execute_pick(
                        robot_api,
                        target_coord5=target_coord5,
                        approach_mm=approach_mm,
                        pick_mm=pick_mm,
                        estimate_width=estimate_width,
                        target=target,
                        target_label=target_label,
                        tracking_mode=tracking_mode,
                    )

                    # Execute drop
                    await self._execute_drop(
                        robot_api,
                        approach_mm=approach_mm,
                        drop_mm=drop_mm,
                    )

                    cycles_completed += 1
                    self.update_message(
                        f"Cycle complete. Total: {cycles_completed} picked, {cycles_failed} failed."
                    )

                except Exception as exc:
                    LOGGER.warning(f"Pick/place cycle failed: {exc}")
                    cycles_failed += 1
                    # Return to drop position on error
                    try:
                        await self._move_to_drop(
                            robot_api,
                            approach_mm=approach_mm,
                            drop_mm=drop_mm,
                            open_gripper=True,
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
        target_label: str,
        tracking_mode: str,
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

        def _pose_from_coord5(coord5_xy: Tuple[float, float], z: float) -> dict:
            return {
                "x": float(coord5_xy[0]),
                "y": float(coord5_xy[1]),
                "z": float(z),
                "a": 0.0,
                "b": 0.0,
                "c": 0.0,
            }

        static_coord5_xy = (float(target_coord5[0]), float(target_coord5[1]))
        tracking_state = {
            "xy": static_coord5_xy,
            "lost_count": 0,
        }

        def _select_tracking_coord5() -> Optional[Tuple[float, float]]:
            detections = SESSION.detections or []
            if target_label:
                detections = [
                    det for det in detections
                    if str(det.get("label", "")).strip() == str(target_label).strip()
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
                coord5_fn=lambda d: _detection_coord5_mm(d, SESSION.workspace_definition),
            )
            coord5_xy = _detection_coord5_mm(coord5, SESSION.workspace_definition)
            if coord5_xy is None:
                tracking_state["lost_count"] += 1
                return None

            tracking_state["lost_count"] = 0
            tracking_state["xy"] = (float(coord5_xy[0]), float(coord5_xy[1]))
            return tracking_state["xy"]

        def _pick_approach_pose_fn() -> Optional[dict]:
            if use_segment_tracking:
                coord5_xy = _select_tracking_coord5()
                if coord5_xy is None:
                    coord5_xy = tracking_state["xy"]
            else:
                coord5_xy = static_coord5_xy
            return _pose_from_coord5(coord5_xy, approach_mm)

        def _pick_pose_fn() -> Optional[dict]:
            if use_segment_tracking:
                coord5_xy = tracking_state["xy"]
            else:
                coord5_xy = static_coord5_xy
            return _pose_from_coord5(coord5_xy, pick_mm)

        # Move to approach
        await wait_for_operation_resume(self._run_id)
        approach_pose = _pick_approach_pose_fn()
        if approach_pose is None:
            raise RuntimeError("Pick approach pose unavailable.")
        
        log_move_target("pick_approach", approach_pose, coordinate_id=coordinate_id)
        
        if use_segment_tracking:
            try:
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
            except asyncio.TimeoutError:
                LOGGER.error("Auto: Pick approach move timed out after 30s")
                raise RuntimeError("Pick approach move timed out")
        else:
            await self.move_with_hold(
                approach_pose,
                coordinate_id=coordinate_id,
                hold_seconds=2.0,
            )

        # Move to pick
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
            await self.move_with_hold(
                pick_pose,
                coordinate_id=coordinate_id,
                hold_seconds=2.0,
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
        retreat_pose = _pick_approach_pose_fn()
        if retreat_pose is None:
            retreat_pose = approach_pose
        await self.move_with_hold(
            retreat_pose,
            coordinate_id=coordinate_id,
            hold_seconds=2.0,
        )

    async def _execute_drop(
        self,
        robot_api: "CodroidAPI",
        *,
        approach_mm: float,
        drop_mm: float,
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
            orientation=SESSION.drop_target_orientation,
            approach_mm=approach_mm,
            drop_mm=drop_mm,
        )

        # Move to approach
        await wait_for_operation_resume(self._run_id)
        log_move_target("drop_approach", approach_pose, coordinate_id=coordinate_id)
        await self.move_with_hold(
            approach_pose,
            coordinate_id=coordinate_id,
            hold_seconds=2.0,
        )

        # Move to drop position
        await wait_for_operation_resume(self._run_id)
        log_move_target("drop_pose", drop_pose, coordinate_id=coordinate_id)
        await self.move_with_hold(
            drop_pose,
            coordinate_id=coordinate_id,
            hold_seconds=2.0,
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
                        "Open width %.1f <= min width %.1f; using max width %.1f",
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
        log_move_target("drop_retreat", approach_pose, coordinate_id=coordinate_id)
        await self.move_with_hold(
            approach_pose,
            coordinate_id=coordinate_id,
            hold_seconds=2.0,
        )

    async def _move_to_drop(
        self,
        robot_api: "CodroidAPI",
        *,
        approach_mm: float,
        drop_mm: float,
        open_gripper: bool,
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
            orientation=SESSION.drop_target_orientation,
            approach_mm=approach_mm,
            drop_mm=drop_mm,
        )

        await wait_for_operation_resume(self._run_id)
        await self.move_with_hold(
            approach_pose,
            coordinate_id=coordinate_id,
            hold_seconds=2.0,
        )

        await wait_for_operation_resume(self._run_id)
        await self.move_with_hold(
            drop_pose,
            coordinate_id=coordinate_id,
            hold_seconds=2.0,
        )

        if open_gripper:
            gripper_open = settings.gripper_profile.open_width_default
            command_gripper(gripper_open, "opened")
            SESSION.robot_has_object = False

    def _filter_targets(
        self,
        detections: list,
        workspace_def,
        *,
        target_label: str,
        drop_target_mm: Optional[Tuple[float, float]],
        drop_ignore_radius_mm: float,
    ) -> list:
        """Filter detections to valid pick targets."""
        targets = []

        for det in detections:
            # Check label
            if det.get("label") != target_label:
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

    def _select_nearest_target(self, targets: list, workspace_def) -> Optional[dict]:
        """Select the nearest valid target from the list."""
        if not targets:
            return None

        # For now, just return the first target
        # Could be improved to calculate actual distance from gripper
        return targets[0]
