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
from app.codroid_picker.app_instance import (
    get_robot_session,
    get_settings,
    get_robot_coordinate_id,
    get_gripper_defaults,
    registry,
)
from app.codroid_picker.robot_motion import (
    ensure_operation_active,
    wait_for_operation_resume,
    move_linear_follow_target,
    command_gripper,
    drop_with_verify,
    ensure_coordinate_active,
)
from app.codroid_picker.pick_place_helpers import build_drop_poses

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
        session = get_robot_session()
        if not session.posture_seen():
            return False, "Robot posture stream not available."

        # Check gripper
        gripper = registry.gripper or SESSION.gripper
        gripper_defaults = get_gripper_defaults()
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

        settings = get_settings()
        coordinate_id = get_robot_coordinate_id()
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
        await robot_api.power_on()
        await ensure_coordinate_active(robot_api, coordinate_id)

        SESSION.auto_waiting_at_drop = False
        cycles_completed = 0
        cycles_failed = 0

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

                # Get detections
                detections = SESSION.detections or []
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
    ) -> None:
        """Execute a pick operation."""
        from app.codroid_picker.session import SESSION

        settings = get_settings()
        coordinate_id = get_robot_coordinate_id()

        # Determine gripper width
        gripper_close = settings.gripper_profile.close_width_default
        if estimate_width and "bbox" in target:
            # Estimate width from bounding box
            bbox = target["bbox"]
            width_px = bbox[2] - bbox[0]
            # Simple estimation - would need calibration for accuracy
            width_mm = width_px * 0.2  # rough estimate
            gripper_close = max(gripper_close, width_mm * 0.9)

        def _fetch_pose(z: float) -> dict:
            return {
                "x": float(target_coord5[0]),
                "y": float(target_coord5[1]),
                "z": float(z),
                "a": 0.0, "b": 0.0, "c": 0.0,
            }

        # Move to approach
        await wait_for_operation_resume(self._run_id)
        await move_linear_follow_target(
            robot_api,
            lambda: _fetch_pose(approach_mm),
            label="auto_pick_approach",
            coordinate_id=coordinate_id,
            run_id=self._run_id,
        )

        # Move to pick
        await wait_for_operation_resume(self._run_id)
        await move_linear_follow_target(
            robot_api,
            lambda: _fetch_pose(pick_mm),
            label="auto_pick_pose",
            coordinate_id=coordinate_id,
            run_id=self._run_id,
        )

        # Close gripper
        await wait_for_operation_resume(self._run_id)
        command_gripper(gripper_close, "closed")
        SESSION.robot_has_object = True

        # Retreat
        await wait_for_operation_resume(self._run_id)
        await move_linear_follow_target(
            robot_api,
            lambda: _fetch_pose(approach_mm),
            label="auto_pick_retreat",
            coordinate_id=coordinate_id,
            run_id=self._run_id,
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

        settings = get_settings()
        coordinate_id = get_robot_coordinate_id()

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
        await move_linear_follow_target(
            robot_api,
            lambda: approach_pose,
            label="auto_drop_approach",
            coordinate_id=coordinate_id,
            run_id=self._run_id,
        )

        # Move to drop position
        await wait_for_operation_resume(self._run_id)
        await move_linear_follow_target(
            robot_api,
            lambda: drop_pose,
            label="auto_drop_pose",
            coordinate_id=coordinate_id,
            run_id=self._run_id,
        )

        # Open gripper
        await wait_for_operation_resume(self._run_id)
        gripper_open = settings.gripper_profile.open_width_default
        await drop_with_verify(gripper_open)
        SESSION.robot_has_object = False

        # Retreat
        await wait_for_operation_resume(self._run_id)
        await move_linear_follow_target(
            robot_api,
            lambda: approach_pose,
            label="auto_drop_retreat",
            coordinate_id=coordinate_id,
            run_id=self._run_id,
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

        settings = get_settings()
        coordinate_id = get_robot_coordinate_id()

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

        await move_linear_follow_target(
            robot_api,
            lambda: approach_pose,
            label="auto_wait_approach",
            coordinate_id=coordinate_id,
            run_id=self._run_id,
        )

        await move_linear_follow_target(
            robot_api,
            lambda: drop_pose,
            label="auto_wait_drop",
            coordinate_id=coordinate_id,
            run_id=self._run_id,
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
