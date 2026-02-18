"""Manual pick and place robot program.

This program performs manual pick and place operations, where the user
triggers individual pick or drop actions on detected objects.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Callable, Optional, Tuple

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
    wait_for_operation_resume,
    move_linear_follow_target,
    command_gripper,
    resolve_gripper_limits,
    ensure_coordinate_active,
    log_move_target,
)

LOGGER = logging.getLogger(__name__)


class ManualPickPlaceProgram(RobotProgram):
    """Manual pick and place robot program.

    This program handles single pick, drop, or pick-and-drop operations
    triggered by the user. Unlike the auto program, it doesn't loop -
    each invocation performs the requested action(s).

    The program supports three actions:
    - 'pick': Move to target, close gripper, retreat
    - 'drop': Move to drop position, open gripper, retreat
    - 'pick_drop': Pick then drop in sequence
    """

    def __init__(
        self,
        *,
        action: str = "pick",
        # Pose fetcher functions for dynamic tracking
        pick_approach_pose_fn: Optional[Callable[[], Optional[dict]]] = None,
        pick_pose_fn: Optional[Callable[[], Optional[dict]]] = None,
        drop_approach_pose: Optional[dict] = None,
        drop_pose: Optional[dict] = None,
        # Fallback static poses
        approach_mm: float = -30.0,
        pick_mm: float = 0.0,
        drop_mm: float = 0.0,
        # Tracking mode configuration
        tracking_mode: str = "move",
        segment_min_mm: float = 5.0,
        segment_max_mm: float = 50.0,
        segment_fraction: float = 0.5,
        # Gripper settings
        gripper_close_width: Optional[float] = None,
        gripper_open_width: Optional[float] = None,
        pick_final_c: Optional[float] = None,
        # Callbacks for state updates
        on_pick_complete: Optional[Callable[[], None]] = None,
        on_drop_complete: Optional[Callable[[], None]] = None,
    ) -> None:
        super().__init__()
        self._action = action
        self._pick_approach_pose_fn = pick_approach_pose_fn
        self._pick_pose_fn = pick_pose_fn
        self._drop_approach_pose = drop_approach_pose
        self._drop_pose = drop_pose
        self._approach_mm = approach_mm
        self._pick_mm = pick_mm
        self._drop_mm = drop_mm
        self._tracking_mode = tracking_mode
        self._segment_min_mm = segment_min_mm
        self._segment_max_mm = segment_max_mm
        self._segment_fraction = segment_fraction
        self._gripper_close_width = gripper_close_width
        self._gripper_open_width = gripper_open_width
        self._pick_final_c = pick_final_c
        self._on_pick_complete = on_pick_complete
        self._on_drop_complete = on_drop_complete

    @property
    def name(self) -> str:
        return "manual_pick"

    @property
    def label(self) -> str:
        action_labels = {
            "pick": "Manual Pick",
            "drop": "Manual Drop",
            "pick_drop": "Manual Pick & Drop",
        }
        return action_labels.get(self._action, "Manual Pick")

    def check_prerequisites(self) -> Tuple[bool, str]:
        """Check if the operation can proceed."""
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

        # Action-specific checks
        if self._action in ("pick", "pick_drop"):
            if self._pick_pose_fn is None:
                return False, "No pick target specified."

        if self._action in ("drop", "pick_drop"):
            if self._drop_pose is None and SESSION.drop_target_robot is None:
                return False, "No drop target defined."

        return True, "Ready."

    async def execute(self, robot_api: "CodroidAPI") -> ProgramResult:
        """Execute the pick, drop, or pick-and-drop action."""
        try:
            if self._action == "pick":
                return await self._execute_pick(robot_api)
            elif self._action == "drop":
                return await self._execute_drop(robot_api)
            elif self._action == "pick_drop":
                pick_result = await self._execute_pick(robot_api)
                if pick_result.status != ProgramStatus.COMPLETED:
                    return pick_result
                await wait_for_operation_resume(self._run_id)
                return await self._execute_drop(robot_api)
            else:
                return ProgramResult(
                    status=ProgramStatus.ERROR,
                    message=f"Unknown action: {self._action}",
                )
        except OperationAbort as exc:
            return ProgramResult(
                status=ProgramStatus.STOPPED,
                message=str(exc),
            )
        except OperationPaused:
            # This shouldn't happen if we're handling pauses correctly
            return ProgramResult(
                status=ProgramStatus.STOPPED,
                message="Operation paused unexpectedly.",
            )

    async def _execute_pick(self, robot_api: "CodroidAPI") -> ProgramResult:
        """Execute a pick operation with optional dynamic target tracking."""
        from app.codroid_picker.session import SESSION

        settings = registry().settings
        coordinate_id = registry().robot_coordinate_id
        use_segment_tracking = self._tracking_mode.lower() == "segment"

        LOGGER.info(f"Manual pick using tracking_mode={self._tracking_mode} (segment={use_segment_tracking})")

        # Power on and ensure coordinate system
        self.update_message("Powering on robot...")
        await robot_api.power_on()
        await ensure_coordinate_active(robot_api, coordinate_id)

        # Get approach pose
        await wait_for_operation_resume(self._run_id)
        self.update_message("Moving to target XY...")
        approach_pose = self._pick_approach_pose_fn(
        ) if self._pick_approach_pose_fn else None
        if approach_pose is None:
            return ProgramResult(
                status=ProgramStatus.ERROR,
                message="Could not compute approach pose - no target detected.",
            )
        posture = registry().get_or_create_robot_session().position_snapshot()
        transit_pose = dict(approach_pose)
        transit_pose["z"] = float(posture.z)
        transit_pose["a"] = float(posture.a)
        transit_pose["b"] = float(posture.b)
        transit_pose["c"] = float(posture.c)
        await self.move_with_hold(
            transit_pose,
            coordinate_id=coordinate_id,
            hold_seconds=2.0,
        )

        self.update_message("Moving to approach position...")

        log_move_target("pick_approach", approach_pose,
                        coordinate_id=coordinate_id)

        if use_segment_tracking:
            # Use move_linear_follow_target for dynamic tracking (segment mode)
            # The pose fetcher will use last known position if target is obscured by robot
            await move_linear_follow_target(
                robot_api,
                lambda: self._pick_approach_pose_fn() if self._pick_approach_pose_fn else approach_pose,
                label="pick_approach_follow",
                coordinate_id=coordinate_id,
                run_id=self._run_id,
            )
        else:
            # Use simple linear move for static target (move target mode)
            await self.move_with_hold(
                approach_pose,
                coordinate_id=coordinate_id,
                hold_seconds=2.0,
            )

        # Rotate only when already above the target.
        await wait_for_operation_resume(self._run_id)
        align_pose = dict(approach_pose)
        if self._pick_final_c is not None:
            align_pose["c"] = float(self._pick_final_c)
        await self.move_with_hold(
            align_pose,
            coordinate_id=coordinate_id,
            hold_seconds=2.0,
        )

        # Get pick pose - for segment tracking, this uses the last tracked position
        await wait_for_operation_resume(self._run_id)
        self.update_message("Moving to pick position...")
        pick_pose = self._pick_pose_fn() if self._pick_pose_fn else None
        if pick_pose is None:
            return ProgramResult(
                status=ProgramStatus.ERROR,
                message="Could not compute pick pose.",
            )
        if self._pick_final_c is not None:
            pick_pose["c"] = float(self._pick_final_c)

        log_move_target("pick_pose", pick_pose, coordinate_id=coordinate_id)

        if use_segment_tracking:
            # Use move_linear_follow_target for dynamic tracking (segment mode)
            await move_linear_follow_target(
                robot_api,
                lambda: self._pick_pose_fn() if self._pick_pose_fn else pick_pose,
                label="pick_pose_follow",
                coordinate_id=coordinate_id,
                run_id=self._run_id,
            )
        else:
            # Use simple linear move for static target (move target mode)
            await self.move_with_hold(
                pick_pose,
                coordinate_id=coordinate_id,
                hold_seconds=2.0,
            )

        # Close gripper
        await wait_for_operation_resume(self._run_id)
        self.update_message("Closing gripper...")
        grip_force = min(
            settings.gripper_profile.force_default * 2,
            settings.gripper_profile.force_max,
        )
        close_width = self._gripper_close_width or settings.gripper_profile.close_width_default
        result = command_gripper(close_width, "closed", force=grip_force)
        LOGGER.info(f"Gripper close result: {result}")

        # Retreat
        await wait_for_operation_resume(self._run_id)
        self.update_message("Retreating...")
        retreat_pose = self._pick_approach_pose_fn(
        ) if self._pick_approach_pose_fn else approach_pose
        if retreat_pose is None:
            retreat_pose = approach_pose
        if self._pick_final_c is not None:
            retreat_pose["c"] = float(self._pick_final_c)
        await self.move_with_hold(
            retreat_pose,
            coordinate_id=coordinate_id,
            hold_seconds=2.0,
        )

        SESSION.robot_has_object = True
        if self._on_pick_complete:
            self._on_pick_complete()

        if self._action != "pick_drop":
            self.finish(status="idle", message="Pick complete.")
        return ProgramResult(
            status=ProgramStatus.COMPLETED,
            message="Pick complete.",
        )

    async def _retreat_to_drop_approach(self, robot_api: "CodroidAPI", coordinate_id: int) -> None:
        """Retreat to the drop approach position."""
        from app.codroid_picker.session import SESSION

        LOGGER.info(f"Retreating to drop approach. drop_approach_pose={self._drop_approach_pose}, "
                   f"drop_target_robot={SESSION.drop_target_robot}")

        # Determine the retreat target - prefer drop_approach_pose, fall back to session data
        retreat_pose = self._drop_approach_pose

        if retreat_pose is None and SESSION.drop_target_robot is not None:
            # Build drop approach from session data
            base = SESSION.drop_target_robot
            retreat_pose = {
                "x": float(base[0]),
                "y": float(base[1]),
                "z": float(base[2]) + self._approach_mm,
                "a": 0.0, "b": 0.0, "c": 0.0,
            }
            if SESSION.drop_target_orientation is not None:
                orient = SESSION.drop_target_orientation
                retreat_pose.update({
                    "a": float(orient[0]),
                    "b": float(orient[1]),
                    "c": float(orient[2]),
                })
            LOGGER.info(f"Built retreat pose from session: {retreat_pose}")

        if retreat_pose is not None:
            log_move_target("tracking_lost_retreat", retreat_pose,
                           coordinate_id=coordinate_id)
            await self.move_with_hold(
                retreat_pose,
                coordinate_id=coordinate_id,
                hold_seconds=2.0,
            )
        else:
            LOGGER.warning("No drop approach position available for retreat - "
                          "neither drop_approach_pose nor SESSION.drop_target_robot are set")

    async def _execute_drop(self, robot_api: "CodroidAPI") -> ProgramResult:
        """Execute a drop operation."""
        from app.codroid_picker.session import SESSION

        settings = registry().settings
        coordinate_id = registry().robot_coordinate_id

        # Power on and ensure coordinate system
        self.update_message("Powering on robot...")
        await robot_api.power_on()
        await ensure_coordinate_active(robot_api, coordinate_id)

        # Get drop poses
        drop_approach = self._drop_approach_pose
        drop_pose = self._drop_pose

        if drop_approach is None or drop_pose is None:
            # Fallback to session data
            if SESSION.drop_target_robot is not None:
                base = SESSION.drop_target_robot
                drop_pose = {
                    "x": float(base[0]),
                    "y": float(base[1]),
                    "z": float(base[2]) + self._drop_mm,
                    "a": 0.0, "b": 0.0, "c": 0.0,
                }
                drop_approach = {
                    "x": float(base[0]),
                    "y": float(base[1]),
                    "z": float(base[2]) + self._approach_mm,
                    "a": 0.0, "b": 0.0, "c": 0.0,
                }
                # Apply stored orientation if available
                if SESSION.drop_target_orientation is not None:
                    orient = SESSION.drop_target_orientation
                    drop_pose.update({
                        "a": float(orient[0]),
                        "b": float(orient[1]),
                        "c": float(orient[2]),
                    })
                    drop_approach.update({
                        "a": float(orient[0]),
                        "b": float(orient[1]),
                        "c": float(orient[2]),
                    })
            else:
                return ProgramResult(
                    status=ProgramStatus.ERROR,
                    message="No drop target defined.",
                )

        # Move to approach
        await wait_for_operation_resume(self._run_id)
        self.update_message("Moving to drop approach...")
        log_move_target("drop_approach", drop_approach,
                        coordinate_id=coordinate_id)
        await self.move_with_hold(
            drop_approach,
            coordinate_id=coordinate_id,
            hold_seconds=2.0,
        )

        # Move to drop position
        await wait_for_operation_resume(self._run_id)
        self.update_message("Moving to drop position...")
        log_move_target("drop_pose", drop_pose, coordinate_id=coordinate_id)
        await self.move_with_hold(
            drop_pose,
            coordinate_id=coordinate_id,
            hold_seconds=2.0,
        )

        # Open gripper
        await wait_for_operation_resume(self._run_id)
        self.update_message("Opening gripper...")
        open_width = self._gripper_open_width or settings.gripper_profile.open_width_default
        if self._gripper_close_width is not None:
            close_width = self._gripper_close_width
        else:
            close_width = settings.gripper_profile.close_width_default
        if open_width <= close_width:
            gripper = registry().gripper
            if gripper is not None:
                _min_width, max_width = resolve_gripper_limits(gripper)
                if max_width > open_width:
                    LOGGER.warning(
                        "Open width %.1f <= close width %.1f; using max width %.1f",
                        open_width,
                        close_width,
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

        # Retreat
        log_move_target("drop_retreat", drop_approach,
                        coordinate_id=coordinate_id)
        await self.move_with_hold(
            drop_approach,
            coordinate_id=coordinate_id,
            hold_seconds=2.0,
        )

        SESSION.robot_has_object = False
        if self._on_drop_complete:
            self._on_drop_complete()

        self.finish(status="idle", message="Drop complete.")
        return ProgramResult(
            status=ProgramStatus.COMPLETED,
            message="Drop complete.",
        )
