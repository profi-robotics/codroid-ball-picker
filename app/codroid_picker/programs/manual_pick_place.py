"""Manual pick and place robot program.

This program performs manual pick and place operations, where the user
triggers individual pick or drop actions on detected objects.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import TYPE_CHECKING, Callable, Optional, Tuple

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
    command_gripper,
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
        # Gripper settings
        gripper_close_width: Optional[float] = None,
        gripper_open_width: Optional[float] = None,
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
        self._gripper_close_width = gripper_close_width
        self._gripper_open_width = gripper_open_width
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
        """Execute a pick operation with dynamic target tracking."""
        from app.codroid_picker.session import SESSION

        settings = registry().settings
        coordinate_id = registry().robot_coordinate_id

        # Power on and ensure coordinate system
        self.update_message("Powering on robot...")
        await robot_api.power_on()
        await ensure_coordinate_active(robot_api, coordinate_id)

        # Get approach pose
        await wait_for_operation_resume(self._run_id)
        self.update_message("Moving to approach position...")
        approach_pose = self._pick_approach_pose_fn(
        ) if self._pick_approach_pose_fn else None
        if approach_pose is None:
            return ProgramResult(
                status=ProgramStatus.ERROR,
                message="Could not compute approach pose.",
            )

        log_move_target("pick_approach", approach_pose,
                        coordinate_id=coordinate_id)
        await self._move_linear_follow(
            robot_api,
            lambda: self._pick_approach_pose_fn() if self._pick_approach_pose_fn else approach_pose,
            label="pick_approach_follow",
            coordinate_id=coordinate_id,
        )

        # Get pick pose
        await wait_for_operation_resume(self._run_id)
        self.update_message("Moving to pick position...")
        pick_pose = self._pick_pose_fn() if self._pick_pose_fn else None
        if pick_pose is None:
            return ProgramResult(
                status=ProgramStatus.ERROR,
                message="Could not compute pick pose.",
            )

        log_move_target("pick_pose", pick_pose, coordinate_id=coordinate_id)
        await self._move_linear_follow(
            robot_api,
            lambda: self._pick_pose_fn() if self._pick_pose_fn else pick_pose,
            label="pick_pose_follow",
            coordinate_id=coordinate_id,
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
        await self._execute_move_with_hold(
            robot_api,
            retreat_pose,
            coordinate_id=coordinate_id,
            hold_seconds=2.0,
        )

        SESSION.robot_has_object = True
        if self._on_pick_complete:
            self._on_pick_complete()

        self.finish(status="idle", message="Pick complete.")
        return ProgramResult(
            status=ProgramStatus.COMPLETED,
            message="Pick complete.",
        )

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
        await self._execute_move_with_hold(
            robot_api,
            drop_approach,
            coordinate_id=coordinate_id,
            hold_seconds=2.0,
        )

        # Move to drop position
        await wait_for_operation_resume(self._run_id)
        self.update_message("Moving to drop position...")
        log_move_target("drop_pose", drop_pose, coordinate_id=coordinate_id)
        await self._execute_move_with_hold(
            robot_api,
            drop_pose,
            coordinate_id=coordinate_id,
            hold_seconds=2.0,
        )

        # Open gripper
        await wait_for_operation_resume(self._run_id)
        self.update_message("Opening gripper...")
        open_width = self._gripper_open_width or settings.gripper_profile.open_width_default
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
        await wait_for_operation_resume(self._run_id)
        self.update_message("Retreating from drop...")
        await self._execute_move_with_hold(
            robot_api,
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

    async def _move_linear_follow(
        self,
        robot_api: "CodroidAPI",
        pose_fn: Callable[[], Optional[dict]],
        *,
        label: str,
        coordinate_id: int,
    ) -> None:
        """Execute a linear move with position updates from pose_fn."""
        settings = registry().settings
        timeout_s = settings.verification_move_timeout_s
        settle_s = settings.verification_settle_s
        motion_eps_mm = settings.verification_motion_eps_mm
        heartbeat_interval = 0.2

        session = registry().get_or_create_robot_session()
        start_time = time.monotonic()
        last_pos = session.position_snapshot()
        moved = False
        settle_deadline: Optional[float] = None

        while time.monotonic() - start_time < timeout_s:
            ensure_operation_active(self._run_id)

            # Get current target pose
            target = pose_fn()
            if target is None:
                raise RuntimeError(f"Lost target during {label}")

            # Send move command
            await robot_api.set_target_pos_type(robot_api.config.target_pos_types.cpos)
            await robot_api.set_target_cpos(
                robot_api.attach_coordinate_to_cpos(
                    robot_api.build_target_cpos(**target),
                    coordinate_id=coordinate_id,
                )
            )
            await robot_api.set_robot_command(robot_api.config.commands.move_target_linear)
            await robot_api.send_command_heartbeat()
            await asyncio.sleep(heartbeat_interval)

            # Check position
            curr_pos = session.position_snapshot()
            delta = np.linalg.norm([
                float(curr_pos.x) - float(last_pos.x),
                float(curr_pos.y) - float(last_pos.y),
                float(curr_pos.z) - float(last_pos.z),
            ])

            if delta > motion_eps_mm:
                moved = True
                settle_deadline = time.monotonic() + settle_s
            elif moved and settle_deadline is not None:
                if time.monotonic() >= settle_deadline:
                    break

            last_pos = curr_pos

        await robot_api.clear_target_position()
        await robot_api.stop_command()

    async def _execute_move_with_hold(
        self,
        robot_api: "CodroidAPI",
        target: dict,
        *,
        coordinate_id: int,
        hold_seconds: float = 2.0,
    ) -> None:
        """Execute a linear move and hold with heartbeats."""
        heartbeat_interval = 0.2

        await robot_api.set_target_pos_type(robot_api.config.target_pos_types.cpos)
        await robot_api.set_target_cpos(
            robot_api.attach_coordinate_to_cpos(
                robot_api.build_target_cpos(**target),
                coordinate_id=coordinate_id,
            )
        )
        await robot_api.set_robot_command(robot_api.config.commands.move_target_linear)

        deadline = time.monotonic() + hold_seconds
        while time.monotonic() < deadline:
            ensure_operation_active(self._run_id)
            await robot_api.send_command_heartbeat()
            await asyncio.sleep(heartbeat_interval)

        await robot_api.clear_target_position()
        await robot_api.stop_command()
