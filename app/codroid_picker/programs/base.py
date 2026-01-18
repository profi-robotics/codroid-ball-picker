"""Robot program base class and utilities.

This module defines the abstract base class for all robot programs
(verification, plate test, manual pick/place, auto pick/place).
Each program can be started, paused, resumed, and stopped through
a unified interface.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Awaitable, Callable, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from codroid_api import CodroidAPI

from app.codroid_picker.state import (
    begin_operation,
    finish_operation,
    pause_operation,
    resume_operation,
    stop_operation,
    update_operation_message,
)
from app.codroid_picker.exceptions import OperationPaused, OperationAbort
from app.codroid_picker.robot_motion import (
    ensure_operation_active,
    wait_for_operation_resume,
    execute_linear_move_with_hold,
    await_motion_settle,
    move_linear_follow_target,
    command_gripper,
    drop_with_verify,
    ensure_coordinate_active,
    log_move_target,
)
from app.codroid_picker.app_instance import registry

LOGGER = logging.getLogger(__name__)


class ProgramStatus(Enum):
    """Status of a robot program."""

    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPING = "stopping"
    STOPPED = "stopped"
    COMPLETED = "completed"
    ERROR = "error"


@dataclass
class ProgramResult:
    """Result of a program execution."""

    status: ProgramStatus
    message: str
    data: dict = field(default_factory=dict)


class RobotProgram(ABC):
    """Abstract base class for robot programs.

    A robot program is a self-contained operation that controls the robot
    to perform a specific task. All programs support:
    - Starting with prerequisites check
    - Pausing and resuming
    - Stopping/aborting
    - Progress updates

    Subclasses must implement:
    - name: Program identifier
    - label: Human-readable program name
    - execute(): Main execution logic
    - check_prerequisites(): Verify system is ready

    The base class provides:
    - Operation lifecycle management (begin/finish)
    - Pause/resume handling
    - Error handling and cleanup
    - Access to shared motion utilities
    """

    # -------------------------------------------------------------------------
    # Abstract Properties (must be defined by subclasses)
    # -------------------------------------------------------------------------

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique program identifier (e.g., 'verification', 'plate_test')."""
        ...

    @property
    @abstractmethod
    def label(self) -> str:
        """Human-readable program name (e.g., 'Verification Move')."""
        ...

    # -------------------------------------------------------------------------
    # Initialization
    # -------------------------------------------------------------------------

    def __init__(self) -> None:
        self._run_id: Optional[int] = None
        self._robot_api: Optional[CodroidAPI] = None
        self._status: ProgramStatus = ProgramStatus.IDLE
        self._cancel_requested: bool = False

    # -------------------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------------------

    @property
    def run_id(self) -> Optional[int]:
        """Current operation run ID, or None if not running."""
        return self._run_id

    @property
    def robot_api(self) -> Optional["CodroidAPI"]:
        """Robot API instance, available during execution."""
        return self._robot_api

    @property
    def status(self) -> ProgramStatus:
        """Current program status."""
        return self._status

    @property
    def is_running(self) -> bool:
        """True if the program is currently running (including paused)."""
        return self._status in {ProgramStatus.RUNNING, ProgramStatus.PAUSED}

    # -------------------------------------------------------------------------
    # Abstract Methods (must be implemented by subclasses)
    # -------------------------------------------------------------------------

    @abstractmethod
    async def execute(self, robot_api: "CodroidAPI") -> ProgramResult:
        """Main execution logic for the program.

        This is called after prerequisites are verified and the operation
        has been registered. Subclasses should implement their specific
        robot control logic here.

        The base class handles:
        - Operation state management
        - Exception handling (OperationPaused, OperationAbort)
        - Cleanup on completion/error

        Args:
            robot_api: Connected robot API instance.

        Returns:
            ProgramResult with status and message.
        """
        ...

    @abstractmethod
    def check_prerequisites(self) -> Tuple[bool, str]:
        """Check if all prerequisites are met to run this program.

        Returns:
            Tuple of (ready, message). If ready is False, message
            explains what's missing.
        """
        ...

    # -------------------------------------------------------------------------
    # Optional Hooks (can be overridden by subclasses)
    # -------------------------------------------------------------------------

    async def on_pause(self) -> None:
        """Called when the program is paused."""
        pass

    async def on_resume(self) -> None:
        """Called when the program is resumed."""
        pass

    async def on_stop(self) -> None:
        """Called when the program is stopped/aborted."""
        pass

    async def cleanup(self) -> None:
        """Called after program completes (success or failure)."""
        pass

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    async def start(
        self,
        robot_api: "CodroidAPI",
        *,
        initial_message: Optional[str] = None,
    ) -> ProgramResult:
        """Start the program.

        This is the main entry point for running a program. It:
        1. Checks prerequisites
        2. Registers the operation
        3. Calls execute()
        4. Handles cleanup

        Args:
            robot_api: Connected robot API instance.
            initial_message: Optional initial status message.

        Returns:
            ProgramResult with final status.
        """
        # Check prerequisites
        ready, message = self.check_prerequisites()
        if not ready:
            return ProgramResult(
                status=ProgramStatus.ERROR,
                message=message,
            )

        # Register the operation
        try:
            self._run_id = begin_operation(
                source=self.name,
                label=self.label,
                message=initial_message or f"{self.label} starting...",
            )
        except RuntimeError as exc:
            return ProgramResult(
                status=ProgramStatus.ERROR,
                message=f"Cannot start {self.label}: {exc}",
            )

        self._robot_api = robot_api
        self._status = ProgramStatus.RUNNING
        self._cancel_requested = False

        try:
            result = await self.execute(robot_api)
            self._status = result.status
            return result

        except OperationAbort as exc:
            self._status = ProgramStatus.STOPPED
            message = f"{self.label} stopped: {exc}"
            finish_operation(self._run_id, status="stopped", message=message)
            return ProgramResult(status=ProgramStatus.STOPPED, message=message)

        except Exception as exc:
            LOGGER.exception(f"{self.label} failed")
            self._status = ProgramStatus.ERROR
            message = f"{self.label} failed: {exc}"
            finish_operation(self._run_id, status="error", message=message)
            return ProgramResult(status=ProgramStatus.ERROR, message=message)

        finally:
            await self.cleanup()
            self._robot_api = None
            if self._status == ProgramStatus.RUNNING:
                self._status = ProgramStatus.IDLE

    def request_pause(self, reason: str = "user_request") -> None:
        """Request the program to pause."""
        if self._run_id is None:
            return
        pause_operation(reason, f"{self.label} paused.")
        self._status = ProgramStatus.PAUSED

    def request_resume(self) -> None:
        """Request the program to resume."""
        if self._run_id is None:
            return
        resume_operation(f"{self.label} resumed.")
        self._status = ProgramStatus.RUNNING

    def request_stop(self) -> None:
        """Request the program to stop."""
        if self._run_id is None:
            return
        self._cancel_requested = True
        stop_operation(f"{self.label} stopping...")

    # -------------------------------------------------------------------------
    # Helper Methods (available to subclasses)
    # -------------------------------------------------------------------------

    def update_message(self, message: str, *, label: Optional[str] = None) -> None:
        """Update the current operation message."""
        update_operation_message(message, label=label)

    async def wait_for_resume(self) -> None:
        """Wait for the operation to resume if paused.

        Raises:
            OperationAbort: If the operation was stopped.
        """
        if self._run_id is None:
            return
        await wait_for_operation_resume(self._run_id)

    def ensure_active(self) -> None:
        """Ensure the operation is still active.

        Raises:
            OperationPaused: If paused.
            OperationAbort: If stopped.
        """
        if self._run_id is None:
            return
        ensure_operation_active(self._run_id)

    def finish(
        self,
        status: str = "idle",
        message: Optional[str] = None,
    ) -> None:
        """Mark the operation as finished."""
        if self._run_id is None:
            return
        finish_operation(
            self._run_id,
            status=status,
            message=message or f"{self.label} complete.",
        )

    # -------------------------------------------------------------------------
    # Motion Utilities (delegated to robot_motion module)
    # -------------------------------------------------------------------------

    async def move_with_hold(
        self,
        target_pose: dict,
        *,
        coordinate_id: Optional[int] = None,
        hold_seconds: float = 2.0,
        heartbeat_interval: float = 0.2,
    ) -> None:
        """Execute a linear move and hold position.

        Wrapper around execute_linear_move_with_hold with automatic run_id.
        """
        if self._robot_api is None:
            raise RuntimeError("Robot API not available")
        await execute_linear_move_with_hold(
            self._robot_api,
            target_pose,
            coordinate_id=coordinate_id,
            hold_seconds=hold_seconds,
            heartbeat_interval=heartbeat_interval,
            run_id=self._run_id,
        )

    async def move_follow_target(
        self,
        target_fetch: Callable[[], Optional[dict]],
        *,
        label: str,
        coordinate_id: Optional[int] = None,
        timeout_s: Optional[float] = None,
        safety_check: Optional[Callable[[], Awaitable[None]]] = None,
    ) -> Optional[dict]:
        """Move to a target with continuous position updates.

        Wrapper around move_linear_follow_target with automatic run_id.
        """
        if self._robot_api is None:
            raise RuntimeError("Robot API not available")
        return await move_linear_follow_target(
            self._robot_api,
            target_fetch,
            label=label,
            coordinate_id=coordinate_id,
            timeout_s=timeout_s,
            run_id=self._run_id,
            safety_check=safety_check,
        )

    async def wait_for_settle(
        self,
        target: dict,
        *,
        label: str = "motion",
        timeout_s: Optional[float] = None,
    ) -> None:
        """Wait for robot motion to complete and settle.

        Wrapper around await_motion_settle with automatic run_id.
        """
        if self._robot_api is None:
            raise RuntimeError("Robot API not available")
        session = registry().get_or_create_robot_session()
        await await_motion_settle(
            self._robot_api,
            session,
            target,
            run_id=self._run_id,
            timeout_s=timeout_s,
            label=label,
        )

    async def ensure_coordinate(self, coordinate_id: Optional[int] = None) -> None:
        """Ensure the specified coordinate system is active."""
        if self._robot_api is None:
            raise RuntimeError("Robot API not available")
        await ensure_coordinate_active(self._robot_api, coordinate_id)

    def gripper_command(
        self,
        target_width: float,
        action_label: str,
        *,
        force: Optional[int] = None,
        speed: Optional[int] = None,
    ) -> str:
        """Issue a gripper command."""
        return command_gripper(
            target_width,
            action_label,
            force=force,
            speed=speed,
        )

    async def gripper_drop_with_verify(
        self,
        open_width: float,
        *,
        max_width_diff: float = 5.0,
    ) -> str:
        """Open gripper for drop with verification."""
        return await drop_with_verify(
            open_width,
            max_width_diff=max_width_diff,
        )

    def log_target(
        self,
        label: str,
        target: dict,
        *,
        coordinate_id: Optional[int] = None,
    ) -> None:
        """Log a move target for debugging."""
        log_move_target(label, target, coordinate_id=coordinate_id)


# Re-export exceptions for convenience
__all__ = [
    "RobotProgram",
    "ProgramStatus",
    "ProgramResult",
    "OperationPaused",
    "OperationAbort",
]
