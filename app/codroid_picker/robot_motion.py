"""Robot motion utilities for the calibration UI.

This module provides reusable async helpers for robot motion operations,
extracted from main.py to share across all robot programs:
- execute_linear_move_with_hold: Move to target with heartbeat hold
- await_motion_settle: Wait for motion to complete and settle
- move_linear_follow_target: Move while updating target position
- command_gripper: Issue gripper command
- drop_with_verify: High-level drop sequence with gripper retry/verify
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Awaitable, Callable, Optional, Tuple, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from codroid_api import CodroidAPI, RobotSession
    from onrobot.twofg import TWOFG

from app.codroid_picker.app_instance import (
    get_settings,
    get_gripper_defaults,
    get_robot_coordinate_id,
    get_robot_session,
    registry,
)
from app.codroid_picker.exceptions import OperationPaused, OperationAbort
from app.codroid_picker.robot_helpers import (
    resolve_gripper_limits as _resolve_gripper_limits,
    is_gripper_connected,
    safe_int,
)

LOGGER = logging.getLogger(__name__)


# =============================================================================
# Operation State Helpers
# =============================================================================


def _operation_status(run_id: int) -> str:
    """Check the current operation status for a given run_id.

    Returns one of: 'running', 'paused', 'stopped', 'idle', 'superseded'
    """
    from app.codroid_picker.state import get_operation_snapshot

    snapshot = get_operation_snapshot()
    if snapshot.get("run_id") != run_id:
        return "superseded"
    return snapshot.get("status", "idle")


def ensure_operation_active(run_id: int) -> None:
    """Ensure the operation is still active.

    Raises OperationPaused if paused, OperationAbort if stopped.
    """
    status = _operation_status(run_id)
    if status in {"stopping", "stopped", "idle", "superseded"}:
        raise OperationAbort("Program stopped.")
    if status == "paused":
        raise OperationPaused()


async def wait_for_operation_resume(run_id: int) -> None:
    """Wait until the operation resumes or is aborted.

    If the operation is currently paused, this blocks until resumed.
    If the operation is stopped/idle, raises OperationAbort.
    """
    while True:
        status = _operation_status(run_id)
        if status in {"stopping", "stopped", "idle", "superseded"}:
            raise OperationAbort("Program stopped.")
        if status == "running":
            return
        await asyncio.sleep(0.1)


# =============================================================================
# Gripper Utilities
# =============================================================================


def resolve_gripper_limits(gripper: "TWOFG") -> Tuple[float, float]:
    """Get gripper min/max width limits.

    Returns (min_width, max_width) in mm.

    Note: This is a wrapper around the shared helper for backwards compatibility.
    """
    return _resolve_gripper_limits(gripper)


def command_gripper(
    target_width: float,
    action_label: str,
    *,
    force: Optional[int] = None,
    speed: Optional[int] = None,
    prevent_overclose: bool = False,
) -> str:
    """Move gripper to target width and return status text.

    Args:
        target_width: Target width in mm.
        action_label: Human-readable label for the action (e.g., "opened", "closed").
        force: Grip force in N (uses default if not specified).
        speed: Grip speed as percentage (uses default if not specified).
        prevent_overclose: If True, don't close beyond current width.

    Returns:
        Status message string.
    """
    from app.codroid_picker.session import SESSION

    gripper = registry.gripper or SESSION.gripper
    gripper_defaults = get_gripper_defaults()
    settings = get_settings()
    gripper_profile = settings.gripper_profile

    if gripper is None or not gripper.isConnected(t_index=gripper_defaults["t_index"]):
        return "Gripper is not connected. Enable gripper in Setup menu."

    try:
        safe_width = float(target_width)
        min_width, max_width = resolve_gripper_limits(gripper)

        if prevent_overclose:
            current_width = gripper.get_ext_width(
                t_index=gripper_defaults["t_index"])
            if isinstance(current_width, (int, float)) and current_width >= 0:
                if safe_width < current_width:
                    safe_width = current_width

        safe_width = min(max(safe_width, min_width), max_width)

        applied_force = force if force is not None else gripper_profile.force_default
        applied_force = min(
            max(applied_force, gripper_profile.force_min),
            gripper_profile.force_max,
        )
        applied_speed = speed if speed is not None else 80

        gripper.grip(
            t_index=gripper_defaults["t_index"],
            t_width=safe_width,
            n_force=applied_force,
            p_speed=applied_speed,
            f_wait=True,
        )

        message = (
            f"Gripper {action_label} to {safe_width:.1f} mm @ "
            f"{applied_force} N / {applied_speed}% speed."
        )
        SESSION.last_action = message
        return message

    except Exception as exc:
        LOGGER.exception("Gripper command failed")
        return f"Gripper action failed: {exc}"


async def drop_with_verify(
    gripper_open_width: float,
    *,
    max_width_diff: float = 5.0,
    max_retries: int = 1,
) -> str:
    """Open gripper for drop with verification and retry logic.

    This is a high-level drop helper that:
    1. Opens the gripper to the target width
    2. Waits for the gripper to finish moving
    3. Verifies the gripper actually opened
    4. Retries with wider opening if needed

    Args:
        gripper_open_width: Target open width in mm.
        max_width_diff: Maximum acceptable difference from target (mm).
        max_retries: Maximum number of retries if width doesn't match.

    Returns:
        Status message string.
    """
    from app.codroid_picker.session import SESSION

    gripper = registry.gripper or SESSION.gripper
    gripper_defaults = get_gripper_defaults()

    # Initial open command
    result = command_gripper(gripper_open_width, "opened")

    if "not connected" in result.lower() or "error" in result.lower():
        LOGGER.warning(f"Gripper command may have failed: {result}")
        return result

    if gripper is None:
        return result

    # Wait for gripper to finish
    t_index = gripper_defaults["t_index"]
    max_polls = 20  # 2 seconds max
    for poll_count in range(max_polls):
        try:
            busy = gripper.isBusy(t_index=t_index)
            if not busy:
                break
        except Exception:
            break
        await asyncio.sleep(0.1)

    LOGGER.info(f"Gripper ready after {(poll_count + 1) * 0.1:.1f}s")

    # Verify gripper opened
    if not gripper.isConnected(t_index=t_index):
        return result

    try:
        actual_width = gripper.get_ext_width(t_index=t_index)
        LOGGER.info(f"Gripper actual width after open: {actual_width} mm")

        if not isinstance(actual_width, (int, float)) or actual_width <= 0:
            LOGGER.warning(f"Could not read gripper width: {actual_width}")
            return result

        width_diff = abs(actual_width - gripper_open_width)
        if width_diff <= max_width_diff:
            LOGGER.info("Gripper opened successfully")
            return result

        # Need to retry with wider opening
        for retry in range(max_retries):
            LOGGER.warning(
                f"Gripper width mismatch (target: {gripper_open_width}, "
                f"actual: {actual_width}), retry {retry + 1}"
            )

            # Try to get max width
            try:
                max_width = gripper.get_max_ext_width(t_index=t_index)
                if isinstance(max_width, (int, float)) and max_width > actual_width:
                    retry_width = max_width
                else:
                    retry_width = gripper_open_width + 10.0
            except Exception:
                retry_width = gripper_open_width + 10.0

            LOGGER.info(f"Retrying with width: {retry_width} mm")
            settings = get_settings()
            result = command_gripper(
                retry_width,
                "opened (retry)",
                force=settings.gripper_profile.force_max,
            )

            # Check final width
            actual_width = gripper.get_ext_width(t_index=t_index)
            LOGGER.info(f"Final gripper width: {actual_width} mm")

            if isinstance(actual_width, (int, float)) and actual_width > 0:
                new_diff = abs(actual_width - gripper_open_width)
                if new_diff <= max_width_diff:
                    break

    except Exception as exc:
        LOGGER.warning(f"Error verifying gripper state: {exc}")

    return result


# =============================================================================
# Robot Motion Utilities
# =============================================================================


async def execute_linear_move_with_hold(
    robot_api: "CodroidAPI",
    target_pose: dict,
    *,
    coordinate_id: Optional[int] = None,
    hold_seconds: float = 2.0,
    heartbeat_interval: float = 0.2,
    run_id: Optional[int] = None,
) -> None:
    """Execute a linear move and hold position with heartbeats.

    This is the core motion pattern used throughout the application:
    1. Set target position type
    2. Set target cartesian position
    3. Issue linear move command
    4. Hold position with heartbeats for specified duration
    5. Clear target and stop command

    Args:
        robot_api: The robot API instance.
        target_pose: Target pose dict with x, y, z, a, b, c keys.
        coordinate_id: Coordinate system ID (defaults to calibrated ID).
        hold_seconds: Duration to hold position with heartbeats.
        heartbeat_interval: Time between heartbeats in seconds.
        run_id: Operation run ID for pause/stop checking.
    """
    if coordinate_id is None:
        coordinate_id = get_robot_coordinate_id()

    if run_id is not None:
        await wait_for_operation_resume(run_id)

    await robot_api.set_target_pos_type(robot_api.config.target_pos_types.cpos)
    await robot_api.set_target_cpos(
        robot_api.attach_coordinate_to_cpos(
            robot_api.build_target_cpos(**target_pose),
            coordinate_id=coordinate_id,
        )
    )
    await robot_api.set_robot_command(robot_api.config.commands.move_target_linear)

    deadline = time.monotonic() + hold_seconds
    while time.monotonic() < deadline:
        if run_id is not None:
            ensure_operation_active(run_id)
        await robot_api.send_command_heartbeat()
        await asyncio.sleep(heartbeat_interval)

    await robot_api.clear_target_position()
    await robot_api.stop_command()


async def await_motion_settle(
    robot_api: "CodroidAPI",
    session: "RobotSession",
    target: dict,
    *,
    run_id: Optional[int] = None,
    timeout_s: Optional[float] = None,
    settle_s: Optional[float] = None,
    motion_eps_mm: Optional[float] = None,
    label: str = "motion",
) -> None:
    """Wait for robot motion to complete and settle.

    Monitors robot position and waits until:
    1. Motion has started (position changed from initial)
    2. Motion has settled (no movement for settle_s seconds)

    Args:
        robot_api: The robot API instance.
        session: The robot session for reading posture.
        target: Target pose for reference.
        run_id: Operation run ID for pause/stop checking.
        timeout_s: Maximum wait time in seconds.
        settle_s: Duration of no movement to consider settled.
        motion_eps_mm: Minimum movement to detect motion.
        label: Label for logging.

    Raises:
        RuntimeError: If timeout occurs without proper motion/settle.
    """
    settings = get_settings()
    timeout_s = timeout_s or settings.verification_move_timeout_s
    settle_s = settle_s or settings.verification_settle_s
    motion_eps_mm = motion_eps_mm or settings.verification_motion_eps_mm

    deadline = time.monotonic() + timeout_s
    start = session.position_snapshot()
    last = start
    moved = False
    settle_deadline: Optional[float] = None

    while time.monotonic() < deadline:
        if run_id is not None:
            ensure_operation_active(run_id)
        await robot_api.send_command_heartbeat()
        await asyncio.sleep(0.2)

        curr = session.position_snapshot()
        delta = np.linalg.norm([
            float(curr.x) - float(last.x),
            float(curr.y) - float(last.y),
            float(curr.z) - float(last.z),
        ])

        if delta > motion_eps_mm:
            moved = True
            settle_deadline = time.monotonic() + settle_s

        if moved and settle_deadline is not None:
            if delta > motion_eps_mm:
                settle_deadline = time.monotonic() + settle_s
            elif time.monotonic() >= settle_deadline:
                return

        last = curr

    total = np.linalg.norm([
        float(last.x) - float(start.x),
        float(last.y) - float(start.y),
        float(last.z) - float(start.z),
    ])

    if not moved:
        raise RuntimeError(
            f"Motion timeout at {label}: no motion detected."
        )
    raise RuntimeError(
        f"Motion timeout at {label}: motion did not settle "
        f"(total={total:.2f} mm)."
    )


async def move_linear_follow_target(
    robot_api: "CodroidAPI",
    target_fetch: Callable[[], Optional[dict]],
    *,
    label: str,
    coordinate_id: Optional[int] = None,
    timeout_s: Optional[float] = None,
    update_interval: Optional[float] = None,
    segment_hold_s: Optional[float] = None,
    settle_s: Optional[float] = None,
    tolerance_mm: Optional[float] = None,
    require_posture: bool = True,
    run_id: Optional[int] = None,
    safety_check: Optional[Callable[[], Awaitable[None]]] = None,
) -> Optional[dict]:
    """Move to a target with continuous position updates.

    This is used for pick operations where the target may be moving
    (e.g., tracking an object on a conveyor or updating based on
    new camera frames).

    Args:
        robot_api: The robot API instance.
        target_fetch: Callable that returns current target pose dict.
        label: Human-readable label for logging.
        coordinate_id: Coordinate system ID.
        timeout_s: Maximum time for the move.
        update_interval: Time between target updates.
        segment_hold_s: Time to hold each segment.
        settle_s: Time to consider position stable.
        tolerance_mm: Distance tolerance for position matching.
        require_posture: Whether to require posture stream.
        run_id: Operation run ID for pause/stop checking.
        safety_check: Optional async callback for safety checks.

    Returns:
        The final target pose dict, or None if no target was ever fetched.
    """
    if coordinate_id is None:
        coordinate_id = get_robot_coordinate_id()

    settings = get_settings()
    timeout_s = timeout_s or settings.pick_follow_timeout_s
    update_interval = update_interval or settings.pick_follow_update_interval_s
    segment_hold_s = segment_hold_s or settings.pick_follow_segment_hold_s
    settle_s = settle_s or settings.pick_follow_settle_s
    tolerance_mm = tolerance_mm or settings.pick_follow_tolerance_mm

    session = get_robot_session()
    last_target: Optional[dict] = None

    def _check_operation() -> None:
        if run_id is not None:
            ensure_operation_active(run_id)

    if require_posture and not session.posture_seen():
        _check_operation()
        LOGGER.info(
            f"move_linear_follow_target '{label}': No posture seen, fetching initial target")
        initial_target = target_fetch()
        LOGGER.info(
            f"move_linear_follow_target '{label}': Got initial target: {initial_target}")
        if initial_target is None:
            raise RuntimeError("No target available for follow move.")
        await robot_api.move_to_cartesian_target_linear(
            robot_api.build_target_cpos(**initial_target)
        )
        return initial_target

    LOGGER.info(f"move_linear_follow_target '{label}': Starting follow loop")
    await robot_api.set_target_pos_type(robot_api.config.target_pos_types.cpos)

    stable_since: Optional[float] = None
    deadline = time.monotonic() + timeout_s

    while True:
        _check_operation()
        now = time.monotonic()

        if now >= deadline:
            LOGGER.warning(
                f"Follow move '{label}' timed out after {timeout_s:.1f}s")
            break

        target = target_fetch()
        if target is not None:
            last_target = target
        if last_target is None:
            raise RuntimeError("No target available for follow move.")

        await robot_api.set_target_cpos(
            robot_api.attach_coordinate_to_cpos(
                robot_api.build_target_cpos(**last_target),
                coordinate_id=coordinate_id,
            )
        )
        await robot_api.set_robot_command(robot_api.config.commands.move_target_linear)

        segment_deadline = time.monotonic() + max(segment_hold_s, 0.0)
        while time.monotonic() < segment_deadline:
            _check_operation()
            if safety_check is not None:
                await safety_check()
            await robot_api.send_command_heartbeat()
            await asyncio.sleep(min(update_interval, 0.05))

        if session.posture_seen():
            posture = session.position_snapshot()
            curr_xyz = np.array(
                [posture.x, posture.y, posture.z], dtype=np.float64)
            target_xyz = np.array([
                float(last_target["x"]),
                float(last_target["y"]),
                float(last_target["z"]),
            ], dtype=np.float64)

            if np.linalg.norm(curr_xyz - target_xyz) <= tolerance_mm:
                if stable_since is None:
                    stable_since = now
                elif now - stable_since >= settle_s:
                    break
            else:
                stable_since = None

        await asyncio.sleep(update_interval)

    await robot_api.clear_target_position()
    await robot_api.stop_command()
    return last_target


async def ensure_coordinate_active(
    robot_api: "CodroidAPI",
    coordinate_id: Optional[int] = None,
) -> None:
    """Ensure the specified coordinate system is active.

    Args:
        robot_api: The robot API instance.
        coordinate_id: Coordinate system ID (defaults to calibrated ID).
    """
    if coordinate_id is None:
        coordinate_id = get_robot_coordinate_id()

    session = get_robot_session()
    status = session.robot_status_snapshot()

    if isinstance(status, dict):
        current_id = status.get("CoordinateId")
        if current_id == coordinate_id:
            return

    await robot_api.set_current_coordinate_id(coordinate_id)


def log_move_target(
    label: str,
    target: dict,
    *,
    coordinate_id: Optional[int] = None,
) -> None:
    """Log a move target for debugging.

    Args:
        label: Human-readable label for the move.
        target: Target pose dict.
        coordinate_id: Coordinate system ID.
    """
    if coordinate_id is None:
        coordinate_id = get_robot_coordinate_id()

    LOGGER.info(
        "Move target '%s' (coord=%d): x=%.2f y=%.2f z=%.2f a=%.2f b=%.2f c=%.2f",
        label,
        coordinate_id,
        target.get("x", 0.0),
        target.get("y", 0.0),
        target.get("z", 0.0),
        target.get("a", 0.0),
        target.get("b", 0.0),
        target.get("c", 0.0),
    )
