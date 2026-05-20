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

from app.codroid_picker.app_instance import registry
from app.codroid_picker.exceptions import OperationPaused, OperationAbort
from app.codroid_picker.robot_helpers import (
    resolve_gripper_limits as _resolve_gripper_limits,
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


def resolve_gripper_close_width(preferred_width: Optional[float] = None) -> float:
    """Resolve the close width used for picks.

    When no explicit width is supplied, use the connected gripper's minimum
    external width. OnRobot grip detection is edge-triggered by an obstructed
    close command, so auto/manual picks should target the narrowest reachable
    width and then verify the grip-detected bit separately.
    """
    from app.codroid_picker.session import SESSION

    settings = registry().settings
    fallback = settings.gripper_profile.close_width_default
    if preferred_width is not None:
        try:
            fallback = float(preferred_width)
        except (TypeError, ValueError):
            fallback = settings.gripper_profile.close_width_default

    gripper = registry().gripper or SESSION.gripper
    if gripper is None:
        return float(fallback)

    try:
        min_width, max_width = resolve_gripper_limits(gripper)
    except Exception as exc:  # noqa: BLE001
        LOGGER.debug("Failed to resolve gripper close width: %s", exc)
        return float(fallback)

    if preferred_width is None:
        return float(min_width)
    return float(min(max(float(fallback), min_width), max_width))


def _wait_for_gripper_motion(gripper: "TWOFG", t_index: int) -> bool:
    """Wait until gripper motion finishes without requiring grip detection."""
    policy = getattr(gripper, "_policy", None)
    timeout_s = float(getattr(policy, "busy_timeout_s", 5.0) or 5.0)
    deadline = time.monotonic() + max(timeout_s, 0.1)
    while time.monotonic() < deadline:
        try:
            busy = gripper.isBusy(t_index=t_index)
        except Exception as exc:  # noqa: BLE001
            LOGGER.debug("Failed to poll gripper busy state: %s", exc)
            return False
        if isinstance(busy, (int, float)) and busy < 0:
            return False
        if not bool(busy):
            return True
        time.sleep(0.05)
    LOGGER.warning("Gripper motion timeout after %.1fs", timeout_s)
    return False


def _gripper_command_failed(result: object) -> bool:
    """Return True when a legacy gripper command result is an error code."""
    return isinstance(result, (int, float)) and result < 0


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

    gripper = registry().gripper or SESSION.gripper
    gripper_defaults = registry().gripper_defaults
    settings = registry().settings
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

        label_lower = action_label.lower()
        if "open" in label_lower and safe_width <= min_width and max_width > min_width:
            LOGGER.warning(
                "Open width %.1f <= min width %.1f; using max width %.1f",
                safe_width,
                min_width,
                max_width,
            )
            safe_width = max_width

        safe_width = min(max(safe_width, min_width), max_width)

        applied_force = force if force is not None else gripper_profile.force_default
        applied_force = min(
            max(applied_force, gripper_profile.force_min),
            gripper_profile.force_max,
        )
        applied_speed = speed if speed is not None else 80

        command_result = gripper.grip(
            t_index=gripper_defaults["t_index"],
            t_width=safe_width,
            n_force=applied_force,
            p_speed=applied_speed,
            f_wait=False,
        )
        if _gripper_command_failed(command_result):
            return f"Gripper action failed: command returned {command_result}."
        _wait_for_gripper_motion(gripper, gripper_defaults["t_index"])

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

    gripper = registry().gripper or SESSION.gripper
    gripper_defaults = registry().gripper_defaults

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
            settings = registry().settings
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

    Note: This function may have issues with asyncio task cancellation
    when called from Dash async callbacks. Programs should consider
    implementing their own movement loop (see PlateTestProgram for example).

    Args:
        robot_api: The robot API instance.
        target_pose: Target pose dict with x, y, z, a, b, c keys.
        coordinate_id: Coordinate system ID (defaults to calibrated ID).
        hold_seconds: Duration to hold position with heartbeats.
        heartbeat_interval: Time between heartbeats in seconds.
        run_id: Operation run ID for pause/stop checking.
    """
    if coordinate_id is None:
        coordinate_id = registry().robot_coordinate_id

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
    settings = registry().settings
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
    settle_on_stall: bool = False,
    stall_eps_mm: Optional[float] = None,
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
        settle_on_stall: Treat a stall (no motion) as a settled move.
        stall_eps_mm: Motion threshold used to detect stalls.
        require_posture: Whether to require posture stream.
        run_id: Operation run ID for pause/stop checking.
        safety_check: Optional async callback for safety checks.

    Returns:
        The final target pose dict, or None if no target was ever fetched.
    """
    if coordinate_id is None:
        coordinate_id = registry().robot_coordinate_id

    settings = registry().settings
    timeout_s = timeout_s or settings.pick_follow_timeout_s
    update_interval = update_interval or settings.pick_follow_update_interval_s
    segment_hold_s = segment_hold_s or settings.pick_follow_segment_hold_s
    settle_s = settle_s or settings.pick_follow_settle_s
    tolerance_mm = tolerance_mm or settings.pick_follow_tolerance_mm
    stall_eps_mm = stall_eps_mm or settings.verification_motion_eps_mm

    session = registry().get_or_create_robot_session()
    last_target: Optional[dict] = None
    last_posture: Optional[np.ndarray] = None
    last_motion_time: Optional[float] = None
    motion_seen = False

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
    LOGGER.info(
        f"move_linear_follow_target '{label}': Setting target pos type to cpos")
    await robot_api.set_target_pos_type(robot_api.config.target_pos_types.cpos)
    LOGGER.info(
        f"move_linear_follow_target '{label}': Target pos type set, entering main loop")

    stable_since: Optional[float] = None
    deadline = time.monotonic() + timeout_s
    consecutive_none_count = 0
    target_acquired = False  # True once we've successfully tracked a target
    iteration_count = 0

    while True:
        iteration_count += 1
        if iteration_count % 10 == 1:  # Log every 10 iterations
            LOGGER.debug(
                f"move_linear_follow_target '{label}': Iteration {iteration_count}")

        _check_operation()
        now = time.monotonic()

        if now >= deadline:
            LOGGER.warning(
                f"Follow move '{label}' timed out after {timeout_s:.1f}s")
            break

        target = target_fetch()
        if iteration_count <= 2:  # Log first 2 target fetches
            LOGGER.info(
                f"move_linear_follow_target '{label}': Fetched target: {target}")

        if target is not None:
            last_target = target
            consecutive_none_count = 0  # Reset counter on successful fetch
            target_acquired = True
        else:
            consecutive_none_count += 1
            if consecutive_none_count == 3:
                # Log that we're losing target but continue with last known position
                # This is expected when robot obscures the target during approach
                LOGGER.info(
                    f"Follow move '{label}': Target not visible for {consecutive_none_count} frames, "
                    f"continuing with last known position"
                )
            # Don't raise TrackingLost - just continue with last_target
            # The robot likely obscures the target when close

        if last_target is None:
            # We never acquired a target - this is a real error
            raise RuntimeError("No target available for follow move.")

        if iteration_count <= 2:
            LOGGER.info(
                f"move_linear_follow_target '{label}': Setting target cpos")
        await robot_api.set_target_cpos(
            robot_api.attach_coordinate_to_cpos(
                robot_api.build_target_cpos(**last_target),
                coordinate_id=coordinate_id,
            )
        )
        if iteration_count <= 2:
            LOGGER.info(
                f"move_linear_follow_target '{label}': Setting robot command to move_target_linear")
        await robot_api.set_robot_command(robot_api.config.commands.move_target_linear)
        if iteration_count <= 2:
            LOGGER.info(
                f"move_linear_follow_target '{label}': Robot command set, entering heartbeat loop")

        segment_deadline = time.monotonic() + max(segment_hold_s, 0.0)
        heartbeat_count = 0
        while time.monotonic() < segment_deadline:
            heartbeat_count += 1
            if iteration_count <= 2 and heartbeat_count == 1:
                LOGGER.info(
                    f"move_linear_follow_target '{label}': Heartbeat loop iteration {heartbeat_count}, checking operation")
            _check_operation()
            if iteration_count <= 2 and heartbeat_count == 1:
                LOGGER.info(
                    f"move_linear_follow_target '{label}': Operation check passed, running safety check")
            if safety_check is not None:
                await safety_check()
            if iteration_count <= 2 and heartbeat_count == 1:
                LOGGER.info(
                    f"move_linear_follow_target '{label}': About to send command heartbeat")
            await robot_api.send_command_heartbeat()
            if iteration_count <= 2 and heartbeat_count == 1:
                LOGGER.info(
                    f"move_linear_follow_target '{label}': Heartbeat sent, about to sleep for {min(update_interval, 0.05)}s")
            await asyncio.sleep(min(update_interval, 0.05))
            if iteration_count <= 2 and heartbeat_count == 1:
                LOGGER.info(
                    f"move_linear_follow_target '{label}': Sleep complete, continuing heartbeat loop")

        if iteration_count <= 2:
            LOGGER.info(
                f"move_linear_follow_target '{label}': Heartbeat loop complete ({heartbeat_count} heartbeats), checking position")

        if session.posture_seen():
            posture = session.position_snapshot()
            curr_xyz = np.array(
                [posture.x, posture.y, posture.z], dtype=np.float64)
            if last_posture is not None:
                move_delta = np.linalg.norm(curr_xyz - last_posture)
                if move_delta > stall_eps_mm:
                    motion_seen = True
                    last_motion_time = now
                elif (
                    settle_on_stall
                    and motion_seen
                    and last_motion_time is not None
                    and now - last_motion_time >= settle_s
                ):
                    LOGGER.info(
                        "move_linear_follow_target '%s': Motion stalled for %.2fs",
                        label,
                        now - last_motion_time,
                    )
                    break
            last_posture = curr_xyz
            target_xyz = np.array([
                float(last_target["x"]),
                float(last_target["y"]),
                float(last_target["z"]),
            ], dtype=np.float64)

            distance = np.linalg.norm(curr_xyz - target_xyz)
            if iteration_count <= 2:
                LOGGER.info(
                    f"move_linear_follow_target '{label}': Position check - distance: {distance:.2f}mm, tolerance: {tolerance_mm}mm")

            if distance <= tolerance_mm:
                if stable_since is None:
                    stable_since = now
                    if iteration_count <= 2:
                        LOGGER.info(
                            f"move_linear_follow_target '{label}': Position within tolerance, starting stability timer")
                elif now - stable_since >= settle_s:
                    LOGGER.info(
                        f"move_linear_follow_target '{label}': Position stable for {settle_s}s, move complete")
                    break
            else:
                stable_since = None
        else:
            if iteration_count <= 2:
                LOGGER.warning(
                    f"move_linear_follow_target '{label}': No posture data available")

        if iteration_count <= 2:
            LOGGER.info(
                f"move_linear_follow_target '{label}': Sleeping for {update_interval}s before next iteration")
        await asyncio.sleep(update_interval)
        if iteration_count <= 2:
            LOGGER.info(
                f"move_linear_follow_target '{label}': Sleep complete, starting iteration {iteration_count + 1}")

    LOGGER.info(
        f"move_linear_follow_target '{label}': Main loop exited after {iteration_count} iterations")
    await robot_api.clear_target_position()
    LOGGER.info(
        f"move_linear_follow_target '{label}': Cleared target position")
    await robot_api.stop_command()
    LOGGER.info(
        f"move_linear_follow_target '{label}': Stopped command, move complete")
    return last_target


def _auto_socket_backend_connected(robot_api: "CodroidAPI") -> bool | None:
    """Return auto-socket connection state when the backend exposes it."""
    backend = getattr(robot_api, "_auto_socket_backend", None)
    if backend is None:
        return None

    is_connected = getattr(backend, "is_connected", None)
    if callable(is_connected):
        return bool(is_connected())

    writer = getattr(backend, "_writer", None)
    if writer is None:
        return False
    return not writer.is_closing()


def _status_coordinate_id(session: "RobotSession") -> int | None:
    """Return the reported robot coordinate ID, if available."""
    try:
        status = session.robot_status_snapshot()
    except Exception:  # noqa: BLE001
        return None
    if not isinstance(status, dict):
        return None
    try:
        return int(status.get("CoordinateId"))
    except (TypeError, ValueError):
        return None


def _auto_socket_posture_xyz_candidates(
    session: "RobotSession",
    *,
    target_coordinate_id: int,
) -> list[tuple[str, tuple[float, float, float]]]:
    """Return current posture candidates in the target coordinate frame.

    The controller can report RobotPosture in base coordinates while an
    auto-socket project is active, even though the target frame is Coord5.
    Comparing both the raw posture and its base-to-Coord5 transform prevents
    false timeouts after a valid auto-mode move.
    """
    posture = session.position_snapshot()
    raw_xyz = (
        float(posture.x),
        float(posture.y),
        float(posture.z),
    )
    candidates = [("reported", raw_xyz)]
    if int(target_coordinate_id) != int(registry().robot_coordinate_id):
        return candidates

    try:
        from app.codroid_picker.transforms import _base_to_coord5

        coord5_xyz = _base_to_coord5(np.array(raw_xyz, dtype=np.float64))
    except Exception as exc:  # noqa: BLE001
        LOGGER.debug("Auto socket posture frame conversion failed: %s", exc)
        return candidates
    if coord5_xyz is None:
        return candidates

    converted_xyz = (
        float(coord5_xyz[0]),
        float(coord5_xyz[1]),
        float(coord5_xyz[2]),
    )
    if converted_xyz != raw_xyz:
        candidates.append(("base_to_coord5", converted_xyz))
    return candidates


async def move_auto_socket_cartesian_target(
    robot_api: "CodroidAPI",
    target_pose: dict,
    *,
    label: str,
    coordinate_id: Optional[int] = None,
    timeout_s: Optional[float] = None,
    tolerance_mm: Optional[float] = None,
    z_tolerance_mm: Optional[float] = None,
    settle_s: Optional[float] = None,
    hold_s: float = 0.0,
    active_check: Optional[Callable[[], None]] = None,
) -> None:
    """Send one auto-socket cartesian target and wait for posture to reach it.

    This intentionally uses the same codroid-api path as the standalone
    auto-mode test script: build a target CPos, then call
    move_to_cartesian_target_linear while the auto-socket project is active.
    """
    if coordinate_id is None:
        coordinate_id = registry().robot_coordinate_id

    settings = registry().settings
    session = registry().get_or_create_robot_session()
    timeout_s = float(timeout_s or getattr(settings, "auto_move_timeout_s", 15.0))
    tolerance_mm = float(tolerance_mm or getattr(settings, "auto_tolerance_mm", 2.0))
    z_tolerance_mm = float(
        z_tolerance_mm or getattr(settings, "auto_z_tolerance_mm", tolerance_mm)
    )
    settle_s = float(
        settle_s
        if settle_s is not None
        else getattr(settings, "auto_settle_s", 0.1)
    )

    log_move_target(label, target_pose, coordinate_id=coordinate_id)
    connected = _auto_socket_backend_connected(robot_api)
    if connected is False:
        raise RuntimeError(
            "Auto socket disconnected before "
            f"{label}: controller TCP client is missing."
        )

    cpos = robot_api.build_target_cpos(**target_pose)
    attach_coordinate = getattr(robot_api, "attach_coordinate_to_cpos", None)
    if int(coordinate_id) != 0 and callable(attach_coordinate):
        cpos = attach_coordinate(cpos, coordinate_id=coordinate_id)
    try:
        from codroid_api import format_auto_socket_frame

        config = getattr(robot_api, "_auto_socket_config", None)
        if config is not None:
            LOGGER.info(
                "Auto socket move '%s': sending frame %s",
                label,
                format_auto_socket_frame(cpos, config),
            )
    except Exception as exc:  # noqa: BLE001
        LOGGER.debug("Auto socket move '%s': frame formatting failed: %s", label, exc)

    try:
        await robot_api.set_current_coordinate_id(coordinate_id)
        await robot_api.move_to_cartesian_target_linear(cpos)
    except (TimeoutError, ConnectionError, OSError, RuntimeError) as exc:
        raise RuntimeError(
            f"Auto socket move failed during {label}: {exc}"
        ) from exc

    target_xyz = (
        float(target_pose["x"]),
        float(target_pose["y"]),
        float(target_pose["z"]),
    )
    deadline = time.monotonic() + max(timeout_s, 0.1)
    settle_since: Optional[float] = None
    last_xyz: tuple[float, float, float] | None = None
    last_xyz_frame = "unavailable"

    while time.monotonic() < deadline:
        if active_check is not None:
            active_check()

        if session.posture_seen():
            candidates = _auto_socket_posture_xyz_candidates(
                session,
                target_coordinate_id=int(coordinate_id),
            )
            last_xyz_frame, last_xyz = candidates[0]
            within_target = False
            for frame, candidate_xyz in candidates:
                candidate_within_target = (
                    abs(candidate_xyz[0] - target_xyz[0]) <= tolerance_mm
                    and abs(candidate_xyz[1] - target_xyz[1]) <= tolerance_mm
                    and abs(candidate_xyz[2] - target_xyz[2]) <= z_tolerance_mm
                )
                if candidate_within_target:
                    last_xyz_frame = frame
                    last_xyz = candidate_xyz
                    within_target = True
                    break
            if within_target:
                if settle_s <= 0:
                    break
                if settle_since is None:
                    settle_since = time.monotonic()
                elif time.monotonic() - settle_since >= settle_s:
                    break
            else:
                settle_since = None

        await asyncio.sleep(0.1)
    else:
        raise TimeoutError(
            "Auto socket move did not reach "
            f"{label} within {timeout_s:.1f}s "
            f"(coord={coordinate_id}, current_xyz={last_xyz}, "
            f"current_frame={last_xyz_frame}, "
            f"reported_coordinate={_status_coordinate_id(session)}, "
            "target_xyz="
            f"({target_xyz[0]:.2f}, {target_xyz[1]:.2f}, {target_xyz[2]:.2f}), "
            f"tolerance_xy={tolerance_mm:.1f}mm, "
            f"tolerance_z={z_tolerance_mm:.1f}mm)."
        )

    hold_deadline = time.monotonic() + max(float(hold_s), 0.0)
    while time.monotonic() < hold_deadline:
        if active_check is not None:
            active_check()
        await asyncio.sleep(0.1)


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
        coordinate_id = registry().robot_coordinate_id

    session = registry().get_or_create_robot_session()
    status = session.robot_status_snapshot()

    if isinstance(status, dict):
        current_id = status.get("CoordinateId")
        if current_id == coordinate_id:
            return

    await robot_api.set_current_coordinate_id(coordinate_id)


async def select_coordinate_and_wait_for_posture(
    robot_api: "CodroidAPI",
    session: "RobotSession",
    *,
    coordinate_id: int,
    timeout_s: float = 5.0,
    settle_s: float = 2.0,
) -> None:
    """Select a coordinate ID and wait for status plus a fresh posture sample."""
    initial_posture = session.position_snapshot()
    initial_timestamp = float(getattr(initial_posture, "timestamp", 0.0) or 0.0)

    await ensure_coordinate_active(robot_api, coordinate_id)
    deadline = time.monotonic() + max(float(timeout_s), 0.1)
    resent = False

    while time.monotonic() < deadline:
        status = session.robot_status_snapshot()
        current_id = status.get("CoordinateId") if isinstance(status, dict) else None
        try:
            matched_coordinate = int(current_id) == int(coordinate_id)
        except (TypeError, ValueError):
            matched_coordinate = False

        posture = session.position_snapshot()
        posture_timestamp = float(getattr(posture, "timestamp", 0.0) or 0.0)
        fresh_posture = (
            posture_timestamp <= 0.0
            or initial_timestamp <= 0.0
            or posture_timestamp > initial_timestamp
        )
        if matched_coordinate and fresh_posture:
            await asyncio.sleep(max(float(settle_s), 0.0))
            return

        if not resent and time.monotonic() > deadline - (
            max(float(timeout_s), 0.1) / 2.0
        ):
            await robot_api.set_current_coordinate_id(coordinate_id)
            resent = True
        await asyncio.sleep(0.1)

    raise RuntimeError(
        f"Failed to confirm CoordinateId {coordinate_id} with fresh posture."
    )


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
        coordinate_id = registry().robot_coordinate_id

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
