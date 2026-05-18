"""Automated pick and place robot program.

This program runs a continuous loop picking detected objects and
dropping them at the designated drop location.
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import suppress
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
    resolve_gripper_close_width,
    resolve_gripper_limits,
)
from app.codroid_picker.pick_place_helpers import (
    build_drop_poses,
    compute_safe_transit_z,
    select_closest_detection,
)
from app.codroid_picker.gripper_utils import await_grip_status
from app.codroid_picker.persistence import _record_pick_cycle
from app.codroid_picker.target_planner import (
    PickPlan,
    TargetCandidate,
    plan_pick_strategy,
    rank_targets,
)
from app.codroid_picker.auto_plan_state import (
    clear_auto_plan_snapshot,
    publish_auto_plan_snapshot,
    update_auto_plan_snapshot,
)

LOGGER = logging.getLogger(__name__)


class EmptyGripError(RuntimeError):
    """Raised when a pick closes without detecting a gripped object."""


def _record_cycle_metrics_safe(success: bool, cycle_started_at: Optional[float]) -> None:
    """Record cycle metrics, with a lightweight fallback for unit tests."""
    try:
        _record_pick_cycle(success, cycle_started_at)
        return
    except RuntimeError:
        # App registry may be intentionally uninitialized in isolated unit tests.
        from app.codroid_picker.session import SESSION

        SESSION.pick_attempts += 1
        if success:
            SESSION.pick_successes += 1
        else:
            SESSION.pick_failures += 1


def _resolve_local_socket_host(robot_api: "CodroidAPI") -> str | None:
    """Resolve the app host IP the robot controller should connect back to."""
    config = getattr(robot_api, "config", None)
    robot_host = getattr(config, "host", None)
    resolver = getattr(robot_api, "_local_source_ip_for", None)
    if callable(resolver) and robot_host:
        try:
            return str(resolver(str(robot_host)))
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("Auto: failed to resolve local socket host: %s", exc)
    return None


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

        clear_auto_plan_snapshot()

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

        try:
            self.update_message("Switching controller to Auto mode...")
            await robot_api.set_auto_mode()
            await robot_api.set_current_coordinate_id(coordinate_id)
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("Auto: failed to switch controller to Auto mode")
            message = f"Auto mode setup failed: {exc}"
            update_auto_plan_snapshot(stage="error", message=message)
            await self._restore_manual_mode(robot_api)
            return ProgramResult(status=ProgramStatus.ERROR, message=message)

        auto_socket_config = self._build_auto_socket_config(settings, robot_api)
        try:
            try:
                await robot_api.stop_project()
                await asyncio.sleep(0.3)
            except Exception as exc:  # noqa: BLE001
                LOGGER.debug("Auto: no active project stopped before socket start: %s", exc)
            auto_socket_context = robot_api.auto_socket_mode(auto_socket_config)
            backend = await auto_socket_context.__aenter__()
            await robot_api.set_current_coordinate_id(coordinate_id)
            project_info = getattr(backend, "project_info", None)
            LOGGER.info(
                "Auto: auto-socket project=%s task=%s run_label=%s socket=%s:%s",
                getattr(project_info, "project_id", None),
                getattr(project_info, "task_id", None),
                getattr(project_info, "run_label_id", None),
                getattr(project_info, "socket_host", None),
                getattr(project_info, "socket_port", None),
            )
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("Auto: failed to activate auto-socket motion")
            message = f"Auto socket motion setup failed: {exc}"
            update_auto_plan_snapshot(stage="error", message=message)
            await self._cleanup_failed_auto_socket_start(robot_api)
            return ProgramResult(status=ProgramStatus.ERROR, message=message)

        try:
            try:
                self.update_message("Auto socket motion active.")
                LOGGER.info(
                    "Auto: auto-socket motion active on port %s",
                    auto_socket_config.socket_port,
                )
                result = await self._run_auto_loop(
                    robot_api,
                    settings=settings,
                    coordinate_id=coordinate_id,
                    workspace_def=workspace_def,
                )
            except BaseException as exc:
                suppress = await auto_socket_context.__aexit__(
                    type(exc), exc, exc.__traceback__
                )
                if not suppress:
                    raise
                return ProgramResult(
                    status=ProgramStatus.STOPPED,
                    message="Auto socket motion context stopped execution.",
                )
            else:
                await auto_socket_context.__aexit__(None, None, None)
                return result
        finally:
            await self._restore_manual_mode(robot_api)

    @staticmethod
    def _build_auto_socket_config(settings, robot_api: "CodroidAPI"):
        """Build the codroid-api auto-socket config for Auto Pick & Place."""
        from codroid_api import AutoSocketConfig

        port = int(getattr(settings, "auto_socket_port", 19093) or 19093)
        return AutoSocketConfig(
            socket_role="server",
            socket_bind_host="0.0.0.0",
            socket_port=port,
            project_socket_host=_resolve_local_socket_host(robot_api),
            project_socket_port=port,
            install_project=True,
            start_project=True,
            stop_project_on_exit=True,
            connect_on_enter=True,
            accept_timeout_s=25.0,
            read_timeout_ms=int(
                getattr(settings, "auto_socket_read_timeout_ms", 120000)
                or 120000
            ),
            move_speed="V100",
            move_acc="ACC100",
        )

    @staticmethod
    async def _cleanup_failed_auto_socket_start(robot_api: "CodroidAPI") -> None:
        """Best-effort cleanup after socket project startup fails."""
        try:
            await robot_api.stop_auto_socket_project()
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning(
                "Auto: failed to stop auto-socket project after setup error: %s",
                exc,
            )
        await AutoPickPlaceProgram._restore_manual_mode(robot_api)

    @staticmethod
    async def _restore_manual_mode(robot_api: "CodroidAPI") -> None:
        """Best-effort return from controller Auto mode after auto operation."""
        try:
            await robot_api.set_manual_mode()
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("Auto: failed to switch controller to Manual mode: %s", exc)

    async def _run_auto_loop(
        self,
        robot_api: "CodroidAPI",
        *,
        settings,
        coordinate_id: int,
        workspace_def,
    ) -> ProgramResult:
        """Run the auto pick/place loop after auto-socket motion is active."""
        from app.codroid_picker.session import SESSION

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
        publish_auto_plan_snapshot(
            stage="starting",
            message="Auto program started.",
            orientation=run_orientation,
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
                    # No targets - wait at drop approach without descending.
                    publish_auto_plan_snapshot(
                        stage="waiting_for_targets",
                        message=(
                            "No targets detected. No object gripped; "
                            "waiting at drop approach."
                        ),
                        ranked_candidates=[],
                        selected_candidate=None,
                        approach_mm=approach_mm,
                        pick_mm=pick_mm,
                        drop_mm=drop_mm,
                        orientation=run_orientation,
                        gripper_open_mm=settings.gripper_profile.open_width_default,
                        meta={"detections": len(detections)},
                    )
                    if not SESSION.auto_waiting_at_drop:
                        self.update_message(
                            "No targets. No object gripped; moving to drop approach."
                        )
                        await self._move_to_drop(
                            robot_api,
                            approach_mm=approach_mm,
                            drop_mm=drop_mm,
                            open_gripper=SESSION.robot_has_object,
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
                missed_grip_this_cycle = False
                cycle_started_at = time.monotonic()
                publish_auto_plan_snapshot(
                    stage="ranked",
                    message="Targets ranked.",
                    ranked_candidates=ranked_candidates,
                    selected_candidate=ranked_candidates[0],
                    approach_mm=approach_mm,
                    pick_mm=pick_mm,
                    drop_mm=drop_mm,
                    orientation=run_orientation,
                    meta={
                        "targets": len(targets),
                        "detections": len(detections),
                    },
                )
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
                    publish_auto_plan_snapshot(
                        stage="candidate_selected",
                        message="Candidate selected for planning.",
                        ranked_candidates=ranked_candidates,
                        selected_candidate=candidate,
                        pick_plan=pick_plan,
                        approach_mm=approach_mm,
                        pick_mm=pick_mm,
                        drop_mm=drop_mm,
                        orientation=run_orientation,
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
                            update_auto_plan_snapshot(
                                stage="separation_attempt",
                                message="Target is clustered. Executing separation nudge.",
                                meta={"attempt": attempt, "max_attempts": max(1, separation_max_attempts)},
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
                            publish_auto_plan_snapshot(
                                stage="replan_after_separation",
                                message="Re-planned after separation attempt.",
                                ranked_candidates=ranked_candidates,
                                selected_candidate=candidate,
                                pick_plan=pick_plan,
                                approach_mm=approach_mm,
                                pick_mm=pick_mm,
                                drop_mm=drop_mm,
                                orientation=run_orientation,
                                meta={"attempt": attempt},
                            )
                            if pick_plan.mode == "rotate_pick":
                                separated = True
                                break
                        if not separated:
                            update_auto_plan_snapshot(
                                stage="separation_failed",
                                message="Unable to isolate candidate after separation attempts.",
                            )
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
                            drop_mm=drop_mm,
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
                        _record_cycle_metrics_safe(True, cycle_started_at)
                        picked_this_cycle = True
                        update_auto_plan_snapshot(
                            stage="cycle_complete",
                            message="Pick/drop cycle completed.",
                            meta={"cycles_completed": cycles_completed, "cycles_failed": cycles_failed},
                        )
                        self.update_message(
                            f"Cycle complete. Total: {cycles_completed} picked, {cycles_failed} failed."
                        )
                        break
                    except EmptyGripError as exc:
                        LOGGER.info("Auto: empty grip detected, rescanning workspace: %s", exc)
                        update_auto_plan_snapshot(
                            stage="empty_grip",
                            message="Grip closed without object; rescanning.",
                            meta={"error": str(exc)},
                        )
                        missed_grip_this_cycle = True
                        break
                    except Exception as exc:
                        LOGGER.warning(f"Pick/place cycle failed for candidate: {exc}")
                        update_auto_plan_snapshot(
                            stage="candidate_failed",
                            message=f"Candidate failed: {exc}",
                        )
                        continue

                if not picked_this_cycle:
                    cycles_failed += 1
                    _record_cycle_metrics_safe(False, cycle_started_at)
                    if missed_grip_this_cycle:
                        continue
                    try:
                        update_auto_plan_snapshot(
                            stage="returning_to_drop",
                            message="No successful pick. Returning to drop pose.",
                        )
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
            update_auto_plan_snapshot(
                stage="stopped",
                message="Auto program stopped.",
            )
            pass

        message = f"Auto operation stopped. {cycles_completed} cycles completed, {cycles_failed} failed."
        update_auto_plan_snapshot(
            stage="stopped",
            message=message,
            meta={"cycles_completed": cycles_completed, "cycles_failed": cycles_failed},
        )
        with suppress(Exception):
            from app.codroid_picker.streaming import set_active_overlay_task

            set_active_overlay_task(None)
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
        drop_mm: float,
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

        # Determine gripper width. Pick closes target the mechanical minimum so
        # OnRobot can set the grip-detected bit when the object blocks travel.
        gripper_close = resolve_gripper_close_width()
        estimated_width_mm: Optional[float] = None
        if estimate_width and "bbox" in target:
            # Estimate width from bounding box
            bbox = target["bbox"]
            width_px = bbox[2] - bbox[0]
            # Simple estimation - would need calibration for accuracy
            estimated_width_mm = width_px * 0.2  # rough estimate

        publish_auto_plan_snapshot(
            stage="executing_pick",
            message="Executing pick motion.",
            selected_candidate=target,
            pick_plan=PickPlan(mode="rotate_pick", final_c=final_c, separation=None),
            approach_mm=approach_mm,
            pick_mm=pick_mm,
            orientation=(orientation[0], orientation[1], final_c),
            gripper_close_mm=gripper_close,
            gripper_open_mm=settings.gripper_profile.open_width_default,
            meta={
                "estimated_object_width_mm": estimated_width_mm,
            },
        )

        transit_z = compute_safe_transit_z(
            approach_mm=approach_mm,
            drop_mm=drop_mm,
            drop_robot_xyz=SESSION.drop_target_robot,
            clearance_mm=float(settings.auto_transit_clearance_mm),
        )
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

        # Align orientation at transit height only when needed.
        await wait_for_operation_resume(self._run_id)
        orientation_pose = dict(transit_pose)
        orientation_pose["a"] = float(final_orientation[0])
        orientation_pose["b"] = float(final_orientation[1])
        orientation_pose["c"] = float(final_orientation[2])
        angle_delta = max(
            abs(float(orientation_pose["a"]) - float(transit_pose["a"])),
            abs(float(orientation_pose["b"]) - float(transit_pose["b"])),
            abs(float(orientation_pose["c"]) - float(transit_pose["c"])),
        )
        if angle_delta > 0.05:
            await self._move_linear_with_hold(
                robot_api,
                orientation_pose,
                coordinate_id,
                label="auto_pick_align_orientation",
                hold_s=settings.auto_hold_s,
            )
        else:
            orientation_pose = transit_pose

        # Move to pick in one descent step.
        LOGGER.debug("Auto: Transit done, preparing pick move")
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
        grip_timeout_s = max(float(settings.plate_test_grip_timeout_s), 0.5)
        gripped = await await_grip_status(grip_timeout_s)
        LOGGER.info("Auto: grip verification result=%s timeout=%.2fs", gripped, grip_timeout_s)

        # Retreat directly to transit height to avoid a redundant lift step before drop.
        await wait_for_operation_resume(self._run_id)
        retreat_pose = dict(orientation_pose)
        retreat_pose["z"] = float(transit_z)
        await self._move_linear_with_hold(
            robot_api,
            retreat_pose,
            coordinate_id,
            label="auto_pick_retreat",
            hold_s=settings.auto_hold_s,
        )

        if not gripped:
            SESSION.robot_has_object = False
            command_gripper(settings.gripper_profile.open_width_default, "opened")
            update_auto_plan_snapshot(
                stage="empty_grip",
                message="No object detected after close command.",
            )
            raise EmptyGripError("No object detected after grip close.")

        SESSION.robot_has_object = True

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
        transit_z = compute_safe_transit_z(
            approach_mm=approach_mm,
            drop_mm=drop_mm,
            drop_robot_xyz=drop_target,
            clearance_mm=float(settings.auto_transit_clearance_mm),
        )
        transit_pose = dict(approach_pose)
        transit_pose["z"] = float(transit_z)
        update_auto_plan_snapshot(
            stage="executing_drop",
            message="Executing drop sequence.",
            approach_mm=approach_mm,
            drop_mm=drop_mm,
            orientation=orientation,
            gripper_open_mm=settings.gripper_profile.open_width_default,
            drop_target_robot=drop_target,
            meta={
                "drop_approach_pose": dict(approach_pose),
                "drop_pose": dict(drop_pose),
            },
        )

        # Traverse to drop at safe transit height.
        await wait_for_operation_resume(self._run_id)
        await self._move_linear_with_hold(
            robot_api,
            transit_pose,
            coordinate_id,
            label="drop_transit",
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
        gripper_close = resolve_gripper_close_width()
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
        """Move to drop waiting position and optionally release an object.

        Idle waiting should stay at drop approach height to reduce unnecessary
        vertical motion and gripper actions while scanning for targets.
        """
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
        transit_z = compute_safe_transit_z(
            approach_mm=approach_mm,
            drop_mm=drop_mm,
            drop_robot_xyz=drop_target,
            clearance_mm=float(settings.auto_transit_clearance_mm),
        )
        transit_pose = dict(approach_pose)
        transit_pose["z"] = float(transit_z)
        should_release = bool(open_gripper and SESSION.robot_has_object)
        xy_tolerance_mm = float(getattr(settings, "auto_tolerance_mm", 2.0))
        z_tolerance_mm = float(getattr(settings, "auto_z_tolerance_mm", 3.0))
        at_drop_xy = False
        at_drop_approach = False
        at_drop_pose = False
        try:
            robot_session = registry().get_or_create_robot_session()
            if robot_session.posture_seen():
                posture = robot_session.position_snapshot()
                at_drop_xy = (
                    abs(float(posture.x) - float(approach_pose["x"])) <= xy_tolerance_mm
                    and abs(float(posture.y) - float(approach_pose["y"])) <= xy_tolerance_mm
                )
                at_drop_approach = (
                    at_drop_xy
                    and abs(float(posture.z) - float(approach_pose["z"])) <= z_tolerance_mm
                )
                at_drop_pose = (
                    at_drop_xy
                    and abs(float(posture.z) - float(drop_pose["z"])) <= z_tolerance_mm
                )
        except Exception as exc:  # noqa: BLE001
            LOGGER.debug("Auto: failed to read posture for move-to-drop optimization: %s", exc)

        if not should_release and at_drop_approach:
            update_auto_plan_snapshot(
                stage="waiting_at_drop",
                message="No object gripped; waiting at drop approach.",
                approach_mm=approach_mm,
                drop_mm=drop_mm,
                orientation=orientation,
            )
            return

        lift_pose = dict(transit_pose)
        try:
            robot_session = registry().get_or_create_robot_session()
            if robot_session.posture_seen():
                posture = robot_session.position_snapshot()
                lift_pose["x"] = float(posture.x)
                lift_pose["y"] = float(posture.y)
        except Exception as exc:  # noqa: BLE001
            LOGGER.debug("Auto: failed to read posture for move-to-drop lift pose: %s", exc)

        if not should_release:
            LOGGER.info("Auto: no object gripped; moving to drop approach only.")

        if not at_drop_xy:
            await wait_for_operation_resume(self._run_id)
            await self._move_linear_with_hold(
                robot_api,
                lift_pose,
                coordinate_id,
                label="move_to_drop_lift",
                hold_s=settings.auto_hold_s,
            )

            await wait_for_operation_resume(self._run_id)
            await self._move_linear_with_hold(
                robot_api,
                transit_pose,
                coordinate_id,
                label="move_to_drop_transit",
                hold_s=settings.auto_hold_s,
            )

        if not should_release:
            if not at_drop_approach:
                await wait_for_operation_resume(self._run_id)
                await self._move_linear_with_hold(
                    robot_api,
                    approach_pose,
                    coordinate_id,
                    label="move_to_drop_approach",
                    hold_s=settings.auto_hold_s,
                )

        if should_release:
            if not at_drop_pose:
                await wait_for_operation_resume(self._run_id)
                await self._move_linear_with_hold(
                    robot_api,
                    drop_pose,
                    coordinate_id,
                    label="move_to_drop_pose",
                    hold_s=settings.auto_hold_s,
                )

            gripper_open = settings.gripper_profile.open_width_default
            command_gripper(gripper_open, "opened")
            SESSION.robot_has_object = False

            await wait_for_operation_resume(self._run_id)
            await self._move_linear_with_hold(
                robot_api,
                approach_pose,
                coordinate_id,
                label="move_to_drop_retreat",
                hold_s=settings.auto_hold_s,
            )
        update_auto_plan_snapshot(
            stage="waiting_at_drop",
            message=(
                "Object dropped; waiting at drop approach."
                if should_release
                else "No object gripped; waiting at drop approach."
            ),
            approach_mm=approach_mm,
            drop_mm=drop_mm,
            orientation=orientation,
        )

    @staticmethod
    def _auto_socket_backend(robot_api: "CodroidAPI"):
        """Return the active codroid-api auto-socket backend, if any."""
        return getattr(robot_api, "_auto_socket_backend", None)

    @staticmethod
    def _require_auto_socket_connected(
        robot_api: "CodroidAPI",
        *,
        label: str,
    ) -> None:
        """Fail fast when auto-socket mode lost its controller TCP client."""
        backend = AutoPickPlaceProgram._auto_socket_backend(robot_api)
        if backend is None:
            return
        is_connected = getattr(backend, "is_connected", None)
        if callable(is_connected):
            connected = bool(is_connected())
        else:
            writer = getattr(backend, "_writer", None)
            connected = writer is not None and not writer.is_closing()
        if not connected:
            raise RuntimeError(
                "Auto socket disconnected before "
                f"{label}: controller TCP client is missing. "
                "The controller project may have stopped or SocketReadStr "
                "may have timed out."
            )

    @staticmethod
    def _socket_move_error(exc: BaseException, *, label: str) -> RuntimeError:
        """Build a clear operator-facing error for socket move failures."""
        return RuntimeError(
            "Auto socket move failed during "
            f"{label}: controller TCP client is missing, the project stopped, "
            f"or SocketReadStr timed out ({exc})."
        )

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
        self._require_auto_socket_connected(robot_api, label=label)

        await robot_api.set_target_pos_type(robot_api.config.target_pos_types.cpos)
        await robot_api.set_target_cpos(
            robot_api.attach_coordinate_to_cpos(
                robot_api.build_target_cpos(**target_pose),
                coordinate_id=coordinate_id,
            )
        )
        try:
            await robot_api.set_robot_command(robot_api.config.commands.move_target_linear)
        except (TimeoutError, ConnectionError, OSError, RuntimeError) as exc:
            raise self._socket_move_error(exc, label=label) from exc

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
