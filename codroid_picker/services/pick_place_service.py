"""Manual pick/drop service routines."""

from __future__ import annotations

import asyncio
import time
from typing import Any, Optional

import numpy as np

from codroid_picker.config import AppSettings
from codroid_picker.core.persistence import write_json
from codroid_picker.core.target_planner import plan_pick_strategy, rank_targets
from codroid_picker.core.transforms import pose_from_xy, robot_to_workspace_mm, workspace_pixel_to_mm
from codroid_picker.services.calibration_service import CalibrationService
from codroid_picker.services.detection_service import DetectionService
from codroid_picker.services.gripper_service import GripperService
from codroid_picker.services.robot_service import RobotService
from codroid_picker.state.store import StateStore


class PickPlaceService:
    """Execute pick/place operations using the new service layer."""

    def __init__(
        self,
        settings: AppSettings,
        state: StateStore,
        detection: DetectionService,
        robot: RobotService,
        gripper: GripperService,
        calibration: CalibrationService,
    ) -> None:
        self.settings = settings
        self.state = state
        self.detection = detection
        self.robot = robot
        self.gripper = gripper
        self.calibration = calibration
        self._lock = asyncio.Lock()

    async def teach_drop_target(self) -> str:
        posture = self._current_robot_pose()
        robot_xyz = (float(posture["x"]), float(posture["y"]), float(posture["z"]))
        workspace_xy = self._robot_to_workspace_or_xy(robot_xyz)
        payload: dict[str, Any] = {
            "x": workspace_xy[0],
            "y": workspace_xy[1],
            "a": float(posture.get("a", 0.0)),
            "b": float(posture.get("b", 0.0)),
            "c": float(posture.get("c", 0.0)),
            "robot_x": robot_xyz[0],
            "robot_y": robot_xyz[1],
            "robot_z": robot_xyz[2],
            "coord_id": int(self.state.domain("robot").get("coordinate_id") or self.settings.robot_coordinate_id),
        }
        write_json(self.settings.drop_target_path, payload)
        self.calibration.drop_target = payload
        self.state.update_domain("auto_pick", drop_target=payload, message="Drop target taught.")
        return f"Drop target taught at ({workspace_xy[0]:.1f}, {workspace_xy[1]:.1f}) mm."

    async def return_to_drop_target(self, *, height_offset_mm: float) -> str:
        drop = self._require_drop_target()
        approach_pose, _drop_pose, coord_id = self._drop_poses(drop, approach_mm=height_offset_mm, drop_mm=0.0)
        await self.robot.move(approach_pose, coordinate_id=coord_id, timeout_s=self.settings.auto_move_timeout_s)
        self.state.update_domain("operation", message="Returned to drop target approach pose.")
        return "Returned to drop target."

    async def drop(self, *, drop_height_mm: float, operation_id: Optional[str] = None) -> str:
        async with self._lock:
            return await self._run_drop(drop_height_mm=drop_height_mm, operation_id=operation_id)

    async def pick_closest(self, *, drop_after: bool, operation_id: Optional[str] = None) -> str:
        async with self._lock:
            return await self._run_pick(drop_after=drop_after, operation_id=operation_id)

    async def pick_candidate(self, candidate: Any, plan: Any, *, operation_id: Optional[str]) -> str:
        async with self._lock:
            return await self._execute_pick_candidate(candidate, plan, drop_after=True, operation_id=operation_id)

    def plan_next_target(self, detections: Optional[list[dict[str, Any]]] = None) -> tuple[Any, Any, list[Any]]:
        candidates = self._ranked_candidates(detections)
        if not candidates:
            raise RuntimeError("No pickable detections are available.")
        candidate = candidates[0]
        posture = self._current_robot_pose(default=True)
        plan = plan_pick_strategy(
            candidate,
            candidates,
            current_c_deg=float(posture.get("c", 0.0)),
            rotation_step_deg=float(self.settings.auto_rotation_step_deg),
            rotation_max_delta_deg=float(self.settings.auto_rotation_max_delta_deg),
            body_radius_mm=float(self.settings.auto_collision_body_radius_mm),
            finger_reach_mm=float(self.settings.auto_collision_finger_reach_mm),
            clearance_mm=float(self.settings.auto_rotation_clearance_mm),
            separation_entry_mm=float(self.settings.auto_separation_entry_mm),
            separation_push_mm=float(self.settings.auto_separation_push_mm),
        )
        self._publish_plan(candidate, plan, candidates)
        return candidate, plan, candidates

    def enrich_detection(self, detection: dict[str, Any]) -> dict[str, Any]:
        self._coord5_for_detection(detection)
        return detection

    async def _run_pick(self, *, drop_after: bool, operation_id: Optional[str]) -> str:
        own_operation = operation_id is None
        started = time.monotonic()
        active_id = operation_id
        if own_operation:
            active_id = self.state.begin_operation(
                source="manual",
                label="Manual pick and drop" if drop_after else "Manual pick",
                message="Manual pick started.",
            )
        try:
            candidate, plan, _candidates = self.plan_next_target()
            message = await self._execute_pick_candidate(candidate, plan, drop_after=drop_after, operation_id=active_id)
            if own_operation:
                self.record_cycle(success=True, started=started)
            if own_operation:
                self.state.finish_operation(active_id, status="idle", message=message)
            return message
        except Exception as exc:
            if own_operation:
                self.record_cycle(success=False, started=started)
            if own_operation:
                self.state.finish_operation(active_id, status="error", message=str(exc))
            raise

    async def _execute_pick_candidate(self, candidate: Any, plan: Any, *, drop_after: bool, operation_id: Optional[str]) -> str:
        if self.state.domain("safety").get("blocked"):
            raise RuntimeError("Safety guard is blocking robot motion.")
        await self._wait_if_paused_or_stopped()
        if getattr(plan, "mode", "") == "separate_then_rescan":
            await self._execute_separation(getattr(plan, "separation", None), final_c=float(getattr(plan, "final_c", 0.0)))
            return "Separation nudge executed; rescan before picking."

        profile = self._profile_for_detection(candidate.detection)
        approach_mm = float(profile.get("approach_mm", self.settings.auto_approach_mm))
        pick_mm = float(profile.get("pick_mm", self.settings.auto_pick_mm))
        final_c = float(getattr(plan, "final_c", self._current_robot_pose(default=True).get("c", 0.0)))
        xy = (float(candidate.coord5_xy[0]), float(candidate.coord5_xy[1]))
        approach_pose = pose_from_xy(xy, z_mm=approach_mm, c=final_c, max_z_mm=self.settings.coord5_max_z_mm)
        pick_pose = pose_from_xy(xy, z_mm=pick_mm, c=final_c, max_z_mm=self.settings.coord5_max_z_mm)
        coord_id = int(self.settings.robot_coordinate_id)
        self._publish_plan_stage("opening_gripper", "Opening gripper before pick.")
        self.state.update_domain("operation", message="Moving to pick approach.")
        await self.gripper.open(width_mm=float(profile.get("gripper_open_mm", 80.0)), speed=int(profile.get("gripper_speed", 80)))
        await self._wait_if_paused_or_stopped()
        self._publish_plan_stage("approach", "Moving to pick approach.", approach_pose=approach_pose)
        await self.robot.move(approach_pose, coordinate_id=coord_id, timeout_s=self.settings.auto_move_timeout_s)
        self.state.update_domain("operation", message="Moving to pick pose.")
        await self._wait_if_paused_or_stopped()
        self._publish_plan_stage("pick", "Moving to pick pose.", pick_pose=pick_pose)
        await self.robot.move(pick_pose, coordinate_id=coord_id, timeout_s=self.settings.auto_move_timeout_s)
        self.state.update_domain("operation", message="Closing gripper.")
        await self._wait_if_paused_or_stopped()
        self._publish_plan_stage("closing_gripper", "Closing gripper on target.")
        await self.gripper.close(
            width_mm=float(profile.get("gripper_close_mm", 20.0)),
            force=int(profile.get("gripper_force", 40)),
            speed=int(profile.get("gripper_speed", 80)),
            gentle=bool(profile.get("soft_gentle", False)),
            vacuum_setpoint=int(profile.get("vacuum_setpoint", 40)),
            vacuum_threshold=float(profile.get("vacuum_threshold", 32.0)),
        )
        holding = self.state.domain("gripper").get("holding")
        if holding is False:
            self._publish_plan_stage("grip_failed", "Gripper did not report a held object.")
            raise RuntimeError("Grip verification failed.")
        await self._wait_if_paused_or_stopped()
        self._publish_plan_stage("retreat", "Retreating from pick pose.", approach_pose=approach_pose)
        await self.robot.move(approach_pose, coordinate_id=coord_id, timeout_s=self.settings.auto_move_timeout_s)
        if not drop_after:
            self._publish_plan_stage("complete", "Pick completed.")
            return "Pick completed."
        return await self._run_drop(drop_height_mm=self.settings.auto_drop_mm, operation_id=operation_id)

    async def _run_drop(self, *, drop_height_mm: float, operation_id: Optional[str]) -> str:
        own_operation = operation_id is None
        active_id = operation_id
        started = time.monotonic()
        if own_operation:
            active_id = self.state.begin_operation(source="manual", label="Drop", message="Drop started.")
        try:
            drop = self._require_drop_target()
            approach_pose, drop_pose, coord_id = self._drop_poses(
                drop,
                approach_mm=float(self.settings.auto_approach_mm),
                drop_mm=float(drop_height_mm),
            )
            self.state.update_domain("operation", message="Moving to drop approach.")
            await self._wait_if_paused_or_stopped()
            self._publish_plan_stage("drop_approach", "Moving to drop approach.", approach_pose=approach_pose)
            await self.robot.move(approach_pose, coordinate_id=coord_id, timeout_s=self.settings.auto_move_timeout_s)
            self.state.update_domain("operation", message="Moving to drop pose.")
            await self._wait_if_paused_or_stopped()
            self._publish_plan_stage("drop", "Moving to drop pose.", drop_pose=drop_pose)
            await self.robot.move(drop_pose, coordinate_id=coord_id, timeout_s=self.settings.auto_move_timeout_s)
            await self._wait_if_paused_or_stopped()
            self._publish_plan_stage("release", "Opening gripper at drop target.")
            await self.gripper.open(width_mm=80.0, speed=80)
            await self._wait_if_paused_or_stopped()
            self._publish_plan_stage("drop_retreat", "Retreating from drop target.", approach_pose=approach_pose)
            await self.robot.move(approach_pose, coordinate_id=coord_id, timeout_s=self.settings.auto_move_timeout_s)
            self.record_cycle(success=True, started=started)
            if own_operation:
                self.state.finish_operation(active_id, status="idle", message="Drop completed.")
            self._publish_plan_stage("complete", "Drop completed.")
            return "Drop completed."
        except Exception as exc:
            self.record_cycle(success=False, started=started)
            if own_operation:
                self.state.finish_operation(active_id, status="error", message=str(exc))
            raise

    async def _execute_separation(self, separation: Any, *, final_c: float) -> None:
        if separation is None:
            raise RuntimeError("No separation trajectory is available.")
        coord_id = int(self.settings.robot_coordinate_id)
        z_entry = float(self.settings.auto_approach_mm)
        z_push = float(self.settings.auto_pick_mm)
        entry_pose = pose_from_xy(separation.entry_xy, z_mm=z_entry, c=final_c, max_z_mm=self.settings.coord5_max_z_mm)
        push_pose = pose_from_xy(separation.push_end_xy, z_mm=z_push, c=final_c, max_z_mm=self.settings.coord5_max_z_mm)
        retreat_pose = pose_from_xy(separation.entry_xy, z_mm=z_entry, c=final_c, max_z_mm=self.settings.coord5_max_z_mm)
        await self._wait_if_paused_or_stopped()
        await self.gripper.open(width_mm=80.0, speed=80)
        await self._wait_if_paused_or_stopped()
        await self.robot.move(entry_pose, coordinate_id=coord_id, timeout_s=self.settings.auto_move_timeout_s)
        await self._wait_if_paused_or_stopped()
        await self.robot.move(push_pose, coordinate_id=coord_id, timeout_s=self.settings.auto_move_timeout_s)
        await self._wait_if_paused_or_stopped()
        await self.robot.move(retreat_pose, coordinate_id=coord_id, timeout_s=self.settings.auto_move_timeout_s)

    async def _wait_if_paused_or_stopped(self) -> None:
        while True:
            operation = self.state.domain("operation")
            if operation.get("stop_requested") or operation.get("status") == "stopping":
                raise RuntimeError("Operation stopped by operator.")
            if operation.get("status") != "paused":
                return
            await asyncio.sleep(0.05)

    def _ranked_candidates(self, detections: Optional[list[dict[str, Any]]] = None) -> list[Any]:
        raw = detections if detections is not None else self._latest_detections()
        filtered = self._filter_detections(raw)
        if filtered:
            self.state.update_domain("detection", detections=filtered)
        posture = self._current_robot_pose(default=True)
        robot_xy = (float(posture.get("x", 0.0)), float(posture.get("y", 0.0)))
        return list(
            rank_targets(
                filtered,
                robot_xy=robot_xy,
                coord5_fn=self._coord5_for_detection,
                neighbor_radius_mm=float(self.settings.auto_cluster_neighbor_mm),
            )
        )

    def _latest_detections(self) -> list[dict[str, Any]]:
        current = self.state.domain("detection").get("detections", [])
        if isinstance(current, list) and current:
            return [dict(item) for item in current if isinstance(item, dict)]
        return self.detection.run_once(force=False)

    def _filter_detections(self, detections: list[dict[str, Any]]) -> list[dict[str, Any]]:
        labels = set(self.state.domain("auto_pick").get("target_labels") or self.settings.auto_target_labels)
        drop = self.calibration.drop_target
        result: list[dict[str, Any]] = []
        for detection in detections:
            label = str(detection.get("label", ""))
            if labels and label not in labels:
                continue
            xy = self._coord5_for_detection(detection)
            if xy is None:
                continue
            if isinstance(drop, dict) and "x" in drop and "y" in drop:
                dx = xy[0] - float(drop["x"])
                dy = xy[1] - float(drop["y"])
                if (dx * dx + dy * dy) ** 0.5 < float(self.settings.auto_drop_ignore_radius_mm):
                    continue
            result.append(detection)
        return result

    def _coord5_for_detection(self, detection: dict[str, Any]) -> Optional[tuple[float, float]]:
        coord5 = detection.get("coord5_mm")
        if isinstance(coord5, (list, tuple)) and len(coord5) >= 2:
            return float(coord5[0]), float(coord5[1])
        workspace_mm = detection.get("workspace_mm")
        if isinstance(workspace_mm, (list, tuple)) and len(workspace_mm) >= 2:
            xy = (float(workspace_mm[0]), float(workspace_mm[1]))
            detection["coord5_mm"] = xy
            return xy
        center = detection.get("center")
        if not (isinstance(center, (list, tuple)) and len(center) >= 2):
            return None
        ws_def = self.calibration.workspace_definition
        if ws_def is None:
            return None
        pixel_to_mm = None if self.calibration.calibration_bundle is None else float(self.calibration.calibration_bundle.pixel_to_mm)
        xy = workspace_pixel_to_mm(ws_def, (float(center[0]), float(center[1])), pixel_to_mm=pixel_to_mm)
        detection["workspace_mm"] = xy
        detection["coord5_mm"] = xy
        return xy

    def _profile_for_detection(self, detection: dict[str, Any]) -> dict[str, Any]:
        label = str(detection.get("label", ""))
        profiles = self.state.domain("auto_pick").get("label_profiles") or self.settings.auto_label_profiles
        profile = profiles.get(label, {}) if isinstance(profiles, dict) else {}
        return dict(profile) if isinstance(profile, dict) else {}

    def _publish_plan(self, candidate: Any, plan: Any, candidates: list[Any]) -> None:
        selected = {
            "label": candidate.detection.get("label"),
            "confidence": candidate.detection.get("confidence"),
            "coord5_xy": list(candidate.coord5_xy),
            "distance_to_robot_mm": candidate.distance_to_robot_mm,
            "nearest_neighbor_mm": candidate.nearest_neighbor_mm,
            "neighbor_count": candidate.neighbor_count,
        }
        plan_snapshot: dict[str, Any] = {
            "available": True,
            "stage": "planned",
            "mode": getattr(plan, "mode", "rotate_pick"),
            "final_c": getattr(plan, "final_c", 0.0),
            "candidate_count": len(candidates),
            "selected": selected,
        }
        separation = getattr(plan, "separation", None)
        if separation is not None:
            plan_snapshot["separation"] = {
                "blocker_xy": list(separation.blocker_xy),
                "entry_xy": list(separation.entry_xy),
                "push_end_xy": list(separation.push_end_xy),
            }
        self.state.update_domain("auto_pick", plan_snapshot=plan_snapshot, selected_candidate=selected)

    def _publish_plan_stage(self, stage: str, message: str, **updates: Any) -> None:
        current = dict(self.state.domain("auto_pick").get("plan_snapshot") or {})
        if not current.get("available"):
            return
        current.update({"stage": stage, "message": message, **updates})
        self.state.update_domain("auto_pick", plan_snapshot=current, message=message)

    def _current_robot_pose(self, *, default: bool = False) -> dict[str, float]:
        posture = self.state.domain("robot").get("posture")
        if isinstance(posture, dict) and all(key in posture for key in ("x", "y", "z")):
            return {
                "x": float(posture.get("x", 0.0)),
                "y": float(posture.get("y", 0.0)),
                "z": float(posture.get("z", 0.0)),
                "a": float(posture.get("a", 0.0)),
                "b": float(posture.get("b", 0.0)),
                "c": float(posture.get("c", 0.0)),
            }
        if default:
            return {"x": 0.0, "y": 0.0, "z": 0.0, "a": 0.0, "b": 0.0, "c": 0.0}
        raise RuntimeError("Robot posture is unavailable.")

    def _robot_to_workspace_or_xy(self, robot_xyz: tuple[float, float, float]) -> tuple[float, float]:
        ws_def = self.calibration.workspace_definition
        if ws_def is None:
            return robot_xyz[0], robot_xyz[1]
        return robot_to_workspace_mm(ws_def, robot_xyz=np.asarray(robot_xyz, dtype=float))

    def _require_drop_target(self) -> dict[str, Any]:
        drop = self.calibration.drop_target or self.state.domain("auto_pick").get("drop_target")
        if not isinstance(drop, dict):
            raise RuntimeError("Teach a drop target first.")
        self.calibration.drop_target = drop
        return drop

    def _drop_poses(self, drop: dict[str, Any], *, approach_mm: float, drop_mm: float) -> tuple[dict[str, float], dict[str, float], int]:
        if all(key in drop for key in ("robot_x", "robot_y", "robot_z")):
            x_val = float(drop["robot_x"])
            y_val = float(drop["robot_y"])
            z_base = float(drop["robot_z"])
        else:
            x_val = float(drop["x"])
            y_val = float(drop["y"])
            z_base = 0.0
        a_val = float(drop.get("a", 0.0))
        b_val = float(drop.get("b", 0.0))
        c_val = float(drop.get("c", 0.0))
        coord_id = int(drop.get("coord_id", self.settings.robot_coordinate_id))
        approach_pose = {
            "x": x_val,
            "y": y_val,
            "z": z_base + float(approach_mm),
            "a": a_val,
            "b": b_val,
            "c": c_val,
        }
        drop_pose = {
            "x": x_val,
            "y": y_val,
            "z": z_base + float(drop_mm),
            "a": a_val,
            "b": b_val,
            "c": c_val,
        }
        return approach_pose, drop_pose, coord_id

    def record_cycle(self, *, success: bool, started: float) -> None:
        monitoring = self.state.domain("monitoring")
        attempts = int(monitoring.get("pick_attempts", 0)) + 1
        successes = int(monitoring.get("pick_successes", 0)) + (1 if success else 0)
        failures = int(monitoring.get("pick_failures", 0)) + (0 if success else 1)
        duration = max(time.monotonic() - started, 0.0)
        cycle_times = list(monitoring.get("cycle_times") or [])
        cycle_times.append(duration)
        self.state.update_domain(
            "monitoring",
            pick_attempts=attempts,
            pick_successes=successes,
            pick_failures=failures,
            last_cycle_s=duration,
            cycle_times=cycle_times[-50:],
        )
