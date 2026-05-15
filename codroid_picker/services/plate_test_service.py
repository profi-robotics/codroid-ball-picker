"""Calibration plate pick-and-place test service."""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from codroid_picker.config import AppSettings
from codroid_picker.core.persistence import write_json
from codroid_picker.core.transforms import rigid_transform
from codroid_picker.services.calibration_service import CalibrationService
from codroid_picker.services.gripper_service import GripperService
from codroid_picker.services.pick_place_service import PickPlaceService
from codroid_picker.services.robot_service import RobotService
from codroid_picker.state.store import StateStore
from vision import plate


class PlateTestService:
    def __init__(
        self,
        settings: AppSettings,
        state: StateStore,
        calibration: CalibrationService,
        robot: RobotService,
        gripper: GripperService,
        pick_place: PickPlaceService,
    ) -> None:
        self.settings = settings
        self.state = state
        self.calibration = calibration
        self.robot = robot
        self.gripper = gripper
        self.pick_place = pick_place

    async def run(
        self,
        *,
        approach_mm: float,
        pick_mm: float,
        drop_mm: float,
        grip_timeout_s: float,
        use_base_coordinates: bool | None = None,
        dry_run: bool = False,
    ) -> dict[str, Any]:
        if not dry_run and not self.calibration.drop_target and not self.state.domain("auto_pick").get("drop_target"):
            raise RuntimeError("Teach a drop target before running the plate test.")
        if len(self.calibration.robot_calibration_points) < 3:
            raise RuntimeError("Load or apply robot calibration before running the plate test.")
        layout = plate.load_plate_layout(self.settings.plate_layout_path)
        targets = layout.robot_calibration_points()
        if not targets:
            raise RuntimeError("Plate layout does not define robot calibration targets.")

        operation_id = self.state.begin_operation(source="plate_test", label="Plate Pick & Place Test", message="Plate test starting.")
        started = time.monotonic()
        use_base = self.settings.plate_test_use_base_coordinates if use_base_coordinates is None else bool(use_base_coordinates)
        coord_id = 0 if use_base else int(self.settings.robot_coordinate_id)
        results: list[dict[str, Any]] = []
        try:
            transform = self._plate_to_robot_transform()
            baseline_centers: dict[int, np.ndarray] | None = None
            for target in targets:
                if await self._wait_if_paused_or_stopped():
                    results.append({"sequence": int(target.sequence), "status": "skipped", "reason": "stop_requested"})
                    break
                plate_check = self._check_plate_alignment(baseline_centers)
                if baseline_centers is None and plate_check.get("centers"):
                    baseline_centers = plate_check["centers"]
                if plate_check.get("moved_too_far"):
                    raise RuntimeError(str(plate_check.get("message") or "Calibration plate moved beyond the allowed threshold."))
                x_val, y_val, z_val, coord_id = self._target_robot_pose(target, transform, use_base=use_base)
                pose_common = {
                    "x": x_val,
                    "y": y_val,
                    "a": float(self.calibration.robot_calibration_points[0].get("a", 0.0)),
                    "b": float(self.calibration.robot_calibration_points[0].get("b", 0.0)),
                    "c": float(self.calibration.robot_calibration_points[0].get("c", 0.0)),
                }
                approach_pose = {**pose_common, "z": z_val + float(approach_mm)}
                pick_pose = {**pose_common, "z": z_val + float(pick_mm)}
                self.state.update_domain("operation", message=f"Plate test target {target.sequence}: approach.")
                target_result = {
                    "sequence": int(target.sequence),
                    "status": "dry_run" if dry_run else "ok",
                    "coordinate_id": coord_id,
                    "approach_pose": approach_pose,
                    "pick_pose": pick_pose,
                    "plate_check": {key: value for key, value in plate_check.items() if key != "centers"},
                }
                if not dry_run:
                    if await self._wait_if_paused_or_stopped():
                        target_result.update({"status": "skipped", "reason": "stop_requested"})
                        results.append(target_result)
                        break
                    await self.robot.move(approach_pose, coordinate_id=coord_id, timeout_s=self.settings.plate_test_move_timeout_s)
                    if await self._wait_if_paused_or_stopped():
                        target_result.update({"status": "skipped", "reason": "stop_requested"})
                        results.append(target_result)
                        break
                    await self.robot.move(pick_pose, coordinate_id=coord_id, timeout_s=self.settings.plate_test_move_timeout_s)
                    if await self._wait_if_paused_or_stopped():
                        target_result.update({"status": "skipped", "reason": "stop_requested"})
                        results.append(target_result)
                        break
                    await self.gripper.close(force=40, speed=80)
                    if grip_timeout_s > 0:
                        await self._sleep(min(float(grip_timeout_s), 10.0))
                    holding = self.state.domain("gripper").get("holding")
                    target_result["grip_result"] = {"holding": holding}
                    if holding is False:
                        target_result["status"] = "failed"
                        target_result["error"] = "Grip verification failed."
                        results.append(target_result)
                        continue
                    if await self._wait_if_paused_or_stopped():
                        target_result.update({"status": "skipped", "reason": "stop_requested"})
                        results.append(target_result)
                        break
                    await self.robot.move(approach_pose, coordinate_id=coord_id, timeout_s=self.settings.plate_test_move_timeout_s)
                    if await self._wait_if_paused_or_stopped():
                        target_result.update({"status": "skipped", "reason": "stop_requested"})
                        results.append(target_result)
                        break
                    await self.pick_place.drop(drop_height_mm=drop_mm, operation_id=operation_id)
                    target_result["grip_checked"] = True
                results.append(target_result)
            success_count = len([item for item in results if item["status"] in {"ok", "dry_run"}])
            skipped_count = len([item for item in results if item["status"] == "skipped"])
            summary = {
                "status": "dry_run" if dry_run else ("stopped" if skipped_count else "completed"),
                "target_count": len(targets),
                "successes": success_count,
                "failures": len([item for item in results if item["status"] == "failed"]),
                "skipped": skipped_count + max(len(targets) - len(results), 0),
                "dry_run": dry_run,
                "coordinate_mode": "base" if use_base else "coord5",
                "duration_s": max(time.monotonic() - started, 0.0),
                "results": results,
                "saved_at": time.time(),
            }
            self._persist(summary)
            self.state.finish_operation(operation_id, status="idle", message="Plate test complete.")
            self.state.update_domain("monitoring", plate_test=summary)
            return summary
        except Exception as exc:
            summary = {
                "status": "error",
                "message": str(exc),
                "duration_s": max(time.monotonic() - started, 0.0),
                "results": results,
                "saved_at": time.time(),
            }
            self._persist(summary)
            self.state.finish_operation(operation_id, status="error", message=f"Plate test failed: {exc}")
            self.state.update_domain("monitoring", plate_test=summary)
            raise

    def _plate_to_robot_transform(self) -> Any:
        layout = plate.load_plate_layout(self.settings.plate_layout_path)
        plate_points = layout.robot_calibration_points()[:3]
        robot_points = self.calibration.robot_calibration_points[:3]
        plate_xyz = np.asarray([[point.center_mm[0], point.center_mm[1], 0.0] for point in plate_points], dtype=float)
        robot_xyz = np.asarray([[point["x"], point["y"], point["z"]] for point in robot_points], dtype=float)
        return rigid_transform(plate_xyz, robot_xyz)

    def _target_robot_pose(self, target: Any, transform: Any, *, use_base: bool) -> tuple[float, float, float, int]:
        if use_base:
            xyz = (transform.rotation @ np.asarray([target.center_mm[0], target.center_mm[1], 0.0], dtype=float)) + transform.translation
            return float(xyz[0]), float(xyz[1]), float(xyz[2]), 0
        layout = plate.load_plate_layout(self.settings.plate_layout_path)
        origin = layout.robot_calibration_points()[0].center_mm
        return (
            float(target.center_mm[0] - origin[0]),
            float(target.center_mm[1] - origin[1]),
            float(self.calibration.robot_calibration_points[0].get("z", 0.0)),
            int(self.settings.robot_coordinate_id),
        )

    def _check_plate_alignment(self, baseline: dict[int, np.ndarray] | None) -> dict[str, Any]:
        try:
            result = self.calibration.detect_plate()
        except Exception as exc:  # noqa: BLE001
            return {"status": "skipped", "message": str(exc)}
        plate_calibration = self.calibration.plate_calibration
        if plate_calibration is None:
            return {"status": "skipped", "message": "No plate calibration data."}
        centers = {
            marker_id: np.asarray(detection.center, dtype=float)
            for marker_id, detection in plate_calibration.detections.items()
        }
        if baseline is None:
            return {"status": "baseline", "centers": centers, "markers": len(centers)}
        common = [marker_id for marker_id in centers if marker_id in baseline]
        max_motion = 0.0
        for marker_id in common:
            max_motion = max(max_motion, float(np.linalg.norm(centers[marker_id] - baseline[marker_id])))
        moved = max_motion > float(self.settings.plate_movement_threshold_px)
        return {
            "status": "checked",
            "centers": centers,
            "markers": len(centers),
            "max_motion_px": max_motion,
            "moved_too_far": moved,
            "message": f"Plate moved {max_motion:.1f} px.",
        }

    def _persist(self, summary: dict[str, Any]) -> None:
        self.settings.plate_test_capture_dir.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        write_json(self.settings.plate_test_capture_dir / f"plate_test_{stamp}.json", summary)

    @staticmethod
    async def _sleep(delay_s: float) -> None:
        import asyncio

        await asyncio.sleep(delay_s)

    async def _wait_if_paused_or_stopped(self) -> bool:
        while True:
            operation = self.state.domain("operation")
            if operation.get("stop_requested") or operation.get("status") == "stopping":
                return True
            if operation.get("status") != "paused":
                return False
            await self._sleep(0.05)
