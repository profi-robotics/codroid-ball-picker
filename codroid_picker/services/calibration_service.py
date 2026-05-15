"""Calibration workflow facade."""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

import cv2
import numpy as np

from codroid_picker.config import AppSettings, DEFAULT_BOARD_PATH
from codroid_picker.core.persistence import read_json, write_json
from codroid_picker.core.transforms import default_transform, rigid_transform
from codroid_picker.state.store import StateStore
from codroid_picker.video.frame_buffer import RawFrameBuffer
from vision import aruco_board, calibration, markers, plate, workspace

if TYPE_CHECKING:
    from codroid_picker.services.robot_service import RobotService


ROBOT_CALIBRATION_POINT_COUNT = 3


class CalibrationService:
    def __init__(self, settings: AppSettings, state: StateStore, frame_buffer: RawFrameBuffer) -> None:
        self.settings = settings
        self.state = state
        self.frame_buffer = frame_buffer
        self.calibration_bundle: Optional[calibration.CalibrationBundle] = None
        self.workspace_definition: Optional[workspace.WorkspaceDefinition] = None
        self.plate_calibration: Optional[plate.PlateCalibration] = None
        self.robot_calibration_points: list[dict[str, Any]] = []
        self.robot_calibration_min_z: Optional[float] = None
        self.robot_controller_frame: dict[str, Any] = {}
        self.drop_target: Optional[dict[str, Any]] = None
        self._load_initial_artifacts()

    def capture_camera(
        self,
        *,
        frame_count: Optional[int] = None,
        delay_s: Optional[float] = None,
        min_usable_frames: Optional[int] = None,
        output_dir: Optional[Path] = None,
        board_path: Path = DEFAULT_BOARD_PATH,
        bundle_path: Optional[Path] = None,
    ) -> str:
        frame_total = int(frame_count or self.settings.camera_calibration_frame_count)
        delay = float(delay_s if delay_s is not None else self.settings.camera_calibration_frame_delay_s)
        min_usable = int(min_usable_frames or self.settings.camera_calibration_min_usable_frames)
        output = output_dir or self.settings.capture_dir
        bundle_output = bundle_path or self.settings.calibration_bundle_path
        if frame_total <= 0:
            raise ValueError("frame_count must be positive.")
        if not board_path.exists():
            raise FileNotFoundError(board_path)
        output.mkdir(parents=True, exist_ok=True)
        self.state.update_domain("calibration", camera_status="capturing", camera_progress=0.0, message="Capturing camera calibration frames.")

        frame_paths: list[Path] = []
        depth_frame: Optional[np.ndarray] = None
        for idx in range(frame_total):
            snapshot = self.frame_buffer.latest(copy=True)
            if snapshot.frame is None:
                raise RuntimeError("Camera stream is not providing frames.")
            path = output / f"calibration_color_{idx + 1:02}.png"
            if not cv2.imwrite(str(path), snapshot.frame):
                raise RuntimeError(f"Failed to write frame to {path}.")
            frame_paths.append(path)
            if snapshot.depth is not None:
                depth_frame = snapshot.depth.astype(np.float32)
            self.state.update_domain(
                "calibration",
                camera_progress=(idx + 1) / frame_total,
                message=f"Captured {idx + 1}/{frame_total} calibration frames.",
            )
            if idx < frame_total - 1 and delay > 0:
                time.sleep(delay)

        if depth_frame is None:
            depth_path = output / "calibration_depth_frame.npz"
            if depth_path.exists():
                data = np.load(depth_path, allow_pickle=True)
                raw_depth = data.get("depth")
                if raw_depth is not None:
                    depth_frame = raw_depth.astype(np.float32)
        if depth_frame is None:
            raise RuntimeError("No depth snapshot is available for workspace plane estimation.")

        spec = aruco_board.load_charuco_board_spec(board_path)
        usable = self._usable_charuco_frame_count(frame_paths, spec)
        if usable < min_usable:
            raise RuntimeError(f"Only {usable} usable calibration frames; need at least {min_usable}.")
        camera_calibration = calibration.calibrate_camera_charuco(frame_paths, spec)
        try:
            pixel_to_mm = calibration.charuco_board_pixel_scale(frame_paths[0], spec)
        except Exception:
            pixel_to_mm = spec.marker_length_mm
        plane = calibration.estimate_workspace_plane(depth_frame, camera_calibration)
        bundle = calibration.CalibrationBundle(camera=camera_calibration, workspace_plane=plane, pixel_to_mm=pixel_to_mm)
        calibration.save_calibration(bundle, bundle_output)
        self.calibration_bundle = bundle
        self.state.update_domain(
            "calibration",
            camera_status="loaded",
            camera_progress=1.0,
            artifacts={**self.state.domain("calibration").get("artifacts", {}), "camera": str(bundle_output)},
            message=f"Calibration bundle saved to {bundle_output}.",
        )
        return f"Calibration bundle saved to {bundle_output}."

    def load_camera(self, path: Optional[Path] = None) -> str:
        path = path or self.settings.calibration_bundle_path
        if not path.exists():
            raise FileNotFoundError(path)
        self.calibration_bundle = calibration.load_calibration(path)
        self.state.update_domain(
            "calibration",
            camera_status="loaded",
            artifacts={**self.state.domain("calibration").get("artifacts", {}), "camera": str(path)},
            message=f"Loaded camera calibration from {path}.",
        )
        return f"Loaded camera calibration from {path}."

    def load_workspace(self, path: Optional[Path] = None) -> str:
        path = path or self.settings.workspace_path
        if not path.exists():
            raise FileNotFoundError(path)
        self.workspace_definition = workspace.load_workspace(path)
        self.state.update_domain(
            "calibration",
            workspace_status="loaded",
            artifacts={**self.state.domain("calibration").get("artifacts", {}), "workspace": str(path)},
            message=f"Loaded workspace from {path}.",
        )
        return f"Loaded workspace from {path}."

    def load_robot(self, path: Optional[Path] = None) -> str:
        path = path or self.settings.robot_calibration_path
        if not path.exists():
            raise FileNotFoundError(path)
        data = read_json(path, default={}) or {}
        points = data.get("points", [])
        if isinstance(points, list):
            self.robot_calibration_points = [dict(point) for point in points if isinstance(point, dict)]
        min_z = data.get("min_z")
        if isinstance(min_z, (int, float)):
            self.robot_calibration_min_z = float(min_z)
        elif self.robot_calibration_points:
            self.robot_calibration_min_z = min(float(point.get("z", 0.0)) for point in self.robot_calibration_points)
        controller_frame = data.get("controller_frame")
        self.robot_controller_frame = dict(controller_frame) if isinstance(controller_frame, dict) else {}
        if self.robot_calibration_points:
            self._apply_robot_points_to_workspace()
        expected = self._robot_expected_point_count()
        self.state.update_domain(
            "calibration",
            robot_status="loaded",
            step=min(len(self.robot_calibration_points), expected),
            robot_points=self.robot_calibration_points,
            robot_expected_points=expected,
            robot_verification_ready=len(self.robot_calibration_points) >= ROBOT_CALIBRATION_POINT_COUNT,
            robot_controller_applied=bool(self.robot_controller_frame),
            robot_controller_frame=self.robot_controller_frame,
            accuracy=data.get("accuracy") if isinstance(data.get("accuracy"), dict) else self.state.domain("calibration").get("accuracy"),
            artifacts={**self.state.domain("calibration").get("artifacts", {}), "robot": str(path)},
            message=f"Loaded {len(self.robot_calibration_points)} robot calibration points from {path}.",
        )
        return f"Loaded robot calibration from {path}."

    async def start_robot_calibration(self, robot: "RobotService") -> str:
        robot_state = self.state.domain("robot")
        if robot_state.get("status") != "online":
            raise RuntimeError("Robot must be connected before robot calibration.")
        expected = self._robot_expected_point_count()
        if expected < ROBOT_CALIBRATION_POINT_COUNT:
            raise RuntimeError("Robot calibration plate layout must define at least 3 calibration points.")
        await robot.set_coordinate_id(0, await_ack=False, timeout_s=self.settings.robot_coordinate_timeout_s)
        self.robot_calibration_points = []
        self.robot_calibration_min_z = None
        self.robot_controller_frame = {}
        self.state.update_domain(
            "calibration",
            robot_status="teaching",
            step=1,
            robot_points=[],
            robot_expected_points=expected,
            robot_verification_ready=False,
            robot_controller_applied=False,
            robot_controller_frame={},
            accuracy=None,
            message="Robot set to CoordinateId 0. Move the TCP to calibration point 1, then capture point 1.",
        )
        return "Robot calibration teaching started."

    def capture_robot_point(self, index: Optional[int] = None) -> str:
        calibration_state = self.state.domain("calibration")
        if calibration_state.get("robot_status") not in {"teaching", "ready_to_apply"}:
            raise RuntimeError("Start robot calibration before capturing points.")
        expected = int(calibration_state.get("robot_expected_points") or ROBOT_CALIBRATION_POINT_COUNT)
        point_index = int(index or (len(self.robot_calibration_points) + 1))
        if point_index < 1 or point_index > expected:
            raise ValueError(f"Point index must be between 1 and {expected}.")
        if point_index > len(self.robot_calibration_points) + 1:
            raise RuntimeError("Capture points sequentially, or replace an already captured point.")

        point = self._current_robot_calibration_point(point_index)
        points = list(self.robot_calibration_points)
        if point_index <= len(points):
            points[point_index - 1] = point
        else:
            points.append(point)
        self.robot_calibration_points = points

        ready = len(points) >= ROBOT_CALIBRATION_POINT_COUNT
        next_step = min(len(points) + 1, expected)
        if ready:
            status = "ready_to_apply"
            message = "Captured 3 robot calibration points. Apply calibration to save the transform."
        else:
            status = "teaching"
            message = f"Captured point {point_index}. Move the robot TCP to calibration point {next_step}."
        self.state.update_domain(
            "calibration",
            robot_status=status,
            step=next_step,
            robot_points=points,
            robot_expected_points=expected,
            robot_verification_ready=False,
            message=message,
        )
        return f"Captured robot calibration point {point_index}."

    async def finish_robot_calibration(self, robot: "RobotService") -> str:
        if len(self.robot_calibration_points) < ROBOT_CALIBRATION_POINT_COUNT:
            raise RuntimeError("Capture 3 robot calibration points before applying calibration.")
        accuracy = self._robot_calibration_accuracy()
        self.robot_calibration_min_z = min(float(point["z"]) for point in self.robot_calibration_points[:ROBOT_CALIBRATION_POINT_COUNT])
        self.state.update_domain(
            "calibration",
            robot_status="applying_controller",
            message=f"Requesting controller coordinate calibration for CoordinateId {self.settings.robot_coordinate_id}.",
        )
        controller_points = self._controller_calibration_points()
        try:
            controller_frame = await robot.calibrate_coordinate(
                controller_points,
                coordinate_id=int(self.settings.robot_coordinate_id),
                timeout_s=float(self.settings.robot_coordinate_timeout_s),
            )
        except Exception as exc:
            self.state.update_domain(
                "calibration",
                robot_status="controller_failed",
                robot_controller_applied=False,
                message=f"Controller coordinate calibration failed: {exc}",
            )
            raise
        self.robot_controller_frame = self._normalize_controller_frame(controller_frame)
        payload = {
            "points": self.robot_calibration_points,
            "min_z": self.robot_calibration_min_z,
            "accuracy": accuracy,
            "controller_coordinate_id": int(self.settings.robot_coordinate_id),
            "controller_frame": self.robot_controller_frame,
            "plate_layout": str(self.settings.plate_layout_path),
            "saved_at": time.time(),
        }
        write_json(self.settings.robot_calibration_path, payload)
        self._apply_robot_points_to_workspace()
        self.state.update_domain(
            "calibration",
            robot_status="accuracy_ready",
            step=ROBOT_CALIBRATION_POINT_COUNT + 1,
            robot_points=self.robot_calibration_points,
            robot_expected_points=self._robot_expected_point_count(),
            robot_verification_ready=True,
            robot_controller_applied=True,
            robot_controller_frame=self.robot_controller_frame,
            accuracy=accuracy,
            artifacts={**self.state.domain("calibration").get("artifacts", {}), "robot": str(self.settings.robot_calibration_path)},
            message=(
                f"Robot calibration saved to {self.settings.robot_calibration_path}; "
                f"controller CoordinateId {self.settings.robot_coordinate_id} updated."
            ),
        )
        return f"Robot calibration saved and controller CoordinateId {self.settings.robot_coordinate_id} updated."

    def cancel_robot_calibration(self) -> str:
        self.robot_calibration_points = []
        self.robot_calibration_min_z = None
        self.robot_controller_frame = {}
        self.state.update_domain(
            "calibration",
            robot_status="idle",
            step=0,
            robot_points=[],
            robot_verification_ready=False,
            robot_controller_applied=False,
            robot_controller_frame={},
            message="Robot calibration canceled.",
        )
        return "Robot calibration canceled."

    async def verify_robot_calibration(self, robot: "RobotService") -> str:
        if len(self.robot_calibration_points) < ROBOT_CALIBRATION_POINT_COUNT:
            raise RuntimeError("Load or apply robot calibration before verification.")
        operation_id = self.state.begin_operation(
            source="calibration",
            label="Robot verification",
            message="Starting robot calibration verification.",
        )
        try:
            self.state.update_domain(
                "calibration",
                robot_status="verifying",
                robot_verification_ready=True,
                message="Robot calibration verification is running.",
            )
            if self._use_controller_coordinate_for_verification():
                await robot.set_coordinate_id(
                    int(self.settings.robot_coordinate_id),
                    await_ack=False,
                    timeout_s=self.settings.robot_coordinate_timeout_s,
                )
            for label, pose, coordinate_id in self._verification_moves():
                await self._wait_if_paused_or_stopped()
                self.state.update_domain(
                    "operation",
                    message=f"Verification move: {label}.",
                    updated_at=time.time(),
                )
                await robot.move(
                    pose,
                    coordinate_id=coordinate_id,
                    timeout_s=self.settings.verification_move_timeout_s,
                )
                if self.settings.verification_settle_s > 0:
                    await asyncio.sleep(self.settings.verification_settle_s)
            self.state.update_domain(
                "calibration",
                robot_status="complete",
                robot_verification_ready=True,
                message="Robot calibration verification complete.",
            )
            self.state.finish_operation(operation_id, status="idle", message="Robot calibration verification complete.")
            return "Robot calibration verification complete."
        except Exception as exc:
            self.state.update_domain(
                "calibration",
                robot_status="verification_failed",
                message=f"Robot calibration verification failed: {exc}",
            )
            self.state.finish_operation(operation_id, status="error", message=f"Robot calibration verification failed: {exc}")
            raise

    def clear_workspace(self) -> str:
        self.workspace_definition = None
        self.state.update_domain("calibration", workspace_status="missing", message="Workspace cleared.")
        return "Workspace cleared."

    def compute_workspace(self, marker_ids: tuple[int, int], marker_size_mm: float) -> str:
        if self.calibration_bundle is None:
            raise RuntimeError("Camera calibration bundle is not loaded.")
        snapshot = self.frame_buffer.latest(copy=True)
        if snapshot.frame is None:
            raise RuntimeError("Camera stream is not providing frames.")
        marker_detections = markers.detect_markers(snapshot.frame, marker_ids)
        ws_def = workspace.compute_workspace(
            marker_detections,
            self.calibration_bundle,
            marker_size_mm=marker_size_mm,
            camera_to_robot=default_transform(),
        )
        if self.robot_calibration_points:
            ws_def = self._workspace_with_robot_transform(ws_def)
        workspace.save_workspace(ws_def, self.settings.workspace_path)
        self.workspace_definition = ws_def
        self.state.update_domain(
            "calibration",
            workspace_status="loaded",
            artifacts={**self.state.domain("calibration").get("artifacts", {}), "workspace": str(self.settings.workspace_path)},
            message=f"Workspace computed and saved ({ws_def.width_mm:.1f} x {ws_def.height_mm:.1f} mm).",
        )
        return f"Workspace computed and saved ({ws_def.width_mm:.1f} x {ws_def.height_mm:.1f} mm)."

    def detect_plate(self) -> dict[str, Any]:
        snapshot = self.frame_buffer.latest(copy=True)
        if snapshot.frame is None:
            raise RuntimeError("Camera stream is not providing frames.")
        if not self.settings.plate_layout_path.exists():
            raise FileNotFoundError(self.settings.plate_layout_path)
        layout = plate.load_plate_layout(self.settings.plate_layout_path)
        result = plate.estimate_plate_calibration(snapshot.frame, layout)
        self.plate_calibration = result
        payload = {
            "ok": True,
            "stable": True,
            "count": len(result.detections),
            "reprojection_error_mm": result.reprojection_error_mm,
        }
        self.state.update_domain("calibration", plate=payload, message=f"Detected {len(result.detections)} plate markers.")
        return payload

    def reset(self, scope: str = "all") -> str:
        update: dict[str, Any] = {"message": f"Calibration reset: {scope}."}
        if scope in {"all", "camera"}:
            update["camera_status"] = "idle"
            self.calibration_bundle = None
        if scope in {"all", "workspace"}:
            update["workspace_status"] = "missing"
            self.workspace_definition = None
        if scope in {"all", "robot"}:
            update["robot_status"] = "idle"
            update["robot_points"] = []
            update["robot_verification_ready"] = False
            update["robot_controller_applied"] = False
            update["robot_controller_frame"] = {}
            update["step"] = 0
            self.robot_calibration_points = []
            self.robot_calibration_min_z = None
            self.robot_controller_frame = {}
        self.state.update_domain("calibration", **update)
        return str(update["message"])

    def _load_initial_artifacts(self) -> None:
        for loader in (self.load_camera, self.load_workspace, self.load_robot):
            try:
                loader()
            except FileNotFoundError:
                continue
            except Exception:
                continue
        self._load_drop_target()

    def _load_drop_target(self) -> None:
        data = read_json(self.settings.drop_target_path, default=None)
        if isinstance(data, dict):
            self.drop_target = data
            self.state.update_domain("auto_pick", drop_target=data)
        elif self.settings.drop_target:
            self.drop_target = {"x": float(self.settings.drop_target[0]), "y": float(self.settings.drop_target[1])}
            self.state.update_domain("auto_pick", drop_target=self.drop_target)

    @staticmethod
    def _usable_charuco_frame_count(frame_paths: list[Path], spec: aruco_board.CharucoBoardSpec) -> int:
        board = spec.create_board()
        dictionary_id = getattr(cv2.aruco, spec.dictionary)
        dictionary = cv2.aruco.getPredefinedDictionary(dictionary_id)
        detector_params = cv2.aruco.DetectorParameters()
        detector_params.detectInvertedMarker = True
        usable = 0
        for frame_path in frame_paths:
            image = cv2.imread(str(frame_path))
            if image is None:
                continue
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            detector = cv2.aruco.ArucoDetector(dictionary, detector_params)
            _marker_corners, marker_ids, _rejected = detector.detectMarkers(gray)
            if marker_ids is None or len(marker_ids) == 0:
                continue
            charuco_detector = cv2.aruco.CharucoDetector(board, cv2.aruco.CharucoParameters(), detector_params)
            charuco_corners, charuco_ids, _corners, _ids = charuco_detector.detectBoard(gray)
            if charuco_ids is not None and charuco_corners is not None:
                obj_points, image_points = board.matchImagePoints(charuco_corners, charuco_ids)
                if obj_points is not None and image_points is not None and len(obj_points) >= 4:
                    usable += 1
        return usable

    def _apply_robot_points_to_workspace(self) -> None:
        if self.workspace_definition is None:
            return
        updated = self._workspace_with_robot_transform(self.workspace_definition)
        self.workspace_definition = updated
        workspace.save_workspace(updated, self.settings.workspace_path)

    def _workspace_with_robot_transform(self, ws_def: workspace.WorkspaceDefinition) -> workspace.WorkspaceDefinition:
        layout = plate.load_plate_layout(self.settings.plate_layout_path)
        plate_points = layout.robot_calibration_points()[:3]
        if len(self.robot_calibration_points) < 3 or len(plate_points) < 3:
            return ws_def
        plate_pts = np.asarray(
            [[float(point.center_mm[0]), float(point.center_mm[1]), 0.0] for point in plate_points],
            dtype=np.float64,
        )
        robot_pts = np.asarray(
            [
                [float(point["x"]), float(point["y"]), float(point["z"])]
                for point in self.robot_calibration_points[:3]
            ],
            dtype=np.float64,
        )
        transform = rigid_transform(plate_pts, robot_pts)
        return workspace.WorkspaceDefinition(
            width_mm=ws_def.width_mm,
            height_mm=ws_def.height_mm,
            origin_camera=ws_def.origin_camera,
            x_axis_camera=ws_def.x_axis_camera,
            y_axis_camera=ws_def.y_axis_camera,
            camera_to_robot=transform,
        )

    def _robot_expected_point_count(self) -> int:
        try:
            return len(plate.load_plate_layout(self.settings.plate_layout_path).robot_calibration_points())
        except Exception:
            return ROBOT_CALIBRATION_POINT_COUNT

    def _current_robot_calibration_point(self, sequence: int) -> dict[str, Any]:
        posture = self.state.domain("robot").get("posture")
        if not isinstance(posture, dict):
            raise RuntimeError("Robot posture is not available.")
        point: dict[str, Any] = {"sequence": int(sequence), "coord_id": 0, "captured_at": time.time()}
        for key in ("x", "y", "z", "a", "b", "c"):
            value = posture.get(key)
            if not isinstance(value, (int, float)):
                raise RuntimeError(f"Robot posture is missing {key}.")
            point[key] = float(value)
        return point

    def _robot_calibration_accuracy(self) -> dict[str, float]:
        layout = plate.load_plate_layout(self.settings.plate_layout_path)
        plate_points = layout.robot_calibration_points()[:ROBOT_CALIBRATION_POINT_COUNT]
        if len(plate_points) < ROBOT_CALIBRATION_POINT_COUNT:
            raise RuntimeError("Robot calibration plate layout must define at least 3 calibration points.")
        plate_pts = np.asarray(
            [[float(point.center_mm[0]), float(point.center_mm[1]), 0.0] for point in plate_points],
            dtype=np.float64,
        )
        robot_pts = np.asarray(
            [[float(point["x"]), float(point["y"]), float(point["z"])] for point in self.robot_calibration_points[:ROBOT_CALIBRATION_POINT_COUNT]],
            dtype=np.float64,
        )
        transform = rigid_transform(plate_pts, robot_pts)
        projected = (transform.rotation @ plate_pts.T).T + transform.translation
        errors = robot_pts - projected
        xy_errors = np.linalg.norm(errors[:, :2], axis=1)
        z_errors = np.abs(errors[:, 2])
        return {
            "mean_xy_mm": float(np.mean(xy_errors)),
            "max_xy_mm": float(np.max(xy_errors)),
            "mean_z_mm": float(np.mean(z_errors)),
            "max_z_mm": float(np.max(z_errors)),
            "point_count": float(len(robot_pts)),
        }

    def _controller_calibration_points(self) -> list[dict[str, float]]:
        points: list[dict[str, float]] = []
        for point in self.robot_calibration_points[:ROBOT_CALIBRATION_POINT_COUNT]:
            points.append(
                {
                    "x": float(point["x"]),
                    "y": float(point["y"]),
                    "z": float(point["z"]),
                    "a": float(point.get("a", 0.0)),
                    "b": float(point.get("b", 0.0)),
                    "c": float(point.get("c", 0.0)),
                }
            )
        return points

    @staticmethod
    def _normalize_controller_frame(frame: dict[str, Any]) -> dict[str, Any]:
        normalized = dict(frame)

        def _axis(*keys: str) -> float:
            for key in keys:
                value = normalized.get(key)
                if isinstance(value, (int, float)):
                    return float(value)
                if isinstance(value, str):
                    try:
                        return float(value)
                    except ValueError:
                        continue
            return 0.0

        normalized["x"] = _axis("x")
        normalized["y"] = _axis("y")
        normalized["z"] = _axis("z")
        normalized["a"] = _axis("a", "rx", "roll")
        normalized["b"] = _axis("b", "ry", "pitch")
        normalized["c"] = _axis("c", "rz", "yaw")
        return normalized

    def _use_controller_coordinate_for_verification(self) -> bool:
        if not self.robot_controller_frame:
            return False
        try:
            layout = plate.load_plate_layout(self.settings.plate_layout_path)
            return len(layout.robot_calibration_points()) >= ROBOT_CALIBRATION_POINT_COUNT
        except Exception:
            return False

    def _verification_moves(self) -> list[tuple[str, dict[str, float], int]]:
        points = self.robot_calibration_points[:ROBOT_CALIBRATION_POINT_COUNT]
        if len(points) < ROBOT_CALIBRATION_POINT_COUNT:
            raise RuntimeError("Robot verification requires 3 calibration points.")
        raise_mm = float(self.settings.verification_raise_mm)
        if self._use_controller_coordinate_for_verification():
            layout = plate.load_plate_layout(self.settings.plate_layout_path)
            plate_points = layout.robot_calibration_points()[:ROBOT_CALIBRATION_POINT_COUNT]
            origin = np.asarray(plate_points[0].center_mm, dtype=np.float64)
            coord_positions = [
                np.asarray(point.center_mm, dtype=np.float64) - origin
                for point in plate_points
            ]

            def coord_pose(point: dict[str, Any], xy_mm: np.ndarray, *, raised: bool) -> dict[str, float]:
                z = float(point["z"])
                return {
                    "x": float(xy_mm[0]),
                    "y": float(xy_mm[1]),
                    "z": z - raise_mm if raised else z,
                    "a": float(point.get("a", 0.0)),
                    "b": float(point.get("b", 0.0)),
                    "c": float(point.get("c", 0.0)),
                }

            coordinate_id = int(self.settings.robot_coordinate_id)
            point_1, point_2, point_3 = points[0], points[1], points[2]
            c1, c2, c3 = coord_positions[0], coord_positions[1], coord_positions[2]
            return [
                ("point 3 safe height", coord_pose(point_3, c3, raised=True), coordinate_id),
                ("point 1 approach", coord_pose(point_1, c1, raised=True), coordinate_id),
                ("point 1 taught position", coord_pose(point_1, c1, raised=False), coordinate_id),
                ("point 1 retreat", coord_pose(point_1, c1, raised=True), coordinate_id),
                ("point 2 approach", coord_pose(point_2, c2, raised=True), coordinate_id),
                ("point 2 taught position", coord_pose(point_2, c2, raised=False), coordinate_id),
                ("point 2 retreat", coord_pose(point_2, c2, raised=True), coordinate_id),
                ("point 3 approach", coord_pose(point_3, c3, raised=True), coordinate_id),
                ("point 3 taught position", coord_pose(point_3, c3, raised=False), coordinate_id),
                ("point 3 retreat", coord_pose(point_3, c3, raised=True), coordinate_id),
                ("point 1 home", coord_pose(point_1, c1, raised=False), coordinate_id),
            ]

        def pose(point: dict[str, Any], *, raised: bool) -> dict[str, float]:
            return {
                "x": float(point["x"]),
                "y": float(point["y"]),
                "z": float(point["z"]) + (raise_mm if raised else 0.0),
                "a": float(point.get("a", 0.0)),
                "b": float(point.get("b", 0.0)),
                "c": float(point.get("c", 0.0)),
            }

        coordinate_id = int(points[0].get("coord_id", 0))
        point_1, point_2, point_3 = points[0], points[1], points[2]
        return [
            ("point 3 safe height", pose(point_3, raised=True), coordinate_id),
            ("point 1 approach", pose(point_1, raised=True), coordinate_id),
            ("point 1 taught position", pose(point_1, raised=False), coordinate_id),
            ("point 1 retreat", pose(point_1, raised=True), coordinate_id),
            ("point 2 approach", pose(point_2, raised=True), coordinate_id),
            ("point 2 taught position", pose(point_2, raised=False), coordinate_id),
            ("point 2 retreat", pose(point_2, raised=True), coordinate_id),
            ("point 3 approach", pose(point_3, raised=True), coordinate_id),
            ("point 3 taught position", pose(point_3, raised=False), coordinate_id),
            ("point 3 retreat", pose(point_3, raised=True), coordinate_id),
            ("point 1 home", pose(point_1, raised=False), coordinate_id),
        ]

    async def _wait_if_paused_or_stopped(self) -> None:
        while True:
            operation = self.state.domain("operation")
            if operation.get("stop_requested") or operation.get("status") == "stopping":
                raise RuntimeError("Operation stopped by operator.")
            if operation.get("status") != "paused":
                return
            await asyncio.sleep(0.05)
