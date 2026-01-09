"""Camera stream helpers and calibration utilities."""

from __future__ import annotations

import logging
import re
import threading
import time
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
from app.orchestrator import BootstrapDemo, DemoConfig
from vision import aruco_board, calibration, markers, plate, workspace
from vision.stream import DepthAIFrameStream, StreamConfig

from .constants import (
    DEFAULT_BUNDLE_PATH,
    DEFAULT_FPS,
    DEFAULT_RESOLUTION,
    DEFAULT_WORKSPACE_PATH,
)
from .helpers import _safe_float, _safe_int
from .state import SESSION, _commit_settings

LOGGER = logging.getLogger(__name__)

CALIBRATION_PROGRESS = {"value": 0, "current_frame": 0, "max_frames": 0}


class MotionJPEGStream:
    """Manager for Motion JPEG video streaming."""

    def __init__(self, frame_rate: float = 30.0, min_push_interval_s: float = 1 / 120):
        self.frame_rate = frame_rate
        self.frame_queue: list[bytes] = []
        self.lock = threading.Lock()
        self.max_queue_size = 3
        self.frame_count = 0
        self.min_push_interval_s = min_push_interval_s
        self._last_push_ts = 0.0
        self._fps = 0.0
        self._last_fps_ts = 0.0

    def add_frame(self, frame: np.ndarray) -> None:
        now = time.time()
        if now - self._last_push_ts < self.min_push_interval_s:
            return
        success, buffer = cv2.imencode(".jpg", frame)
        if not success:
            return
        with self.lock:
            self.frame_queue.append(buffer.tobytes())
            self.frame_count += 1
            self._last_push_ts = now
            if self._last_fps_ts > 0.0:
                dt = now - self._last_fps_ts
                if dt > 0:
                    inst_fps = 1.0 / dt
                    self._fps = 0.9 * self._fps + 0.1 * inst_fps
            self._last_fps_ts = now
            if len(self.frame_queue) > self.max_queue_size:
                self.frame_queue.pop(0)

    def get_frame(self) -> Optional[bytes]:
        with self.lock:
            if self.frame_queue:
                return self.frame_queue[-1]
        return None

    def get_fps(self) -> float:
        with self.lock:
            return self._fps


MJPEG_STREAM = MotionJPEGStream()
RAW_STREAM = MotionJPEGStream()


def _frame_index(path: Path) -> int:
    match = re.search(r"calibration_color_(\d+)\.png", path.name)
    if match:
        return int(match.group(1))
    return 0


def _list_capture_frames(output_dir: Path) -> list[Path]:
    frames = list(output_dir.glob("calibration_color_*.png"))
    frames.sort(key=_frame_index)
    return frames


def _update_capture_state(output_dir: Path) -> list[Path]:
    SESSION.capture.output_dir = output_dir
    frames = _list_capture_frames(output_dir)
    SESSION.capture.frames = frames
    return frames


def _charuco_detection_counts(image_paths: list[Path], board_path: Path) -> Tuple[List[int], int]:
    spec = aruco_board.load_charuco_board_spec(board_path)
    board = spec.create_board()
    dictionary_id = getattr(cv2.aruco, spec.dictionary)
    detector_params = cv2.aruco.DetectorParameters()
    detector_params.detectInvertedMarker = True
    dictionary = cv2.aruco.getPredefinedDictionary(dictionary_id)
    counts: list[int] = []
    usable_frames = 0
    for path in image_paths:
        image = cv2.imread(str(path))
        if image is None:
            counts.append(0)
            continue
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        detector = cv2.aruco.ArucoDetector(dictionary, detector_params)
        marker_corners, marker_ids, _rejected = detector.detectMarkers(gray)
        corner_count = 0
        if marker_ids is not None and len(marker_ids) > 0:
            charuco_params = cv2.aruco.CharucoParameters()
            charuco_detector = cv2.aruco.CharucoDetector(
                board, charuco_params, detector_params)
            charuco_corners, charuco_ids, _, _ = charuco_detector.detectBoard(gray)
            corner_count = 0 if charuco_ids is None else int(len(charuco_ids))
            if charuco_ids is not None and charuco_corners is not None:
                obj, img = board.matchImagePoints(charuco_corners, charuco_ids)
                if obj is not None and img is not None and len(obj) >= 4:
                    usable_frames += 1
        counts.append(corner_count)
    return counts, usable_frames


def compute_calibration_bundle(board_path: Path, bundle_path: Path, output_dir: Path) -> str:
    if not board_path.exists():
        raise FileNotFoundError(
            f"ChArUco board layout not found: {board_path}")
    frames = _update_capture_state(output_dir)
    if not frames:
        raise RuntimeError(
            "No calibration frames found. Capture frames before calibrating.")
    depth_frame = (SESSION.capture.depth_frame
                   if SESSION.capture.depth_frame is not None
                   else load_depth_snapshot(output_dir))
    if depth_frame is None:
        raise RuntimeError(
            "No depth snapshot found. Capture a depth frame before calibrating.")
    config = DemoConfig(
        calibration_path=bundle_path,
        aruco_board_layout_path=board_path,
    )
    demo = BootstrapDemo(config)
    demo.load_or_create_calibration(
        image_paths=frames, depth_frame=depth_frame, force_rebuild=True)
    SESSION.calibration_bundle = demo.calibration
    _commit_settings(calibration_bundle_path=str(bundle_path))
    return f"Calibration bundle saved to {bundle_path}."


def load_calibration_bundle(bundle_path: Path) -> str:
    if not bundle_path.exists():
        raise FileNotFoundError(f"Calibration bundle not found: {bundle_path}")
    SESSION.calibration_bundle = calibration.load_calibration(bundle_path)
    _commit_settings(calibration_bundle_path=str(bundle_path))
    return f"Loaded calibration bundle from {bundle_path}."


def _default_transform() -> workspace.RigidTransform:
    return workspace.RigidTransform(rotation=np.eye(3), translation=np.zeros(3))


def compute_workspace_from_latest(
    marker_ids: Tuple[int, int], marker_size_mm: float
) -> workspace.WorkspaceDefinition:
    frame, _depth = get_latest_frame()
    if frame is None:
        raise RuntimeError("Camera stream is not providing frames yet.")
    if SESSION.calibration_bundle is None:
        raise RuntimeError("Calibration bundle is not loaded.")
    detections = markers.detect_markers(frame, marker_ids)
    ws = workspace.compute_workspace(
        detections,
        SESSION.calibration_bundle,
        marker_size_mm=marker_size_mm,
        camera_to_robot=_default_transform(),
    )
    SESSION.workspace_definition = ws
    save_workspace_definition(ws, DEFAULT_WORKSPACE_PATH)
    return ws


def save_workspace_definition(
    workspace_def: workspace.WorkspaceDefinition,
    workspace_path: Path = DEFAULT_WORKSPACE_PATH,
) -> str:
    workspace.save_workspace(workspace_def, workspace_path)
    _commit_settings(workspace_path=str(workspace_path))
    return f"Workspace saved to {workspace_path}."


def load_workspace_definition(workspace_path: Path = DEFAULT_WORKSPACE_PATH) -> workspace.WorkspaceDefinition:
    if not workspace_path.exists():
        raise FileNotFoundError(
            f"Workspace definition not found: {workspace_path}")
    SESSION.workspace_definition = workspace.load_workspace(workspace_path)
    _commit_settings(workspace_path=str(workspace_path))
    return SESSION.workspace_definition


def compute_plate_calibration_from_latest(layout_path: Path) -> plate.PlateCalibration:
    frame, _depth = get_latest_frame()
    if frame is None:
        raise RuntimeError("Camera stream is not providing frames yet.")
    if not layout_path.exists():
        raise FileNotFoundError(
            f"Calibration plate layout not found: {layout_path}")
    layout = plate.load_plate_layout(layout_path)
    result = plate.estimate_plate_calibration(frame, layout)
    SESSION.plate_calibration = result
    return result


def attach_plate_calibration(bundle_path: Path, layout_path: Path) -> str:
    if SESSION.calibration_bundle is None:
        raise RuntimeError("Calibration bundle is not loaded.")
    if SESSION.plate_calibration is None:
        raise RuntimeError("Calibration plate has not been detected yet.")
    updated = calibration.CalibrationBundle(
        camera=SESSION.calibration_bundle.camera,
        workspace_plane=SESSION.calibration_bundle.workspace_plane,
        pixel_to_mm=SESSION.calibration_bundle.pixel_to_mm,
        plate_homography=SESSION.plate_calibration.homography,
        plate_layout_path=str(layout_path),
    )
    calibration.save_calibration(updated, bundle_path)
    SESSION.calibration_bundle = updated
    return f"Updated bundle saved to {bundle_path}."


def _clear_calibration_outputs(output_dir: Path) -> None:
    if not output_dir.exists():
        return
    for pattern in ("calibration_color_*.png", "calibration_depth_frame.npz"):
        for path in output_dir.glob(pattern):
            try:
                path.unlink()
            except OSError:
                LOGGER.warning(f"[CALIBRATE] Unable to remove {path}")


def start_camera_stream(
    width: Optional[float],
    height: Optional[float],
    fps: Optional[float],
    lock_af: bool = False,
) -> str:
    stop_camera_stream()
    resolution = (
        _safe_int(width, DEFAULT_RESOLUTION[0]),
        _safe_int(height, DEFAULT_RESOLUTION[1]),
    )
    fps_value = _safe_float(fps, DEFAULT_FPS)
    stream = DepthAIFrameStream(
        StreamConfig(resolution=resolution, fps=fps_value, lock_af=lock_af),
        frame_hook=_publish_raw_frame,
    )
    stream.start()
    SESSION.camera_stream = stream
    device_id = stream.device_id() or "unknown"
    return f"Camera stream running ({resolution[0]}x{resolution[1]} @ {fps_value:.1f} FPS, MXID {device_id})."


def stop_camera_stream() -> None:
    if SESSION.camera_stream is None:
        return
    SESSION.camera_stream.stop()
    SESSION.camera_stream = None


def get_latest_frame() -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    if SESSION.camera_stream is None or not SESSION.camera_stream.is_running():
        return None, None
    return SESSION.camera_stream.get_latest()


def capture_color_frame(output_dir: Path) -> Path:
    max_retries = 5
    frame = None
    for attempt in range(max_retries):
        frame, _depth = get_latest_frame()
        if frame is not None and frame.size > 0:
            break
        if attempt < max_retries - 1:
            time.sleep(0.1)
    if frame is None or frame.size == 0:
        raise RuntimeError(
            f"Camera stream not providing frames after {max_retries} attempts.")
    output_dir.mkdir(parents=True, exist_ok=True)
    frames = _update_capture_state(output_dir)
    index = _frame_index(frames[-1]) + 1 if frames else 1
    path = output_dir / f"calibration_color_{index:02}.png"
    success = cv2.imwrite(str(path), frame)
    if not success:
        raise RuntimeError(f"Failed to write frame to {path}.")
    verify = cv2.imread(str(path))
    if verify is None:
        raise RuntimeError(f"Failed to read back saved frame from {path}.")
    SESSION.capture.frames.append(path)
    return path


def capture_depth_snapshot(output_dir: Path) -> Path:
    _frame, depth = get_latest_frame()
    if depth is None:
        raise RuntimeError("No depth frame available from the camera stream.")
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "calibration_depth_frame.npz"
    np.savez_compressed(path, depth=depth.astype(np.float32))
    SESSION.capture.depth_frame = depth
    return path


def load_depth_snapshot(output_dir: Path) -> Optional[np.ndarray]:
    path = output_dir / "calibration_depth_frame.npz"
    if not path.exists():
        return None
    data = np.load(path, allow_pickle=True)
    depth = data.get("depth")
    if depth is None:
        return None
    return depth.astype(np.float32)


def _publish_raw_frame(frame: np.ndarray) -> None:
    RAW_STREAM.add_frame(frame)
