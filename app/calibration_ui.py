"""
Dash calibration UI for the Codroid bring-up workflow.
"""

from __future__ import annotations

import asyncio
import base64
import io
import logging
import os
import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Awaitable, Callable, Optional, Tuple
from urllib.parse import urlparse

import cv2
import numpy as np
from dash import Dash, Input, Output, State, callback_context, dcc, html, no_update
import dash_mantine_components as dmc

from app.orchestrator import BootstrapDemo, DemoConfig
from codroid_api import CodroidAPI, CodroidSettings
from vision import aruco_board, calibration, markers, plate, workspace
from vision.camera_check import list_oak_devices
from vision.stream import DepthAIFrameStream, StreamConfig

LOGGER = logging.getLogger(__name__)

DEFAULT_BOARD_PATH = Path(
    "data/markers/camera_calibration/camera_calibration_board.json")
DEFAULT_PLATE_PATH = Path(
    "data/markers/robot_calibration/calibration_plate_layout.json")
DEFAULT_BUNDLE_PATH = Path("data/calibration_bundle.npz")
DEFAULT_CAPTURE_DIR = Path("data/calibration")
DEFAULT_RESOLUTION = (1280, 720)
DEFAULT_FPS = 30.0
DEFAULT_MODEL_DIR = Path("data/models")
DEFAULT_DROP_MARKER_ID = 30


def _load_codroid_defaults() -> dict:
    try:
        settings = CodroidSettings()
        host = settings.host
        robot_port = settings.robot_port
        ws_port = settings.ws_port
    except Exception:
        host = os.getenv("CODROID_HOST", "192.168.101.100")
        robot_port = int(os.getenv("CODROID_ROBOT_PORT", "9000"))
        ws_port = int(os.getenv("CODROID_WS_PORT", "9098"))
    robot_ws_url = f"ws://{host}:{robot_port}/"
    user_ws_url = f"ws://{host}:{ws_port}/"
    return {
        "host": host,
        "robot_port": robot_port,
        "ws_port": ws_port,
        "robot_ws_url": robot_ws_url,
        "user_ws_url": user_ws_url,
    }


CODROID_DEFAULTS = _load_codroid_defaults()
DEFAULT_ROBOT_URI = CODROID_DEFAULTS["robot_ws_url"]


@dataclass
class CaptureState:
    output_dir: Path = DEFAULT_CAPTURE_DIR
    frames: list[Path] = field(default_factory=list)
    depth_frame: Optional[np.ndarray] = None


@dataclass
class UISession:
    config: DemoConfig = field(default_factory=DemoConfig)
    camera_stream: Optional[DepthAIFrameStream] = None
    capture: CaptureState = field(default_factory=CaptureState)
    calibration_bundle: Optional[calibration.CalibrationBundle] = None
    workspace_definition: Optional[workspace.WorkspaceDefinition] = None
    plate_calibration: Optional[plate.PlateCalibration] = None
    active_model_path: Optional[Path] = None
    yolo_model: Optional[object] = None
    model_loaded: bool = False
    detections: list[dict] = field(default_factory=list)
    last_detection_error: Optional[str] = None
    drop_marker_id: int = DEFAULT_DROP_MARKER_ID
    drop_target_mm: Optional[Tuple[float, float]] = None
    last_pick_mm: Optional[Tuple[float, float]] = None
    last_action: Optional[str] = None


SESSION = UISession(
    config=DemoConfig(
        robot_host=CODROID_DEFAULTS["host"],
        robot_port=CODROID_DEFAULTS["robot_port"],
        user_port=CODROID_DEFAULTS["ws_port"],
    )
)


class MotionJPEGStream:
    """Manager for Motion JPEG video streaming for fullscreen video playback."""

    def __init__(self, frame_rate: float = 30.0):
        self.frame_rate = frame_rate
        self.frame_queue: list[bytes] = []
        self.lock = threading.Lock()
        self.max_queue_size = 3

    def add_frame(self, frame: np.ndarray) -> None:
        """Add a frame to the stream queue."""
        success, buffer = cv2.imencode(".jpg", frame)
        if not success:
            return

        with self.lock:
            self.frame_queue.append(buffer.tobytes())
            # Keep only recent frames
            if len(self.frame_queue) > self.max_queue_size:
                self.frame_queue.pop(0)

    def get_frame(self) -> Optional[bytes]:
        """Get the most recent frame from the queue."""
        with self.lock:
            if self.frame_queue:
                return self.frame_queue[-1]
        return None


MJPEG_STREAM = MotionJPEGStream()


def _safe_int(value: Optional[float], fallback: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return fallback


def _safe_float(value: Optional[float], fallback: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return fallback


def _safe_path(value: Optional[str], fallback: Path) -> Path:
    if value is None:
        return fallback
    candidate = str(value).strip()
    return Path(candidate) if candidate else fallback


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


def _sanitize_filename(name: str, fallback: str = "model.bin") -> str:
    base = Path(name).name
    safe = re.sub(r"[^A-Za-z0-9._-]", "_", base)
    return safe or fallback


def _list_model_files(output_dir: Path = DEFAULT_MODEL_DIR) -> list[Path]:
    if not output_dir.exists():
        return []
    files = [path for path in output_dir.iterdir() if path.is_file()]
    return sorted(files, key=lambda path: path.name.lower())


def _model_options() -> list[dict]:
    options = []
    for path in _list_model_files():
        options.append({"value": str(path), "label": path.name})
    return options


def save_uploaded_model(contents: str, filename: str, output_dir: Path) -> Path:
    if "," not in contents:
        raise ValueError("Uploaded model payload is not base64 encoded.")
    _, encoded = contents.split(",", 1)
    data = base64.b64decode(encoded)
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_name = _sanitize_filename(filename or "model.bin")
    path = output_dir / safe_name
    path.write_bytes(data)
    return path


def _load_ultralytics_model(model_path: Path) -> object:
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError(
            "ultralytics is not installed. Run `uv add ultralytics` to enable YOLO inference.") from exc
    return YOLO(str(model_path))


def activate_model(model_path: Optional[Path]) -> str:
    if model_path is None:
        SESSION.active_model_path = None
        SESSION.yolo_model = None
        SESSION.model_loaded = False
        return "No model selected."
    if not model_path.exists():
        SESSION.active_model_path = None
        SESSION.yolo_model = None
        SESSION.model_loaded = False
        raise FileNotFoundError(f"Model not found: {model_path}")
    SESSION.active_model_path = model_path
    try:
        SESSION.yolo_model = _load_ultralytics_model(model_path)
        SESSION.model_loaded = True
        SESSION.last_detection_error = None
        return f"Model loaded: {model_path.name}."
    except Exception as exc:
        SESSION.yolo_model = None
        SESSION.model_loaded = False
        SESSION.last_detection_error = str(exc)
        return f"Model ready but not loaded: {exc}"


def _workspace_pixel_to_mm(
    workspace_def: workspace.WorkspaceDefinition,
    pixel_xy: Tuple[float, float],
) -> Tuple[float, float]:
    scale = float(workspace_def.x_axis_camera[0])
    if scale == 0:
        raise RuntimeError(
            "Workspace scale is zero; recompute workspace markers.")
    origin_px_x = workspace_def.origin_camera[0] / scale
    origin_px_y = workspace_def.origin_camera[1] / scale
    x_mm = (pixel_xy[0] - origin_px_x) * scale
    y_mm = (pixel_xy[1] - origin_px_y) * scale
    return x_mm, y_mm


def _pose_from_workspace_mm(
    workspace_def: workspace.WorkspaceDefinition,
    x_mm: float,
    y_mm: float,
    z_offset_mm: float,
) -> dict:
    robot_xyz = workspace_def.workspace_to_robot(x_mm, y_mm, z_offset_mm)
    return {
        "x": float(robot_xyz[0]),
        "y": float(robot_xyz[1]),
        "z": float(robot_xyz[2]),
        "a": 0.0,
        "b": 0.0,
        "c": 0.0,
    }


def _run_detection(
    frame: np.ndarray,
    confidence: float,
    iou: float,
    max_detections: int,
    class_filter: Optional[str],
) -> list[dict]:
    if SESSION.yolo_model is None:
        raise RuntimeError(
            "No YOLO model loaded. Upload a model or install ultralytics.")
    classes = None
    if class_filter:
        classes = [int(val.strip())
                   for val in class_filter.split(",") if val.strip().isdigit()]
    results = SESSION.yolo_model.predict(
        frame,
        conf=confidence,
        iou=iou,
        max_det=max_detections,
        classes=classes,
        verbose=False,
    )
    detections: list[dict] = []
    if not results:
        return detections
    for result in results:
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            continue
        xyxy = boxes.xyxy.cpu().numpy() if hasattr(boxes.xyxy, "cpu") else boxes.xyxy
        confs = boxes.conf.cpu().numpy() if hasattr(boxes.conf, "cpu") else boxes.conf
        clss = boxes.cls.cpu().numpy() if hasattr(boxes.cls, "cpu") else boxes.cls
        for box, conf, cls_idx in zip(xyxy, confs, clss):
            x1, y1, x2, y2 = [float(val) for val in box]
            center = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
            label = str(int(cls_idx))
            if hasattr(SESSION.yolo_model, "names"):
                label = SESSION.yolo_model.names.get(int(cls_idx), label)
            detection = {
                "bbox": (x1, y1, x2, y2),
                "center": center,
                "label": label,
                "confidence": float(conf),
            }
            if SESSION.workspace_definition is not None:
                try:
                    detection["workspace_mm"] = _workspace_pixel_to_mm(
                        SESSION.workspace_definition, center)
                except Exception:
                    detection["workspace_mm"] = None
            detections.append(detection)
    return detections


def _format_detection_list(detections: list[dict]) -> list:
    if not detections:
        return [dmc.Text("No detections yet.", size="sm", c="dimmed")]
    rows = []
    for idx, det in enumerate(detections, start=1):
        label = det.get("label", "object")
        conf = det.get("confidence", 0.0)
        workspace_mm = det.get("workspace_mm")
        if workspace_mm:
            coord = f"{workspace_mm[0]:.1f}, {workspace_mm[1]:.1f} mm"
        else:
            coord = "workspace: --"
        rows.append(
            dmc.Text(f"{idx}. {label} ({conf:.2f}) @ {coord}", size="sm"))
    return rows


def _detect_drop_marker(
    frame: np.ndarray,
    marker_id: int,
) -> Tuple[markers.MarkerDetection, Tuple[float, float]]:
    detections = markers.detect_all_markers(frame)
    if marker_id not in detections:
        raise RuntimeError(f"Drop marker {marker_id} not detected.")
    marker = detections[marker_id]
    workspace_mm = None
    if SESSION.workspace_definition is not None:
        workspace_mm = _workspace_pixel_to_mm(
            SESSION.workspace_definition, marker.center)
    if workspace_mm is None:
        raise RuntimeError(
            "Workspace is not defined; run workspace detection first.")
    return marker, workspace_mm


def _placeholder_frame(message: str, size: Tuple[int, int] = (720, 1280)) -> np.ndarray:
    height, width = size
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    cv2.putText(
        frame,
        message,
        (40, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 255),
        2,
    )
    return frame


def _encode_frame(frame: np.ndarray) -> str:
    success, buffer = cv2.imencode(".jpg", frame)
    if not success:
        raise RuntimeError("Failed to encode camera frame.")
    encoded = base64.b64encode(buffer).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def _default_transform() -> workspace.RigidTransform:
    return workspace.RigidTransform(rotation=np.eye(3), translation=np.zeros(3))


def start_camera_stream(width: Optional[float], height: Optional[float], fps: Optional[float]) -> str:
    stop_camera_stream()
    resolution = (_safe_int(width, DEFAULT_RESOLUTION[0]), _safe_int(
        height, DEFAULT_RESOLUTION[1]))
    fps_value = _safe_float(fps, DEFAULT_FPS)
    stream = DepthAIFrameStream(StreamConfig(
        resolution=resolution, fps=fps_value))
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
    frame, _depth = get_latest_frame()
    if frame is None:
        raise RuntimeError("Camera stream is not providing frames yet.")
    output_dir.mkdir(parents=True, exist_ok=True)
    frames = _update_capture_state(output_dir)
    index = _frame_index(frames[-1]) + 1 if frames else 1
    path = output_dir / f"calibration_color_{index:02}.png"
    cv2.imwrite(str(path), frame)
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


def compute_calibration_bundle(board_path: Path, bundle_path: Path, output_dir: Path) -> str:
    if not board_path.exists():
        raise FileNotFoundError(
            f"ChArUco board layout not found: {board_path}")
    frames = _update_capture_state(output_dir)
    if not frames:
        raise RuntimeError(
            "No calibration frames found. Capture frames before calibrating.")
    depth_frame = SESSION.capture.depth_frame or load_depth_snapshot(
        output_dir)
    if depth_frame is None:
        raise RuntimeError(
            "No depth snapshot found. Capture a depth frame before calibrating.")
    config = DemoConfig(calibration_path=bundle_path,
                        aruco_board_layout_path=board_path)
    demo = BootstrapDemo(config)
    demo.load_or_create_calibration(
        image_paths=frames, depth_frame=depth_frame, force_rebuild=True)
    SESSION.calibration_bundle = demo.calibration
    return f"Calibration bundle saved to {bundle_path}."


def load_calibration_bundle(bundle_path: Path) -> str:
    if not bundle_path.exists():
        raise FileNotFoundError(f"Calibration bundle not found: {bundle_path}")
    SESSION.calibration_bundle = calibration.load_calibration(bundle_path)
    return f"Loaded calibration bundle from {bundle_path}."


def compute_workspace_from_latest(marker_ids: Tuple[int, int], marker_size_mm: float) -> workspace.WorkspaceDefinition:
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
    return ws


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


def _draw_label(frame: np.ndarray, text: str, origin: Tuple[int, int], color: Tuple[int, int, int]) -> None:
    cv2.putText(frame, text, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


def draw_charuco_overlay(frame: np.ndarray, board_path: Path) -> Tuple[np.ndarray, str, str]:
    vis = frame.copy()
    if not board_path.exists():
        status = "ChArUco layout not found."
        return vis, status, "ChArUco corners: --"
    board_spec = aruco_board.load_charuco_board_spec(board_path)
    board = board_spec.create_board()
    charuco_params = cv2.aruco.CharucoParameters()
    detector_params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.CharucoDetector(
        board, charuco_params, detector_params)
    gray = cv2.cvtColor(vis, cv2.COLOR_BGR2GRAY)
    charuco_corners, charuco_ids, marker_corners, marker_ids = detector.detectBoard(
        gray)
    marker_count = 0 if marker_ids is None else len(marker_ids)
    corner_count = 0 if charuco_ids is None else len(charuco_ids)
    if marker_ids is not None and marker_corners is not None:
        cv2.aruco.drawDetectedMarkers(vis, marker_corners, marker_ids)
    if charuco_ids is not None and charuco_corners is not None:
        cv2.aruco.drawDetectedCornersCharuco(vis, charuco_corners, charuco_ids)
    _draw_label(vis, f"Markers: {marker_count}", (10, 30), (255, 255, 255))
    _draw_label(vis, f"Corners: {corner_count}", (10, 55), (255, 255, 255))
    status = "Camera calibration overlay ready."
    metric = f"ChArUco corners: {corner_count}"
    return vis, status, metric


def draw_workspace_overlay(
    frame: np.ndarray,
    marker_ids: Tuple[int, int],
    marker_size_mm: float,
) -> Tuple[np.ndarray, str, str]:
    vis = frame.copy()
    try:
        detections = markers.detect_markers(frame, marker_ids)
        for det, color in zip((detections.marker_a, detections.marker_b), ((0, 255, 0), (0, 140, 255))):
            pts = det.corners.astype(int)
            cv2.polylines(vis, [pts], isClosed=True, color=color, thickness=2)
            center = tuple(np.round(det.center).astype(int))
            cv2.circle(vis, center, 4, color, -1)
            cv2.putText(vis, f"ID {det.marker_id}", center,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        status = "Workspace markers detected."
    except Exception as exc:
        return vis, f"Workspace detection: {exc}", "Workspace: --"

    metric = "Workspace: --"
    if SESSION.calibration_bundle is not None:
        try:
            ws = workspace.compute_workspace(
                detections,
                SESSION.calibration_bundle,
                marker_size_mm=marker_size_mm,
                camera_to_robot=_default_transform(),
            )
            metric = f"Workspace: {ws.width_mm:.1f} x {ws.height_mm:.1f} mm"
            _draw_label(vis, metric, (10, 30), (255, 255, 255))
        except Exception as exc:
            metric = f"Workspace: {exc}"
    return vis, status, metric


def draw_plate_overlay(frame: np.ndarray, layout_path: Path) -> Tuple[np.ndarray, str, str]:
    vis = frame.copy()
    if not layout_path.exists():
        return vis, "Calibration plate layout not found.", "Plate error: --"
    plate_layout = plate.load_plate_layout(layout_path)
    try:
        calibration_result = plate.estimate_plate_calibration(
            frame, plate_layout)
    except Exception as exc:
        return vis, f"Plate detection: {exc}", "Plate error: --"

    for det in calibration_result.detections.values():
        pts = det.corners.astype(int)
        cv2.polylines(vis, [pts], isClosed=True,
                      color=(255, 204, 0), thickness=2)
        center = tuple(np.round(det.center).astype(int))
        cv2.circle(vis, center, 4, (255, 204, 0), -1)
        cv2.putText(vis, f"ID {det.marker_id}", center,
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 204, 0), 2)

    metric = f"Reprojection error: {calibration_result.reprojection_error_mm:.2f} mm"
    _draw_label(vis, metric, (10, 30), (255, 255, 255))
    status = f"Plate markers detected: {len(calibration_result.detections)}"
    return vis, status, metric


def draw_detection_overlay(frame: np.ndarray) -> Tuple[np.ndarray, str, str]:
    vis = frame.copy()
    detections = SESSION.detections or []
    if not detections:
        return vis, "No detections available.", "Detections: 0"
    for det in detections:
        x1, y1, x2, y2 = [int(val) for val in det["bbox"]]
        label = det.get("label", "object")
        conf = det.get("confidence", 0.0)
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 180, 255), 2)
        cv2.putText(
            vis,
            f"{label} {conf:.2f}",
            (x1, max(0, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 180, 255),
            2,
        )
    metric = f"Detections: {len(detections)}"
    _draw_label(vis, metric, (10, 30), (255, 255, 255))
    return vis, "Detection overlay ready.", metric


def draw_drop_marker_overlay(frame: np.ndarray, marker_id: int) -> Tuple[np.ndarray, str, str]:
    vis = frame.copy()
    try:
        marker, workspace_mm = _detect_drop_marker(frame, marker_id)
    except Exception as exc:
        return vis, f"Drop marker: {exc}", "Drop marker: --"
    pts = marker.corners.astype(int)
    cv2.polylines(vis, [pts], isClosed=True, color=(255, 128, 0), thickness=2)
    center = tuple(np.round(marker.center).astype(int))
    cv2.circle(vis, center, 4, (255, 128, 0), -1)
    cv2.putText(
        vis,
        f"Drop ID {marker.marker_id}",
        center,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 128, 0),
        2,
    )
    metric = f"Drop marker: {workspace_mm[0]:.1f}, {workspace_mm[1]:.1f} mm"
    _draw_label(vis, metric, (10, 30), (255, 255, 255))
    return vis, "Drop marker detected.", metric


def draw_monitor_overlay(frame: np.ndarray, marker_id: int) -> Tuple[np.ndarray, str, str, str]:
    vis, status, detection_metric = draw_detection_overlay(frame)
    vis, drop_status, drop_metric = draw_drop_marker_overlay(vis, marker_id)
    status = f"{status} | {drop_status}"
    return vis, status, detection_metric, drop_metric


def build_header() -> dmc.AppShellHeader:
    return dmc.AppShellHeader(
        dmc.Group(
            [
                dmc.Stack(
                    [
                        dmc.Text("Codroid Calibration Console",
                                 className="app-title"),
                        dmc.Text("Vision + robot bring-up dashboard",
                                 size="sm", c="dimmed"),
                    ],
                    gap=2,
                ),
                dmc.Group(
                    [
                        dmc.Badge("Camera: off", id="header-camera-badge",
                                  color="gray", variant="light"),
                        dmc.Badge("Robot: idle", id="header-robot-badge",
                                  color="gray", variant="light"),
                        dmc.Button("Setup", id="open-setup",
                                   variant="outline"),
                    ],
                    gap="sm",
                ),
            ],
            justify="space-between",
            align="center",
            h=64,
            px="md",
        )
    )


def build_setup_drawer() -> dmc.Drawer:
    return dmc.Drawer(
        id="setup-drawer",
        opened=False,
        title="Setup & Connections",
        padding="md",
        size="lg",
        children=dmc.Stack(
            [
                dmc.Text("Camera connection", fw=600),
                dmc.Group(
                    [
                        dmc.Button("Scan for camera",
                                   id="camera-scan", variant="light"),
                        dmc.Switch(
                            id="camera-enable",
                            label="Enable live camera stream",
                            checked=False,
                            color="teal",
                        ),
                    ],
                    gap="md",
                ),
                dmc.Group(
                    [
                        dmc.NumberInput(
                            id="camera-width",
                            label="Width",
                            value=DEFAULT_RESOLUTION[0],
                            min=320,
                            max=3840,
                        ),
                        dmc.NumberInput(
                            id="camera-height",
                            label="Height",
                            value=DEFAULT_RESOLUTION[1],
                            min=240,
                            max=2160,
                        ),
                        dmc.NumberInput(
                            id="camera-fps",
                            label="FPS",
                            value=DEFAULT_FPS,
                            min=1,
                            max=120,
                            step=1,
                        ),
                    ],
                    grow=True,
                ),
                dmc.Text(id="camera-status-text", size="sm", c="dimmed"),
                html.Div(id="camera-device-info"),
                dmc.Divider(),
                dmc.Text("Robot connection", fw=600),
                dmc.TextInput(
                    id="robot-uri",
                    label="WebSocket URI",
                    value=DEFAULT_ROBOT_URI,
                ),
                dmc.Button("Test robot connection",
                           id="robot-test", variant="light"),
                dmc.Text(id="robot-status-text", size="sm", c="dimmed"),
                dmc.Group(
                    [
                        dmc.Button("Close", id="setup-close",
                                   variant="subtle"),
                    ],
                    justify="flex-end",
                ),
            ],
            gap="sm",
        ),
    )


def build_stepper_card() -> dmc.Card:
    return dmc.Card(
        [
            dmc.Group(
                [
                    dmc.Text("Calibration Steps", fw=600, size="lg"),
                    dmc.Group(
                        [
                            dmc.Button("Back", id="step-prev",
                                       variant="light", size="sm"),
                            dmc.Button("Next", id="step-next", size="sm"),
                        ],
                        gap="xs",
                    ),
                ],
                justify="space-between",
                align="center",
            ),
            dmc.Divider(),
            dmc.Stepper(
                id="calibration-stepper",
                active=0,
                orientation="vertical",
                children=[
                    dmc.StepperStep(
                        label="Camera calibration",
                        description="Intrinsics + workspace plane",
                        children=dmc.Stack(
                            [
                                dmc.Text(
                                    "Capture the ChArUco board frames and a depth snapshot.",
                                    size="sm",
                                    c="dimmed",
                                ),
                                dmc.TextInput(
                                    id="calibration-board-path",
                                    label="Board layout",
                                    value=str(DEFAULT_BOARD_PATH),
                                ),
                                dmc.TextInput(
                                    id="calibration-output-dir",
                                    label="Capture output dir",
                                    value=str(DEFAULT_CAPTURE_DIR),
                                ),
                                dmc.TextInput(
                                    id="calibration-bundle-path",
                                    label="Calibration bundle",
                                    value=str(DEFAULT_BUNDLE_PATH),
                                ),
                                dmc.Group(
                                    [
                                        dmc.Button(
                                            "Capture frame", id="capture-frame", variant="outline"),
                                        dmc.Button(
                                            "Capture depth", id="capture-depth", variant="outline"),
                                    ],
                                    grow=True,
                                ),
                                dmc.Group(
                                    [
                                        dmc.Button(
                                            "Compute calibration", id="compute-calibration"),
                                        dmc.Button(
                                            "Load bundle", id="load-calibration", variant="light"),
                                    ],
                                    grow=True,
                                ),
                                dmc.Text(id="camera-calibration-action",
                                         size="sm", c="dimmed"),
                                dmc.Text(
                                    id="camera-calibration-metrics", size="sm"),
                            ],
                            gap="xs",
                        ),
                    ),
                    dmc.StepperStep(
                        label="Workarea definition",
                        description="Detect workspace markers",
                        children=dmc.Stack(
                            [
                                dmc.Text(
                                    "Use the two corner markers to define the workarea size.",
                                    size="sm",
                                    c="dimmed",
                                ),
                                dmc.Group(
                                    [
                                        dmc.NumberInput(
                                            id="marker-id-a",
                                            label="Marker ID A",
                                            value=SESSION.config.marker_ids[0],
                                            min=0,
                                            max=49,
                                        ),
                                        dmc.NumberInput(
                                            id="marker-id-b",
                                            label="Marker ID B",
                                            value=SESSION.config.marker_ids[1],
                                            min=0,
                                            max=49,
                                        ),
                                    ],
                                    grow=True,
                                ),
                                dmc.NumberInput(
                                    id="marker-size",
                                    label="Marker size (mm)",
                                    value=SESSION.config.marker_size_mm,
                                    min=1,
                                    max=100,
                                ),
                                dmc.Button(
                                    "Compute workspace", id="compute-workspace", variant="light"),
                                dmc.Text(id="workspace-action-status",
                                         size="sm", c="dimmed"),
                                dmc.Text(id="workspace-metrics", size="sm"),
                            ],
                            gap="xs",
                        ),
                    ),
                    dmc.StepperStep(
                        label="Robot calibration",
                        description="Detect plate markers",
                        children=dmc.Stack(
                            [
                                dmc.Text(
                                    "Detect the calibration plate and attach the homography.",
                                    size="sm",
                                    c="dimmed",
                                ),
                                dmc.TextInput(
                                    id="plate-layout-path",
                                    label="Plate layout",
                                    value=str(DEFAULT_PLATE_PATH),
                                ),
                                dmc.Group(
                                    [
                                        dmc.Button(
                                            "Detect plate", id="compute-plate", variant="light"),
                                        dmc.Button("Attach to bundle",
                                                   id="attach-plate"),
                                    ],
                                    grow=True,
                                ),
                                dmc.Text(id="plate-action-status",
                                         size="sm", c="dimmed"),
                                dmc.Text(id="plate-metrics", size="sm"),
                            ],
                            gap="xs",
                        ),
                    ),
                ],
            ),
        ],
        withBorder=True,
        radius="lg",
        className="panel-card",
    )


def build_camera_panel(title: str = "Camera View", badge: str = "Overlay") -> dmc.Card:
    placeholder_src = _encode_frame(
        _placeholder_frame("Camera stream offline"))
    return dmc.Card(
        [
            dmc.Group(
                [
                    dmc.Group(
                        [
                            dmc.Text(title, fw=600, size="lg"),
                            dmc.Badge(badge, variant="light"),
                        ],
                        gap="xs",
                    ),
                    dmc.ActionIcon(
                        html.Span(
                            "⛶", style={"fontSize": "1.2rem"}, title="Fullscreen"),
                        id="fullscreen-btn",
                        variant="light",
                    ),
                ],
                justify="space-between",
                align="center",
            ),
            dmc.Text(id="overlay-status", size="sm", c="dimmed"),
            html.Video(
                id="camera-feed",
                src="/video_feed",
                controls=False,
                autoPlay=True,
                muted=True,
                poster=placeholder_src,
                style={
                    "width": "100%",
                    "height": "auto",
                    "borderRadius": "16px",
                    "border": "1px solid rgba(15, 118, 110, 0.2)",
                    "marginTop": "12px",
                    "backgroundColor": "#0b0f14",
                    "display": "block",
                },
            ),
        ],
        withBorder=True,
        radius="lg",
        className="panel-card camera-panel",
    )


def build_calibration_tab() -> dmc.Stack:
    return dmc.Stack([build_stepper_card()], gap="md")


def build_detection_tab() -> dmc.Stack:
    model_card = dmc.Card(
        [
            dmc.Text("Model Loader", fw=600, size="lg"),
            dmc.Text(
                "Upload a YOLO model (.pt/.onnx) to enable detection.", size="sm", c="dimmed"),
            dcc.Upload(
                id="model-upload",
                children=dmc.Button("Upload YOLO model", variant="light"),
                multiple=False,
            ),
            dmc.Select(
                id="model-select",
                label="Active model",
                data=_model_options(),
                value=str(
                    SESSION.active_model_path) if SESSION.active_model_path else None,
                searchable=True,
                clearable=True,
            ),
            dmc.Button("Load selected model",
                       id="load-model", variant="outline"),
            dmc.Text(id="model-status", size="sm", c="dimmed"),
        ],
        withBorder=True,
        radius="lg",
        className="panel-card",
    )

    detection_card = dmc.Card(
        [
            dmc.Text("Detection Controls", fw=600, size="lg"),
            dmc.Group(
                [
                    dmc.NumberInput(
                        id="detection-confidence",
                        label="Confidence",
                        value=0.4,
                        min=0.0,
                        max=1.0,
                        step=0.05,
                    ),
                    dmc.NumberInput(
                        id="detection-iou",
                        label="IOU",
                        value=0.45,
                        min=0.0,
                        max=1.0,
                        step=0.05,
                    ),
                ],
                grow=True,
            ),
            dmc.Group(
                [
                    dmc.NumberInput(
                        id="detection-max",
                        label="Max detections",
                        value=10,
                        min=1,
                        max=200,
                    ),
                    dmc.TextInput(
                        id="detection-classes",
                        label="Class filter (comma IDs)",
                        placeholder="e.g. 0, 32",
                    ),
                ],
                grow=True,
            ),
            dmc.Button("Run detection", id="run-detection"),
            dmc.Text(id="detection-status", size="sm", c="dimmed"),
            dmc.Text(id="detection-metrics", size="sm"),
        ],
        withBorder=True,
        radius="lg",
        className="panel-card",
    )

    detections_card = dmc.Card(
        [
            dmc.Text("Detected Objects", fw=600, size="lg"),
            dmc.Stack(id="detection-list", gap="xs"),
        ],
        withBorder=True,
        radius="lg",
        className="panel-card",
    )

    return dmc.Stack([model_card, detection_card, detections_card], gap="md")


def build_pick_place_tab() -> dmc.Stack:
    target_card = dmc.Card(
        [
            dmc.Text("Pick Targets", fw=600, size="lg"),
            dmc.Select(
                id="pick-detection-select",
                label="Detection to pick",
                data=[],
                placeholder="Run detection first",
                clearable=True,
            ),
            dmc.Group(
                [
                    dmc.NumberInput(
                        id="pick-approach-height",
                        label="Approach height (mm)",
                        value=30.0,
                        min=0,
                        max=200,
                    ),
                    dmc.NumberInput(
                        id="pick-height",
                        label="Pick height (mm)",
                        value=0.0,
                        min=-50,
                        max=200,
                    ),
                ],
                grow=True,
            ),
        ],
        withBorder=True,
        radius="lg",
        className="panel-card",
    )

    drop_card = dmc.Card(
        [
            dmc.Text("Drop Target", fw=600, size="lg"),
            dmc.NumberInput(
                id="drop-marker-id",
                label="Drop marker ID",
                value=SESSION.drop_marker_id,
                min=0,
                max=49,
            ),
            dmc.Button("Detect drop marker",
                       id="detect-drop-marker", variant="light"),
            dmc.Text(id="drop-marker-metrics", size="sm"),
            dmc.NumberInput(
                id="drop-height",
                label="Drop height (mm)",
                value=0.0,
                min=-50,
                max=200,
            ),
        ],
        withBorder=True,
        radius="lg",
        className="panel-card",
    )

    gripper_card = dmc.Card(
        [
            dmc.Text("Gripper Control", fw=600, size="lg"),
            dmc.TextInput(
                id="gripper-path",
                label="Gripper control path",
                placeholder="Robot/IO/gripper",
            ),
            dmc.Group(
                [
                    dmc.NumberInput(
                        id="gripper-open-value",
                        label="Open value",
                        value=1,
                    ),
                    dmc.NumberInput(
                        id="gripper-close-value",
                        label="Close value",
                        value=0,
                    ),
                ],
                grow=True,
            ),
            dmc.Group(
                [
                    dmc.Button("Open gripper", id="gripper-open",
                               variant="light"),
                    dmc.Button("Close gripper", id="gripper-close",
                               variant="light"),
                ],
                grow=True,
            ),
        ],
        withBorder=True,
        radius="lg",
        className="panel-card",
    )

    action_card = dmc.Card(
        [
            dmc.Text("Robot Actions", fw=600, size="lg"),
            dmc.Group(
                [
                    dmc.Button("Pick", id="execute-pick"),
                    dmc.Button("Drop", id="execute-drop", variant="light"),
                ],
                grow=True,
            ),
            dmc.Button("Pick & Drop", id="execute-pick-drop",
                       variant="outline"),
            dmc.Text(id="pick-place-status", size="sm", c="dimmed"),
        ],
        withBorder=True,
        radius="lg",
        className="panel-card",
    )

    return dmc.Stack([target_card, drop_card, gripper_card, action_card], gap="md")


def build_monitoring_tab() -> dmc.Stack:
    summary_card = dmc.Card(
        [
            dmc.Text("System Summary", fw=600, size="lg"),
            dmc.Stack(id="monitor-system-summary", gap="xs"),
        ],
        withBorder=True,
        radius="lg",
        className="panel-card",
    )
    detection_card = dmc.Card(
        [
            dmc.Text("Detection Summary", fw=600, size="lg"),
            dmc.Stack(id="monitor-detection-summary", gap="xs"),
        ],
        withBorder=True,
        radius="lg",
        className="panel-card",
    )
    pick_card = dmc.Card(
        [
            dmc.Text("Pick & Drop Summary", fw=600, size="lg"),
            dmc.Stack(id="monitor-pick-summary", gap="xs"),
        ],
        withBorder=True,
        radius="lg",
        className="panel-card",
    )

    return dmc.Stack([summary_card, detection_card, pick_card], gap="md")


def build_layout() -> dmc.MantineProvider:
    theme = {
        "fontFamily": "Space Grotesk, IBM Plex Sans, sans-serif",
        "headings": {"fontFamily": "Space Grotesk, IBM Plex Sans, sans-serif"},
        "primaryColor": "teal",
    }
    return dmc.MantineProvider(
        theme=theme,
        withGlobalClasses=True,
        withCssVariables=True,
        children=html.Div(
            [
                dmc.NotificationContainer(
                    id="app-notifications", position="top-right", limit=3),
                dcc.Store(id="camera-connection-state"),
                dcc.Store(id="robot-connection-state"),
                dcc.Store(id="notification-state",
                          data={"camera": None, "robot": None}),
                dcc.Interval(id="auto-init", interval=400,
                             n_intervals=0, max_intervals=1),
                dcc.Interval(id="camera-interval",
                             interval=300, n_intervals=0),
                build_setup_drawer(),
                dmc.AppShell(
                    [
                        build_header(),
                        dmc.AppShellMain(
                            dmc.Container(
                                dmc.Grid(
                                    [
                                        dmc.GridCol(
                                            dmc.Tabs(
                                                [
                                                    dmc.TabsList(
                                                        [
                                                            dmc.TabsTab(
                                                                "Calibration", value="calibration"),
                                                            dmc.TabsTab(
                                                                "Detection", value="detection"),
                                                            dmc.TabsTab(
                                                                "Pick & Place", value="pick-place"),
                                                            dmc.TabsTab(
                                                                "Monitoring", value="monitoring"),
                                                        ],
                                                        style={
                                                            "width": "100%",
                                                            "justifyContent": "flex-start",
                                                            "marginTop": "0px",
                                                            "marginBottom": "12px",
                                                        },
                                                    ),
                                                    dmc.TabsPanel(
                                                        build_calibration_tab(), value="calibration"),
                                                    dmc.TabsPanel(
                                                        build_detection_tab(), value="detection"),
                                                    dmc.TabsPanel(
                                                        build_pick_place_tab(), value="pick-place"),
                                                    dmc.TabsPanel(
                                                        build_monitoring_tab(), value="monitoring"),
                                                ],
                                                id="main-tabs",
                                                value="calibration",
                                                variant="pills",
                                                color="teal",
                                                style={"width": "100%",
                                                       "marginBottom": "12px"},
                                            ),
                                            span={"base": 12, "md": 4},
                                        ),
                                        dmc.GridCol(
                                            dmc.Stack(
                                                [build_camera_panel()], gap="md"),
                                            span={"base": 12, "md": 8},
                                        ),
                                    ],
                                    gutter="md",
                                ),
                                fluid=True,
                                px=0,
                                mx=0,
                                mt="sm",
                                className="main-panel",
                            ),
                        ),
                    ],
                    header={"height": 64},
                    navbar={"width": 220, "breakpoint": "md",
                            "collapsed": {"mobile": True}},
                    padding="md",
                ),
            ]
        ),
    )


def _format_device_list(devices: list[object]) -> html.Ul:
    if not devices:
        return html.Ul([html.Li("No OAK-D devices detected.")])
    items = []
    for idx, device in enumerate(devices, start=1):
        name = None
        mx_id_fn = getattr(device, "getMxId", None)
        device_id_fn = getattr(device, "getDeviceId", None)
        if callable(mx_id_fn):
            name = mx_id_fn()
        elif callable(device_id_fn):
            name = device_id_fn()
        else:
            name = str(device)
        state = getattr(device, "state", None)
        items.append(html.Li(f"{idx}. {name} ({state})"))
    return html.Ul(items)


def _apply_robot_uri(settings: CodroidSettings, uri: str) -> CodroidSettings:
    if not uri:
        return settings
    parsed = urlparse(uri)
    if parsed.hostname:
        settings.host = parsed.hostname
    if parsed.port:
        settings.robot_port = parsed.port
    return settings


async def _probe_robot(uri: str) -> dict:
    settings = _apply_robot_uri(CodroidSettings(), uri)
    robot_config = settings.build_robot_config()
    robot_api = CodroidAPI(robot_config)
    await robot_api.connect()
    try:
        await robot_api.robot_login()
        await robot_api.read_system_data()
    finally:
        await robot_api.close()
    return {"host": settings.host, "port": settings.robot_port}


async def _with_robot_api(uri: str, action: Callable[[CodroidAPI], Awaitable[None]]) -> None:
    settings = _apply_robot_uri(CodroidSettings(), uri)
    user_config = settings.build_user_config()
    robot_config = settings.build_robot_config()
    user_api = CodroidAPI(user_config)
    robot_api = CodroidAPI(robot_config)
    await user_api.connect()
    if user_config.user_password:
        await user_api.ws_login_with_password()
    elif user_config.usercode:
        await user_api.ws_login()
    await robot_api.connect()
    await robot_api.robot_login()
    try:
        await action(robot_api)
    finally:
        await robot_api.close()
        await user_api.close()


def register_callbacks(app: Dash) -> None:
    app.clientside_callback(
        """
        function(n_clicks) {
            if (!n_clicks) return;
            const video = document.getElementById('camera-feed');
            if (!video) return;
            const fsFn = video.requestFullscreen || 
                         video.webkitRequestFullscreen ||
                         video.mozRequestFullScreen ||
                         video.msRequestFullscreen;
            if (fsFn) {
                fsFn.call(video).catch(err => {
                    if (video.webkitEnterFullscreen) video.webkitEnterFullscreen();
                });
            } else if (video.webkitEnterFullscreen) {
                video.webkitEnterFullscreen();
            }
        }
        """,
        Output("fullscreen-btn", "n_clicks"),
        Input("fullscreen-btn", "n_clicks"),
        prevent_initial_call=True,
    )

    @app.callback(
        Output("setup-drawer", "opened"),
        Input("open-setup", "n_clicks"),
        Input("setup-close", "n_clicks"),
        State("setup-drawer", "opened"),
    )
    def _toggle_setup_drawer(open_clicks: int, close_clicks: int, opened: bool) -> bool:
        trigger = callback_context.triggered_id
        if trigger == "open-setup":
            return True
        if trigger == "setup-close":
            return False
        return opened

    @app.callback(
        Output("calibration-stepper", "active"),
        Input("step-prev", "n_clicks"),
        Input("step-next", "n_clicks"),
        State("calibration-stepper", "active"),
    )
    def _update_stepper(prev_clicks: int, next_clicks: int, active: Optional[int]) -> int:
        current = active or 0
        trigger = callback_context.triggered_id
        if trigger == "step-next":
            return min(current + 1, 2)
        if trigger == "step-prev":
            return max(current - 1, 0)
        return current

    @app.callback(
        Output("camera-device-info", "children"),
        Input("camera-scan", "n_clicks"),
        prevent_initial_call=True,
    )
    def _scan_camera(_clicks: int) -> html.Ul:
        devices = list_oak_devices()
        return _format_device_list(devices)

    @app.callback(
        Output("camera-status-text", "children"),
        Output("header-camera-badge", "children"),
        Output("header-camera-badge", "color"),
        Output("camera-enable", "checked"),
        Output("camera-connection-state", "data"),
        Input("camera-enable", "checked"),
        Input("auto-init", "n_intervals"),
        State("camera-width", "value"),
        State("camera-height", "value"),
        State("camera-fps", "value"),
    )
    def _toggle_camera(
        checked: bool,
        _init: int,
        width: Optional[float],
        height: Optional[float],
        fps: Optional[float],
    ) -> Tuple[str, str, str, bool, dict]:
        trigger = callback_context.triggered_id
        desired_checked = checked
        if trigger == "auto-init":
            desired_checked = True
        if desired_checked:
            try:
                status = start_camera_stream(width, height, fps)
                return status, "Camera: live", "teal", True, {"status": "live", "message": status}
            except Exception as exc:
                stop_camera_stream()
                error = f"Camera error: {exc}"
                return error, "Camera: error", "red", False, {"status": "error", "message": error}
        stop_camera_stream()
        stopped = "Camera stream stopped."
        return stopped, "Camera: off", "gray", False, {"status": "off", "message": stopped}

    @app.callback(
        Output("robot-status-text", "children"),
        Output("header-robot-badge", "children"),
        Output("header-robot-badge", "color"),
        Output("robot-connection-state", "data"),
        Input("robot-test", "n_clicks"),
        Input("auto-init", "n_intervals"),
        State("robot-uri", "value"),
        prevent_initial_call=True,
    )
    def _test_robot(_clicks: int, _init: int, uri: str) -> Tuple[str, str, str, dict]:
        trigger = callback_context.triggered_id
        if trigger == "auto-init":
            try:
                response = asyncio.run(_probe_robot(uri))
                status = f"Robot connected to {response.get('host')}:{response.get('port')}."
                return status, "Robot: online", "teal", {"status": "online", "message": status}
            except Exception as exc:
                error = f"Robot connection failed: {exc}"
                return error, "Robot: error", "red", {"status": "error", "message": error}
        try:
            response = asyncio.run(_probe_robot(uri))
            status = f"Robot connected to {response.get('host')}:{response.get('port')}."
            return status, "Robot: online", "teal", {"status": "online", "message": status}
        except Exception as exc:
            error = f"Robot connection failed: {exc}"
            return error, "Robot: error", "red", {"status": "error", "message": error}

    @app.callback(
        Output("model-status", "children"),
        Output("model-select", "data"),
        Output("model-select", "value"),
        Input("model-upload", "contents"),
        Input("load-model", "n_clicks"),
        State("model-upload", "filename"),
        State("model-select", "value"),
        prevent_initial_call=True,
    )
    def _handle_model_actions(
        contents: Optional[str],
        _load_clicks: int,
        filename: Optional[str],
        selected: Optional[str],
    ) -> Tuple[str, list, Optional[str]]:
        trigger = callback_context.triggered_id
        options = _model_options()
        if trigger == "model-upload" and contents:
            try:
                model_path = save_uploaded_model(
                    contents, filename or "model.bin", DEFAULT_MODEL_DIR)
                status = activate_model(model_path)
                options = _model_options()
                return status, options, str(model_path)
            except Exception as exc:
                return f"Model upload failed: {exc}", options, selected
        if trigger == "load-model":
            if selected:
                try:
                    status = activate_model(Path(selected))
                except Exception as exc:
                    status = f"Model load failed: {exc}"
                return status, options, selected
            status = activate_model(None)
            return status, options, None
        return no_update, options, no_update

    @app.callback(
        Output("detection-status", "children"),
        Output("detection-list", "children"),
        Output("pick-detection-select", "data"),
        Output("pick-detection-select", "value"),
        Input("run-detection", "n_clicks"),
        State("detection-confidence", "value"),
        State("detection-iou", "value"),
        State("detection-max", "value"),
        State("detection-classes", "value"),
        prevent_initial_call=True,
    )
    def _run_detection_action(
        _clicks: int,
        confidence: Optional[float],
        iou: Optional[float],
        max_detections: Optional[float],
        class_filter: Optional[str],
    ) -> Tuple[str, list, list, Optional[str]]:
        frame, _depth = get_latest_frame()
        if frame is None:
            return (
                "Camera stream is not running.",
                _format_detection_list([]),
                [],
                None,
            )
        conf_value = _safe_float(confidence, 0.4)
        iou_value = _safe_float(iou, 0.45)
        max_value = _safe_int(max_detections, 10)
        try:
            detections = _run_detection(
                frame, conf_value, iou_value, max_value, class_filter)
            SESSION.detections = detections
            SESSION.last_detection_error = None
            SESSION.last_action = "Detection run completed."
        except Exception as exc:
            SESSION.detections = []
            SESSION.last_detection_error = str(exc)
            return (
                f"Detection failed: {exc}",
                _format_detection_list([]),
                [],
                None,
            )

        options = []
        for idx, det in enumerate(detections):
            label = det.get("label", "object")
            options.append({"value": str(idx), "label": f"{idx + 1}: {label}"})
        selected = options[0]["value"] if options else None
        return (
            f"Detection complete: {len(detections)} object(s).",
            _format_detection_list(detections),
            options,
            selected,
        )

    @app.callback(
        Output("pick-place-status", "children"),
        Input("detect-drop-marker", "n_clicks"),
        Input("execute-pick", "n_clicks"),
        Input("execute-drop", "n_clicks"),
        Input("execute-pick-drop", "n_clicks"),
        Input("gripper-open", "n_clicks"),
        Input("gripper-close", "n_clicks"),
        State("pick-detection-select", "value"),
        State("drop-marker-id", "value"),
        State("pick-approach-height", "value"),
        State("pick-height", "value"),
        State("drop-height", "value"),
        State("gripper-path", "value"),
        State("gripper-open-value", "value"),
        State("gripper-close-value", "value"),
        State("robot-uri", "value"),
        prevent_initial_call=True,
    )
    def _handle_pick_place(
        _drop_clicks: int,
        _pick_clicks: int,
        _drop_action: int,
        _pick_drop_clicks: int,
        _gripper_open: int,
        _gripper_close: int,
        selected_detection: Optional[str],
        drop_marker_id: Optional[float],
        approach_height: Optional[float],
        pick_height: Optional[float],
        drop_height: Optional[float],
        gripper_path: Optional[str],
        gripper_open_value: Optional[float],
        gripper_close_value: Optional[float],
        robot_uri: Optional[str],
    ) -> str:
        trigger = callback_context.triggered_id
        frame, _depth = get_latest_frame()
        if frame is None:
            return "Camera stream is not running."

        drop_marker = _safe_int(drop_marker_id, SESSION.drop_marker_id)
        approach_mm = _safe_float(approach_height, 30.0)
        pick_mm = _safe_float(pick_height, 0.0)
        drop_mm = _safe_float(drop_height, 0.0)
        gripper_open_val = _safe_float(gripper_open_value, 1.0)
        gripper_close_val = _safe_float(gripper_close_value, 0.0)
        gripper_path = (gripper_path or "").strip() or None

        if trigger == "detect-drop-marker":
            try:
                marker, workspace_mm = _detect_drop_marker(frame, drop_marker)
                SESSION.drop_marker_id = drop_marker
                SESSION.drop_target_mm = workspace_mm
                SESSION.last_action = f"Drop marker {marker.marker_id} detected."
                return f"Drop marker at {workspace_mm[0]:.1f}, {workspace_mm[1]:.1f} mm."
            except Exception as exc:
                return f"Drop marker detection failed: {exc}"

        if trigger in ("gripper-open", "gripper-close"):
            if not gripper_path:
                return "Gripper path is empty; configure a control path."
            value = gripper_open_val if trigger == "gripper-open" else gripper_close_val

            async def _action(robot_api: CodroidAPI) -> None:
                await robot_api.set_param(gripper_path, value)

            try:
                asyncio.run(_with_robot_api(
                    robot_uri or DEFAULT_ROBOT_URI, _action))
                SESSION.last_action = f"Gripper command sent ({value})."
                return f"Gripper command sent ({value})."
            except Exception as exc:
                return f"Gripper action failed: {exc}"

        if SESSION.workspace_definition is None:
            return "Workspace is not defined; run workspace calibration first."
        if not SESSION.detections:
            return "No detections available; run detection first."
        if selected_detection is None:
            return "Select a detection to pick."

        try:
            det_idx = int(selected_detection)
            detection = SESSION.detections[det_idx]
        except (ValueError, IndexError):
            return "Invalid detection selection."

        center = detection.get("center")
        if center is None:
            return "Detection is missing center coordinates."
        workspace_mm = detection.get("workspace_mm")
        if workspace_mm is None:
            workspace_mm = _workspace_pixel_to_mm(
                SESSION.workspace_definition, center)

        pick_pose = _pose_from_workspace_mm(
            SESSION.workspace_definition, workspace_mm[0], workspace_mm[1], pick_mm)
        approach_pose = _pose_from_workspace_mm(
            SESSION.workspace_definition, workspace_mm[0], workspace_mm[1], approach_mm
        )
        SESSION.last_pick_mm = workspace_mm

        def _resolve_drop_target() -> Tuple[float, float]:
            if SESSION.drop_target_mm and SESSION.drop_marker_id == drop_marker:
                return SESSION.drop_target_mm
            _marker, drop_mm_target = _detect_drop_marker(frame, drop_marker)
            SESSION.drop_target_mm = drop_mm_target
            SESSION.drop_marker_id = drop_marker
            return drop_mm_target

        async def _pick_action(robot_api: CodroidAPI) -> None:
            await robot_api.power_on()
            await robot_api.move_to_cartesian_target_linear(robot_api.build_target_cpos(**approach_pose))
            await robot_api.move_to_cartesian_target_linear(robot_api.build_target_cpos(**pick_pose))
            if gripper_path:
                await robot_api.set_param(gripper_path, gripper_close_val)
            await robot_api.move_to_cartesian_target_linear(robot_api.build_target_cpos(**approach_pose))

        async def _drop_action(robot_api: CodroidAPI, drop_target_mm: Tuple[float, float]) -> None:
            drop_pose = _pose_from_workspace_mm(
                SESSION.workspace_definition, drop_target_mm[0], drop_target_mm[1], drop_mm
            )
            drop_approach = _pose_from_workspace_mm(
                SESSION.workspace_definition, drop_target_mm[0], drop_target_mm[1], approach_mm
            )
            await robot_api.move_to_cartesian_target_linear(robot_api.build_target_cpos(**drop_approach))
            await robot_api.move_to_cartesian_target_linear(robot_api.build_target_cpos(**drop_pose))
            if gripper_path:
                await robot_api.set_param(gripper_path, gripper_open_val)
            await robot_api.move_to_cartesian_target_linear(robot_api.build_target_cpos(**drop_approach))
            await robot_api.power_off()

        async def _pick_drop_action(robot_api: CodroidAPI) -> None:
            await _pick_action(robot_api)
            drop_target_mm = _resolve_drop_target()
            await _drop_action(robot_api, drop_target_mm)

        try:
            if trigger == "execute-pick":
                asyncio.run(_with_robot_api(
                    robot_uri or DEFAULT_ROBOT_URI, _pick_action))
                SESSION.last_action = "Pick command sent."
                return "Pick command sent."
            if trigger == "execute-drop":
                drop_target_mm = _resolve_drop_target()

                async def _drop_only(robot_api: CodroidAPI) -> None:
                    await _drop_action(robot_api, drop_target_mm)

                asyncio.run(_with_robot_api(
                    robot_uri or DEFAULT_ROBOT_URI, _drop_only))
                SESSION.last_action = "Drop command sent."
                return "Drop command sent."
            if trigger == "execute-pick-drop":
                asyncio.run(_with_robot_api(
                    robot_uri or DEFAULT_ROBOT_URI, _pick_drop_action))
                SESSION.last_action = "Pick & drop command sent."
                return "Pick & drop command sent."
        except Exception as exc:
            return f"Robot action failed: {exc}"

        return "Awaiting pick & place action."

    @app.callback(
        Output("monitor-system-summary", "children"),
        Output("monitor-detection-summary", "children"),
        Output("monitor-pick-summary", "children"),
        Input("camera-interval", "n_intervals"),
        State("camera-connection-state", "data"),
        State("robot-connection-state", "data"),
    )
    def _update_monitoring_summary(
        _ticks: int,
        camera_state: Optional[dict],
        robot_state: Optional[dict],
    ) -> Tuple[list, list, list]:
        camera_status = camera_state.get("status") if camera_state else "off"
        robot_status = robot_state.get("status") if robot_state else "offline"
        calibration_state = "loaded" if SESSION.calibration_bundle is not None else "missing"
        workspace_state = "ready" if SESSION.workspace_definition is not None else "pending"
        plate_state = "ready" if SESSION.plate_calibration is not None else "pending"
        model_state = SESSION.active_model_path.name if SESSION.active_model_path else "none"
        detection_count = len(SESSION.detections)
        last_action = SESSION.last_action or "—"

        system_items = [
            dmc.Text(f"Camera: {camera_status}", size="sm"),
            dmc.Text(f"Robot: {robot_status}", size="sm"),
            dmc.Text(f"Calibration bundle: {calibration_state}", size="sm"),
            dmc.Text(f"Workspace: {workspace_state}", size="sm"),
            dmc.Text(f"Plate calibration: {plate_state}", size="sm"),
        ]
        detection_items = [
            dmc.Text(f"Model: {model_state}", size="sm"),
            dmc.Text(f"Detections: {detection_count}", size="sm"),
        ]
        if SESSION.last_detection_error:
            detection_items.append(
                dmc.Text(f"Last error: {SESSION.last_detection_error}", size="sm", c="red"))
        pick_items = [
            dmc.Text(f"Drop marker ID: {SESSION.drop_marker_id}", size="sm"),
            dmc.Text(f"Last action: {last_action}", size="sm"),
        ]
        if SESSION.last_pick_mm:
            pick_items.append(
                dmc.Text(
                    f"Last pick: {SESSION.last_pick_mm[0]:.1f}, {SESSION.last_pick_mm[1]:.1f} mm",
                    size="sm",
                )
            )
        if SESSION.drop_target_mm:
            pick_items.append(
                dmc.Text(
                    f"Drop target: {SESSION.drop_target_mm[0]:.1f}, {SESSION.drop_target_mm[1]:.1f} mm",
                    size="sm",
                )
            )
        return system_items, detection_items, pick_items

    @app.callback(
        Output("app-notifications", "sendNotifications"),
        Output("notification-state", "data"),
        Input("camera-connection-state", "data"),
        Input("robot-connection-state", "data"),
        State("notification-state", "data"),
    )
    def _notify_connections(
        camera_state: Optional[dict],
        robot_state: Optional[dict],
        previous_state: Optional[dict],
    ) -> Tuple[list, dict]:
        prev = previous_state or {"camera": None, "robot": None}
        current = {"camera": prev.get("camera"), "robot": prev.get("robot")}
        notifications = []

        def _append_notification(name: str, state: Optional[dict]) -> None:
            if not state:
                return
            status = state.get("status")
            if prev.get(name) == status:
                return
            current[name] = status
            color = "teal"
            title = f"{name.capitalize()} connected"
            if status in ("error", "off"):
                color = "red" if status == "error" else "gray"
                title = f"{name.capitalize()} disconnected"
            notifications.append(
                {
                    "action": "show",
                    "id": f"{name}-connection",
                    "title": title,
                    "message": state.get("message", ""),
                    "color": color,
                    "autoClose": 4500,
                }
            )

        _append_notification("camera", camera_state)
        _append_notification("robot", robot_state)

        if not notifications:
            return no_update, prev
        return notifications, current

    @app.callback(
        Output("camera-calibration-action", "children"),
        Input("capture-frame", "n_clicks"),
        Input("capture-depth", "n_clicks"),
        Input("compute-calibration", "n_clicks"),
        Input("load-calibration", "n_clicks"),
        State("calibration-output-dir", "value"),
        State("calibration-board-path", "value"),
        State("calibration-bundle-path", "value"),
        prevent_initial_call=True,
    )
    def _handle_camera_calibration_actions(
        _capture_frame: int,
        _capture_depth: int,
        _compute: int,
        _load: int,
        output_dir: str,
        board_path: str,
        bundle_path: str,
    ) -> str:
        trigger = callback_context.triggered_id
        output_path = _safe_path(output_dir, DEFAULT_CAPTURE_DIR)
        board_path = _safe_path(board_path, DEFAULT_BOARD_PATH)
        bundle_path = _safe_path(bundle_path, DEFAULT_BUNDLE_PATH)
        try:
            if trigger == "capture-frame":
                captured = capture_color_frame(output_path)
                return f"Captured {captured.name}."
            if trigger == "capture-depth":
                captured = capture_depth_snapshot(output_path)
                return f"Saved depth snapshot to {captured.name}."
            if trigger == "compute-calibration":
                return compute_calibration_bundle(board_path, bundle_path, output_path)
            if trigger == "load-calibration":
                return load_calibration_bundle(bundle_path)
        except Exception as exc:
            return f"Calibration action failed: {exc}"
        return "Awaiting calibration action."

    @app.callback(
        Output("workspace-action-status", "children"),
        Input("compute-workspace", "n_clicks"),
        State("marker-id-a", "value"),
        State("marker-id-b", "value"),
        State("marker-size", "value"),
        prevent_initial_call=True,
    )
    def _handle_workspace_action(
        _clicks: int,
        marker_a: Optional[float],
        marker_b: Optional[float],
        marker_size: Optional[float],
    ) -> str:
        marker_ids = (
            _safe_int(marker_a, SESSION.config.marker_ids[0]),
            _safe_int(marker_b, SESSION.config.marker_ids[1]),
        )
        marker_size_mm = _safe_float(
            marker_size, SESSION.config.marker_size_mm)
        try:
            ws = compute_workspace_from_latest(marker_ids, marker_size_mm)
            return f"Workspace locked: {ws.width_mm:.1f} x {ws.height_mm:.1f} mm."
        except Exception as exc:
            return f"Workspace detection failed: {exc}"

    @app.callback(
        Output("plate-action-status", "children"),
        Input("compute-plate", "n_clicks"),
        Input("attach-plate", "n_clicks"),
        State("plate-layout-path", "value"),
        State("calibration-bundle-path", "value"),
        prevent_initial_call=True,
    )
    def _handle_plate_action(
        _compute_clicks: int,
        _attach_clicks: int,
        layout_path: str,
        bundle_path: str,
    ) -> str:
        trigger = callback_context.triggered_id
        layout_path = _safe_path(layout_path, DEFAULT_PLATE_PATH)
        bundle_path = _safe_path(bundle_path, DEFAULT_BUNDLE_PATH)
        try:
            if trigger == "compute-plate":
                result = compute_plate_calibration_from_latest(layout_path)
                return f"Plate detected. Error {result.reprojection_error_mm:.2f} mm."
            if trigger == "attach-plate":
                return attach_plate_calibration(bundle_path, layout_path)
        except Exception as exc:
            return f"Plate calibration failed: {exc}"
        return "Awaiting plate action."

    @app.callback(
        Output("overlay-status", "children"),
        Output("camera-calibration-metrics", "children"),
        Output("workspace-metrics", "children"),
        Output("plate-metrics", "children"),
        Output("detection-metrics", "children"),
        Output("drop-marker-metrics", "children"),
        Input("camera-interval", "n_intervals"),
        State("main-tabs", "value"),
        State("calibration-stepper", "active"),
        State("calibration-board-path", "value"),
        State("marker-id-a", "value"),
        State("marker-id-b", "value"),
        State("marker-size", "value"),
        State("plate-layout-path", "value"),
        State("drop-marker-id", "value"),
    )
    def _update_camera_view(
        _ticks: int,
        active_tab: Optional[str],
        active_step: Optional[int],
        board_path: str,
        marker_a: Optional[float],
        marker_b: Optional[float],
        marker_size: Optional[float],
        plate_layout: str,
        drop_marker_id: Optional[float],
    ) -> Tuple[str, str, str, str, str, str]:
        frame, _depth = get_latest_frame()
        if frame is None:
            placeholder = _placeholder_frame("Camera stream offline")
            MJPEG_STREAM.add_frame(placeholder)
            return (
                "Camera stream is stopped.",
                "ChArUco corners: --",
                "Workspace: --",
                "Plate error: --",
                "Detections: --",
                "Drop marker: --",
            )

        step = active_step or 0
        tab = active_tab or "calibration"
        board_path = _safe_path(board_path, DEFAULT_BOARD_PATH)
        plate_path = _safe_path(plate_layout, DEFAULT_PLATE_PATH)
        marker_ids = (
            _safe_int(marker_a, SESSION.config.marker_ids[0]),
            _safe_int(marker_b, SESSION.config.marker_ids[1]),
        )
        marker_size_mm = _safe_float(
            marker_size, SESSION.config.marker_size_mm)
        drop_marker = _safe_int(drop_marker_id, SESSION.drop_marker_id)

        if tab == "calibration":
            if step == 0:
                vis, status, metric = draw_charuco_overlay(frame, board_path)
                MJPEG_STREAM.add_frame(vis)
                return (
                    status,
                    metric,
                    "Workspace: --",
                    "Plate error: --",
                    "Detections: --",
                    "Drop marker: --",
                )
            if step == 1:
                vis, status, metric = draw_workspace_overlay(
                    frame, marker_ids, marker_size_mm)
                MJPEG_STREAM.add_frame(vis)
                return (
                    status,
                    "ChArUco corners: --",
                    metric,
                    "Plate error: --",
                    "Detections: --",
                    "Drop marker: --",
                )
            vis, status, metric = draw_plate_overlay(frame, plate_path)
            MJPEG_STREAM.add_frame(vis)
            return (
                status,
                "ChArUco corners: --",
                "Workspace: --",
                metric,
                "Detections: --",
                "Drop marker: --",
            )

        if tab == "detection":
            vis, status, metric = draw_detection_overlay(frame)
            MJPEG_STREAM.add_frame(vis)
            return (
                status,
                "ChArUco corners: --",
                "Workspace: --",
                "Plate error: --",
                metric,
                "Drop marker: --",
            )

        if tab == "pick-place":
            vis, status, metric = draw_drop_marker_overlay(frame, drop_marker)
            MJPEG_STREAM.add_frame(vis)
            return (
                status,
                "ChArUco corners: --",
                "Workspace: --",
                "Plate error: --",
                "Detections: --",
                metric,
            )

        vis, status, detection_metric, drop_metric = draw_monitor_overlay(
            frame, drop_marker)
        MJPEG_STREAM.add_frame(vis)
        return (
            status,
            "ChArUco corners: --",
            "Workspace: --",
            "Plate error: --",
            detection_metric,
            drop_metric,
        )


def create_app() -> Dash:
    app = Dash(__name__)
    app.title = "Codroid Calibration Console"
    app.layout = build_layout()
    register_callbacks(app)

    @app.server.route("/video_feed")
    def video_feed():
        """Stream Motion JPEG video for fullscreen playback."""
        def generate():
            boundary = b'--frame\r\n'
            while True:
                frame_data = MJPEG_STREAM.get_frame()
                if frame_data is None:
                    blank = np.zeros((720, 1280, 3), dtype=np.uint8)
                    success, buffer = cv2.imencode(".jpg", blank)
                    frame_data = buffer.tobytes() if success else b''

                if frame_data:
                    yield boundary
                    yield b'Content-Type: image/jpeg\r\n'
                    yield b'Content-Length: ' + str(len(frame_data)).encode() + b'\r\n\r\n'
                    yield frame_data
                    yield b'\r\n'
                else:
                    time.sleep(0.033)

        return app.server.response_class(
            generate(),
            mimetype='multipart/x-mixed-replace; boundary=frame'
        )

    return app


app = create_app()
server = app.server


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    app.run(debug=True)
