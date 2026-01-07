"""
Dash calibration UI for the Codroid bring-up workflow.
"""

from __future__ import annotations
from vision.stream import DepthAIFrameStream, StreamConfig
from vision.camera_check import list_oak_devices
from vision import aruco_board, calibration, markers, plate, workspace
from onrobot.twofg import TWOFG as TwoFG7Gripper
from onrobot.device import Device as GripperDevice
from codroid_api import CodroidAPI, CodroidSettings, RobotSession
from app.orchestrator import BootstrapDemo, DemoConfig

import asyncio
import base64
import io
import json
import logging
import os
import re
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Awaitable, Callable, Deque, Optional, Tuple

import cv2
import numpy as np
from dash import Dash, Input, Output, State, callback_context, dcc, html, no_update
import dash_mantine_components as dmc

LOGGER = logging.getLogger(__name__)


try:
    import pynvml
    NVML_AVAILABLE = True
except ImportError:
    NVML_AVAILABLE = False

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

DEFAULT_BOARD_PATH = Path(
    "data/markers/camera_calibration/camera_calibration_board.json")
DEFAULT_PLATE_PATH = Path(
    "data/markers/robot_calibration/calibration_plate_layout.json")
DEFAULT_BUNDLE_PATH = Path("data/calibration_bundle.npz")
DEFAULT_WORKSPACE_PATH = Path("data/workspace_definition.npz")
DEFAULT_ROBOT_CALIBRATION_PATH = Path("data/robot_calibration.json")
DEFAULT_CAPTURE_DIR = Path("data/calibration")
DEFAULT_RESOLUTION = (1920, 1080)
DEFAULT_FPS = 30.0
DEFAULT_MODEL_DIR = Path("data/models")
DEFAULT_DROP_MARKER_ID = 30
ROBOT_COORDINATE_ID = 5
MIN_PLATE_MARKERS = 2
PLATE_MOVEMENT_THRESHOLD_PX = 500.0
PLATE_VISIBILITY_MAX_AGE_S = 5.0
ROBOT_POSTURE_MAX_AGE_S = 2.0
VERIFICATION_RAISE_MM = 300.0
ROBOT_COORDINATE_TIMEOUT_S = 60.0
VERIFICATION_HOLD_SECONDS = 5.0
FLANGE_BUTTON_PORT = 41
FLANGE_BUTTON_PORTS = (40, 41, 42, 43)
GRIPPER_FORCE_MIN = 20
GRIPPER_FORCE_MAX = 140
GRIPPER_FORCE_DEFAULT = 80
GRIPPER_SPEED_DEFAULT = 80
GRIPPER_SPEED_MIN = 5
GRIPPER_CALIBRATION_WIDTH_MM = 5.0


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


def _load_gripper_defaults() -> dict:
    """Load OnRobot gripper default configuration."""
    return {
        "compute_box_ip": os.getenv("ONROBOT_CB_IP", "192.168.101.180"),
        "t_index": int(os.getenv("ONROBOT_T_INDEX", "0")),
    }


GRIPPER_DEFAULTS = _load_gripper_defaults()
DEFAULT_GRIPPER_IP = GRIPPER_DEFAULTS["compute_box_ip"]


@dataclass
class CaptureState:
    output_dir: Path = DEFAULT_CAPTURE_DIR
    frames: list[Path] = field(default_factory=list)
    depth_frame: Optional[np.ndarray] = None


@dataclass
class UISession:
    config: DemoConfig = field(default_factory=DemoConfig)
    camera_stream: Optional[DepthAIFrameStream] = None
    gripper_device: Optional[GripperDevice] = None
    gripper: Optional[TwoFG7Gripper] = None
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
    robot_session: Optional[RobotSession] = None
    robot_connection_attempted: bool = False
    robot_last_connection_error: Optional[str] = None
    last_pick_mm: Optional[Tuple[float, float]] = None
    last_action: Optional[str] = None
    last_frame_id: int = 0  # Track frame changes to avoid duplicate stream pushes
    continuous_detection: bool = False  # Enable continuous detection mode
    detection_confidence: float = 0.7
    detection_iou: float = 0.45
    detection_max: int = 10
    detection_classes: Optional[str] = None
    gripper_sync: dict = field(default_factory=dict)
    robot_speed_rate: float = 0.25
    robot_speed_last_set: float = 0.0
    robot_calibration_points: list[dict] = field(default_factory=list)
    robot_calibration_min_z: Optional[float] = None
    robot_calibration_baseline_px: Optional[np.ndarray] = None
    robot_calibration_gripper_width: Optional[float] = None


SESSION = UISession(
    config=DemoConfig(
        robot_host=CODROID_DEFAULTS["host"],
        robot_port=CODROID_DEFAULTS["robot_port"],
        user_port=CODROID_DEFAULTS["ws_port"],
    )
)

ROOT_DIR = Path(__file__).resolve().parent.parent
ASSETS_DIR = ROOT_DIR / "assets"


def _get_robot_session() -> RobotSession:
    if SESSION.robot_session is None:
        SESSION.robot_session = RobotSession(
            CodroidSettings(),
            flange_button_port=FLANGE_BUTTON_PORT,
            flange_button_ports=FLANGE_BUTTON_PORTS,
        )
    return SESSION.robot_session


def _set_robot_guidance(
    message: str,
    *,
    step: Optional[int] = None,
    status: Optional[str] = None,
    verification_ready: Optional[bool] = None,
) -> None:
    ROBOT_CALIBRATION_GUIDANCE["message"] = message
    if step is not None:
        ROBOT_CALIBRATION_GUIDANCE["step"] = step
    if status is not None:
        ROBOT_CALIBRATION_GUIDANCE["status"] = status
    if verification_ready is not None:
        ROBOT_CALIBRATION_GUIDANCE["verification_ready"] = verification_ready
    ROBOT_CALIBRATION_GUIDANCE["timestamp"] = time.time()


def _load_robot_calibration(path: Path = DEFAULT_ROBOT_CALIBRATION_PATH) -> None:
    if not path.exists():
        return
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        LOGGER.warning("Failed to parse robot calibration data: %s", exc)
        return
    points = data.get("points", [])
    if isinstance(points, list):
        SESSION.robot_calibration_points = points
    min_z = data.get("min_z")
    if isinstance(min_z, (int, float)):
        SESSION.robot_calibration_min_z = float(min_z)
    elif SESSION.robot_calibration_points:
        SESSION.robot_calibration_min_z = min(
            float(pt.get("z", 0.0)) for pt in SESSION.robot_calibration_points
        )
    LOGGER.info("Loaded robot calibration data from %s", path)


def _save_robot_calibration(points: list[dict], path: Path = DEFAULT_ROBOT_CALIBRATION_PATH) -> None:
    min_z = None
    if points:
        min_z = min(float(pt.get("z", 0.0)) for pt in points)
    payload = {
        "points": points,
        "min_z": min_z,
        "timestamp": time.time(),
        "coordinate_id": ROBOT_COORDINATE_ID,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def _update_robot_calibration_meta(robot_points: list[dict]) -> None:
    SESSION.robot_calibration_points = list(robot_points)
    if robot_points:
        SESSION.robot_calibration_min_z = min(
            float(pt.get("z", 0.0)) for pt in robot_points
        )
    else:
        SESSION.robot_calibration_min_z = None
    _save_robot_calibration(robot_points)


def _robot_posture_ready(
    max_age_s: float = ROBOT_POSTURE_MAX_AGE_S,
) -> Tuple[bool, str]:
    session = _get_robot_session()
    if not session.posture_seen():
        return False, "Robot posture stream not available yet."
    age = session.position_snapshot().age_seconds()
    if age > max_age_s:
        return False, f"Robot posture stale ({age:.1f}s old)."
    return True, "Robot posture stream healthy."


def _plate_center_px(detections: dict) -> Optional[np.ndarray]:
    if not detections:
        return None
    centers = [det.center for det in detections.values()]
    if not centers:
        return None
    return np.mean(np.asarray(centers, dtype=np.float64), axis=0)


def _plate_is_stable(
    max_age_s: float = PLATE_VISIBILITY_MAX_AGE_S,
    require_baseline: bool = False,
    allow_lost: bool = False,
) -> Tuple[bool, str]:
    if require_baseline and SESSION.robot_calibration_baseline_px is None:
        return False, "Calibration plate baseline not set."
    if not PLATE_STATUS.get("ok"):
        if allow_lost and SESSION.robot_calibration_baseline_px is not None:
            return True, "Plate markers not visible; waiting for re-acquire."
        return False, "Calibration plate markers are not visible."
    movement_px = PLATE_STATUS.get("movement_px")
    if movement_px is not None and movement_px > PLATE_MOVEMENT_THRESHOLD_PX:
        return False, f"Calibration plate moved ({movement_px:.1f}px)."
    if PLATE_STATUS.get("stable") is False:
        return False, PLATE_STATUS.get("message", "Calibration plate not stable.")
    return True, PLATE_STATUS.get("message", "Plate markers stable.")

PLATE_STATUS = {
    "count": 0,
    "ok": False,
    "stable": False,
    "center_px": None,
    "movement_px": None,
    "timestamp": time.time(),
    "message": "Waiting for plate markers",
}
ROBOT_CALIBRATION_GUIDANCE = {
    "status": "idle",
    "step": 0,
    "message": "Robot calibration idle. Click 'Begin calibrating robot' to start.",
    "verification_ready": False,
    "timestamp": time.time(),
}


def _restore_persisted_state() -> None:
    if SESSION.calibration_bundle is None and DEFAULT_BUNDLE_PATH.exists():
        try:
            SESSION.calibration_bundle = calibration.load_calibration(
                DEFAULT_BUNDLE_PATH)
            LOGGER.info("Loaded calibration bundle from %s",
                        DEFAULT_BUNDLE_PATH)
        except Exception as exc:
            LOGGER.warning(
                "Failed to load calibration bundle from %s: %s",
                DEFAULT_BUNDLE_PATH,
                exc,
            )
    if SESSION.workspace_definition is None and DEFAULT_WORKSPACE_PATH.exists():
        try:
            SESSION.workspace_definition = workspace.load_workspace(
                DEFAULT_WORKSPACE_PATH)
            LOGGER.info("Loaded workspace definition from %s",
                        DEFAULT_WORKSPACE_PATH)
        except Exception as exc:
            LOGGER.warning(
                "Failed to load workspace definition from %s: %s",
                DEFAULT_WORKSPACE_PATH,
                exc,
            )
    if DEFAULT_ROBOT_CALIBRATION_PATH.exists():
        _load_robot_calibration(DEFAULT_ROBOT_CALIBRATION_PATH)


_restore_persisted_state()

# Global calibration progress tracker
CALIBRATION_PROGRESS = {"value": 0, "current_frame": 0, "max_frames": 0}


class MotionJPEGStream:
    """Manager for Motion JPEG video streaming for fullscreen video playback."""

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
        """Add a frame to the stream queue, dropping frames if pushing too fast."""
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
            # Keep only recent frames
            if len(self.frame_queue) > self.max_queue_size:
                self.frame_queue.pop(0)

    def get_frame(self) -> Optional[bytes]:
        """Get the most recent frame from the queue."""
        with self.lock:
            if self.frame_queue:
                return self.frame_queue[-1]
        return None

    def get_fps(self) -> float:
        with self.lock:
            return self._fps


@dataclass
class OverlayResult:
    vis: np.ndarray
    status: str
    metric: str


@dataclass
class OverlayTask:
    key: str
    runner: Callable[[np.ndarray], OverlayResult]


class FrameProcessor:
    """Background processor that keeps only the newest request and result."""

    def __init__(self, max_queue: int = 2) -> None:
        self._queue: Deque[tuple[str, np.ndarray,
                                 Callable[[np.ndarray], OverlayResult]]]
        self._queue = deque(maxlen=max_queue)
        self._lock = threading.Lock()
        self._latest: dict[str, OverlayResult] = {}
        self._running = True
        self._thread = threading.Thread(
            target=self._worker, name="FrameProcessor", daemon=True
        )
        self._thread.start()

    def submit(self, task: OverlayTask, frame: np.ndarray) -> None:
        # Drop older pending tasks and keep the newest request only.
        with self._lock:
            self._queue.clear()
            self._queue.append((task.key, frame.copy(), task.runner))

    def get_latest(self, key: str) -> Optional[OverlayResult]:
        with self._lock:
            result = self._latest.get(key)
            if result is None:
                return None
            # Copy the image so callers cannot mutate internal state.
            return OverlayResult(result.vis.copy(), result.status, result.metric)

    def stop(self) -> None:
        self._running = False
        if self._thread.is_alive():
            self._thread.join(timeout=1.0)

    def _worker(self) -> None:
        while self._running:
            task: Optional[tuple[str, np.ndarray,
                                 Callable[[np.ndarray], OverlayResult]]] = None
            with self._lock:
                if self._queue:
                    # Pop newest to drop stale work.
                    task = self._queue.pop()
            if task is None:
                time.sleep(0.003)
                continue
            key, frame, runner = task
            try:
                result = runner(frame)
            except Exception as exc:  # noqa: BLE001
                logging.error("FrameProcessor error for %s: %s",
                              key, exc, exc_info=True)
                result = OverlayResult(frame, f"Processing error: {exc}", "--")
            with self._lock:
                self._latest[key] = result


def _format_fps(capture_fps: float, stream_fps: float) -> str:
    cap_text = f"{capture_fps:.1f}" if capture_fps > 0 else "--"
    stream_text = f"{stream_fps:.1f}" if stream_fps > 0 else "--"
    return f"FPS — capture: {cap_text}, stream: {stream_text}"


MJPEG_STREAM = MotionJPEGStream()
FRAME_PROCESSOR = FrameProcessor()
RAW_STREAM = MotionJPEGStream()


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


def _recent_gripper_sync(
    key: str,
    value: Optional[float],
    *,
    window_s: float = 0.75,
) -> bool:
    sync_data = SESSION.gripper_sync
    if not sync_data or value is None:
        return False
    try:
        updated = float(sync_data.get("updated", 0.0))
        if time.time() - updated > window_s:
            return False
        synced_value = sync_data.get(key)
        if synced_value is None:
            return False
        return abs(float(value) - float(synced_value)) < 0.01
    except (TypeError, ValueError):
        return False


def _record_gripper_sync(width: Optional[float], force: Optional[float]) -> None:
    payload = dict(SESSION.gripper_sync)
    payload["updated"] = time.time()
    if width is not None:
        payload["width"] = width
    if force is not None:
        payload["force"] = force
    SESSION.gripper_sync = payload


def get_gpu_info() -> dict:
    """Get GPU status information including availability, clock speed, usage, and memory.

    Supports CUDA (NVIDIA), Metal Performance Shaders (Apple Silicon), and CPU fallback.
    """
    info = {
        "available": False,
        "name": "N/A",
        "driver_version": "N/A",
        "cuda_available": False,
        "cuda_version": "N/A",
        "mps_available": False,
        "gpu_count": 0,
        "devices": [],
    }

    # Check CUDA availability via PyTorch
    if TORCH_AVAILABLE:
        info["cuda_available"] = torch.cuda.is_available()
        if info["cuda_available"]:
            info["cuda_version"] = torch.version.cuda or "N/A"
            info["gpu_count"] = torch.cuda.device_count()
            info["available"] = True

        # Check for Metal Performance Shaders (Apple Silicon)
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            info["mps_available"] = True
            info["available"] = True
            info["name"] = "Metal Performance Shaders"
            info["gpu_count"] = 1

    # Get detailed GPU info via NVML (NVIDIA only)
    if not NVML_AVAILABLE or not info["cuda_available"]:
        return info

    try:
        pynvml.nvmlInit()
        device_count = pynvml.nvmlDeviceGetCount()
        info["available"] = device_count > 0

        for i in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            name = pynvml.nvmlDeviceGetName(handle)
            if isinstance(name, bytes):
                name = name.decode("utf-8")

            # Get memory info
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            mem_used_mb = mem_info.used / (1024 ** 2)
            mem_total_mb = mem_info.total / (1024 ** 2)
            mem_percent = (mem_info.used / mem_info.total) * \
                100 if mem_info.total > 0 else 0

            # Get utilization
            try:
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                gpu_util = util.gpu
            except Exception:
                gpu_util = 0

            # Get clock speeds
            try:
                clock_graphics = pynvml.nvmlDeviceGetClockInfo(
                    handle, pynvml.NVML_CLOCK_GRAPHICS)
                clock_mem = pynvml.nvmlDeviceGetClockInfo(
                    handle, pynvml.NVML_CLOCK_MEM)
            except Exception:
                clock_graphics = 0
                clock_mem = 0

            # Get temperature
            try:
                temp = pynvml.nvmlDeviceGetTemperature(
                    handle, pynvml.NVML_TEMPERATURE_GPU)
            except Exception:
                temp = 0

            # Get power usage
            try:
                power = pynvml.nvmlDeviceGetPowerUsage(
                    handle) / 1000.0  # mW to W
            except Exception:
                power = 0

            device_info = {
                "index": i,
                "name": name,
                "memory_used_mb": round(mem_used_mb, 1),
                "memory_total_mb": round(mem_total_mb, 1),
                "memory_percent": round(mem_percent, 1),
                "gpu_utilization": gpu_util,
                "clock_graphics_mhz": clock_graphics,
                "clock_memory_mhz": clock_mem,
                "temperature_c": temp,
                "power_watts": round(power, 1),
            }
            info["devices"].append(device_info)

            if i == 0:
                info["name"] = name
                try:
                    driver_version = pynvml.nvmlSystemGetDriverVersion()
                    if isinstance(driver_version, bytes):
                        driver_version = driver_version.decode("utf-8")
                    info["driver_version"] = driver_version
                except Exception:
                    pass

        pynvml.nvmlShutdown()
    except Exception as exc:
        # Only log at debug level for expected failures on non-NVIDIA platforms
        LOGGER.debug(
            "NVML GPU detection failed (expected on non-NVIDIA systems): %s", exc)

    return info


def format_gpu_status() -> list:
    """Format GPU status for display in the UI."""
    gpu_info = get_gpu_info()

    if not gpu_info["available"]:
        return [
            dmc.Alert(
                "No NVIDIA GPU detected or NVML not available",
                color="yellow",
                title="GPU Status",
            )
        ]

    elements = []

    # Overall status
    cuda_status = "✓ Available" if gpu_info["cuda_available"] else "✗ Not available"
    elements.append(
        dmc.Group(
            [
                dmc.Badge(
                    f"CUDA: {cuda_status}", color="green" if gpu_info["cuda_available"] else "red"),
                dmc.Badge(
                    f"Driver: {gpu_info['driver_version']}", color="blue"),
                dmc.Badge(f"{gpu_info['gpu_count']} GPU(s)", color="teal"),
            ],
            gap="xs",
        )
    )

    # Per-device info
    for dev in gpu_info["devices"]:
        elements.append(
            dmc.Card(
                [
                    dmc.Text(
                        f"GPU {dev['index']}: {dev['name']}", fw=600, size="sm"),
                    dmc.Group(
                        [
                            dmc.Stack(
                                [
                                    dmc.Text("Memory", size="xs", c="dimmed"),
                                    dmc.Text(
                                        f"{dev['memory_used_mb']:.0f} / {dev['memory_total_mb']:.0f} MB ({dev['memory_percent']:.1f}%)",
                                        size="sm",
                                    ),
                                ],
                                gap=2,
                            ),
                            dmc.Stack(
                                [
                                    dmc.Text(
                                        "GPU Usage", size="xs", c="dimmed"),
                                    dmc.Text(
                                        f"{dev['gpu_utilization']}%", size="sm"),
                                ],
                                gap=2,
                            ),
                            dmc.Stack(
                                [
                                    dmc.Text("Clock", size="xs", c="dimmed"),
                                    dmc.Text(
                                        f"{dev['clock_graphics_mhz']} MHz", size="sm"),
                                ],
                                gap=2,
                            ),
                            dmc.Stack(
                                [
                                    dmc.Text("Temp", size="xs", c="dimmed"),
                                    dmc.Text(
                                        f"{dev['temperature_c']}°C", size="sm"),
                                ],
                                gap=2,
                            ),
                            dmc.Stack(
                                [
                                    dmc.Text("Power", size="xs", c="dimmed"),
                                    dmc.Text(
                                        f"{dev['power_watts']:.1f} W", size="sm"),
                                ],
                                gap=2,
                            ),
                        ],
                        gap="md",
                    ),
                ],
                withBorder=True,
                padding="sm",
                radius="md",
            )
        )

    return elements


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
    robot_xyz = _apply_robot_z_floor(robot_xyz)
    return {
        "x": float(robot_xyz[0]),
        "y": float(robot_xyz[1]),
        "z": float(robot_xyz[2]),
        "a": 0.0,
        "b": 0.0,
        "c": 0.0,
    }


def _apply_robot_z_floor(robot_xyz: np.ndarray) -> np.ndarray:
    min_z = SESSION.robot_calibration_min_z
    if min_z is None or robot_xyz[2] >= min_z:
        return robot_xyz
    adjusted = np.array(robot_xyz, dtype=np.float64, copy=True)
    adjusted[2] = min_z
    return adjusted


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

    # Crop to workspace region if workspace is defined
    crop_offset_x = 0
    crop_offset_y = 0
    detection_frame = frame

    if SESSION.workspace_definition is not None:
        try:
            corners = _workspace_corners_px(SESSION.workspace_definition)
            # Get bounding box of workspace region
            x_min = int(np.min(corners[:, 0]))
            y_min = int(np.min(corners[:, 1]))
            x_max = int(np.max(corners[:, 0]))
            y_max = int(np.max(corners[:, 1]))

            # Clamp to frame bounds
            x_min = max(0, x_min)
            y_min = max(0, y_min)
            x_max = min(frame.shape[1], x_max)
            y_max = min(frame.shape[0], y_max)

            # Crop frame to workspace region
            detection_frame = frame[y_min:y_max, x_min:x_max]
            crop_offset_x = x_min
            crop_offset_y = y_min
        except Exception as exc:
            logging.debug(
                "Could not crop to workspace, using full frame: %s", exc)

    results = SESSION.yolo_model.predict(
        detection_frame,
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
            # Adjust coordinates back to full frame
            x1, y1, x2, y2 = [float(val) for val in box]
            x1 += crop_offset_x
            y1 += crop_offset_y
            x2 += crop_offset_x
            y2 += crop_offset_y
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


def _rigid_transform(from_pts: np.ndarray, to_pts: np.ndarray) -> workspace.RigidTransform:
    """Compute best-fit rigid transform from from_pts -> to_pts."""
    if from_pts.shape != to_pts.shape:
        raise ValueError("Point clouds must have the same shape.")
    if from_pts.shape[0] < 3:
        raise ValueError(
            "At least 3 points are required for a rigid transform.")
    a = np.asarray(from_pts, dtype=np.float64)
    b = np.asarray(to_pts, dtype=np.float64)
    centroid_a = np.mean(a, axis=0)
    centroid_b = np.mean(b, axis=0)
    aa = a - centroid_a
    bb = b - centroid_b
    h = aa.T @ bb
    u, _s, vt = np.linalg.svd(h)
    r = vt.T @ u.T
    if np.linalg.det(r) < 0:
        vt[-1, :] *= -1
        r = vt.T @ u.T
    t = centroid_b - r @ centroid_a
    return workspace.RigidTransform(rotation=r, translation=t)


def _ensure_plate_visible(max_age_s: float = PLATE_VISIBILITY_MAX_AGE_S) -> None:
    ok, message = _plate_is_stable(
        max_age_s=max_age_s,
        require_baseline=False,
        allow_lost=False,
    )
    if not ok:
        raise RuntimeError(message)


def _ensure_plate_stable(
    max_age_s: float = PLATE_VISIBILITY_MAX_AGE_S,
    *,
    allow_lost: bool = False,
) -> None:
    ok, message = _plate_is_stable(
        max_age_s=max_age_s,
        require_baseline=True,
        allow_lost=allow_lost,
    )
    if not ok:
        raise RuntimeError(message)


def _publish_raw_frame(frame: np.ndarray) -> None:
    RAW_STREAM.add_frame(frame)


def start_camera_stream(width: Optional[float], height: Optional[float], fps: Optional[float], lock_af: bool = False) -> str:
    stop_camera_stream()
    resolution = (_safe_int(width, DEFAULT_RESOLUTION[0]), _safe_int(
        height, DEFAULT_RESOLUTION[1]))
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
    # Retry logic: frame might not be available immediately, so poll briefly
    max_retries = 5
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

    # Verify by reading back immediately
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


def _clear_calibration_outputs(output_dir: Path) -> None:
    if not output_dir.exists():
        return
    for pattern in ("calibration_color_*.png", "calibration_depth_frame.npz"):
        for path in output_dir.glob(pattern):
            try:
                path.unlink()
            except OSError:
                logging.warning(f"[CALIBRATE] Unable to remove {path}")


def _charuco_detection_counts(image_paths: list[Path], board_path: Path) -> Tuple[list[int], int]:
    spec = aruco_board.load_charuco_board_spec(board_path)
    board = spec.create_board()

    # Get dictionary for initial marker detection
    dictionary_id = getattr(cv2.aruco, spec.dictionary)
    dictionary = cv2.aruco.getPredefinedDictionary(dictionary_id)
    detector_params = cv2.aruco.DetectorParameters()
    detector_params.detectInvertedMarker = True  # Match live overlay behavior

    counts: list[int] = []
    usable_frames = 0
    for path in image_paths:
        image = cv2.imread(str(path))
        if image is None:
            counts.append(0)
            continue
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # First detect markers
        detector = cv2.aruco.ArucoDetector(dictionary, detector_params)
        marker_corners, marker_ids, _rejected = detector.detectMarkers(gray)

        # Then detect ChArUco corners if markers were found
        corner_count = 0
        if marker_ids is not None and len(marker_ids) > 0:
            charuco_params = cv2.aruco.CharucoParameters()
            charuco_detector = cv2.aruco.CharucoDetector(
                board, charuco_params, detector_params)
            charuco_corners, charuco_ids, _, _ = charuco_detector.detectBoard(
                gray)
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
    depth_frame = SESSION.capture.depth_frame if SESSION.capture.depth_frame is not None else load_depth_snapshot(
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
    save_workspace_definition(ws, DEFAULT_WORKSPACE_PATH)
    return ws


def save_workspace_definition(
    workspace_def: workspace.WorkspaceDefinition,
    workspace_path: Path = DEFAULT_WORKSPACE_PATH,
) -> str:
    workspace.save_workspace(workspace_def, workspace_path)
    return f"Workspace saved to {workspace_path}."


def load_workspace_definition(workspace_path: Path = DEFAULT_WORKSPACE_PATH) -> workspace.WorkspaceDefinition:
    if not workspace_path.exists():
        raise FileNotFoundError(
            f"Workspace definition not found: {workspace_path}")
    SESSION.workspace_definition = workspace.load_workspace(workspace_path)
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


async def _run_robot_plate_calibration(
    robot_uri: str,
    plate_layout: plate.CalibrationPlate,
) -> Tuple[list[dict], dict]:
    """Capture three robot poses via flange button 1 and calibrate coord ID 5."""
    session = _get_robot_session()

    async def _ensure_coordinate_id(
        robot_api: CodroidAPI,
        expected_id: int,
        timeout_s: float = 5.0,
        require: bool = True,
    ) -> None:
        await robot_api.set_current_coordinate_id(expected_id)
        deadline = time.monotonic() + timeout_s
        resent = False
        while time.monotonic() < deadline:
            status = session.robot_status_snapshot()
            coord_id = None
            if isinstance(status, dict):
                coord_id = status.get("CoordinateId")
            if coord_id is not None:
                try:
                    if int(coord_id) == expected_id:
                        return
                except (TypeError, ValueError):
                    pass
            if not resent and time.monotonic() > deadline - (timeout_s / 2):
                await robot_api.set_current_coordinate_id(expected_id)
                resent = True
            await asyncio.sleep(0.1)
        if not require:
            LOGGER.warning(
                "CoordinateId did not switch to %s within %.1fs (last=%s).",
                expected_id,
                timeout_s,
                coord_id,
            )
            return
        raise RuntimeError(
            "Calibration requires the robot to be on CoordinateId=0 (base). "
            f"Current CoordinateId={coord_id}. Switch to 0 and retry."
        )

    async def _await_flange_press_with_plate_guard(
        timeout_s: float,
    ) -> dict:
        deadline = time.monotonic() + timeout_s
        while True:
            plate_ok, plate_message = _plate_is_stable(
                require_baseline=True,
                allow_lost=True,
            )
            if not plate_ok:
                raise RuntimeError(f"Calibration canceled: {plate_message}")
            posture_ok, posture_message = _robot_posture_ready()
            if not posture_ok:
                raise RuntimeError(f"Calibration canceled: {posture_message}")
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise asyncio.TimeoutError()
            try:
                return await session.next_flange_press(timeout=min(0.5, remaining))
            except asyncio.TimeoutError:
                continue

    async def _runner(robot_api: CodroidAPI) -> Tuple[list[dict], dict]:
        _set_robot_guidance(
            "Selecting CoordinateId=0 (base) for calibration...",
            status="running",
            step=1,
            verification_ready=False,
        )
        await _ensure_coordinate_id(robot_api, 0)
        await session.reset_flange_queue()

        robot_points: list[dict] = []
        calib_points = plate_layout.robot_calibration_points()
        if len(calib_points) < 3:
            raise RuntimeError(
                "Calibration plate layout is missing calibration points.")

        point_labels = [
            "Point 1 (Origin)",
            "Point 2 (+X)",
            "Point 3 (+X/+Y)",
        ]
        for idx in range(3):
            _ensure_plate_stable(allow_lost=True)
            point_label = point_labels[idx] if idx < len(
                point_labels) else f"Point {idx + 1}"
            _set_robot_guidance(
                "Step {step}/3: Place the gripper fingers in indentation {number}. "
                "Keep the flange vertical. When aligned, press flange button 1.".format(
                    step=idx + 1,
                    number=idx + 1,
                ),
                step=idx + 1,
                status="running",
                verification_ready=False,
            )
            LOGGER.info(
                "Step %d/3: Waiting for gripper to align with %s...",
                idx + 1,
                point_label,
            )

            try:
                posture = await _await_flange_press_with_plate_guard(timeout_s=90.0)
            except asyncio.TimeoutError as exc:
                _set_robot_guidance(
                    f"Timed out waiting for flange button at {point_label}.",
                    status="error",
                    verification_ready=False,
                )
                raise RuntimeError(
                    f"Timed out waiting for flange button 1 press at {point_label}. "
                    f"Move gripper to the groove target and press the flange button."
                ) from exc

            LOGGER.info(
                "Captured %s. Position: X=%.1f, Y=%.1f, Z=%.1f",
                point_label,
                posture.get("x", 0.0),
                posture.get("y", 0.0),
                posture.get("z", 0.0),
            )
            robot_points.append(posture)
            if idx < 2:
                next_label = point_labels[idx + 1]
                _set_robot_guidance(
                    (
                        f"Captured {point_label}. "
                        "Next: place the gripper in indentation "
                        f"{idx + 2} and press flange button 1."
                    ),
                    step=idx + 1,
                    status="running",
                    verification_ready=False,
                )

        _set_robot_guidance(
            f"Switching to CoordinateId {ROBOT_COORDINATE_ID} for calibration...",
            status="running",
            verification_ready=False,
        )
        await _ensure_coordinate_id(
            robot_api, ROBOT_COORDINATE_ID, timeout_s=5.0, require=False
        )

        _set_robot_guidance(
            "Computing coordinate transform from 3 calibration points...",
            status="running",
            verification_ready=False,
        )
        LOGGER.info(
            "Computing coordinate transformation from 3 calibration points...")
        cpos_points = [
            CodroidAPI.build_target_cpos(
                x=float(pt["x"]),
                y=float(pt["y"]),
                z=float(pt["z"]),
                a=float(pt["a"]),
                b=float(pt["b"]),
                c=float(pt["c"]),
            )
            for pt in robot_points
        ]
        _set_robot_guidance(
            "Waiting for robot controller to return the calibrated frame...",
            status="running",
            verification_ready=False,
        )
        LOGGER.info("Requesting coordinate calibration from robot controller...")

        async def _await_listener_frame(
            snapshot: Callable[[], dict],
            timestamp: Callable[[], Optional[float]],
            since: float,
            timeout_s: float,
            label: str,
        ) -> dict:
            deadline = time.monotonic() + timeout_s
            while time.monotonic() < deadline:
                updated_at = timestamp()
                if updated_at is not None and updated_at >= since:
                    frame = snapshot()
                    if frame:
                        return frame
                await asyncio.sleep(0.1)
            raise asyncio.TimeoutError(f"Timed out waiting for {label}.")

        calibration_request_time = time.time()
        payload_points = [
            robot_api.attach_coordinate_to_cpos(
                point, coordinate_id=ROBOT_COORDINATE_ID
            )
            for point in cpos_points
        ]
        calibration_request = robot_api.build_message(
            message_type="Robot",
            action="CoordinateCalibration",
            data=payload_points,
        )
        await robot_api.send_message(calibration_request)

        try:
            coord_frame = await _await_listener_frame(
                session.calibration_frame_snapshot,
                session.calibration_frame_timestamp,
                calibration_request_time,
                ROBOT_COORDINATE_TIMEOUT_S,
                "coordinate calibration response",
            )
        except asyncio.TimeoutError as exc:
            raise RuntimeError(
                "Timed out waiting for coordinate calibration response."
            ) from exc

        if not coord_frame:
            raise RuntimeError("Robot controller returned empty calibration frame.")

        coord_frame["a"] = 0.0
        coord_frame["b"] = 0.0
        coord_frame["c"] = 0.0

        change_request_time = time.time()
        change_payload = {
            "id": ROBOT_COORDINATE_ID,
            "x": coord_frame["x"],
            "y": coord_frame["y"],
            "z": coord_frame["z"],
            "a": coord_frame["a"],
            "b": coord_frame["b"],
            "c": coord_frame["c"],
        }
        change_request = robot_api.build_message(
            message_type="Robot",
            action="ChangeCoordinateParameter",
            data=change_payload,
        )
        await robot_api.send_message(change_request)

        publish_timeout = min(ROBOT_COORDINATE_TIMEOUT_S, 5.0)
        try:
            await _await_listener_frame(
                session.coordinate_frame_snapshot,
                session.coordinate_frame_timestamp,
                change_request_time,
                publish_timeout,
                "coordinate frame publish",
            )
        except asyncio.TimeoutError:
            LOGGER.warning(
                "No RobotCoordinate publish after calibration; reselecting CoordinateId."
            )
            await robot_api.set_current_coordinate_id(ROBOT_COORDINATE_ID)
            try:
                await _await_listener_frame(
                    session.coordinate_frame_snapshot,
                    session.coordinate_frame_timestamp,
                    time.time(),
                    publish_timeout,
                    "coordinate frame publish after reselect",
                )
            except asyncio.TimeoutError:
                LOGGER.warning(
                    "Timed out waiting for coordinate frame publish after reselect."
                )
        _set_robot_guidance(
            "Coordinate frame updated. Preparing verification step...",
            status="running",
            step=3,
            verification_ready=False,
        )
        LOGGER.info("Robot calibration coordinate ID 5 updated.")
        return robot_points, coord_frame or {}

    return await session.run_with_robot(robot_uri, _runner)


def _apply_robot_calibration_results(
    robot_points: list[dict],
    plate_layout: plate.CalibrationPlate,
) -> workspace.RigidTransform:
    plate_pts = []
    for calib_point in plate_layout.robot_calibration_points()[:3]:
        plate_pts.append([float(calib_point.center_mm[0]),
                         float(calib_point.center_mm[1]), 0.0])
    robot_pts = []
    for pt in robot_points[:3]:
        robot_pts.append([float(pt["x"]), float(
            pt["y"]), float(pt["z"])])
    plate_pts_np = np.asarray(plate_pts, dtype=np.float64)
    robot_pts_np = np.asarray(robot_pts, dtype=np.float64)
    transform = _rigid_transform(plate_pts_np, robot_pts_np)

    if SESSION.workspace_definition is not None:
        ws = SESSION.workspace_definition
        updated_ws = workspace.WorkspaceDefinition(
            width_mm=ws.width_mm,
            height_mm=ws.height_mm,
            origin_camera=ws.origin_camera,
            x_axis_camera=ws.x_axis_camera,
            y_axis_camera=ws.y_axis_camera,
            camera_to_robot=transform,
        )
    else:
        # Fallback: use the plate dimensions as the workspace basis.
        updated_ws = workspace.WorkspaceDefinition(
            width_mm=float(plate_layout.plate_size_mm[0]),
            height_mm=float(plate_layout.plate_size_mm[1]),
            origin_camera=np.array([0.0, 0.0, 0.0]),
            x_axis_camera=np.array([1.0, 0.0, 0.0]),
            y_axis_camera=np.array([0.0, 1.0, 0.0]),
            camera_to_robot=transform,
        )

    SESSION.workspace_definition = updated_ws
    save_workspace_definition(updated_ws, DEFAULT_WORKSPACE_PATH)
    return transform


def _draw_label(frame: np.ndarray, text: str, origin: Tuple[int, int], color: Tuple[int, int, int]) -> None:
    cv2.putText(frame, text, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


def _workspace_corners_px(workspace_def: workspace.WorkspaceDefinition) -> np.ndarray:
    # Workspace definition stores everything in mm in camera coordinates
    # The axes are normalized direction vectors
    # We need calibration to convert mm back to pixels
    if SESSION.calibration_bundle is None:
        raise RuntimeError("Calibration bundle required to display workspace")

    scale = SESSION.calibration_bundle.pixel_to_mm
    if scale == 0:
        raise RuntimeError(
            "Workspace scale is zero; recompute workspace markers.")

    # Convert origin from mm to pixels
    origin_px = np.array([workspace_def.origin_camera[0],
                         workspace_def.origin_camera[1]]) / scale

    # Convert width and height from mm to pixels
    width_px = workspace_def.width_mm / scale
    height_px = workspace_def.height_mm / scale

    # Extract 2D axes (they are already normalized directions in mm-space)
    x_dir = np.array([workspace_def.x_axis_camera[0],
                     workspace_def.x_axis_camera[1]])
    y_dir = np.array([workspace_def.y_axis_camera[0],
                     workspace_def.y_axis_camera[1]])

    # Normalize to ensure they are unit vectors
    x_len = np.linalg.norm(x_dir)
    y_len = np.linalg.norm(y_dir)
    if x_len > 0:
        x_dir = x_dir / x_len
    if y_len > 0:
        y_dir = y_dir / y_len

    # Build rotated rectangle corners
    points = np.asarray(
        [
            origin_px,  # top-left
            origin_px + width_px * x_dir,  # top-right
            origin_px + width_px * x_dir + height_px * y_dir,  # bottom-right
            origin_px + height_px * y_dir,  # bottom-left
        ],
        dtype=np.float64,
    )
    return np.round(points).astype(int)


def _draw_dotted_polyline(
    frame: np.ndarray,
    points: np.ndarray,
    color: Tuple[int, int, int] = (0, 215, 255),
    thickness: int = 2,
    gap: int = 10,
) -> None:
    for idx in range(len(points)):
        start = points[idx]
        end = points[(idx + 1) % len(points)]
        vec = end - start
        length = float(np.linalg.norm(vec))
        if length == 0:
            continue
        direction = vec / length
        dist = 0.0
        while dist < length:
            segment_start = start + direction * dist
            dist = min(length, dist + gap)
            segment_end = start + direction * dist
            cv2.line(
                frame,
                tuple(np.round(segment_start).astype(int)),
                tuple(np.round(segment_end).astype(int)),
                color,
                thickness,
                lineType=cv2.LINE_AA,
            )
            dist += gap


def _apply_workspace_overlay(frame: np.ndarray) -> None:
    if SESSION.workspace_definition is None:
        return
    try:
        corners_px = _workspace_corners_px(SESSION.workspace_definition)
    except Exception as exc:
        logging.debug("Workspace overlay skipped: %s", exc)
        return
    _draw_dotted_polyline(frame, corners_px)


def draw_charuco_overlay(frame: np.ndarray, board_path: Path) -> Tuple[np.ndarray, str, str]:
    vis = frame.copy()
    if not board_path.exists():
        status = "ChArUco layout not found."
        return vis, status, "ChArUco corners: --"

    try:
        board_spec = aruco_board.load_charuco_board_spec(board_path)
        board = board_spec.create_board()
        gray = cv2.cvtColor(vis, cv2.COLOR_BGR2GRAY)

        # Get dictionary and detector
        dictionary_id = getattr(cv2.aruco, board_spec.dictionary)
        dictionary = cv2.aruco.getPredefinedDictionary(dictionary_id)
        detector_params = cv2.aruco.DetectorParameters()
        detector_params.detectInvertedMarker = True  # Enable inverted marker detection

        # Detect markers
        detector = cv2.aruco.ArucoDetector(dictionary, detector_params)
        marker_corners, marker_ids, rejected = detector.detectMarkers(gray)

        marker_count = 0 if marker_ids is None else len(marker_ids)

        if marker_ids is not None:
            vis = cv2.aruco.drawDetectedMarkers(
                vis, marker_corners, marker_ids)

        # Try ChArUco detection
        corner_count = 0

        if marker_ids is not None and len(marker_ids) > 0:
            charuco_params = cv2.aruco.CharucoParameters()
            charuco_detector = cv2.aruco.CharucoDetector(
                board, charuco_params, detector_params)
            charuco_corners, charuco_ids, _, _ = charuco_detector.detectBoard(
                gray)
            corner_count = 0 if charuco_ids is None else len(charuco_ids)
            if charuco_ids is not None:
                vis = cv2.aruco.drawDetectedCornersCharuco(
                    vis, charuco_corners, charuco_ids)

        _draw_label(vis, f"Markers: {marker_count}", (10, 30), (255, 255, 255))
        _draw_label(vis, f"Corners: {corner_count}", (10, 55), (255, 255, 255))
        status = f"Detected {marker_count} markers and {corner_count} corners."
        metric = f"ChArUco corners: {corner_count}"
        return vis, status, metric
    except Exception as exc:
        logging.error(f"ChArUco detection error: {exc}", exc_info=True)
        _draw_label(vis, f"Error: {str(exc)[:50]}", (10, 30), (255, 0, 0))
        return vis, f"Detection error: {exc}", "ChArUco corners: --"


# Global buffer for smoothing workspace rectangle (reduce jitter)
_workspace_rect_buffer = []
_WORKSPACE_SMOOTH_FRAMES = 5


def draw_workspace_overlay(
    frame: np.ndarray,
    marker_ids: Tuple[int, int],
    marker_size_mm: float,
) -> Tuple[np.ndarray, str, str]:
    global _workspace_rect_buffer
    vis = frame.copy()
    try:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect ArUco markers with inverted color support
        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        detector_params = cv2.aruco.DetectorParameters()
        detector_params.detectInvertedMarker = True

        detector = cv2.aruco.ArucoDetector(dictionary, detector_params)
        marker_corners, marker_ids_detected, _rejected = detector.detectMarkers(
            gray)

        if marker_ids_detected is None or len(marker_ids_detected) == 0:
            raise RuntimeError(
                "No markers detected. Ensure markers are in frame and high-contrast.")

        # Map detected markers by ID
        detected_map = {}
        for corners, mid in zip(marker_corners, marker_ids_detected.flatten()):
            pts = corners.reshape((4, 2))
            center = tuple(np.mean(pts, axis=0).tolist())
            detected_map[int(mid)] = {
                'corners': pts,
                'center': center,
                'id': int(mid)
            }

        # Check if we have both required markers
        if marker_ids[0] not in detected_map or marker_ids[1] not in detected_map:
            found_ids = sorted(detected_map.keys())
            raise RuntimeError(
                f"Missing required markers {marker_ids}. Found: {found_ids}")

        marker_a_center = None
        marker_b_center = None

        # Draw detected markers with colors
        colors = {marker_ids[0]: (0, 255, 0), marker_ids[1]: (0, 140, 255)}
        for mid, color in colors.items():
            if mid in detected_map:
                det = detected_map[mid]
                pts = det['corners'].astype(int)
                cv2.polylines(vis, [pts], isClosed=True,
                              color=color, thickness=2)
                center = tuple(np.round(det['center']).astype(int))
                cv2.circle(vis, center, 4, color, -1)
                cv2.putText(vis, f"ID {mid}", center,
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                if mid == marker_ids[0]:
                    marker_a_center = center
                elif mid == marker_ids[1]:
                    marker_b_center = center

        # Draw rotated rectangle between the two marker centers
        if marker_a_center is not None and marker_b_center is not None:
            # Use marker_a orientation to define the rectangle axes
            marker_a_corners = detected_map[marker_ids[0]]['corners']

            # X-axis: direction from left edge to right edge of marker_a
            # top-right - top-left
            x_axis_px = marker_a_corners[1] - marker_a_corners[0]
            x_axis_norm = x_axis_px / \
                np.linalg.norm(x_axis_px) if np.linalg.norm(
                    x_axis_px) > 0 else np.array([1.0, 0.0])

            # Y-axis: perpendicular to x-axis (rotate 90 degrees)
            y_axis_norm = np.array([-x_axis_norm[1], x_axis_norm[0]])

            # Diagonal from marker_a to marker_b
            diagonal = np.array(marker_b_center) - np.array(marker_a_center)

            # Project diagonal onto the rotated axes
            width_px = np.dot(diagonal, x_axis_norm)
            height_px = np.dot(diagonal, y_axis_norm)

            # Compute the four corners of the rotated rectangle
            # Start at marker_a center and build rectangle using the projected dimensions
            origin = np.array(marker_a_center, dtype=float)
            corner_tl = origin  # top-left at marker_a
            corner_tr = origin + width_px * x_axis_norm  # move right
            corner_br = origin + width_px * x_axis_norm + height_px * y_axis_norm  # diagonal
            corner_bl = origin + height_px * y_axis_norm  # move down

            # Compute current rectangle corners
            rect_corners_float = np.array(
                [corner_tl, corner_tr, corner_br, corner_bl], dtype=np.float64)

            # Apply temporal smoothing to reduce jitter
            _workspace_rect_buffer.append(rect_corners_float)
            if len(_workspace_rect_buffer) > _WORKSPACE_SMOOTH_FRAMES:
                _workspace_rect_buffer.pop(0)

            # Average the buffered corner positions
            if len(_workspace_rect_buffer) > 0:
                rect_corners_smoothed = np.mean(_workspace_rect_buffer, axis=0)
                rect_corners = np.round(rect_corners_smoothed).astype(np.int32)
            else:
                rect_corners = np.round(rect_corners_float).astype(np.int32)

            # Draw the rotated rectangle
            cv2.polylines(vis, [rect_corners], isClosed=True,
                          color=(0, 255, 255), thickness=2)

            # Draw diagonal for reference
            cv2.line(vis, tuple(rect_corners[0]), tuple(
                rect_corners[2]), (0, 255, 255), 1)

        status = "Workspace markers detected."

        # Compute workspace metrics if calibration is available
        metric = "Workspace: --"
        if SESSION.calibration_bundle is not None:
            try:
                # Create a simple detections object for workspace computation
                det_a = type('MarkerDetection', (), {
                    'marker_id': marker_ids[0],
                    'corners': detected_map[marker_ids[0]]['corners'],
                    'center': detected_map[marker_ids[0]]['center']
                })()
                det_b = type('MarkerDetection', (), {
                    'marker_id': marker_ids[1],
                    'corners': detected_map[marker_ids[1]]['corners'],
                    'center': detected_map[marker_ids[1]]['center']
                })()
                detections = type('WorkspaceMarkers', (), {
                    'marker_a': det_a,
                    'marker_b': det_b
                })()

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

    except Exception as exc:
        return vis, f"Workspace detection: {exc}", "Workspace: --"


def draw_plate_overlay(frame: np.ndarray, layout_path: Path) -> Tuple[np.ndarray, str, str]:
    vis = frame.copy()
    if not layout_path.exists():
        PLATE_STATUS.update(
            {
                "count": 0,
                "ok": False,
                "stable": False,
                "center_px": None,
                "movement_px": None,
                "timestamp": time.time(),
                "message": "Layout not found",
            }
        )
        return vis, "Calibration plate layout not found.", "Plate error: --"
    plate_layout = plate.load_plate_layout(layout_path)
    try:
        calibration_result = plate.estimate_plate_calibration(
            frame, plate_layout)
    except Exception as exc:
        PLATE_STATUS.update(
            {
                "count": 0,
                "ok": False,
                "stable": False,
                "center_px": None,
                "movement_px": None,
                "timestamp": time.time(),
                "message": str(exc),
            }
        )
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
    center_px = _plate_center_px(calibration_result.detections)
    center_list = None
    if center_px is not None:
        center_list = [float(center_px[0]), float(center_px[1])]
    movement_px = None
    stable = len(calibration_result.detections) >= MIN_PLATE_MARKERS
    baseline_px = SESSION.robot_calibration_baseline_px
    if baseline_px is not None and center_px is not None:
        movement_px = float(np.linalg.norm(center_px - baseline_px))
        if movement_px > PLATE_MOVEMENT_THRESHOLD_PX:
            stable = False
            status = f"Plate moved ({movement_px:.1f}px); hold it steady."
    PLATE_STATUS.update(
        {
            "count": len(calibration_result.detections),
            "ok": len(calibration_result.detections) >= MIN_PLATE_MARKERS,
            "stable": stable,
            "center_px": center_list,
            "movement_px": movement_px,
            "timestamp": time.time(),
            "message": status,
        }
    )
    return vis, status, metric


def draw_detection_overlay(frame: np.ndarray) -> Tuple[np.ndarray, str, str]:
    vis = frame.copy()

    # Run detection continuously if enabled and model is loaded
    if SESSION.continuous_detection and SESSION.model_loaded and SESSION.yolo_model is not None:
        try:
            detections = _run_detection(
                frame,
                SESSION.detection_confidence,
                SESSION.detection_iou,
                SESSION.detection_max,
                SESSION.detection_classes,
            )
            SESSION.detections = detections
            SESSION.last_detection_error = None
        except Exception as exc:
            SESSION.last_detection_error = str(exc)
            LOGGER.warning("Continuous detection failed: %s", exc)

    detections = SESSION.detections or []
    if not detections:
        msg = "Continuous detection: waiting..." if SESSION.continuous_detection else "No detections available."
        return vis, msg, "Detections: 0"

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
    status = "Continuous detection active." if SESSION.continuous_detection else "Detection overlay ready."
    return vis, status, metric


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
    branding = dmc.Stack(
        [
            dmc.Text("Estun Codroid Pick-and-Place", className="app-title"),
            dmc.Text("Vision + robot bring-up dashboard", size="sm", c="dimmed",
                     className="app-tagline"),
        ],
        gap="xs",
    )
    badges = dmc.Group(
        [
            dmc.Badge("Camera: off", id="header-camera-badge",
                      color="gray", variant="light"),
            dmc.Badge("Robot: idle", id="header-robot-badge",
                      color="gray", variant="light"),
            dmc.Badge("Gripper: off", id="header-gripper-badge",
                      color="gray", variant="light"),
        ],
        className="header-badges",
    )
    actions = dmc.Group(
        [
            badges,
            dmc.Button("Setup", id="open-setup", variant="filled",
                       color="brand", radius="xl", size="md"),
        ],
        className="header-actions",
        align="center",
    )
    return dmc.AppShellHeader(
        dmc.Group(
            [
                dmc.Group(
                    [branding],
                    align="center",
                    gap="sm",
                ),
                actions,
            ],
            justify="space-between",
            align="center",
            className="header-bar",
        ),
        className="app-header",
        px="lg",
        h=90,
    )


def build_footer() -> dmc.AppShellFooter:
    stop_button = dmc.Button(
        "E-STOP (POWER OFF)",
        id="safety-stop",
        color="red",
        size="md",
        radius="xl",
        variant="filled",
        fullWidth=False,
        style={
            "fontWeight": 800,
            "letterSpacing": "0.06em",
            "padding": "14px 26px",
            "minWidth": "210px",
            "fontSize": "16px",
            "backgroundImage": "repeating-linear-gradient(45deg, #ff4747 0, #ff4747 12px, #ffd447 12px, #ffd447 24px)",
            "color": "#1f1f1f",
            "border": "2px solid #8b0000",
        },
    )
    power_toggle = dmc.Switch(
        id="robot-power-toggle",
        label="Robot power",
        checked=False,
        color="red",
        size="md",
    )

    robot_speed_slider = dmc.Slider(
        id="robot-speed-slider",
        min=0,
        max=100,
        step=5,
        value=25,
        labelAlwaysOn=False,
        size="sm",
        marks=[
            {"value": 0, "label": "0"},
            {"value": 50, "label": "50"},
            {"value": 100, "label": "100"},
        ],
        style={"flexGrow": 1, "minWidth": "220px"},
    )

    robot_speed_controls = dmc.Group(
        [
            dmc.ActionIcon(
                "-",
                id="robot-speed-decrease",
                variant="filled",
                color="gray",
                radius="xl",
                size="md",
            ),
            robot_speed_slider,
            dmc.ActionIcon(
                "+",
                id="robot-speed-increase",
                variant="filled",
                color="gray",
                radius="xl",
                size="md",
            ),
        ],
        gap="sm",
        align="center",
        justify="flex-start",
        style={"flexGrow": 1},
    )

    auto_start_button = dmc.Button(
        "Start Auto",
        id="auto-start",
        color="green",
        size="sm",
        radius="md",
        variant="filled",
        leftSection=dmc.ActionIcon(
            "▶",
            size="sm",
            variant="transparent",
            color="white",
        ),
        style={
            "fontWeight": 700,
            "minWidth": "110px",
        },
    )

    auto_stop_button = dmc.Button(
        "Stop Auto",
        id="auto-stop",
        color="orange",
        size="sm",
        radius="md",
        variant="filled",
        leftSection=dmc.ActionIcon(
            "■",
            size="sm",
            variant="transparent",
            color="white",
        ),
        style={
            "fontWeight": 700,
            "minWidth": "110px",
        },
    )

    stop_column = dmc.Stack(
        [
            dmc.Text("Emergency stop", fw=700, size="sm", c="red"),
            stop_button,
            power_toggle,
        ],
        gap=6,
        align="center",
    )

    auto_control_column = dmc.Stack(
        [
            dmc.Text("Auto pick & place", fw=700, size="sm", c="blue"),
            dmc.Group(
                [auto_start_button, auto_stop_button],
                gap="xs",
            ),
        ],
        gap=6,
        align="center",
    )

    robot_speed_block = dmc.Stack(
        [
            dmc.Text("Robot speed", fw=600, size="sm"),
            robot_speed_controls,
        ],
        gap=4,
        align="flex-start",
        style={"minWidth": "280px"},
    )

    fine_width_slider = dmc.Slider(
        id="footer-gripper-width",
        min=0,
        max=40,
        step=0.5,
        value=10.0,
        labelAlwaysOn=False,
        size="sm",
        marks=[],
        style={"minWidth": "220px"},
    )

    fine_force_slider = dmc.Slider(
        id="footer-gripper-force",
        min=GRIPPER_FORCE_MIN,
        max=GRIPPER_FORCE_MAX,
        step=5,
        value=GRIPPER_FORCE_DEFAULT,
        labelAlwaysOn=False,
        size="sm",
        marks=[],
        style={"minWidth": "220px"},
    )

    fine_speed_slider = dmc.Slider(
        id="footer-gripper-speed",
        min=10,
        max=100,
        step=5,
        value=60,
        labelAlwaysOn=False,
        size="sm",
        marks=[],
        style={"minWidth": "220px"},
    )

    width_controls = dmc.Group(
        [
            dmc.ActionIcon("-", id="footer-gripper-width-decrease",
                           variant="light", color="gray", radius="xl", size="md"),
            fine_width_slider,
            dmc.ActionIcon("+", id="footer-gripper-width-increase",
                           variant="light", color="gray", radius="xl", size="md"),
        ],
        gap="xs",
        align="center",
        justify="flex-start",
        className="footer-slider-controls",
    )

    force_controls = dmc.Group(
        [
            dmc.ActionIcon("-", id="footer-gripper-force-decrease",
                           variant="light", color="gray", radius="xl", size="md"),
            fine_force_slider,
            dmc.ActionIcon("+", id="footer-gripper-force-increase",
                           variant="light", color="gray", radius="xl", size="md"),
        ],
        gap="xs",
        align="center",
        justify="flex-start",
        className="footer-slider-controls",
    )

    speed_controls = dmc.Group(
        [
            dmc.ActionIcon("-", id="footer-gripper-speed-decrease",
                           variant="light", color="gray", radius="xl", size="md"),
            fine_speed_slider,
            dmc.ActionIcon("+", id="footer-gripper-speed-increase",
                           variant="light", color="gray", radius="xl", size="md"),
        ],
        gap="xs",
        align="center",
        justify="flex-start",
        className="footer-slider-controls",
    )

    gripper_block = dmc.Stack(
        [
            dmc.Text("Gripper control", fw=600, size="sm"),
            dmc.Group(
                [
                    dmc.Button(
                        "Open",
                        id="footer-gripper-open",
                        variant="outline",
                        color="teal",
                        size="sm",
                    ),
                    dmc.Button(
                        "Close",
                        id="footer-gripper-close",
                        variant="filled",
                        color="teal",
                        size="sm",
                    ),
                ],
                gap="xs",
                align="center",
            ),
            dmc.Text(id="footer-gripper-status", size="sm", c="dimmed"),
            dmc.Divider(variant="dashed", mt=4, mb=2),
            dmc.Text("Fine control", fw=600, size="xs"),
            html.Div(
                [
                    dmc.Text("Width (mm)", size="xs", fw=600),
                    width_controls,
                ],
                className="footer-fine-row",
            ),
            html.Div(
                [
                    dmc.Text("Force (N)", size="xs", fw=600),
                    force_controls,
                ],
                className="footer-fine-row",
            ),
            html.Div(
                [
                    dmc.Text("Speed (%)", size="xs", fw=600),
                    speed_controls,
                ],
                className="footer-fine-row",
            ),
        ],
        gap=6,
        align="flex-start",
        style={"flexGrow": 1, "minWidth": "320px"},
    )

    robot_position_text = dmc.Text(
        id="footer-robot-position",
        size="sm",
        className="footer-robot-position",
    )

    flange_indicators = html.Div(
        [
            html.Div(id="footer-flange-indicator-1",
                     className="footer-flange-indicator"),
            html.Div(id="footer-flange-indicator-2",
                     className="footer-flange-indicator"),
            html.Div(id="footer-flange-indicator-3",
                     className="footer-flange-indicator"),
            html.Div(id="footer-flange-indicator-4",
                     className="footer-flange-indicator"),
        ],
        className="footer-flange-indicators",
    )

    flange_block = html.Div(
        [
            dmc.Text("Flange press", fw=600, size="sm"),
            flange_indicators,
        ],
        className="footer-flange-block",
    )

    footer_toggle = dmc.Button(
        "Hide footer",
        id="footer-toggle",
        variant="light",
        color="gray",
        size="sm",
        radius="xl",
    )

    footer_grid = html.Div(
        [robot_speed_block, gripper_block],
        className="footer-body-grid",
    )

    telemetry_block = dmc.Stack(
        [
            dmc.Text("Robot telemetry", fw=600, size="xs"),
            robot_position_text,
        ],
        gap=2,
    )

    status_group = dmc.Group(
        [telemetry_block, flange_block],
        gap="md",
        align="center",
        className="footer-status-group",
        wrap="wrap",
    )

    footer_header = html.Div(
        [
            dmc.Group(
                [stop_column, auto_control_column],
                gap="lg",
                align="center",
                wrap="wrap",
            ),
            dmc.Group(
                [status_group, footer_toggle],
                gap="md",
                align="center",
                wrap="wrap",
            ),
        ],
        className="footer-header",
    )

    footer_body = html.Div(
        footer_grid,
        className="footer-body",
    )

    return dmc.AppShellFooter(
        dmc.Container(
            dmc.Card(
                [footer_header, footer_body],
                className="footer-panel",
                padding="sm",
            ),
            fluid=True,
            px="lg",
            py="sm",
        ),
        id="app-footer",
        className="app-footer",
    )


def build_setup_drawer() -> dmc.Drawer:
    return dmc.Drawer(
        id="setup-drawer",
        opened=False,
        title="Setup & Connections",
        padding="md",
        size="lg",
        zIndex=2000,
        withOverlay=True,
        overlayProps={"opacity": 0.55, "color": "#000"},
        withinPortal=True,
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
                            id="camera-fps-input",
                            label="FPS",
                            value=DEFAULT_FPS,
                            min=1,
                            max=120,
                            step=1,
                        ),
                    ],
                    grow=True,
                ),
                dmc.Switch(
                    id="camera-lock-af",
                    label="Lock autofocus",
                    checked=False,
                    color="teal",
                    size="md",
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
                dmc.Group(
                    [
                        dmc.Button("Test robot connection",
                                   id="robot-test", variant="light"),
                        dmc.Button(
                            "Disconnect robot",
                            id="robot-disconnect",
                            variant="outline",
                            color="red",
                        ),
                    ],
                    gap="sm",
                ),
                dmc.Text(id="robot-status-text", size="sm", c="dimmed"),
                dmc.Divider(),
                dmc.Text("OnRobot 2FG7 Gripper", fw=600),
                dmc.TextInput(
                    id="gripper-cb-ip",
                    label="Compute Box IP Address",
                    value=DEFAULT_GRIPPER_IP,
                    placeholder="192.168.101.95",
                    description="IP address of the OnRobot Compute Box",
                ),
                dmc.Group(
                    [
                        dmc.Switch(
                            id="gripper-enable",
                            label="Enable gripper connection",
                            checked=True,
                            color="teal",
                        ),
                        dmc.Button("Test gripper",
                                   id="gripper-test", variant="light"),
                    ],
                    gap="md",
                ),
                dmc.Text(id="gripper-status-text", size="sm", c="dimmed"),
                html.Div(id="gripper-device-info"),
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


def build_robot_calibration_modal() -> dmc.Modal:
    return dmc.Modal(
        id="robot-calibration-modal",
        opened=False,
        title="Robot calibration guide",
        centered=True,
        size="lg",
        withCloseButton=False,
        closeOnClickOutside=False,
        children=dmc.Stack(
            [
                dmc.Stepper(
                    id="robot-calibration-modal-stepper",
                    active=0,
                    orientation="vertical",
                    children=[
                        dmc.StepperStep(
                            label="Point 1",
                            description="Indentation 1 (origin)",
                        ),
                        dmc.StepperStep(
                            label="Point 2",
                            description="Indentation 2 (+X)",
                        ),
                        dmc.StepperStep(
                            label="Point 3",
                            description="Indentation 3 (+X/+Y)",
                        ),
                        dmc.StepperStep(
                            label="Verification",
                            description="Safe motion check",
                        ),
                    ],
                ),
                dmc.Alert(
                    id="robot-calibration-modal-alert",
                    title="Calibration status",
                    color="blue",
                    variant="light",
                    children="Waiting for calibration to start.",
                ),
                dmc.Group(
                    [
                        dmc.Button(
                            "Run verification move",
                            id="robot-calibration-verify",
                            color="teal",
                            disabled=True,
                        ),
                        dmc.Button(
                            "Close",
                            id="robot-calibration-modal-close",
                            variant="outline",
                        ),
                    ],
                    justify="flex-end",
                ),
            ],
            gap="md",
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
                                    "Capture the ChArUco board frames to calibrate the camera.",
                                    size="sm",
                                    c="dimmed",
                                ),
                                dmc.Card(
                                    dmc.Group(
                                        [
                                            dmc.ThemeIcon(
                                                dmc.Text("✓", size="lg"),
                                                size="lg",
                                                radius="md",
                                                color="gray",
                                                id="markers-detected-icon",
                                            ),
                                            dmc.Stack(
                                                [
                                                    dmc.Text(
                                                        "Markers detected", size="sm", fw=600),
                                                    dmc.Text(
                                                        "Position the ChArUco board in view",
                                                        size="xs",
                                                        c="dimmed",
                                                    ),
                                                ],
                                                gap="xs",
                                            ),
                                        ],
                                        gap="md",
                                    ),
                                    p="md",
                                    withBorder=True,
                                ),
                                dmc.Button(
                                    "Calibrate",
                                    id="calibrate-camera",
                                    fullWidth=True,
                                    disabled=True,
                                    color="teal",
                                ),
                                dmc.Button(
                                    "Use saved calibration",
                                    id="use-existing-calibration",
                                    fullWidth=True,
                                    variant="outline",
                                    style={"display": "none"},
                                ),
                                dmc.Box(
                                    id="camera-calibration-action",
                                    style={"minHeight": "0px"},
                                ),
                                dmc.Text(
                                    id="calibration-progress-text",
                                    size="sm",
                                    c="dimmed",
                                    style={"display": "none",
                                           "marginBottom": "8px"},
                                ),
                                dmc.Progress(
                                    id="calibration-progress",
                                    value=0,
                                    size="sm",
                                    striped=True,
                                    animated=True,
                                    style={"display": "none"},
                                ),
                                dmc.Accordion(
                                    children=[
                                        dmc.AccordionItem(
                                            children=[
                                                dmc.AccordionControl(
                                                    "Advanced options"),
                                                dmc.AccordionPanel(
                                                    dmc.Stack(
                                                        [
                                                            dmc.TextInput(
                                                                id="calibration-board-path",
                                                                label="Board layout path",
                                                                value=str(
                                                                    DEFAULT_BOARD_PATH),
                                                                size="sm",
                                                            ),
                                                            dmc.TextInput(
                                                                id="calibration-output-dir",
                                                                label="Output directory",
                                                                value=str(
                                                                    DEFAULT_CAPTURE_DIR),
                                                                size="sm",
                                                            ),
                                                            dmc.TextInput(
                                                                id="calibration-bundle-path",
                                                                label="Bundle save path",
                                                                value=str(
                                                                    DEFAULT_BUNDLE_PATH),
                                                                size="sm",
                                                            ),
                                                            dmc.Group(
                                                                [
                                                                    dmc.Button(
                                                                        "Load bundle",
                                                                        id="load-calibration",
                                                                        variant="light",
                                                                        size="sm",
                                                                    ),
                                                                ],
                                                                grow=True,
                                                            ),
                                                        ],
                                                        gap="sm",
                                                    ),
                                                ),
                                            ],
                                            value="advanced",
                                        ),
                                    ],
                                    variant="separated",
                                ),
                                dmc.Text(
                                    id="camera-calibration-metrics", size="sm"),
                            ],
                            gap="md",
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
                                dmc.Alert(
                                    id="plate-detection-summary",
                                    title="Plate visibility",
                                    color="yellow",
                                    children="Waiting for plate markers...",
                                    variant="light",
                                ),
                                dmc.Button(
                                    "Begin calibrating robot",
                                    id="start-robot-calibration",
                                    variant="filled",
                                    color="teal",
                                    disabled=True,
                                ),
                                dmc.Text(id="current-robot-position",
                                         size="xs", c="dimmed"),
                                dmc.Text(id="plate-action-status",
                                         size="sm", c="dimmed"),
                                dmc.Text(id="plate-metrics", size="sm"),
                                dmc.Text(
                                    "Step-by-step calibration guide:\n"
                                    "1. Start the camera stream and confirm the plate markers are visible (>= 2)\n"
                                    "2. Keep the plate steady and click 'Begin calibrating robot'\n"
                                    "3. Follow the modal instructions for points 1, 2, and 3\n"
                                    "4. Press flange button 1 at each indentation\n"
                                    "5. Run the verification move once prompted",
                                    size="xs",
                                    c="dimmed",
                                    style={"whiteSpace": "pre-line"},
                                ),
                                dmc.Text(
                                    id="robot-calibration-status",
                                    size="sm",
                                    c="teal",
                                ),
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


def build_workspace_step_content() -> dmc.Stack:
    return dmc.Stack(
        [
            dmc.Text(
                "Use the two corner markers to define the workarea size.",
                size="sm",
                c="dimmed",
            ),
            dmc.Accordion(
                children=[
                    dmc.AccordionItem(
                        children=[
                            dmc.AccordionControl("Advanced options"),
                            dmc.AccordionPanel(
                                dmc.Stack(
                                    [
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
                                    ],
                                    gap="sm",
                                ),
                            ),
                        ],
                        value="workspace-advanced",
                    ),
                ],
                variant="separated",
            ),
            dmc.Button(
                "Compute workspace", id="compute-workspace", variant="light"),
            dmc.Button(
                "Use saved workspace",
                id="use-existing-workspace",
                variant="outline",
                style={"display": "none"},
            ),
            dmc.Text(id="workspace-action-status", size="sm", c="dimmed"),
            dmc.Text(id="workspace-metrics", size="sm"),
        ],
        gap="xs",
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
            dmc.Text(id="camera-fps", size="sm", c="dimmed"),
            html.Div(
                id="camera-feed-container",
                children=html.Img(
                    id="camera-feed",
                    src="/video_feed",  # Use MJPEG stream to avoid per-frame HTTP polling
                    style={
                        "width": "100%",
                        "height": "auto",
                        "borderRadius": "16px",
                        "border": "1px solid rgba(15, 118, 110, 0.2)",
                        "marginTop": "12px",
                        "backgroundColor": "#0b0f14",
                    },
                ),
                style={"position": "relative"}
            ),
        ],
        withBorder=True,
        radius="lg",
        className="panel-card camera-panel",
    )


def build_calibration_tab() -> dmc.Stack:
    return dmc.Stack([build_stepper_card()], gap="md")


def build_detection_tab() -> dmc.Stack:
    gpu_card = dmc.Card(
        [
            dmc.Text("GPU Status", fw=600, size="lg"),
            dmc.Stack(id="gpu-status-display", gap="xs",
                      children=format_gpu_status()),
        ],
        withBorder=True,
        radius="lg",
        className="panel-card",
    )

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
            dmc.Switch(
                id="continuous-detection-toggle",
                label="Continuous detection",
                checked=False,
                color="teal",
                size="md",
            ),
            dmc.Group(
                [
                    dmc.NumberInput(
                        id="detection-confidence",
                        label="Confidence",
                        value=0.7,
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
            dmc.Button("Run detection once",
                       id="run-detection", variant="light"),
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

    return dmc.Stack([gpu_card, model_card, detection_card, detections_card], gap="md")


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


def build_pick_place_application() -> dmc.Stack:
    return dmc.Stack(
        [
            dmc.Tabs(
                [
                    dmc.TabsList(
                        [
                            dmc.TabsTab(
                                "Detection", value="pick-place-detection"),
                            dmc.TabsTab(
                                "Pick & Place", value="pick-place-actions"),
                        ],
                        style={
                            "width": "100%",
                            "justifyContent": "flex-start",
                            "marginBottom": "12px",
                        },
                    ),
                    dmc.TabsPanel(
                        build_detection_tab(), value="pick-place-detection"),
                    dmc.TabsPanel(
                        build_pick_place_tab(), value="pick-place-actions"),
                ],
                id="pick-place-tabs",
                value="pick-place-detection",
                variant="pills",
                color="brand",
                style={"width": "100%"},
            ),
        ],
        gap="md",
    )


def build_application_selector() -> dmc.Stack:
    return dmc.Stack(
        [
            dmc.Text(
                "Choose an application to configure and run.",
                size="sm",
                c="dimmed",
            ),
            dmc.Tabs(
                [
                    dmc.TabsList(
                        [
                            dmc.TabsTab(
                                "Pick & Place", value="pick-place"),
                        ],
                        style={
                            "width": "100%",
                            "justifyContent": "flex-start",
                            "marginBottom": "12px",
                        },
                    ),
                    dmc.TabsPanel(
                        build_pick_place_application(), value="pick-place"),
                ],
                id="application-tabs",
                value="pick-place",
                variant="pills",
                color="brand",
                style={"width": "100%"},
            ),
        ],
        gap="md",
    )


def build_application_stepper_card() -> dmc.Card:
    return dmc.Card(
        [
            dmc.Group(
                [
                    dmc.Text("Application Steps", fw=600, size="lg"),
                    dmc.Group(
                        [
                            dmc.Button("Back", id="app-step-prev",
                                       variant="light", size="sm"),
                            dmc.Button("Next", id="app-step-next", size="sm"),
                        ],
                        gap="xs",
                    ),
                ],
                justify="space-between",
                align="center",
            ),
            dmc.Divider(),
            dmc.Stepper(
                id="application-stepper",
                active=0,
                orientation="vertical",
                children=[
                    dmc.StepperStep(
                        label="Workspace definition",
                        description="Detect workspace markers",
                        children=build_workspace_step_content(),
                    ),
                    dmc.StepperStep(
                        label="Applications",
                        description="Configure and run an application",
                        children=build_application_selector(),
                    ),
                ],
            ),
        ],
        withBorder=True,
        radius="lg",
        className="panel-card",
    )


def build_application_tab() -> dmc.Stack:
    return dmc.Stack([build_application_stepper_card()], gap="md")


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
    brand_palette = [
        "#e6f7f5",
        "#c2f0ea",
        "#99e2dd",
        "#67d5d0",
        "#3cc9c5",
        "#22b3a7",
        "#188f88",
        "#0f6b66",
        "#0a4b4a",
        "#062b2e",
    ]
    theme = {
        "fontFamily": "Space Grotesk, IBM Plex Sans, sans-serif",
        "headings": {"fontFamily": "Space Grotesk, IBM Plex Sans, sans-serif"},
        "primaryColor": "brand",
        "colors": {"brand": brand_palette},
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
                dcc.Store(
                    id="gripper-connection-state",
                    data={"status": "offline", "message": ""},
                ),
                dcc.Store(
                    id="notification-state",
                    data={"camera": None, "robot": None},
                ),
                dcc.Store(id="calibration-progress-store",
                          data={"progress": 0}),
                dcc.Store(
                    id="plate-detection-state",
                    data={
                        "count": 0,
                        "ok": False,
                        "stable": False,
                        "center_px": None,
                        "movement_px": None,
                        "timestamp": 0.0,
                        "message": "",
                    },
                ),
                dcc.Store(
                    id="robot-calibration-state",
                    data={
                        "status": "idle",
                        "message": "",
                        "step": 0,
                        "verification_ready": False,
                    },
                ),
                dcc.Store(
                    id="auto-operation-state",
                    data={"running": False, "message": ""},
                ),
                dcc.Store(
                    id="prerequisites-state",
                    data={
                        "camera_calibrated": False,
                        "workspace_defined": False,
                        "robot_calibrated": False,
                        "model_loaded": False,
                        "gripper_ready": False,
                    },
                ),
                dcc.Store(
                    id="footer-visibility",
                    data={"collapsed": False},
                ),
                dcc.Store(
                    id="robot-speed-state",
                    data={"value": 25},
                ),
                dcc.Store(
                    id="robot-power-state",
                    data={"value": None, "timestamp": 0.0},
                ),
                dcc.Interval(id="auto-init", interval=400,
                             n_intervals=0, max_intervals=1),
                dcc.Interval(id="robot-status-interval",
                             interval=400, n_intervals=0),
                dcc.Interval(
                    id="camera-interval",
                    interval=33,  # ~30 FPS UI poll; stream push happens each tick
                    n_intervals=0,
                    disabled=True,
                ),
                dcc.Interval(id="calibration-progress-interval",
                             interval=100, n_intervals=0),
                html.Div(id="camera-refresh-container",
                         style={"display": "none"}),
                build_setup_drawer(),
                build_robot_calibration_modal(),
                dmc.AppShell(
                    padding="md",
                    className="app-shell",
                    header={"height": 90},
                    footer={"height": 340},
                    children=[
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
                                                                "Applications", value="applications"),
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
                                                        build_application_tab(), value="applications"),
                                                    dmc.TabsPanel(
                                                        build_monitoring_tab(), value="monitoring"),
                                                ],
                                                id="main-tabs",
                                                value="calibration",
                                                variant="pills",
                                                color="brand",
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
                            id="app-main",
                            className="app-main footer-expanded",
                        ),
                        build_footer(),
                    ],
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


async def _probe_robot(uri: str) -> dict:
    session = _get_robot_session()
    return await session.probe(uri or DEFAULT_ROBOT_URI)


async def _with_robot_api(uri: str, action: Callable[[CodroidAPI], Awaitable[None]]) -> None:
    session = _get_robot_session()
    await session.run_with_robot(uri or DEFAULT_ROBOT_URI, action)


def _run_charuco_task(board_path: Path) -> Callable[[np.ndarray], OverlayResult]:
    def _runner(frame: np.ndarray) -> OverlayResult:
        vis, status, metric = draw_charuco_overlay(frame, board_path)
        _apply_workspace_overlay(vis)
        return OverlayResult(vis, status, metric)

    return _runner


def _run_workspace_task(marker_ids: Tuple[int, int], marker_size_mm: float) -> Callable[[np.ndarray], OverlayResult]:
    def _runner(frame: np.ndarray) -> OverlayResult:
        vis, status, metric = draw_workspace_overlay(
            frame, marker_ids, marker_size_mm)
        _apply_workspace_overlay(vis)
        return OverlayResult(vis, status, metric)

    return _runner


def _run_plate_task(layout_path: Path) -> Callable[[np.ndarray], OverlayResult]:
    def _runner(frame: np.ndarray) -> OverlayResult:
        vis, status, metric = draw_plate_overlay(frame, layout_path)
        _apply_workspace_overlay(vis)
        return OverlayResult(vis, status, metric)

    return _runner


def _run_detection_task() -> Callable[[np.ndarray], OverlayResult]:
    def _runner(frame: np.ndarray) -> OverlayResult:
        vis, status, metric = draw_detection_overlay(frame)
        _apply_workspace_overlay(vis)
        return OverlayResult(vis, status, metric)

    return _runner


def _run_drop_marker_task(marker_id: int) -> Callable[[np.ndarray], OverlayResult]:
    def _runner(frame: np.ndarray) -> OverlayResult:
        vis, status, metric = draw_drop_marker_overlay(frame, marker_id)
        _apply_workspace_overlay(vis)
        return OverlayResult(vis, status, metric)

    return _runner


def _select_overlay_task(
    tab: str,
    calibration_step: int,
    application_step: int,
    application: Optional[str],
    application_view: Optional[str],
    board_path: Path,
    marker_ids: Tuple[int, int],
    marker_size_mm: float,
    plate_path: Path,
    drop_marker_id: int,
) -> Optional[OverlayTask]:
    if tab == "calibration":
        if calibration_step == 0:
            return OverlayTask("calibration-charuco", _run_charuco_task(board_path))
        return OverlayTask("calibration-plate", _run_plate_task(plate_path))
    if tab == "applications":
        if application_step == 0:
            return OverlayTask(
                "application-workspace",
                _run_workspace_task(marker_ids, marker_size_mm),
            )
        if application == "pick-place":
            if application_view == "pick-place-detection":
                return OverlayTask("pick-place-detection", _run_detection_task())
            return OverlayTask("pick-place", _run_drop_marker_task(drop_marker_id))
    return None


def register_callbacks(app: Dash) -> None:
    # Client-side toggle for the Setup drawer (faster UI response).
    app.clientside_callback(
        """
        function(openClicks, closeClicks, opened) {
            const ctx = dash_clientside.callback_context;
            if (!ctx.triggered || ctx.triggered.length === 0) {
                return opened;
            }
            const trigger = ctx.triggered[0].prop_id.split('.')[0];
            if (trigger === "open-setup") {
                return true;
            }
            if (trigger === "setup-close") {
                return false;
            }
            // Fallback: compare counts.
            const o = openClicks || 0;
            const c = closeClicks || 0;
            if (o > c) return true;
            if (c > o) return false;
            return opened;
        }
        """,
        Output("setup-drawer", "opened"),
        Input("open-setup", "n_clicks"),
        Input("setup-close", "n_clicks"),
        State("setup-drawer", "opened"),
        prevent_initial_call=False,
    )

    app.clientside_callback(
        """
        function(n_clicks) {
            if (!n_clicks) return;
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

    # Auto-refresh camera image (enabled only when camera is live)
    # (Client refresh removed; MJPEG stream handles updates)

    @app.callback(
        Output("calibration-stepper", "active"),
        Input("step-prev", "n_clicks"),
        Input("step-next", "n_clicks"),
        Input("use-existing-calibration", "n_clicks"),
        State("calibration-stepper", "active"),
    )
    def _update_stepper(
        prev_clicks: int,
        next_clicks: int,
        use_saved_calib: int,
        active: Optional[int],
    ) -> int:
        current = active or 0
        trigger = callback_context.triggered_id
        if trigger == "step-next":
            return min(current + 1, 1)
        if trigger == "step-prev":
            return max(current - 1, 0)
        if trigger == "use-existing-calibration":
            return 1  # Advance to robot calibration step
        return current

    @app.callback(
        Output("application-stepper", "active"),
        Input("app-step-prev", "n_clicks"),
        Input("app-step-next", "n_clicks"),
        Input("use-existing-workspace", "n_clicks"),
        State("application-stepper", "active"),
    )
    def _update_application_stepper(
        prev_clicks: int,
        next_clicks: int,
        use_saved_workspace: int,
        active: Optional[int],
    ) -> int:
        current = active or 0
        trigger = callback_context.triggered_id
        if trigger == "app-step-next":
            return min(current + 1, 1)
        if trigger == "app-step-prev":
            return max(current - 1, 0)
        if trigger == "use-existing-workspace":
            return 1  # Advance to application tasks
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
        Output("camera-interval", "disabled"),
        Input("camera-enable", "checked"),
        Input("auto-init", "n_intervals"),
        Input("camera-lock-af", "checked"),
        State("camera-width", "value"),
        State("camera-height", "value"),
        State("camera-fps-input", "value"),
    )
    def _toggle_camera(
        checked: bool,
        _init: int,
        lock_af: Optional[bool],
        width: Optional[float],
        height: Optional[float],
        fps: Optional[float],
    ) -> Tuple[str, str, str, bool, dict, bool]:
        trigger = callback_context.triggered_id
        desired_checked = checked
        stream_running = SESSION.camera_stream is not None and SESSION.camera_stream.is_running()

        if trigger == "camera-lock-af" and stream_running:
            try:
                assert SESSION.camera_stream is not None
                SESSION.camera_stream.set_focus_lock(bool(lock_af))
                state = "locked" if lock_af else "unlocked"
                status = f"Camera stream running; focus {state}."
                return (
                    status,
                    "Camera: live",
                    "teal",
                    True,
                    {"status": "live", "message": status},
                    False,
                )
            except Exception as exc:  # noqa: BLE001
                error = f"Camera focus control error: {exc}"
                return (
                    error,
                    "Camera: error",
                    "red",
                    True,
                    {"status": "error", "message": error},
                    False,
                )

        if trigger == "auto-init":
            desired_checked = True
        if desired_checked:
            try:
                status = start_camera_stream(width, height, fps, bool(lock_af))
                return (
                    status,
                    "Camera: live",
                    "teal",
                    True,
                    {"status": "live", "message": status},
                    False,
                )
            except Exception as exc:
                stop_camera_stream()
                error = f"Camera error: {exc}"
                return (
                    error,
                    "Camera: error",
                    "red",
                    False,
                    {"status": "error", "message": error},
                    True,
                )
        stop_camera_stream()
        stopped = "Camera stream stopped."
        return (
            stopped,
            "Camera: off",
            "gray",
            False,
            {"status": "off", "message": stopped},
            True,
        )

    @app.callback(
        Output("robot-speed-slider", "value"),
        Input("robot-speed-decrease", "n_clicks"),
        Input("robot-speed-increase", "n_clicks"),
        State("robot-speed-slider", "value"),
        prevent_initial_call=True,
    )
    def _nudge_robot_speed(dec_clicks: Optional[int], inc_clicks: Optional[int], current: Optional[float]) -> float:
        current_value = float(current or 0.0)
        trigger = callback_context.triggered_id
        step = 5.0
        if trigger == "robot-speed-decrease":
            return max(0.0, current_value - step)
        if trigger == "robot-speed-increase":
            return min(100.0, current_value + step)
        return current_value

    @app.callback(
        Output("robot-speed-state", "data"),
        Input("robot-speed-slider", "value"),
        State("robot-connection-state", "data"),
        State("robot-uri", "value"),
        prevent_initial_call=True,
    )
    async def _apply_robot_speed(
        value: Optional[float],
        robot_state: Optional[dict],
        robot_uri: Optional[str],
    ) -> dict:
        if value is None:
            return no_update
        rate = max(0.0, min(1.0, float(value) / 100.0))
        now = time.time()
        if abs(rate - SESSION.robot_speed_rate) < 0.001 and (now - SESSION.robot_speed_last_set) < 0.4:
            return no_update

        SESSION.robot_speed_rate = rate
        if (robot_state or {}).get("status") != "online":
            LOGGER.info("Robot speed requested: %.0f%% (robot offline)", value)
            return {"status": "offline", "value": value, "rate": rate, "updated": now}

        async def _set_speed(api: CodroidAPI) -> None:
            await api.set_manual_move_rate(rate)

        try:
            await _with_robot_api(robot_uri or DEFAULT_ROBOT_URI, _set_speed)
            SESSION.robot_speed_last_set = now
            SESSION.last_action = f"Robot speed set to {rate:.2f}"
            LOGGER.info("Robot speed set to %.0f%%", value)
            return {"status": "ok", "value": value, "rate": rate, "updated": now}
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("Failed to set robot speed")
            return {"status": "error", "value": value, "rate": rate, "error": str(exc), "updated": now}

    @app.callback(
        Output("robot-status-text", "children"),
        Output("header-robot-badge", "children"),
        Output("header-robot-badge", "color"),
        Output("robot-connection-state", "data"),
        Input("robot-test", "n_clicks"),
        Input("robot-disconnect", "n_clicks"),
        Input("auto-init", "n_intervals"),
        State("robot-uri", "value"),
        prevent_initial_call=True,
    )
    async def _test_robot(
        _clicks: int,
        _disconnect_clicks: int,
        _init: int,
        uri: str,
    ) -> Tuple[str, str, str, dict]:
        trigger = callback_context.triggered_id
        session = _get_robot_session()
        if trigger == "robot-disconnect":
            await session.close()
            SESSION.robot_connection_attempted = False
            SESSION.robot_last_connection_error = None
            session.clear_position()
            _set_robot_guidance(
                "Robot calibration idle. Click 'Begin calibrating robot' to start.",
                status="idle",
                step=0,
                verification_ready=False,
            )
            status = "Robot disconnected."
            return status, "Robot: offline", "gray", {"status": "offline", "message": status}
        if trigger == "auto-init":
            if session.is_connected(uri or DEFAULT_ROBOT_URI):
                target = session.resolve_robot_target(uri or DEFAULT_ROBOT_URI)
                status = f"Robot connected to {target['host']}:{target['port']}."
                SESSION.robot_connection_attempted = True
                SESSION.robot_last_connection_error = None
                return status, "Robot: online", "teal", {"status": "online", "message": status}
            if SESSION.robot_connection_attempted:
                if SESSION.robot_last_connection_error:
                    error = SESSION.robot_last_connection_error
                    return error, "Robot: error", "red", {"status": "error", "message": error}
                return no_update, no_update, no_update, no_update
        try:
            response = await _probe_robot(uri)
            status = f"Robot connected to {response.get('host')}:{response.get('port')}."
            SESSION.robot_connection_attempted = True
            SESSION.robot_last_connection_error = None
            return status, "Robot: online", "teal", {"status": "online", "message": status}
        except Exception as exc:
            error = f"Robot connection failed: {exc}"
            SESSION.robot_connection_attempted = True
            SESSION.robot_last_connection_error = error
            return error, "Robot: error", "red", {"status": "error", "message": error}

    def _command_gripper(
        target_width: float,
        action_label: str,
        *,
        force: Optional[int] = None,
        speed: Optional[int] = None,
        prevent_overclose: bool = False,
    ) -> str:
        """Move gripper to target width and return status text."""
        if SESSION.gripper is None or not SESSION.gripper.isConnected(t_index=GRIPPER_DEFAULTS["t_index"]):
            return "Gripper is not connected. Enable gripper in Setup menu."

        try:
            safe_width = float(target_width)
            if prevent_overclose:
                current_width = SESSION.gripper.get_ext_width(
                    t_index=GRIPPER_DEFAULTS["t_index"]
                )
                if isinstance(current_width, (int, float)) and current_width >= 0:
                    if safe_width < current_width:
                        safe_width = current_width

            applied_force = _safe_int(force, GRIPPER_FORCE_DEFAULT)
            applied_force = min(
                max(applied_force, GRIPPER_FORCE_MIN), GRIPPER_FORCE_MAX
            )
            applied_speed = _safe_int(speed, 80)

            SESSION.gripper.grip(
                t_index=GRIPPER_DEFAULTS["t_index"],
                t_width=safe_width,
                n_force=applied_force,
                p_speed=applied_speed,
                f_wait=True,
            )
            SESSION.last_action = (
                f"Gripper {action_label} to {safe_width:.1f} mm @ "
                f"{applied_force} N / {applied_speed}% speed."
            )
            return SESSION.last_action
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("Gripper command failed")
            return f"Gripper action failed: {exc}"


def _prepare_gripper_for_calibration() -> None:
    """Set gripper to calibration defaults and remember prior width."""
    SESSION.robot_calibration_gripper_width = None
    gripper = SESSION.gripper
    if gripper is not None:
        try:
            width = gripper.get_ext_width(t_index=GRIPPER_DEFAULTS["t_index"])
        except Exception as exc:  # noqa: BLE001
            LOGGER.debug("Unable to read gripper width before calibration: %s", exc)
        else:
            if isinstance(width, (int, float)):
                SESSION.robot_calibration_gripper_width = float(width)
    response = _command_gripper(
        GRIPPER_CALIBRATION_WIDTH_MM,
        "prepping for calibration",
        force=GRIPPER_FORCE_MIN,
        speed=GRIPPER_SPEED_MIN,
    )
    LOGGER.debug("Calibration gripper prep: %s", response)


def _restore_calibration_gripper_width() -> None:
    """Restore gripper width saved prior to calibration."""
    width = SESSION.robot_calibration_gripper_width
    if width is None:
        return
    response = _command_gripper(
        width,
        "restored after calibration",
        force=GRIPPER_FORCE_DEFAULT,
        speed=GRIPPER_SPEED_DEFAULT,
        prevent_overclose=True,
    )
    LOGGER.debug("Calibration gripper restored: %s", response)
    SESSION.robot_calibration_gripper_width = None

    @app.callback(
        Output("gripper-status-text", "children"),
        Output("gripper-device-info", "children"),
        Output("header-gripper-badge", "children"),
        Output("header-gripper-badge", "color"),
        Output("gripper-connection-state", "data"),
        Input("gripper-test", "n_clicks"),
        Input("gripper-enable", "checked"),
        Input("auto-init", "n_intervals"),
        State("gripper-cb-ip", "value"),
    )
    def _handle_gripper(
        _test_clicks: int,
        enabled: bool,
        _init: int,
        cb_ip: str,
    ) -> Tuple[str, html.Div, str, str, dict]:
        """Handle gripper connection and testing."""
        trigger = callback_context.triggered_id

        if not enabled and trigger == "gripper-enable":
            # Disable gripper
            if SESSION.gripper is not None:
                SESSION.gripper = None
                SESSION.gripper_device = None
            return (
                "Gripper disabled",
                html.Div(),
                "Gripper: off",
                "gray",
                {"status": "disabled", "message": "Gripper disabled"},
            )

        # Auto-connect once on load
        if trigger == "auto-init":
            enabled = True
        if not enabled:
            return no_update, no_update, no_update, no_update, no_update

        # Enable or test gripper
        try:
            if SESSION.gripper_device is None or SESSION.gripper_device.Global_cbip != cb_ip:
                # Create new connection
                device = GripperDevice(Global_cbip=cb_ip)
                cb = device.getCB()

                if cb is None:
                    return (
                        f"Failed to connect to Compute Box at {cb_ip}",
                        html.Div(
                            dmc.Alert(
                                "Cannot reach the gripper control box. Check power and network.",
                                title="Gripper Unavailable",
                                color="red",
                            )
                        ),
                        "Gripper: error",
                        "red",
                        {
                            "status": "error",
                            "message": f"Failed to connect to Compute Box at {cb_ip}",
                        },
                    )

                SESSION.gripper_device = device
                SESSION.gripper = TwoFG7Gripper(device)

            # Check gripper connection
            try:
                is_connected = SESSION.gripper.isConnected(
                    t_index=GRIPPER_DEFAULTS["t_index"]
                )
            except Exception as exc:  # noqa: BLE001
                SESSION.gripper = None
                SESSION.gripper_device = None
                return (
                    f"Failed to query gripper at {cb_ip}: {exc}",
                    html.Div(
                        dmc.Alert(
                            "Could not reach the gripper. Verify the Compute Box IP and network connection.",
                            title="Gripper Unavailable",
                            color="red",
                        )
                    ),
                    "Gripper: error",
                    "red",
                    {
                        "status": "error",
                        "message": f"Failed to query gripper at {cb_ip}: {exc}",
                    },
                )

            if not is_connected:
                return (
                    f"No 2FG7 gripper detected at Compute Box {cb_ip}",
                    html.Div(
                        dmc.Alert(
                            "Ensure the 2FG7 gripper is connected to the Compute Box "
                            "and powered on. Check the tool index setting if using "
                            "a dual gripper setup.",
                            title="Gripper Not Detected",
                            color="yellow",
                        )
                    ),
                    "Gripper: error",
                    "yellow",
                    {
                        "status": "error",
                        "message": f"No 2FG7 gripper detected at Compute Box {cb_ip}",
                    },
                )

            # Get gripper status and information
            status_text = f"2FG7 connected at {cb_ip}"
            t_index = GRIPPER_DEFAULTS["t_index"]

            # Query gripper state
            width = SESSION.gripper.get_ext_width(t_index=t_index)
            force = SESSION.gripper.get_force(t_index=t_index)
            is_gripped = SESSION.gripper.isGripped(t_index=t_index)

            info_card = html.Div(
                dmc.Card(
                    [
                        dmc.Group(
                            [
                                dmc.Stack(
                                    [
                                        dmc.Text(
                                            "Width", size="sm", c="dimmed"),
                                        dmc.Text(
                                            f"{width:.1f} mm" if isinstance(
                                                width, (int, float)) and width > -2 else "N/A",
                                            fw=500,
                                        ),
                                    ],
                                    gap=0,
                                ),
                                dmc.Stack(
                                    [
                                        dmc.Text(
                                            "Force", size="sm", c="dimmed"),
                                        dmc.Text(
                                            f"{force:.1f} N" if isinstance(
                                                force, (int, float)) and force > -2 else "N/A",
                                            fw=500,
                                        ),
                                    ],
                                    gap=0,
                                ),
                                dmc.Stack(
                                    [
                                        dmc.Text(
                                            "Status", size="sm", c="dimmed"),
                                        dmc.Badge(
                                            "Gripped" if is_gripped else "Open",
                                            color="teal" if is_gripped else "gray",
                                            variant="light",
                                        ),
                                    ],
                                    gap=0,
                                ),
                            ],
                            grow=True,
                        ),
                    ],
                    withBorder=True,
                    padding="sm",
                    mt="xs",
                )
            )

            return (
                status_text,
                info_card,
                "Gripper: online",
                "teal",
                {"status": "online", "message": status_text},
            )

        except Exception as exc:
            LOGGER.exception("Gripper test failed")
            return (
                f"Gripper test failed: {exc}",
                html.Div(),
                "Gripper: error",
                "red",
                {"status": "error", "message": f"Gripper test failed: {exc}"},
            )

    @app.callback(
        Output("footer-gripper-open", "disabled"),
        Output("footer-gripper-close", "disabled"),
        Output("footer-gripper-width", "disabled"),
        Output("footer-gripper-force", "disabled"),
        Output("footer-gripper-speed", "disabled"),
        Input("gripper-connection-state", "data"),
    )
    def _sync_footer_gripper_disabled(gripper_state: Optional[dict]) -> Tuple[bool, bool, bool, bool, bool]:
        status = (gripper_state or {}).get("status", "offline")
        disabled = status != "online"
        return (
            disabled,
            disabled,
            disabled,
            disabled,
            disabled,
        )

    @app.callback(
        Output("footer-visibility", "data"),
        Input("footer-toggle", "n_clicks"),
        State("footer-visibility", "data"),
        prevent_initial_call=True,
    )
    def _toggle_footer_visibility(
        _clicks: int,
        state: Optional[dict],
    ) -> dict:
        collapsed = bool((state or {}).get("collapsed", False))
        return {"collapsed": not collapsed}

    @app.callback(
        Output("app-footer", "className"),
        Output("footer-toggle", "children"),
        Output("app-main", "className"),
        Input("footer-visibility", "data"),
    )
    def _sync_footer_visibility(state: Optional[dict]) -> Tuple[str, str, str]:
        collapsed = bool((state or {}).get("collapsed", False))
        class_name = "app-footer is-collapsed" if collapsed else "app-footer"
        label = "Show footer" if collapsed else "Hide footer"
        main_class = "app-main footer-collapsed" if collapsed else "app-main footer-expanded"
        return class_name, label, main_class

    @app.callback(
        Output("footer-robot-position", "children"),
        Input("robot-connection-state", "data"),
        Input("robot-status-interval", "n_intervals"),
    )
    def _update_footer_robot_position(
        robot_conn_state: Optional[dict],
        _intervals: int,
    ) -> str:
        session = _get_robot_session()
        if not session.posture_seen():
            if (robot_conn_state or {}).get("status") == "online":
                return "X=--.--- Y=--.--- Z=--.---  A=--.--- B=--.--- C=--.---"
            return "X=--.--- Y=--.--- Z=--.---  A=--.--- B=--.--- C=--.---"

        pos = session.position_snapshot()
        text = (
            f"X={pos.x:.3f} Y={pos.y:.3f} Z={pos.z:.3f}  "
            f"A={pos.a:.3f} B={pos.b:.3f} C={pos.c:.3f}"
        )
        age_s = pos.age_seconds()
        if age_s > 2.0:
            text = f"{text} · stale {age_s:.1f}s"
        return text

    @app.callback(
        Output("footer-flange-indicator-1", "className"),
        Output("footer-flange-indicator-2", "className"),
        Output("footer-flange-indicator-3", "className"),
        Output("footer-flange-indicator-4", "className"),
        Input("robot-status-interval", "n_intervals"),
    )
    def _update_footer_flange_indicators(_intervals: int) -> Tuple[str, str, str, str]:
        session = _get_robot_session()
        states = session.flange_button_states()
        classes = []
        for port in FLANGE_BUTTON_PORTS:
            active = states.get(port, 0) == 1
            classes.append(
                "footer-flange-indicator active" if active else "footer-flange-indicator"
            )
        return tuple(classes)

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
        Output("gpu-status-display", "children"),
        Input("camera-interval", "n_intervals"),
    )
    def _update_gpu_status(_n: int) -> list:
        """Update GPU status display periodically."""
        return format_gpu_status()

    @app.callback(
        Output("continuous-detection-toggle", "checked"),
        Input("continuous-detection-toggle", "checked"),
        State("detection-confidence", "value"),
        State("detection-iou", "value"),
        State("detection-max", "value"),
        State("detection-classes", "value"),
    )
    def _toggle_continuous_detection(
        enabled: bool,
        confidence: Optional[float],
        iou: Optional[float],
        max_detections: Optional[float],
        class_filter: Optional[str],
    ) -> bool:
        """Enable/disable continuous detection and store parameters."""
        SESSION.continuous_detection = enabled
        SESSION.detection_confidence = _safe_float(confidence, 0.7)
        SESSION.detection_iou = _safe_float(iou, 0.45)
        SESSION.detection_max = _safe_int(max_detections, 10)
        SESSION.detection_classes = class_filter
        if enabled:
            LOGGER.info("Continuous detection enabled")
        else:
            LOGGER.info("Continuous detection disabled")
        return enabled

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
        Output("detection-list", "children", allow_duplicate=True),
        Output("pick-detection-select", "data", allow_duplicate=True),
        Output("detection-metrics", "children", allow_duplicate=True),
        Input("camera-interval", "n_intervals"),
        State("continuous-detection-toggle", "checked"),
        prevent_initial_call=True,
    )
    def _update_detection_display(
        _n: int,
        continuous_enabled: bool,
    ) -> Tuple[list, list, str]:
        """Update detection display from continuous detection results."""
        if not continuous_enabled:
            return no_update, no_update, no_update

        detections = SESSION.detections or []
        detection_list = _format_detection_list(detections)

        options = []
        for idx, det in enumerate(detections):
            label = det.get("label", "object")
            conf = det.get("confidence", 0.0)
            options.append(
                {"value": str(idx), "label": f"{idx + 1}: {label} ({conf:.2f})"})

        metrics = f"Live: {len(detections)} object(s)"
        if SESSION.last_detection_error:
            metrics = f"Error: {SESSION.last_detection_error}"

        return detection_list, options, metrics

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
    async def _handle_pick_place(
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

        if trigger == "gripper-open":
            return _command_gripper(gripper_open_val, "opened")

        if trigger == "gripper-close":
            return _command_gripper(gripper_close_val, "closed")

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
                await _with_robot_api(
                    robot_uri or DEFAULT_ROBOT_URI, _pick_action)
                SESSION.last_action = "Pick command sent."
                return "Pick command sent."
            if trigger == "execute-drop":
                drop_target_mm = _resolve_drop_target()

                async def _drop_only(robot_api: CodroidAPI) -> None:
                    await _drop_action(robot_api, drop_target_mm)

                await _with_robot_api(
                    robot_uri or DEFAULT_ROBOT_URI, _drop_only)
                SESSION.last_action = "Drop command sent."
                return "Drop command sent."
            if trigger == "execute-pick-drop":
                await _with_robot_api(
                    robot_uri or DEFAULT_ROBOT_URI, _pick_drop_action)
                SESSION.last_action = "Pick & drop command sent."
                return "Pick & drop command sent."
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("Pick/place action failed")
            return f"Pick/place action failed: {exc}"

    @app.callback(
        Output("footer-gripper-status", "children"),
        Input("footer-gripper-open", "n_clicks"),
        Input("footer-gripper-close", "n_clicks"),
        Input("footer-gripper-width", "value"),
        Input("footer-gripper-force", "value"),
        Input("footer-gripper-speed", "value"),
        State("gripper-open-value", "value"),
        State("gripper-close-value", "value"),
        prevent_initial_call=True,
    )
    def _footer_gripper_actions(
        _open_clicks: int,
        _close_clicks: int,
        footer_width: Optional[float],
        footer_force: Optional[float],
        footer_speed: Optional[float],
        gripper_open_value: Optional[float],
        gripper_close_value: Optional[float],
    ) -> str:
        try:
            trigger = callback_context.triggered_id
            if trigger == "footer-gripper-width":
                if _recent_gripper_sync("width", footer_width):
                    return no_update
            if trigger == "footer-gripper-force":
                if _recent_gripper_sync("force", footer_force):
                    return no_update

            applied_force = _safe_int(footer_force, GRIPPER_FORCE_DEFAULT)
            applied_force = min(
                max(applied_force, GRIPPER_FORCE_MIN), GRIPPER_FORCE_MAX
            )
            applied_speed = _safe_int(footer_speed, 80)
            if trigger == "footer-gripper-open":
                width = _safe_float(gripper_open_value,
                                    _safe_float(footer_width, 1.0))
                return _command_gripper(
                    width,
                    "opened",
                    force=applied_force,
                    speed=applied_speed,
                )
            if trigger == "footer-gripper-close":
                width = _safe_float(gripper_close_value,
                                    _safe_float(footer_width, 0.0))
                return _command_gripper(
                    width,
                    "closed",
                    force=applied_force,
                    speed=applied_speed,
                    prevent_overclose=True,
                )
            if trigger in {"footer-gripper-width", "footer-gripper-force", "footer-gripper-speed"}:
                width = _safe_float(
                    footer_width, _safe_float(gripper_close_value, 0.0))
                return _command_gripper(
                    width,
                    "moved",
                    force=applied_force,
                    speed=applied_speed,
                )
            return no_update
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("Footer gripper action failed")
            return f"Robot action failed: {exc}"

    @app.callback(
        Output("app-notifications", "children", allow_duplicate=True),
        Input("safety-stop", "n_clicks"),
        State("robot-uri", "value"),
        prevent_initial_call=True,
    )
    async def _handle_safety_stop(
        _clicks: int,
        robot_uri: Optional[str],
    ) -> dmc.Notification:
        try:
            async def _power_off(api: CodroidAPI) -> None:
                await api.stop_command()
                await api.power_off()

            await _with_robot_api(robot_uri or DEFAULT_ROBOT_URI, _power_off)
            SESSION.last_action = "Emergency stop: power off command sent."
            return dmc.Notification(
                action="show",
                title="Emergency stop",
                message="Stop command + power off sent to robot controller.",
                color="red",
                autoClose=5000,
            )
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("Emergency stop failed")
            return dmc.Notification(
                action="show",
                title="Emergency stop failed",
                message=str(exc),
                color="red",
                autoClose=7000,
            )

    @app.callback(
        Output("app-notifications", "children", allow_duplicate=True),
        Input("robot-power-toggle", "checked"),
        State("robot-uri", "value"),
        State("robot-power-state", "data"),
        prevent_initial_call=True,
    )
    async def _handle_power_toggle(
        checked: bool,
        robot_uri: Optional[str],
        power_state: Optional[dict],
    ) -> dmc.Notification:
        if power_state and power_state.get("value") == checked:
            return no_update
        try:
            async def _apply_power(api: CodroidAPI) -> None:
                if checked:
                    await api.power_on()
                else:
                    await api.power_off()

            await _with_robot_api(robot_uri or DEFAULT_ROBOT_URI, _apply_power)
            action = "Power on" if checked else "Power off"
            SESSION.last_action = f"{action} command sent."
            return dmc.Notification(
                action="show",
                title="Robot power",
                message=f"{action} command sent to robot controller.",
                color="red" if not checked else "teal",
                autoClose=4000,
            )
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("Power toggle failed")
            return dmc.Notification(
                action="show",
                title="Robot power failed",
                message=str(exc),
                color="red",
                autoClose=7000,
            )

    @app.callback(
        Output("footer-gripper-width", "value"),
        Output("footer-gripper-force", "value"),
        Input("robot-status-interval", "n_intervals"),
        State("gripper-connection-state", "data"),
        State("footer-gripper-width", "value"),
        State("footer-gripper-force", "value"),
    )
    def _sync_footer_gripper_values(
        _intervals: int,
        gripper_state: Optional[dict],
        current_width: Optional[float],
        current_force: Optional[float],
    ) -> Tuple[Optional[float], Optional[float]]:
        if (gripper_state or {}).get("status") != "online":
            return no_update, no_update
        if SESSION.gripper is None:
            return no_update, no_update

        try:
            t_index = GRIPPER_DEFAULTS["t_index"]
            width = SESSION.gripper.get_ext_width(t_index=t_index)
            force = SESSION.gripper.get_force(t_index=t_index)
        except Exception as exc:  # noqa: BLE001
            LOGGER.debug("Failed to poll gripper values: %s", exc)
            return no_update, no_update

        width_update: Optional[float] = no_update
        force_update: Optional[float] = no_update
        if isinstance(width, (int, float)) and width > -2:
            if current_width is None or abs(width - current_width) > 0.05:
                width_update = float(width)
        if (
            isinstance(force, (int, float))
            and GRIPPER_FORCE_MIN <= force <= GRIPPER_FORCE_MAX
        ):
            if current_force is None or abs(force - current_force) > 0.1:
                force_update = float(force)

        if width_update is no_update and force_update is no_update:
            return no_update, no_update

        width_value = None if width_update is no_update else float(width_update)
        force_value = None if force_update is no_update else float(force_update)
        _record_gripper_sync(width_value, force_value)
        return width_update, force_update

    @app.callback(
        Output("monitor-system-summary", "children"),
        Output("monitor-detection-summary", "children"),
        Output("monitor-pick-summary", "children"),
        Output("use-existing-calibration", "style"),
        Output("use-existing-workspace", "style"),
        Input("camera-interval", "n_intervals"),
        State("camera-connection-state", "data"),
        State("robot-connection-state", "data"),
        State("calibration-bundle-path", "value"),
    )
    def _update_monitoring_summary(
        _ticks: int,
        camera_state: Optional[dict],
        robot_state: Optional[dict],
        bundle_path_value: Optional[str],
    ) -> Tuple[list, list, list, dict, dict]:
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
        bundle_path = _safe_path(bundle_path_value, DEFAULT_BUNDLE_PATH)
        bundle_style = {} if bundle_path.exists() else {"display": "none"}
        workspace_style = {} if DEFAULT_WORKSPACE_PATH.exists() else {
            "display": "none"}

        return system_items, detection_items, pick_items, bundle_style, workspace_style

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
        Output("safety-stop", "disabled"),
        Output("robot-power-toggle", "disabled"),
        Output("robot-speed-slider", "disabled"),
        Output("robot-speed-decrease", "disabled"),
        Output("robot-speed-increase", "disabled"),
        Input("robot-connection-state", "data"),
    )
    def _sync_robot_controls(robot_state: Optional[dict]) -> Tuple[bool, bool, bool, bool, bool]:
        status = (robot_state or {}).get("status") or "offline"
        disabled = status != "online"
        return disabled, disabled, disabled, disabled, disabled

    @app.callback(
        Output("calibrate-camera", "disabled"),
        Output("markers-detected-icon", "color"),
        Input("camera-calibration-metrics", "children"),
        prevent_initial_call=True,
    )
    def _update_calibrate_button_state(metrics: str) -> Tuple[bool, str]:
        """Enable Calibrate button and update icon when markers are detected (>= 16)"""
        if not metrics or "ChArUco corners:" not in metrics:
            return True, "gray"
        try:
            # Extract corner count from "ChArUco corners: X"
            parts = metrics.split(":")
            if len(parts) >= 2:
                count = int(parts[-1].strip().split()[0])
                is_disabled = count < 16
                color = "red" if is_disabled else "green"
                return is_disabled, color
        except (ValueError, IndexError):
            pass
        return True, "gray"

    @app.callback(
        Output("camera-calibration-action", "children"),
        Output("calibration-progress-store", "data"),
        Input("calibrate-camera", "n_clicks"),
        Input("load-calibration", "n_clicks"),
        Input("use-existing-calibration", "n_clicks"),
        State("calibration-output-dir", "value"),
        State("calibration-board-path", "value"),
        State("calibration-bundle-path", "value"),
        prevent_initial_call=True,
    )
    def _handle_camera_calibration_actions(
        _calibrate: int,
        _load: int,
        _use_existing: int,
        output_dir: str,
        board_path: str,
        bundle_path: str,
    ) -> dmc.Alert:
        trigger = callback_context.triggered_id

        output_path = _safe_path(output_dir, DEFAULT_CAPTURE_DIR)
        board_path = _safe_path(board_path, DEFAULT_BOARD_PATH)
        bundle_path = _safe_path(bundle_path, DEFAULT_BUNDLE_PATH)

        if trigger == "load-calibration":
            try:
                result = load_calibration_bundle(bundle_path)
                alert = dmc.Alert(
                    result,
                    icon="✓",
                    title="Bundle loaded",
                    color="green",
                )
                return alert, {"progress": 0}
            except Exception as exc:
                logging.error(
                    f"[CALIBRATE] Error loading bundle: {exc}", exc_info=True)
                alert = dmc.Alert(
                    str(exc)[:200],
                    icon="!",
                    title="Load error",
                    color="red",
                )
                return alert, {"progress": 0}

        if trigger == "use-existing-calibration":
            try:
                result = load_calibration_bundle(bundle_path)
                alert = dmc.Alert(
                    result,
                    icon="✓",
                    title="Using saved calibration",
                    color="green",
                )
                return alert, {"progress": 0}
            except Exception as exc:
                logging.error(
                    f"[CALIBRATE] Error loading saved calibration: {exc}", exc_info=True)
                alert = dmc.Alert(
                    str(exc)[:200],
                    icon="!",
                    title="Load error",
                    color="red",
                )
                return alert, {"progress": 0}

        if trigger == "calibrate-camera":
            try:
                # Clear prior calibration artifacts on disk to avoid mixing old frames
                _clear_calibration_outputs(output_path)

                # Clear previous captures
                SESSION.capture.frames = []
                SESSION.capture.depth_frame = None

                # Capture 30 frames for more robust calibration
                num_frames = 30
                captured_frames = []
                CALIBRATION_PROGRESS["max_frames"] = num_frames
                CALIBRATION_PROGRESS["value"] = 1  # Start progress
                for i in range(num_frames):
                    captured = capture_color_frame(output_path)
                    captured_frames.append(captured.name)
                    # Update progress (1-30 frames = 3-98%)
                    frame_num = i + 1
                    CALIBRATION_PROGRESS["current_frame"] = frame_num
                    CALIBRATION_PROGRESS["value"] = int(
                        (frame_num / num_frames) * 95) + 3
                    # Small delay between captures to avoid motion blur
                    if i < num_frames - 1:
                        time.sleep(0.1)

                # Capture depth snapshot
                depth_path = capture_depth_snapshot(output_path)

                # Pre-flight check: ensure the saved frames actually contain ChArUco corners
                detection_counts, usable_frames = _charuco_detection_counts(
                    _list_capture_frames(output_path), board_path
                )
                if usable_frames == 0:
                    raise RuntimeError(
                        "No usable ChArUco detections in captured frames. "
                        f"Corner counts: {detection_counts}. "
                        "Ensure the board is visible, sharp, and well-lit."
                    )

                # Compute calibration bundle
                result = compute_calibration_bundle(
                    board_path, bundle_path, output_path)

                alert = dmc.Alert(
                    children=[
                        dmc.Stack(
                            [
                                dmc.Text(
                                    f"✓ Captured {len(captured_frames)} RGB frames", size="sm"
                                ),
                                dmc.Text(
                                    "✓ Captured 1 depth snapshot", size="sm"),
                                dmc.Text(
                                    f"✓ Detected {usable_frames}/{num_frames} usable frames", size="sm"),
                                dmc.Text(result, size="sm", fw=600, c="teal"),
                            ],
                            gap="xs",
                        ),
                    ],
                    icon="✓",
                    title="Calibration complete",
                    color="green",
                )
                # Reset progress tracker
                CALIBRATION_PROGRESS["value"] = 0
                CALIBRATION_PROGRESS["current_frame"] = 0
                return alert, {"progress": 0}
            except Exception as exc:
                logging.error(
                    f"[CALIBRATE] Error during calibration: {exc}", exc_info=True)
                alert = dmc.Alert(
                    str(exc)[:200],
                    icon="!",
                    title="Calibration error",
                    color="red",
                )
                # Reset progress tracker on error
                CALIBRATION_PROGRESS["value"] = 0
                CALIBRATION_PROGRESS["current_frame"] = 0
                return alert, {"progress": 0}

        return dmc.Alert("", icon="•", title="Ready", color="gray"), {"progress": 0}

    @app.callback(
        Output("workspace-action-status", "children"),
        Input("compute-workspace", "n_clicks"),
        Input("use-existing-workspace", "n_clicks"),
        State("marker-id-a", "value"),
        State("marker-id-b", "value"),
        State("marker-size", "value"),
        prevent_initial_call=True,
    )
    def _handle_workspace_action(
        _clicks: int,
        _use_existing: int,
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
            # Load calibration bundle from disk if not already in memory
            if SESSION.calibration_bundle is None and DEFAULT_BUNDLE_PATH.exists():
                load_calibration_bundle(DEFAULT_BUNDLE_PATH)
            if callback_context.triggered_id == "use-existing-workspace":
                ws = load_workspace_definition(DEFAULT_WORKSPACE_PATH)
                return (
                    f"Workspace loaded from {DEFAULT_WORKSPACE_PATH}: "
                    f"{ws.width_mm:.1f} x {ws.height_mm:.1f} mm"
                )
            ws = compute_workspace_from_latest(marker_ids, marker_size_mm)
            return (
                f"Workspace computed and saved ({DEFAULT_WORKSPACE_PATH}): "
                f"{ws.width_mm:.1f} x {ws.height_mm:.1f} mm"
            )
        except Exception as exc:
            return f"Workspace error: {exc}"

    @app.callback(
        Output("plate-action-status", "children"),
        Output("plate-metrics", "children", allow_duplicate=True),
        Input("compute-plate", "n_clicks"),
        Input("attach-plate", "n_clicks"),
        State("plate-layout-path", "value"),
        prevent_initial_call=True,
    )
    def _handle_plate_actions(
        _compute_clicks: int,
        _attach_clicks: int,
        plate_layout_path: str,
    ) -> Tuple[str, str]:
        trigger = callback_context.triggered_id
        layout_path = _safe_path(plate_layout_path, DEFAULT_PLATE_PATH)
        if trigger == "compute-plate":
            try:
                result = compute_plate_calibration_from_latest(layout_path)
                metric = f"Reprojection error: {result.reprojection_error_mm:.2f} mm"
                return (
                    f"Plate detected with {len(result.detections)} markers.",
                    metric,
                )
            except Exception as exc:
                return f"Plate detection failed: {exc}", "Plate error: --"
        if trigger == "attach-plate":
            try:
                status = attach_plate_calibration(
                    DEFAULT_BUNDLE_PATH, layout_path)
                return status, "Plate attached to bundle."
            except Exception as exc:
                return f"Attach failed: {exc}", "Plate error: --"
        return no_update, no_update

    @app.callback(
        Output("overlay-status", "children"),
        Output("camera-fps", "children"),
        Output("camera-calibration-metrics", "children"),
        Output("workspace-metrics", "children"),
        Output("plate-metrics", "children"),
        Output("detection-metrics", "children"),
        Output("drop-marker-metrics", "children"),
        Output("plate-detection-state", "data"),
        Input("camera-interval", "n_intervals"),
        State("main-tabs", "value"),
        State("calibration-stepper", "active"),
        State("application-stepper", "active"),
        State("application-tabs", "value"),
        State("pick-place-tabs", "value"),
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
        calibration_step: Optional[int],
        application_step: Optional[int],
        application_tab: Optional[str],
        pick_place_tab: Optional[str],
        board_path: str,
        marker_a: Optional[float],
        marker_b: Optional[float],
        marker_size: Optional[float],
        plate_layout: str,
        drop_marker_id: Optional[float],
    ) -> Tuple[str, str, str, str, str, str, str, dict]:
        frame, _depth = get_latest_frame()
        plate_state = {
            "count": 0,
            "ok": False,
            "stable": False,
            "center_px": None,
            "movement_px": None,
            "timestamp": time.time(),
            "message": "Waiting for plate markers",
        }
        capture_fps = SESSION.camera_stream.get_fps() if SESSION.camera_stream else 0.0
        overlay_fps_text = _format_fps(capture_fps, MJPEG_STREAM.get_fps())
        if _ticks % 30 == 0:
            logging.debug(
                "_update_camera_view tick; frame available=%s", frame is not None)

        if frame is None:
            # Don't add placeholder to MJPEG stream - let video element show its poster image
            # Only log every 30 calls to reduce noise
            if _ticks % 30 == 0:
                is_running = SESSION.camera_stream and SESSION.camera_stream.is_running()
                logging.info(
                    "No frame from camera. Stream running: %s", is_running)
            return (
                "Camera stream is stopped.",
                overlay_fps_text,
                "ChArUco corners: --",
                "Workspace: --",
                "Plate error: --",
                "Detections: --",
                "Drop marker: --",
                plate_state,
            )

        # Check if this is a new frame by comparing object identity
        current_frame_id = id(frame)
        is_new_frame = current_frame_id != SESSION.last_frame_id
        if is_new_frame:
            SESSION.last_frame_id = current_frame_id

        # Frame received, dispatch to background processor for heavy overlays.
        tab = active_tab or "calibration"
        calibration_step = calibration_step or 0
        application_step = application_step or 0
        application_tab = application_tab or "pick-place"
        pick_place_tab = pick_place_tab or "pick-place-detection"
        board_path = _safe_path(board_path, DEFAULT_BOARD_PATH)
        plate_path = _safe_path(plate_layout, DEFAULT_PLATE_PATH)
        marker_ids = (
            _safe_int(marker_a, SESSION.config.marker_ids[0]),
            _safe_int(marker_b, SESSION.config.marker_ids[1]),
        )
        marker_size_mm = _safe_float(
            marker_size, SESSION.config.marker_size_mm)
        drop_marker = _safe_int(drop_marker_id, SESSION.drop_marker_id)

        task = _select_overlay_task(
            tab,
            calibration_step,
            application_step,
            application_tab,
            pick_place_tab,
            board_path,
            marker_ids,
            marker_size_mm,
            plate_path,
            drop_marker,
        )
        overlay: Optional[OverlayResult] = None
        if task is not None:
            FRAME_PROCESSOR.submit(task, frame)
            overlay = FRAME_PROCESSOR.get_latest(task.key)

        vis = overlay.vis if overlay else frame
        status_text = overlay.status if overlay else "Processing..."
        metric_text = overlay.metric if overlay else "--"
        plate_state = PLATE_STATUS.copy()

        if tab == "calibration":
            if calibration_step == 0:
                if is_new_frame:
                    MJPEG_STREAM.add_frame(vis)
                overlay_fps_text = _format_fps(
                    capture_fps, MJPEG_STREAM.get_fps())
                return (
                    status_text,
                    overlay_fps_text,
                    metric_text,
                    "Workspace: --",
                    "Plate error: --",
                    "Detections: --",
                    "Drop marker: --",
                    plate_state,
                )
            if is_new_frame:
                MJPEG_STREAM.add_frame(vis)
            overlay_fps_text = _format_fps(
                capture_fps, MJPEG_STREAM.get_fps())
            return (
                status_text,
                overlay_fps_text,
                "ChArUco corners: --",
                "Workspace: --",
                metric_text,
                "Detections: --",
                "Drop marker: --",
                plate_state,
            )

        if tab == "applications":
            if application_step == 0:
                if is_new_frame:
                    MJPEG_STREAM.add_frame(vis)
                overlay_fps_text = _format_fps(
                    capture_fps, MJPEG_STREAM.get_fps())
                return (
                    status_text,
                    overlay_fps_text,
                    "ChArUco corners: --",
                    metric_text,
                    "Plate error: --",
                    "Detections: --",
                    "Drop marker: --",
                    plate_state,
                )
            if application_tab == "pick-place":
                if pick_place_tab == "pick-place-detection":
                    if is_new_frame:
                        MJPEG_STREAM.add_frame(vis)
                    overlay_fps_text = _format_fps(
                        capture_fps, MJPEG_STREAM.get_fps())
                    return (
                        status_text,
                        overlay_fps_text,
                        "ChArUco corners: --",
                        "Workspace: --",
                        "Plate error: --",
                        metric_text,
                        "Drop marker: --",
                        plate_state,
                    )
                if is_new_frame:
                    MJPEG_STREAM.add_frame(vis)
                overlay_fps_text = _format_fps(
                    capture_fps, MJPEG_STREAM.get_fps())
                return (
                    status_text,
                    overlay_fps_text,
                    "ChArUco corners: --",
                    "Workspace: --",
                    "Plate error: --",
                    "Detections: --",
                    metric_text,
                    plate_state,
                )

        # Monitoring or any other tab: keep stream alive and surface metrics.
        if is_new_frame:
            MJPEG_STREAM.add_frame(vis)
        overlay_fps_text = _format_fps(capture_fps, MJPEG_STREAM.get_fps())
        return (
            status_text,
            overlay_fps_text,
            "ChArUco corners: --",
            "Workspace: --",
            "Plate error: --",
            "Detections: --",
            "Drop marker: --",
            plate_state,
        )

    @app.callback(
        Output("plate-detection-summary", "children"),
        Output("plate-detection-summary", "color"),
        Output("start-robot-calibration", "disabled"),
        Output("start-robot-calibration", "variant"),
        Input("plate-detection-state", "data"),
        Input("robot-connection-state", "data"),
        State("robot-calibration-state", "data"),
    )
    def _update_plate_summary(
        plate_state: Optional[dict],
        robot_conn_state: Optional[dict],
        calib_state: Optional[dict],
    ) -> Tuple[str, str, bool, str]:
        state = plate_state or {}
        count = state.get("count", 0)
        ok = bool(state.get("ok"))
        stable = bool(state.get("stable", ok))
        calib_status = (calib_state or {}).get("status")
        in_progress = calib_status in {
            "running",
            "awaiting_verification",
            "verifying",
        }
        if ROBOT_CALIBRATION_GUIDANCE.get("status") in {
            "running",
            "awaiting_verification",
            "verifying",
        }:
            in_progress = True
        robot_online = (robot_conn_state or {}).get("status") == "online"
        posture_ready, posture_message = _robot_posture_ready()

        # Determine message based on the actual blocking issue
        if in_progress:
            if calib_status == "awaiting_verification":
                message = "Awaiting verification move."
                color = "yellow"
            elif calib_status == "verifying":
                message = "Verification in progress..."
                color = "blue"
            else:
                message = "Calibration in progress..."
                color = "blue"
        elif not robot_online:
            message = "Robot not connected. Click 'Test robot connection' to establish connection."
            color = "red"
        elif not ok:
            message = state.get("message") or f"Detected {count} marker(s)."
            color = "yellow"
        elif not stable:
            message = state.get("message") or "Calibration plate moved; hold it steady."
            color = "yellow"
        elif not posture_ready:
            message = posture_message
            color = "yellow"
        else:
            message = "Plate markers stable. Ready to calibrate."
            color = "green"

        disabled = not ok or not stable or not posture_ready or in_progress or not robot_online
        variant = "filled" if not disabled else "light"
        return message, color, disabled, variant

    @app.callback(
        Output("current-robot-position", "children"),
        Input("robot-connection-state", "data"),
        Input("robot-status-interval", "n_intervals"),
    )
    def _update_robot_position_display(
        robot_conn_state: Optional[dict],
        _intervals: int,
    ) -> str:
        """Display current robot position with real-time updates."""
        session = _get_robot_session()
        pos = session.position_snapshot()
        age_s = pos.age_seconds()

        # Show position if we have recent data (from any source)
        if session.posture_seen() and age_s < 2.0:
            return (
                f"🤖 Robot position: "
                f"X={pos.x:.1f}, "
                f"Y={pos.y:.1f}, "
                f"Z={pos.z:.1f}"
            )

        # Check if robot is connected via test connection
        if (robot_conn_state or {}).get("status") == "online":
            if not session.posture_seen():
                return "Robot: Awaiting posture stream..."
            return f"Robot: Position stale (last seen {age_s:.1f}s ago)"

        return "Robot: --"

    @app.callback(
        Output("robot-power-state", "data"),
        Input("robot-status-interval", "n_intervals"),
        prevent_initial_call=True,
    )
    def _sync_robot_power_state(_intervals: int) -> dict:
        session = _get_robot_session()
        power_on = session.robot_power_on()
        if power_on is None:
            return no_update
        return {"value": bool(power_on), "timestamp": time.time()}

    @app.callback(
        Output("robot-power-toggle", "checked"),
        Input("robot-power-state", "data"),
        State("robot-power-toggle", "checked"),
        prevent_initial_call=True,
    )
    def _sync_power_toggle(
        power_state: Optional[dict],
        current: Optional[bool],
    ) -> bool:
        if not power_state or power_state.get("value") is None:
            return no_update
        desired = bool(power_state.get("value"))
        if current is None or current != desired:
            return desired
        return no_update

    @app.callback(
        Output("robot-calibration-status", "children"),
        Input("robot-status-interval", "n_intervals"),
    )
    def _update_robot_calibration_guidance(_intervals: int) -> str:
        return ROBOT_CALIBRATION_GUIDANCE.get("message", "")

    @app.callback(
        Output("robot-calibration-state", "data", allow_duplicate=True),
        Input("robot-status-interval", "n_intervals"),
        State("robot-calibration-state", "data"),
        prevent_initial_call=True,
    )
    def _sync_robot_calibration_state(_intervals: int, current: Optional[dict]) -> dict:
        desired = {
            "status": ROBOT_CALIBRATION_GUIDANCE.get("status", "idle"),
            "message": ROBOT_CALIBRATION_GUIDANCE.get("message", ""),
            "step": ROBOT_CALIBRATION_GUIDANCE.get("step", 0),
            "verification_ready": ROBOT_CALIBRATION_GUIDANCE.get(
                "verification_ready", False
            ),
        }
        if current == desired:
            return no_update
        return desired

    @app.callback(
        Output("robot-calibration-modal", "opened"),
        Input("robot-calibration-state", "data"),
        Input("robot-calibration-modal-close", "n_clicks"),
    )
    def _toggle_robot_calibration_modal(
        calibration_state: Optional[dict],
        _close_clicks: int,
    ) -> bool:
        status = (calibration_state or {}).get("status", "idle")
        trigger = callback_context.triggered_id
        if status in {"running", "awaiting_verification", "verifying"}:
            return True
        if status in {"complete", "error", "canceled"}:
            return trigger != "robot-calibration-modal-close"
        if trigger == "robot-calibration-modal-close":
            return False
        return False

    @app.callback(
        Output("robot-calibration-modal-stepper", "active"),
        Output("robot-calibration-modal-alert", "children"),
        Output("robot-calibration-modal-alert", "color"),
        Output("robot-calibration-verify", "disabled"),
        Output("robot-calibration-modal-stepper", "children"),
        Input("robot-calibration-state", "data"),
    )
    def _update_robot_calibration_modal(
        calibration_state: Optional[dict],
    ) -> Tuple[int, str, str, bool, list[dmc.StepperStep]]:
        state = calibration_state or {}
        status = state.get("status", "idle")
        step = int(state.get("step", 0) or 0)
        message = state.get("message", "")
        active = max(0, min(3, step - 1)) if step else 0
        color_map = {
            "running": "blue",
            "awaiting_verification": "yellow",
            "verifying": "blue",
            "complete": "green",
            "error": "red",
            "canceled": "red",
        }
        color = color_map.get(status, "gray")
        verify_enabled = (
            status == "awaiting_verification"
            and state.get("verification_ready", False)
        )
        verification_completed = status == "complete"

        stepper_children = [
            dmc.StepperStep(
                label="Point 1",
                description="Indentation 1 (origin)",
                completed=step > 1,
            ),
            dmc.StepperStep(
                label="Point 2",
                description="Indentation 2 (+X)",
                completed=step > 2,
            ),
            dmc.StepperStep(
                label="Point 3",
                description="Indentation 3 (+X/+Y)",
                completed=step > 3,
            ),
            dmc.StepperStep(
                label="Verification",
                description="Safe motion check",
                completed=verification_completed,
            ),
        ]
        return (
            active,
            message or "Waiting for calibration to start.",
            color,
            not verify_enabled,
            stepper_children,
        )

    @app.callback(
        Output("robot-calibration-state", "data", allow_duplicate=True),
        Input("start-robot-calibration", "n_clicks"),
        State("plate-layout-path", "value"),
        State("plate-detection-state", "data"),
        State("robot-connection-state", "data"),
        State("robot-uri", "value"),
        prevent_initial_call=True,
    )
    async def _start_robot_calibration(
        _clicks: int,
        plate_layout_path: str,
        plate_state: Optional[dict],
        robot_state: Optional[dict],
        robot_uri: Optional[str],
    ) -> dict:
        LOGGER.info("=== Robot calibration button clicked ===")
        LOGGER.info(f"Plate state: {plate_state}")
        LOGGER.info(f"Robot URI: {robot_uri}")

        if (robot_state or {}).get("status") != "online":
            _set_robot_guidance(
                "Robot not connected. Click 'Test robot connection' to establish connection.",
                status="error",
                verification_ready=False,
            )
            return {"status": "error", "message": "Robot not connected."}

        plate_ok, plate_message = _plate_is_stable(require_baseline=False)
        if not plate_ok:
            LOGGER.warning("Plate not ready: %s", plate_message)
            _set_robot_guidance(
                plate_message,
                status="error",
                verification_ready=False,
            )
            return {"status": "error", "message": plate_message}

        posture_ok, posture_message = _robot_posture_ready()
        if not posture_ok:
            _set_robot_guidance(
                posture_message,
                status="error",
                verification_ready=False,
            )
            return {"status": "error", "message": posture_message}

        _prepare_gripper_for_calibration()

        LOGGER.info("Plate state OK, starting calibration procedure...")
        try:
            layout_path = _safe_path(plate_layout_path, DEFAULT_PLATE_PATH)
            LOGGER.info(f"Loading plate layout from: {layout_path}")
            plate_layout = plate.load_plate_layout(layout_path)
            LOGGER.info("Plate layout loaded successfully")

            center = (plate_state or {}).get("center_px")
            if center is None:
                raise RuntimeError("Plate center not available; keep markers visible.")
            SESSION.robot_calibration_baseline_px = np.asarray(center, dtype=np.float64)
            _ensure_plate_stable()
            LOGGER.info("Plate stability baseline set")

            _set_robot_guidance(
                "Robot calibration started. Step 1/3: Place the gripper fingers in "
                "indentation 1, keep the flange vertical, and press flange button 1.",
                step=1,
                status="running",
                verification_ready=False,
            )
            SESSION.robot_calibration_points = []
            LOGGER.info(
                f"Starting robot plate calibration with URI: {robot_uri or DEFAULT_ROBOT_URI}")
            robot_points, _coord_frame = await _run_robot_plate_calibration(
                robot_uri or DEFAULT_ROBOT_URI, plate_layout)
            LOGGER.info(
                f"Robot plate calibration completed. Captured {len(robot_points)} points.")

            LOGGER.info("Applying robot calibration results...")
            _apply_robot_calibration_results(
                robot_points,
                plate_layout,
            )
            _update_robot_calibration_meta(robot_points)
            SESSION.robot_calibration_baseline_px = None
            LOGGER.info("Calibration results applied successfully")
            _set_robot_guidance(
                "Calibration complete. Move away from the robot and run the verification move.",
                status="awaiting_verification",
                step=4,
                verification_ready=True,
            )
            return {
                "status": "awaiting_verification",
                "message": "Awaiting verification move.",
                "step": 4,
                "verification_ready": True,
            }
        except Exception as exc:
            SESSION.robot_calibration_baseline_px = None
            _restore_calibration_gripper_width()
            LOGGER.error("Robot calibration failed: %s", exc, exc_info=True)
            _set_robot_guidance(
                f"Robot calibration failed: {exc}",
                status="error",
                verification_ready=False,
            )
            return {"status": "error", "message": str(exc)}

    @app.callback(
        Output("robot-calibration-state", "data", allow_duplicate=True),
        Input("robot-calibration-verify", "n_clicks"),
        State("robot-uri", "value"),
        prevent_initial_call=True,
    )
    async def _run_robot_calibration_verification(
        _clicks: int,
        robot_uri: Optional[str],
    ) -> dict:
        if not SESSION.robot_calibration_points or len(SESSION.robot_calibration_points) < 3:
            message = "Verification requires three calibration points."
            _set_robot_guidance(message, status="error",
                                step=4, verification_ready=False)
            return {"status": "error", "message": message, "step": 4}

        points = SESSION.robot_calibration_points[:3]
        min_z = SESSION.robot_calibration_min_z
        if min_z is None:
            min_z = min(float(pt.get("z", 0.0)) for pt in points)
            SESSION.robot_calibration_min_z = min_z

        _set_robot_guidance(
            "Verification move running. Keep clear of the robot.",
            status="verifying",
            step=4,
            verification_ready=False,
        )

        async def _verification_motion(api: CodroidAPI) -> None:
            await api.power_on()
            await api.set_current_coordinate_id(0)

            async def _move_linear_with_hold(target: dict, hold_seconds: float = VERIFICATION_HOLD_SECONDS) -> None:
                await api.set_target_pos_type(api.config.target_pos_types.cpos)
                await api.set_target_cpos(target)
                await api.set_robot_command(api.config.commands.move_target_linear)
                if hold_seconds > 0:
                    deadline = time.monotonic() + hold_seconds
                    while time.monotonic() < deadline:
                        await api.send_command_heartbeat()
                        await asyncio.sleep(0.2)
                await api.clear_target_position()
                await api.stop_command()

            def _safe_z(value: float) -> float:
                return max(float(value), float(min_z))

            def _target(point: dict, z_value: float) -> dict:
                poscfg = CodroidAPI.build_poscfg(
                    mode=int(point.get("mode", 0))
                )
                return api.build_target_cpos(
                    x=float(point["x"]),
                    y=float(point["y"]),
                    z=_safe_z(z_value),
                    a=float(point.get("a", 0.0)),
                    b=float(point.get("b", 0.0)),
                    c=float(point.get("c", 0.0)),
                    poscfg=poscfg,
                )

            p1, p2, p3 = points
            z1 = _safe_z(p1.get("z", 0.0))
            z2 = _safe_z(p2.get("z", 0.0))
            z3 = _safe_z(p3.get("z", 0.0))
            raise_1 = z1 + VERIFICATION_RAISE_MM
            raise_2 = z2 + VERIFICATION_RAISE_MM
            raise_3 = z3 + VERIFICATION_RAISE_MM

            await _move_linear_with_hold(_target(p3, raise_3))
            await _move_linear_with_hold(_target(p1, raise_1))
            await _move_linear_with_hold(_target(p1, z1))
            await _move_linear_with_hold(_target(p1, raise_1))
            await _move_linear_with_hold(_target(p2, raise_2))
            await _move_linear_with_hold(_target(p2, z2))
            await _move_linear_with_hold(_target(p2, raise_2))
            await _move_linear_with_hold(_target(p3, raise_3))
            await _move_linear_with_hold(_target(p3, z3))
            await _move_linear_with_hold(_target(p3, raise_3))
            await _move_linear_with_hold(_target(p1, z1), hold_seconds=0.0)
            await api.set_current_coordinate_id(ROBOT_COORDINATE_ID)

        try:
            await _with_robot_api(robot_uri or DEFAULT_ROBOT_URI, _verification_motion)
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("Verification move failed")
            message = f"Verification move failed: {exc}"
            _set_robot_guidance(
                message,
                status="error",
                step=4,
                verification_ready=False,
            )
            _restore_calibration_gripper_width()
            return {"status": "error", "message": message, "step": 4}

        _restore_calibration_gripper_width()
        _set_robot_guidance(
            "Calibration verified. Coordinate ID 5 is active.",
            status="complete",
            step=4,
            verification_ready=False,
        )
        return {"status": "complete", "message": "Calibration verified.", "step": 4}


def create_app() -> Dash:
    app = Dash(
        __name__,
        assets_folder=str(ASSETS_DIR),
        suppress_callback_exceptions=True,
    )
    app.title = "Estun Codroid Pick-and-Place"
    app._favicon = "favicon.ico"
    app.layout = build_layout()
    register_callbacks(app)

    @app.callback(
        Output("prerequisites-state", "data"),
        Input("camera-interval", "n_intervals"),
        Input("gripper-connection-state", "data"),
    )
    def _update_prerequisites(_n: int, gripper_state: Optional[dict]) -> dict:
        """Update prerequisites state based on current system status."""
        return {
            "camera_calibrated": SESSION.calibration_bundle is not None,
            "workspace_defined": SESSION.workspace_definition is not None,
            "robot_calibrated": SESSION.robot_calibration_min_z is not None,
            "model_loaded": SESSION.model_loaded,
            "gripper_ready": (gripper_state or {}).get("status") == "online",
        }

    @app.callback(
        Output("auto-start", "disabled"),
        Output("auto-stop", "disabled"),
        Input("prerequisites-state", "data"),
        Input("auto-operation-state", "data"),
    )
    def _update_auto_buttons(
        prerequisites: Optional[dict],
        auto_state: Optional[dict],
    ) -> Tuple[bool, bool]:
        """Enable/disable auto operation buttons based on prerequisites and operation state."""
        prereqs = prerequisites or {}
        all_ready = all([
            prereqs.get("camera_calibrated", False),
            prereqs.get("workspace_defined", False),
            prereqs.get("robot_calibrated", False),
            prereqs.get("model_loaded", False),
            prereqs.get("gripper_ready", False),
        ])

        running = (auto_state or {}).get("running", False)

        # Start button: enabled only when all prerequisites met and not running
        start_disabled = not all_ready or running
        # Stop button: enabled only when running
        stop_disabled = not running

        return start_disabled, stop_disabled

    @app.callback(
        Output("auto-operation-state", "data"),
        Output("app-notifications", "children", allow_duplicate=True),
        Input("auto-start", "n_clicks"),
        Input("auto-stop", "n_clicks"),
        State("auto-operation-state", "data"),
        prevent_initial_call=True,
    )
    def _handle_auto_operation(
        start_clicks: int,
        stop_clicks: int,
        current_state: Optional[dict],
    ) -> Tuple[dict, dmc.Notification]:
        """Handle auto operation start/stop."""
        ctx = callback_context
        if not ctx.triggered:
            return no_update, no_update

        trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]

        if trigger_id == "auto-start":
            # TODO: Start the auto pick & place operation
            LOGGER.info("Starting auto pick & place operation")
            return (
                {"running": True, "message": "Auto operation started"},
                dmc.Notification(
                    title="Auto Operation",
                    message="Auto pick & place operation started",
                    color="green",
                    action="show",
                ),
            )
        elif trigger_id == "auto-stop":
            # TODO: Stop the auto pick & place operation
            LOGGER.info("Stopping auto pick & place operation")
            return (
                {"running": False, "message": "Auto operation stopped"},
                dmc.Notification(
                    title="Auto Operation",
                    message="Auto pick & place operation stopped",
                    color="orange",
                    action="show",
                ),
            )

        return no_update, no_update

    @app.server.route("/video_feed")
    def video_feed():
        """Stream Motion JPEG video for fullscreen playback."""
        print("DEBUG: /video_feed endpoint accessed")

        def generate():
            boundary = b'--frame\r\n'
            frame_count = 0
            while True:
                frame_data = MJPEG_STREAM.get_frame()

                if not frame_data:
                    time.sleep(0.033)
                    continue

                frame_count += 1
                if frame_count == 1:
                    logging.debug(
                        "/video_feed serving first frame, size=%s bytes", len(frame_data))

                yield boundary
                yield b'Content-Type: image/jpeg\r\n'
                yield b'Content-Length: ' + str(len(frame_data)).encode() + b'\r\n\r\n'
                yield frame_data
                yield b'\r\n'
                time.sleep(0.01)  # Small delay to avoid hammering

        return app.server.response_class(
            generate(),
            mimetype='multipart/x-mixed-replace; boundary=frame'
        )

    @app.callback(
        Output("calibration-progress", "value"),
        Output("calibration-progress", "style"),
        Output("calibration-progress-text", "children"),
        Output("calibration-progress-text", "style"),
        Input("calibration-progress-interval", "n_intervals"),
    )
    def _update_calibration_progress(_):
        """Update progress bar and text from global calibration progress tracker."""
        progress = CALIBRATION_PROGRESS.get("value", 0)
        current_frame = CALIBRATION_PROGRESS.get("current_frame", 0)
        max_frames = CALIBRATION_PROGRESS.get("max_frames", 0)

        # Show progress bar and text only if capturing (1-99%)
        if progress > 0 and progress < 100:
            text = f"Capturing images: {current_frame}/{max_frames}"
            return progress, {"display": "block"}, text, {"display": "block", "marginBottom": "8px"}

        return 0, {"display": "none"}, "", {"display": "none"}

    @app.server.route("/camera_frame.jpg")
    def camera_frame():
        """Get the latest camera frame as a JPEG image."""
        frame_data = MJPEG_STREAM.get_frame()
        if not frame_data:
            # Return placeholder as JPEG bytes
            placeholder = _placeholder_frame("Camera stream offline")
            success, buffer = cv2.imencode(".jpg", placeholder)
            if success:
                frame_data = buffer.tobytes()
            else:
                return app.server.response_class(
                    b"",
                    mimetype='image/jpeg'
                )
        return app.server.response_class(
            frame_data,
            mimetype='image/jpeg'
        )

    @app.server.route("/raw_camera_frame.jpg")
    def raw_camera_frame():
        """Get the latest raw camera frame (no overlays) as a JPEG image."""
        frame_data = RAW_STREAM.get_frame()
        if not frame_data:
            placeholder = _placeholder_frame("Camera stream offline")
            success, buffer = cv2.imencode(".jpg", placeholder)
            if success:
                frame_data = buffer.tobytes()
            else:
                return app.server.response_class(
                    b"",
                    mimetype='image/jpeg'
                )
        return app.server.response_class(
            frame_data,
            mimetype='image/jpeg'
        )

    return app


app = create_app()
server = app.server


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    app.run(debug=True)
