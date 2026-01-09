"""
Dash calibration UI for the Codroid bring-up workflow.
"""

from __future__ import annotations
from vision.camera_check import list_oak_devices
from vision import calibration, markers, plate, workspace
from onrobot.twofg import TWOFG as TwoFG7Gripper
from onrobot.device import Device as GripperDevice
from codroid_api import CodroidAPI, RobotSession
from codroid_api.robot_session import RobotPosture
from app.orchestrator import BootstrapDemo, DemoConfig

import asyncio
import base64
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

from . import constants as cfg
from .constants import (
    ASSETS_DIR,
    DEFAULT_BOARD_PATH,
    DEFAULT_BUNDLE_PATH,
    DEFAULT_CAPTURE_DIR,
    DEFAULT_FPS,
    DEFAULT_MODEL_DIR,
    DEFAULT_PLATE_PATH,
    DEFAULT_RESOLUTION,
    DEFAULT_ROBOT_CALIBRATION_PATH,
    DEFAULT_WORKSPACE_PATH,
    DEFAULT_DROP_MARKER_ID,
    DROP_TARGET_PATH,
    FLANGE_BUTTON_PORT,
    FLANGE_BUTTON_PORTS,
    GRIPPER_CALIBRATION_WIDTH_MM,
    GRIPPER_CLOSE_WIDTH_DEFAULT,
    GRIPPER_FORCE_DEFAULT,
    GRIPPER_FORCE_MAX,
    GRIPPER_FORCE_MIN,
    GRIPPER_OPEN_WIDTH_DEFAULT,
    GRIPPER_SPEED_DEFAULT,
    GRIPPER_SPEED_MIN,
    MIN_PLATE_MARKERS,
    PLATE_MOVEMENT_THRESHOLD_PX,
    PLATE_TEST_CAPTURE_DIR,
    PLATE_TEST_GRIP_TIMEOUT_S,
    PLATE_TEST_MOVE_HOLD_S,
    PLATE_TEST_MOVE_TIMEOUT_S,
    PLATE_TEST_MOTION_EPS_MM,
    PLATE_TEST_SETTLE_S,
    PLATE_TEST_TOLERANCE_MM,
    PLATE_TEST_Z_TOLERANCE_MM,
    ROBOT_COORDINATE_ID,
    ROBOT_POSTURE_MAX_AGE_S,
    VERIFICATION_HOLD_SECONDS,
    VERIFICATION_MOVE_TIMEOUT_S,
    VERIFICATION_MOTION_EPS_MM,
    VERIFICATION_RAISE_MM,
    VERIFICATION_SETTLE_S,
    VERIFICATION_TOLERANCE_MM,
    VERIFICATION_Z_TOLERANCE_MM,
    ROBOT_COORDINATE_TIMEOUT_S,
)
from .state import (
    CODROID_DEFAULTS,
    DEFAULT_GRIPPER_IP,
    DEFAULT_ROBOT_URI,
    GRIPPER_DEFAULTS,
    PLATE_STATUS,
    ROBOT_CALIBRATION_GUIDANCE,
    SETTINGS,
    SESSION,
    _commit_settings,
)
from .overlays import (
    _apply_workspace_overlay,
    _draw_label,
    _plate_center_px,
    draw_charuco_overlay,
    draw_plate_overlay,
    draw_workspace_overlay,
)
from .camera import (
    CALIBRATION_PROGRESS,
    MJPEG_STREAM,
    RAW_STREAM,
    _charuco_detection_counts,
    _clear_calibration_outputs,
    _default_transform,
    _format_fps,
    attach_plate_calibration,
    capture_color_frame,
    capture_depth_snapshot,
    compute_calibration_bundle,
    compute_plate_calibration_from_latest,
    compute_workspace_from_latest,
    get_latest_frame,
    load_calibration_bundle,
    load_depth_snapshot,
    load_workspace_definition,
    save_workspace_definition,
    start_camera_stream,
    stop_camera_stream,
)
from .helpers import (
    _analyze_plate_test_capture,
    _current_coordinate_id,
    _ensure_posture_axes,
    _get_robot_session,
    _load_drop_target,
    _load_robot_calibration,
    _plate_is_stable,
    _plate_test_capture_path,
    _posture_to_dict,
    _record_pick_cycle,
    _save_drop_target,
    _save_plate_test_capture,
    _save_robot_calibration,
    _set_plate_test_state,
    _set_robot_guidance,
    _update_robot_calibration_meta,
    _robot_posture_ready,
    _safe_float,
    _safe_int,
    _safe_path,
)
from .layout import build_layout
from .ui_helpers import (
    _encode_frame,
    _format_device_list,
    _model_options,
    _placeholder_frame,
    format_gpu_status,
)

LOGGER = logging.getLogger(__name__)


if SETTINGS.calibration_bundle_path:
    cfg.DEFAULT_BUNDLE_PATH = Path(SETTINGS.calibration_bundle_path)
    DEFAULT_BUNDLE_PATH = cfg.DEFAULT_BUNDLE_PATH
if SETTINGS.workspace_path:
    cfg.DEFAULT_WORKSPACE_PATH = Path(SETTINGS.workspace_path)
    DEFAULT_WORKSPACE_PATH = cfg.DEFAULT_WORKSPACE_PATH
if SETTINGS.robot_calibration_path:
    cfg.DEFAULT_ROBOT_CALIBRATION_PATH = Path(SETTINGS.robot_calibration_path)
    DEFAULT_ROBOT_CALIBRATION_PATH = cfg.DEFAULT_ROBOT_CALIBRATION_PATH

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
    _load_drop_target()
    if SETTINGS.last_model:
        try:
            activate_model(Path(SETTINGS.last_model))
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("Failed to restore last model: %s", exc)
    if DEFAULT_ROBOT_CALIBRATION_PATH.exists():
        _load_robot_calibration(DEFAULT_ROBOT_CALIBRATION_PATH)


_restore_persisted_state()

# Global calibration progress tracker

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

def _format_fps(capture_fps: float, stream_fps: float) -> str:
    cap_text = f"{capture_fps:.1f}" if capture_fps > 0 else "--"
    stream_text = f"{stream_fps:.1f}" if stream_fps > 0 else "--"
    return f"FPS — capture: {cap_text}, stream: {stream_text}"


FRAME_PROCESSOR = FrameProcessor()


def _format_fps(capture_fps: float, stream_fps: float) -> str:
    cap_text = f"{capture_fps:.1f}" if capture_fps > 0 else "--"
    stream_text = f"{stream_fps:.1f}" if stream_fps > 0 else "--"
    return f"FPS — capture: {cap_text}, stream: {stream_text}"


FRAME_PROCESSOR = FrameProcessor()


def _format_fps(capture_fps: float, stream_fps: float) -> str:
    cap_text = f"{capture_fps:.1f}" if capture_fps > 0 else "--"
    stream_text = f"{stream_fps:.1f}" if stream_fps > 0 else "--"
    return f"FPS — capture: {cap_text}, stream: {stream_text}"


FRAME_PROCESSOR = FrameProcessor()









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
                dmc.Badge(f"{gpu_info['gpu_count']} GPU(s)", color="brand"),
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


def _sanitize_filename(name: str, fallback: str = "model.bin") -> str:
    base = Path(name).name
    safe = re.sub(r"[^A-Za-z0-9._-]", "_", base)
    return safe or fallback


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
        _commit_settings(last_model=None)
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
        _commit_settings(last_model=str(model_path))
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


def _pixel_to_workspace_mm(
    workspace_def: workspace.WorkspaceDefinition,
    pixel_xy: np.ndarray,
) -> Tuple[float, float]:
    bundle = SESSION.calibration_bundle
    if bundle is None or bundle.pixel_to_mm <= 0:
        return _workspace_pixel_to_mm(
            workspace_def, (float(pixel_xy[0]), float(pixel_xy[1]))
        )
    scale = float(bundle.pixel_to_mm)
    camera_xyz = np.array(
        [float(pixel_xy[0]) * scale, float(pixel_xy[1]) * scale, 0.0],
        dtype=np.float64,
    )
    origin = workspace_def.origin_camera
    basis = np.column_stack(
        [workspace_def.x_axis_camera, workspace_def.y_axis_camera]
    )
    coeffs, *_ = np.linalg.lstsq(basis, camera_xyz - origin, rcond=None)
    return float(coeffs[0]), float(coeffs[1])


def _resolve_plate_test_targets(
    layout_path: Path,
    workspace_def: workspace.WorkspaceDefinition,
) -> Tuple[list[dict], dict]:
    calibration_result = compute_plate_calibration_from_latest(layout_path)
    plate_points = calibration_result.plate.robot_calibration_points()
    if len(plate_points) < 3:
        raise RuntimeError("Calibration plate layout is missing targets 1-3.")
    plate_xy = np.array(
        [[pt.center_mm[0], pt.center_mm[1]] for pt in plate_points],
        dtype=np.float64,
    ).reshape(-1, 1, 2)
    homography = calibration_result.homography
    if homography is None:
        raise RuntimeError("Plate homography unavailable.")
    inv_h = np.linalg.inv(homography)
    pixel_xy = cv2.perspectiveTransform(plate_xy, inv_h).reshape(-1, 2)
    targets = []
    for point, px in zip(plate_points, pixel_xy):
        workspace_xy = _pixel_to_workspace_mm(workspace_def, px)
        sequence = point.sequence if point.sequence is not None else len(targets) + 1
        targets.append(
            {
                "sequence": int(sequence),
                "workspace_mm": workspace_xy,
                "plate_mm": (float(point.center_mm[0]), float(point.center_mm[1])),
                "pixel": (float(px[0]), float(px[1])),
            }
        )
    PLATE_STATUS.update(
        {
            "count": len(calibration_result.detections),
            "ok": len(calibration_result.detections) >= MIN_PLATE_MARKERS,
            "stable": len(calibration_result.detections) >= MIN_PLATE_MARKERS,
            "center_px": None,
            "movement_px": None,
            "timestamp": time.time(),
            "message": f"Plate markers detected: {len(calibration_result.detections)}",
        }
    )
    return targets, {
        "count": len(calibration_result.detections),
        "reprojection_error_mm": calibration_result.reprojection_error_mm,
    }


def _resolve_plate_test_coord5_targets(
    layout_path: Path,
) -> Tuple[list[dict], dict, workspace.RigidTransform, bool]:
    if not SESSION.robot_calibration_points or len(SESSION.robot_calibration_points) < 3:
        raise RuntimeError("Robot calibration points are missing; re-run robot calibration.")

    calibration_result = compute_plate_calibration_from_latest(layout_path)
    plate_layout = calibration_result.plate
    plate_points = plate_layout.robot_calibration_points()
    if len(plate_points) < 3:
        raise RuntimeError("Calibration plate layout is missing targets 1-3.")

    reference_initialized = False
    if SESSION.plate_reference_homography is None:
        bundle = SESSION.calibration_bundle
        if bundle is not None and bundle.plate_homography is not None:
            SESSION.plate_reference_homography = bundle.plate_homography.copy()
        else:
            SESSION.plate_reference_homography = calibration_result.homography.copy()
            reference_initialized = True

    origin_mm = np.asarray(plate_points[0].center_mm, dtype=np.float64)
    marker_base: list[np.ndarray] = []
    marker_curr: list[np.ndarray] = []
    marker_info: dict[int, dict] = {}
    reference_h = SESSION.plate_reference_homography
    for marker_id, detection in calibration_result.detections.items():
        marker = plate_layout.markers.get(marker_id)
        if marker is None:
            continue
        base_pt = marker.center_mm - origin_mm
        curr_pt = plate.image_points_to_plate_mm(
            reference_h, np.asarray([detection.center], dtype=np.float64)
        )[0]
        marker_base.append(np.asarray(base_pt, dtype=np.float64))
        marker_curr.append(np.asarray(curr_pt - origin_mm, dtype=np.float64))
        marker_info[int(marker_id)] = {
            "plate_mm": (
                float(marker.center_mm[0]),
                float(marker.center_mm[1]),
            ),
            "base_xy": (float(base_pt[0]), float(base_pt[1])),
        }

    if len(marker_base) < 2:
        raise RuntimeError("Not enough plate markers detected for offset tracking.")

    r2, t2 = _rigid_transform_2d(
        np.vstack(marker_base),
        np.vstack(marker_curr),
    )

    robot_pts = np.asarray(
        [
            [
                float(pt["x"]),
                float(pt["y"]),
                float(pt["z"]),
            ]
            for pt in SESSION.robot_calibration_points[:3]
        ],
        dtype=np.float64,
    )
    baseline_transform = _rigid_transform(
        np.asarray(
            [
                [
                    float(pt.center_mm[0] - origin_mm[0]),
                    float(pt.center_mm[1] - origin_mm[1]),
                    0.0,
                ]
                for pt in plate_points[:3]
            ],
            dtype=np.float64,
        ),
        robot_pts,
    )

    targets = []
    for point in plate_points:
        base_xy = np.array(
            [
                float(point.center_mm[0] - origin_mm[0]),
                float(point.center_mm[1] - origin_mm[1]),
            ],
            dtype=np.float64,
        )
        adj_xy = (r2 @ base_xy) + t2
        sequence = point.sequence if point.sequence is not None else len(targets) + 1
        targets.append(
            {
                "sequence": int(sequence),
                "coord5_xy": (float(adj_xy[0]), float(adj_xy[1])),
                "plate_mm": (float(point.center_mm[0]), float(point.center_mm[1])),
            }
        )

    for marker_id, meta in marker_info.items():
        base_xy = np.asarray(meta["base_xy"], dtype=np.float64)
        adj_xy = (r2 @ base_xy) + t2
        meta["coord5_xy"] = (float(adj_xy[0]), float(adj_xy[1]))

    PLATE_STATUS.update(
        {
            "count": len(calibration_result.detections),
            "ok": len(calibration_result.detections) >= MIN_PLATE_MARKERS,
            "stable": len(calibration_result.detections) >= MIN_PLATE_MARKERS,
            "center_px": None,
            "movement_px": None,
            "timestamp": time.time(),
            "message": f"Plate markers detected: {len(calibration_result.detections)}",
        }
    )

    return targets, {
        "count": len(calibration_result.detections),
        "reprojection_error_mm": calibration_result.reprojection_error_mm,
        "reference_initialized": reference_initialized,
        "markers_coord5": marker_info,
    }, baseline_transform, reference_initialized


async def _await_grip_status(timeout_s: float) -> bool:
    gripper = SESSION.gripper
    if gripper is None:
        return False
    t_index = GRIPPER_DEFAULTS["t_index"]
    deadline = time.monotonic() + max(timeout_s, 0.1)
    while time.monotonic() < deadline:
        try:
            if gripper.isGripped(t_index=t_index):
                return True
        except Exception as exc:  # noqa: BLE001
            LOGGER.debug("Failed to poll gripper grip state: %s", exc)
            return False
        await asyncio.sleep(0.2)
    return False


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


def _robot_to_workspace_mm(
    workspace_def: workspace.WorkspaceDefinition,
    robot_xyz: np.ndarray,
) -> Tuple[float, float]:
    """Map a robot XYZ point back into workspace (mm) assuming z on the plane."""
    cam_to_robot = workspace_def.camera_to_robot.as_matrix()
    robot_h = np.hstack([robot_xyz, 1.0])
    camera_h = np.linalg.inv(cam_to_robot) @ robot_h
    camera_xyz = camera_h[:3]
    origin = workspace_def.origin_camera
    basis = np.column_stack([workspace_def.x_axis_camera, workspace_def.y_axis_camera])
    try:
        coeffs, *_ = np.linalg.lstsq(basis, camera_xyz - origin, rcond=None)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Unable to project robot pose into workspace: {exc}") from exc
    return float(coeffs[0]), float(coeffs[1])


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


def _rigid_transform_2d(
    from_pts: np.ndarray,
    to_pts: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    if from_pts.shape != to_pts.shape:
        raise ValueError("Point clouds must have the same shape.")
    if from_pts.shape[0] < 2:
        raise ValueError("At least 2 points are required for a 2D rigid transform.")
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
    return r, t


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


async def _run_robot_plate_calibration(
    robot_uri: str,
    plate_layout: plate.CalibrationPlate,
) -> Tuple[list[dict], dict]:
    """Capture three robot poses via flange button 1 and calibrate coord ID 5."""
    session = _get_robot_session()

    def _ensure_plate_baseline() -> bool:
        if SESSION.robot_calibration_baseline_px is not None:
            return True
        center = PLATE_STATUS.get("center_px")
        if center is not None:
            SESSION.robot_calibration_baseline_px = np.asarray(
                center, dtype=np.float64
            )
            return True
        frame, _depth = get_latest_frame()
        if frame is None:
            return False
        try:
            calibration_result = plate.estimate_plate_calibration(frame, plate_layout)
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("Plate baseline detection failed: %s", exc)
            return False
        center_px = _plate_center_px(calibration_result.detections)
        if center_px is None:
            return False
        SESSION.robot_calibration_baseline_px = np.asarray(
            center_px, dtype=np.float64
        )
        PLATE_STATUS.update(
            {
                "count": len(calibration_result.detections),
                "ok": len(calibration_result.detections) >= MIN_PLATE_MARKERS,
                "stable": len(calibration_result.detections) >= MIN_PLATE_MARKERS,
                "center_px": (float(center_px[0]), float(center_px[1])),
                "movement_px": None,
                "timestamp": time.time(),
                "message": "Plate baseline refreshed from live detection.",
            }
        )
        return True

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
            if SESSION.robot_calibration_baseline_px is None:
                _ensure_plate_baseline()
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
            "Powering on robot and selecting CoordinateId=0 (base) for calibration...",
            status="running",
            step=1,
            verification_ready=False,
        )
        await robot_api.power_on()
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

            posture = _ensure_posture_axes(posture, session.position_snapshot())
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

        def _coord_axis(keys: Tuple[str, ...]) -> float:
            for key in keys:
                if key in coord_frame:
                    return _safe_float(coord_frame.get(key), 0.0)
            return 0.0

        coord_frame["a"] = _coord_axis(("a", "rx", "roll"))
        coord_frame["b"] = _coord_axis(("b", "ry", "pitch"))
        coord_frame["c"] = _coord_axis(("c", "rz", "yaw"))

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
                    "brand",
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
                    "brand",
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
                    "yellow",
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
                return status, "Robot: online", "brand", {"status": "online", "message": status}
            if SESSION.robot_connection_attempted:
                if SESSION.robot_last_connection_error:
                    error = SESSION.robot_last_connection_error
                    return error, "Robot: error", "yellow", {"status": "error", "message": error}
                return no_update, no_update, no_update, no_update
        try:
            response = await _probe_robot(uri)
            status = f"Robot connected to {response.get('host')}:{response.get('port')}."
            SESSION.robot_connection_attempted = True
            SESSION.robot_last_connection_error = None
            return status, "Robot: online", "brand", {"status": "online", "message": status}
        except Exception as exc:
            error = f"Robot connection failed: {exc}"
            SESSION.robot_connection_attempted = True
            SESSION.robot_last_connection_error = error
            return error, "Robot: error", "yellow", {"status": "error", "message": error}

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
            min_width, max_width = _resolve_gripper_limits()
            if min_width > max_width:
                min_width, max_width = max_width, min_width
            if prevent_overclose:
                current_width = SESSION.gripper.get_ext_width(
                    t_index=GRIPPER_DEFAULTS["t_index"]
                )
                if isinstance(current_width, (int, float)) and current_width >= 0:
                    if safe_width < current_width:
                        safe_width = current_width
            safe_width = min(max(safe_width, min_width), max_width)

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

    def _resolve_gripper_limits() -> Tuple[float, float]:
        min_width = SESSION.gripper_min_width
        max_width = SESSION.gripper_max_width
        if SESSION.gripper is not None:
            t_index = GRIPPER_DEFAULTS["t_index"]
            status_snapshot = SESSION.gripper.get_status_snapshot(t_index=t_index)
            if status_snapshot:
                snapshot_min = status_snapshot.get("min_external_width")
                snapshot_max = status_snapshot.get("max_external_width")
                if isinstance(snapshot_min, (int, float)) and snapshot_min >= 0:
                    min_width = float(snapshot_min)
                    SESSION.gripper_min_width = min_width
                if isinstance(snapshot_max, (int, float)) and snapshot_max >= 0:
                    max_width = float(snapshot_max)
                    SESSION.gripper_max_width = max_width
            if min_width is None:
                try:
                    value = SESSION.gripper.get_min_ext_width(t_index=t_index)
                except Exception as exc:  # noqa: BLE001
                    LOGGER.debug("Failed to read gripper min width: %s", exc)
                else:
                    if isinstance(value, (int, float)) and value >= 0:
                        min_width = float(value)
            if max_width is None:
                try:
                    value = SESSION.gripper.get_max_ext_width(t_index=t_index)
                except Exception as exc:  # noqa: BLE001
                    LOGGER.debug("Failed to read gripper max width: %s", exc)
                else:
                    if isinstance(value, (int, float)) and value >= 0:
                        max_width = float(value)
        if min_width is None:
            min_width = GRIPPER_CLOSE_WIDTH_DEFAULT
        if max_width is None:
            max_width = GRIPPER_OPEN_WIDTH_DEFAULT
        if min_width > max_width:
            LOGGER.warning(
                "Gripper width limits inverted (min=%.1f, max=%.1f); swapping.",
                min_width,
                max_width,
            )
            min_width, max_width = max_width, min_width
        return min_width, max_width


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
                                color="yellow",
                            )
                        ),
                        "Gripper: error",
                        "yellow",
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
                            color="yellow",
                        )
                    ),
                    "Gripper: error",
                    "yellow",
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
            SESSION.gripper.start_status_stream()

            # Query gripper state
            width = SESSION.gripper.get_ext_width(t_index=t_index)
            force = SESSION.gripper.get_force(t_index=t_index)
            is_gripped = SESSION.gripper.isGripped(t_index=t_index)
            SESSION.robot_has_object = bool(is_gripped)

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
                "brand",
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
        Output("footer-gripper-width", "disabled"),
        Output("footer-gripper-force", "disabled"),
        Output("footer-gripper-speed", "disabled"),
        Output("footer-gripper-width-decrease", "disabled"),
        Output("footer-gripper-width-increase", "disabled"),
        Output("footer-gripper-force-decrease", "disabled"),
        Output("footer-gripper-force-increase", "disabled"),
        Output("footer-gripper-speed-decrease", "disabled"),
        Output("footer-gripper-speed-increase", "disabled"),
        Input("gripper-connection-state", "data"),
    )
    def _sync_footer_gripper_disabled(
        gripper_state: Optional[dict],
    ) -> Tuple[bool, bool, bool, bool, bool, bool, bool, bool, bool]:
        status = (gripper_state or {}).get("status", "offline")
        disabled = status != "online" or bool(SESSION.gripper_busy)
        return (
            disabled,
            disabled,
            disabled,
            disabled,
            disabled,
            disabled,
            disabled,
            disabled,
            disabled,
        )

    @app.callback(
        Output("footer-gripper-width", "min", allow_duplicate=True),
        Output("footer-gripper-width", "max", allow_duplicate=True),
        Input("gripper-connection-state", "data"),
        prevent_initial_call=True,
    )
    def _sync_gripper_width_limits(gripper_state: Optional[dict]) -> Tuple[float, float]:
        if (gripper_state or {}).get("status") != "online":
            return no_update, no_update
        if SESSION.gripper is None:
            return no_update, no_update
        min_width, max_width = _resolve_gripper_limits()
        return min_width, max_width

    @app.callback(
        Output("footer-visibility", "data"),
        Output("app-footer", "className"),
        Output("footer-toggle", "children"),
        Output("app-main", "className"),
        Output("footer-body", "style"),
        Input("footer-toggle", "n_clicks"),
        State("footer-visibility", "data"),
    )
    def _toggle_footer_visibility(
        n_clicks: Optional[int],
        state: Optional[dict],
    ) -> Tuple[dict, str, str, str, dict]:
        collapsed = bool((state or {}).get("collapsed", False))
        if not n_clicks:
            # honor current state without toggling on first render
            class_name = "app-footer is-collapsed" if collapsed else "app-footer"
            label = "Show footer" if collapsed else "Hide footer"
            main_class = "app-main footer-collapsed" if collapsed else "app-main footer-expanded"
            style = {"display": "none"} if collapsed else {}
            return state or {"collapsed": False}, class_name, label, main_class, style

        new_collapsed = not collapsed
        class_name = "app-footer is-collapsed" if new_collapsed else "app-footer"
        label = "Show footer" if new_collapsed else "Hide footer"
        main_class = "app-main footer-collapsed" if new_collapsed else "app-main footer-expanded"
        style = {"display": "none"} if new_collapsed else {}
        return {"collapsed": new_collapsed}, class_name, label, main_class, style

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
        Output("settings-status", "children"),
        Output("settings-robot-host", "value"),
        Output("settings-marker-a", "value"),
        Output("settings-marker-b", "value"),
        Output("settings-marker-size", "value"),
        Output("settings-calib-path", "value"),
        Output("settings-workspace-path", "value"),
        Output("settings-robot-calib-path", "value"),
        Input("settings-save", "n_clicks"),
        Input("settings-reset", "n_clicks"),
        State("settings-robot-host", "value"),
        State("settings-marker-a", "value"),
        State("settings-marker-b", "value"),
        State("settings-marker-size", "value"),
        State("settings-calib-path", "value"),
        State("settings-workspace-path", "value"),
        State("settings-robot-calib-path", "value"),
        prevent_initial_call=True,
    )
    def _handle_settings(
        _save: int,
        _reset: int,
        robot_host: Optional[str],
        marker_a: Optional[float],
        marker_b: Optional[float],
        marker_size: Optional[float],
        calib_path: Optional[str],
        workspace_path: Optional[str],
        robot_calib_path: Optional[str],
    ) -> Tuple[str, Optional[str], Optional[int], Optional[int], Optional[float], str, str, str]:
        trigger = callback_context.triggered_id
        global CODROID_DEFAULTS, DEFAULT_ROBOT_URI, GRIPPER_DEFAULTS, DEFAULT_GRIPPER_IP
        global DEFAULT_BUNDLE_PATH, DEFAULT_WORKSPACE_PATH, DEFAULT_ROBOT_CALIBRATION_PATH
        default_marker_a = 1
        default_marker_b = 2
        default_marker_size = 40.0
        default_calib = str(DEFAULT_BUNDLE_PATH)
        default_workspace = str(DEFAULT_WORKSPACE_PATH)
        default_robot_calib = str(DEFAULT_ROBOT_CALIBRATION_PATH)

        if trigger == "settings-reset":
            _commit_settings(
                marker_ids=(default_marker_a, default_marker_b),
                marker_size_mm=default_marker_size,
                calibration_bundle_path=None,
                workspace_path=None,
                robot_calibration_path=str(DEFAULT_ROBOT_CALIBRATION_PATH),
            )
            return (
                "Settings reset to defaults (env + built-ins).",
                default_marker_a,
                default_marker_b,
                default_marker_size,
                default_calib,
                default_workspace,
                default_robot_calib,
            )

        # Save path: sanitize inputs
        host_val = (robot_host or "").strip() or None
        marker_a_val = int(marker_a) if marker_a is not None else default_marker_a
        marker_b_val = int(marker_b) if marker_b is not None else default_marker_b
        marker_size_val = float(marker_size) if marker_size is not None else default_marker_size
        calib_path_val = (calib_path or "").strip() or str(DEFAULT_BUNDLE_PATH)
        workspace_path_val = (workspace_path or "").strip() or str(DEFAULT_WORKSPACE_PATH)
        robot_calib_path_val = (robot_calib_path or "").strip() or str(DEFAULT_ROBOT_CALIBRATION_PATH)

        _commit_settings(
            robot_host=host_val,
            marker_ids=(marker_a_val, marker_b_val),
            marker_size_mm=marker_size_val,
            calibration_bundle_path=calib_path_val,
            workspace_path=workspace_path_val,
            robot_calibration_path=robot_calib_path_val,
        )

        CODROID_DEFAULTS = _load_codroid_defaults()
        DEFAULT_ROBOT_URI = CODROID_DEFAULTS["robot_ws_url"]
        GRIPPER_DEFAULTS = _load_gripper_defaults()
        DEFAULT_GRIPPER_IP = GRIPPER_DEFAULTS["compute_box_ip"]
        DEFAULT_BUNDLE_PATH = Path(calib_path_val)
        DEFAULT_WORKSPACE_PATH = Path(workspace_path_val)
        DEFAULT_ROBOT_CALIBRATION_PATH = Path(robot_calib_path_val)
        SESSION.config.marker_ids = (marker_a_val, marker_b_val)
        SESSION.config.marker_size_mm = marker_size_val

        return (
            "Settings saved.",
            marker_a_val,
            marker_b_val,
            marker_size_val,
            calib_path_val,
            workspace_path_val,
            robot_calib_path_val,
        )

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
        Output("drop-position-metrics", "children", allow_duplicate=True),
        Input("teach-drop-position", "n_clicks"),
        Input("execute-pick", "n_clicks"),
        Input("execute-drop", "n_clicks"),
        Input("execute-pick-drop", "n_clicks"),
        Input("manual-pick-closest", "n_clicks"),
        Input("manual-release", "n_clicks"),
        Input("manual-pick-place-target", "n_clicks"),
        Input("gripper-open", "n_clicks"),
        Input("gripper-close", "n_clicks"),
        State("pick-detection-select", "value"),
        State("pick-approach-height", "value"),
        State("pick-height", "value"),
        State("drop-height", "value"),
        State("estimate-object-width-toggle", "checked"),
        State("robot-uri", "value"),
        prevent_initial_call=True,
    )
    async def _handle_pick_place(
        _teach_drop: int,
        _pick_clicks: int,
        _drop_action: int,
        _pick_drop_clicks: int,
        _manual_pick: int,
        _manual_release: int,
        _manual_pick_place: int,
        _gripper_open: int,
        _gripper_close: int,
        selected_detection: Optional[str],
        approach_height: Optional[float],
        pick_height: Optional[float],
        drop_height: Optional[float],
        estimate_width: Optional[bool],
        robot_uri: Optional[str],
    ) -> Tuple[str, str]:
        trigger = callback_context.triggered_id
        frame, _depth = get_latest_frame()
        drop_text = no_update
        gripper_min_width, gripper_max_width = _resolve_gripper_limits()
        gripper_open_val = GRIPPER_OPEN_WIDTH_DEFAULT
        gripper_close_val = GRIPPER_CLOSE_WIDTH_DEFAULT
        if trigger is None:
            return no_update, no_update
        if frame is None and trigger not in {"teach-drop-position", "manual-release", "gripper-open", "gripper-close"}:
            return "Camera stream is not running.", drop_text

        workspace_def = SESSION.workspace_definition
        approach_mm = _safe_float(approach_height, 30.0)
        pick_mm = _safe_float(pick_height, 0.0)
        drop_mm = _safe_float(drop_height, 0.0)

        def _estimate_width_mm(det: dict) -> Optional[float]:
            if not estimate_width or workspace_def is None:
                return None
            bbox = det.get("bbox")
            if not bbox:
                return None
            x1, y1, x2, y2 = bbox
            width_px = max(abs(x2 - x1), abs(y2 - y1))
            if width_px <= 0:
                return None
            x_scale = np.linalg.norm(workspace_def.x_axis_camera)
            y_scale = np.linalg.norm(workspace_def.y_axis_camera)
            denom = (x_scale + y_scale) / 2.0
            if denom <= 0:
                return None
            mm_per_px = 1.0 / denom
            return width_px * mm_per_px

        if trigger == "teach-drop-position":
            if workspace_def is None:
                return "Workspace is not defined; compute it first.", drop_text
            session = _get_robot_session()
            if not session.posture_seen():
                return "Robot posture stream not available yet.", drop_text
            pos = session.position_snapshot()
            robot_xyz = np.array([pos.x, pos.y, pos.z], dtype=float)
            try:
                drop_mm_xy = _robot_to_workspace_mm(workspace_def, robot_xyz)
            except Exception as exc:  # noqa: BLE001
                return f"Failed to record drop position: {exc}", drop_text
            SESSION.drop_target_mm = drop_mm_xy
            SESSION.drop_target_orientation = (pos.a, pos.b, pos.c)
            SESSION.drop_target_robot = (pos.x, pos.y, pos.z)
            _save_drop_target(
                drop_mm_xy,
                SESSION.drop_target_orientation,
                SESSION.drop_target_robot,
            )
            _save_settings(SETTINGS.model_copy(update={"drop_target": drop_mm_xy}))
            SESSION.last_action = "Drop position taught."
            drop_text = (
                f"Drop at {drop_mm_xy[0]:.1f}, {drop_mm_xy[1]:.1f} mm · "
                f"a={pos.a:.1f}, b={pos.b:.1f}, c={pos.c:.1f}"
            )
            return "Drop position recorded from robot pose.", drop_text

        if trigger == "gripper-open":
            return _command_gripper(gripper_max_width, "opened"), drop_text

        if trigger == "gripper-close":
            return _command_gripper(gripper_min_width, "closed"), drop_text

        if workspace_def is None:
            return "Workspace is not defined; run workspace calibration first.", drop_text
        approach_mm = _safe_float(approach_height, 30.0)
        pick_mm = _safe_float(pick_height, 0.0)
        drop_mm = _safe_float(drop_height, 0.0)
        gripper_open_val = GRIPPER_OPEN_WIDTH_DEFAULT
        gripper_close_val = GRIPPER_CLOSE_WIDTH_DEFAULT
        if not SESSION.detections and trigger in {"execute-pick", "execute-pick-drop", "manual-pick-closest", "manual-pick-place-target"}:
            return "No detections available; run detection first.", drop_text

        if trigger in {"execute-pick", "execute-pick-drop"} and selected_detection is None:
            return "Select a detection to pick.", drop_text

        def _select_detection_for_action() -> dict:
            if trigger in {"manual-pick-closest", "manual-pick-place-target"}:
                session = _get_robot_session()
                pos = session.position_snapshot()
                robot_xy = np.array([pos.x, pos.y], dtype=float)
                try:
                    workspace_xy = _robot_to_workspace_mm(
                        workspace_def, np.array([pos.x, pos.y, pos.z], dtype=float)
                    )
                except Exception:
                    workspace_xy = None
                best_det = SESSION.detections[0]
                best_dist = float("inf")
                for det in SESSION.detections:
                    wp = det.get("workspace_mm")
                    if wp is None:
                        continue
                    dist = np.linalg.norm(np.array(wp[:2]) - np.array(workspace_xy or wp[:2]))
                    if dist < best_dist:
                        best_dist = dist
                        best_det = det
                return best_det
            det_idx = int(selected_detection)
            return SESSION.detections[det_idx]

        try:
            detection = _select_detection_for_action()
        except Exception:
            return "Invalid detection selection.", drop_text

        center = detection.get("center")
        if center is None:
            return "Detection is missing center coordinates.", drop_text
        workspace_mm = detection.get("workspace_mm")
        if workspace_mm is None:
            workspace_mm = _workspace_pixel_to_mm(workspace_def, center)

        width_guess = _estimate_width_mm(detection)
        if width_guess is not None:
            gripper_close_val = max(gripper_close_val, width_guess * 0.9)
            gripper_open_val = max(gripper_open_val, width_guess + 5.0)

        pick_pose = _pose_from_workspace_mm(
            workspace_def, workspace_mm[0], workspace_mm[1], pick_mm)
        approach_pose = _pose_from_workspace_mm(
            workspace_def, workspace_mm[0], workspace_mm[1], approach_mm
        )
        SESSION.last_pick_mm = workspace_mm

        def _resolve_drop_target() -> Tuple[float, float]:
            if SESSION.drop_target_mm:
                return SESSION.drop_target_mm
            raise RuntimeError("No drop location taught yet.")

        async def _pick_action(robot_api: CodroidAPI) -> None:
            await robot_api.power_on()
            await robot_api.move_to_cartesian_target_linear(robot_api.build_target_cpos(**approach_pose))
            await robot_api.move_to_cartesian_target_linear(robot_api.build_target_cpos(**pick_pose))
            _command_gripper(gripper_close_val, "closed")
            await robot_api.move_to_cartesian_target_linear(robot_api.build_target_cpos(**approach_pose))
            SESSION.robot_has_object = True

        async def _drop_action(robot_api: CodroidAPI, drop_target_mm: Tuple[float, float]) -> None:
            drop_pose = _pose_from_workspace_mm(
                workspace_def, drop_target_mm[0], drop_target_mm[1], drop_mm
            )
            drop_approach = _pose_from_workspace_mm(
                workspace_def, drop_target_mm[0], drop_target_mm[1], approach_mm
            )
            if SESSION.drop_target_orientation is not None:
                drop_pose.update(
                    {
                        "a": float(SESSION.drop_target_orientation[0]),
                        "b": float(SESSION.drop_target_orientation[1]),
                        "c": float(SESSION.drop_target_orientation[2]),
                    }
                )
                drop_approach.update(
                    {
                        "a": float(SESSION.drop_target_orientation[0]),
                        "b": float(SESSION.drop_target_orientation[1]),
                        "c": float(SESSION.drop_target_orientation[2]),
                    }
                )
            await robot_api.move_to_cartesian_target_linear(robot_api.build_target_cpos(**drop_approach))
            await robot_api.move_to_cartesian_target_linear(robot_api.build_target_cpos(**drop_pose))
            _command_gripper(gripper_open_val, "opened")
            await robot_api.move_to_cartesian_target_linear(robot_api.build_target_cpos(**drop_approach))
            await robot_api.power_off()
            SESSION.robot_has_object = False

        async def _pick_drop_action(robot_api: CodroidAPI) -> None:
            await _pick_action(robot_api)
            drop_target_mm = _resolve_drop_target()
            await _drop_action(robot_api, drop_target_mm)

        pick_cycle = trigger in {
            "execute-pick",
            "execute-pick-drop",
            "manual-pick-closest",
            "manual-pick-place-target",
        }
        cycle_started_at = time.monotonic() if pick_cycle else None

        try:
            if trigger in {"execute-pick", "manual-pick-closest"}:
                await _with_robot_api(
                    robot_uri or DEFAULT_ROBOT_URI, _pick_action)
                SESSION.last_action = "Pick command sent."
                if pick_cycle:
                    _record_pick_cycle(True, cycle_started_at)
                return "Pick command sent.", drop_text
            if trigger == "execute-drop":
                drop_target_mm = _resolve_drop_target()

                async def _drop_only(robot_api: CodroidAPI) -> None:
                    await _drop_action(robot_api, drop_target_mm)

                await _with_robot_api(
                    robot_uri or DEFAULT_ROBOT_URI, _drop_only)
                SESSION.last_action = "Drop command sent."
                drop_text = f"Drop at {drop_target_mm[0]:.1f}, {drop_target_mm[1]:.1f} mm"
                return "Drop command sent.", drop_text
            if trigger in {"execute-pick-drop", "manual-pick-place-target"}:
                if trigger == "manual-pick-place-target" and SESSION.drop_target_mm is None:
                    return "Teach a drop position first.", drop_text
                await _with_robot_api(
                    robot_uri or DEFAULT_ROBOT_URI, _pick_drop_action)
                SESSION.last_action = "Pick & drop command sent."
                drop_text = (
                    f"Drop at {SESSION.drop_target_mm[0]:.1f}, {SESSION.drop_target_mm[1]:.1f} mm"
                    if SESSION.drop_target_mm
                    else drop_text
                )
                if pick_cycle:
                    _record_pick_cycle(True, cycle_started_at)
                return "Pick & drop command sent.", drop_text
            if trigger == "manual-release":
                if not SESSION.robot_has_object:
                    return "No object held.", drop_text

                async def _release_only(api: CodroidAPI) -> None:
                    _command_gripper(gripper_open_val, "opened")
                    SESSION.robot_has_object = False

                await _with_robot_api(robot_uri or DEFAULT_ROBOT_URI, _release_only)
                SESSION.last_action = "Release command sent."
                return "Release command sent.", drop_text
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("Pick/place action failed")
            if pick_cycle:
                _record_pick_cycle(False, cycle_started_at)
            return f"Pick/place action failed: {exc}", drop_text

        return no_update, drop_text

    @app.callback(
        Output("plate-test-status", "children"),
        Output("plate-test-analysis", "value"),
        Input("plate-test-run", "n_clicks"),
        State("plate-test-approach-height", "value"),
        State("plate-test-pick-height", "value"),
        State("plate-test-drop-height", "value"),
        State("plate-test-grip-timeout", "value"),
        State("robot-uri", "value"),
        State("robot-connection-state", "data"),
        State("gripper-connection-state", "data"),
        prevent_initial_call=True,
    )
    async def _run_plate_pick_place_test(
        _clicks: int,
        approach_height: Optional[float],
        pick_height: Optional[float],
        drop_height: Optional[float],
        grip_timeout_s: Optional[float],
        robot_uri: Optional[str],
        robot_state: Optional[dict],
        gripper_state: Optional[dict],
    ) -> Tuple[str, str]:
        def _plate_test_abort(message: str) -> Tuple[str, str]:
            _set_plate_test_state(message, running=False)
            return message, ""

        if (robot_state or {}).get("status") != "online":
            return _plate_test_abort(
                "Robot is not connected. Click 'Test robot connection' first."
            )
        if (gripper_state or {}).get("status") != "online":
            return _plate_test_abort(
                "Gripper is not connected. Enable gripper in Setup menu."
            )
        if (
            SESSION.gripper is None
            or not SESSION.gripper.isConnected(
                t_index=GRIPPER_DEFAULTS["t_index"]
            )
        ):
            return _plate_test_abort(
                "Gripper session is not active; re-enable the gripper."
            )
        if SESSION.workspace_definition is None:
            return _plate_test_abort("Workspace is not defined; compute it first.")
        if SESSION.drop_target_mm is None:
            return _plate_test_abort("Teach a drop position first.")
        if SESSION.robot_calibration_min_z is None:
            return _plate_test_abort(
                "Robot calibration is missing; run robot calibration first."
            )
        frame, _depth = get_latest_frame()
        if frame is None:
            return _plate_test_abort("Camera stream is not running.")

        approach_mm = _safe_float(approach_height, 30.0)
        pick_mm = _safe_float(pick_height, 0.0)
        drop_mm = _safe_float(drop_height, 0.0)
        grip_timeout = _safe_float(grip_timeout_s, PLATE_TEST_GRIP_TIMEOUT_S)
        grip_timeout = max(grip_timeout, 0.5)
        workspace_def = SESSION.workspace_definition
        session = _get_robot_session()
        if not session.posture_seen():
            return _plate_test_abort("Robot posture stream not available yet.")
        if SESSION.drop_target_orientation is not None:
            orientation = {
                "a": float(SESSION.drop_target_orientation[0]),
                "b": float(SESSION.drop_target_orientation[1]),
                "c": float(SESSION.drop_target_orientation[2]),
            }
        else:
            posture = session.position_snapshot()
            orientation = {
                "a": float(posture.a),
                "b": float(posture.b),
                "c": float(posture.c),
            }

        _set_plate_test_state("Plate test starting...", running=True)
        try:
            (
                targets,
                plate_meta,
                baseline_transform,
                reference_initialized,
            ) = _resolve_plate_test_coord5_targets(DEFAULT_PLATE_PATH)
        except Exception as exc:  # noqa: BLE001
            return _plate_test_abort(f"Plate detection failed: {exc}")

        if not targets:
            return _plate_test_abort("No calibration targets found on the plate.")

        drop_target_mm = SESSION.drop_target_mm
        poscfg = CodroidAPI.build_poscfg(
            mode=int(SESSION.robot_calibration_points[0].get("mode", 0))
        )
        baseline_inv_rot = baseline_transform.rotation.T
        baseline_inv_trans = -baseline_inv_rot @ baseline_transform.translation

        def _base_to_coord(base_xyz: np.ndarray) -> np.ndarray:
            return baseline_inv_rot @ base_xyz + baseline_inv_trans

        def _resolve_coord5_pose(
            posture: RobotPosture,
            target: Optional[dict] = None,
        ) -> Tuple[np.ndarray, str]:
            raw_xyz = np.array(
                [float(posture.x), float(posture.y), float(posture.z)],
                dtype=np.float64,
            )
            coord_id = _current_coordinate_id(session)
            if coord_id == ROBOT_COORDINATE_ID:
                return raw_xyz, "robot_coord5"
            converted_xyz = _base_to_coord(raw_xyz)
            if coord_id == 0:
                return converted_xyz, "base_to_coord"
            if target is not None:
                target_xyz = np.array(
                    [float(target["x"]), float(target["y"]), float(target["z"])],
                    dtype=np.float64,
                )
                if np.linalg.norm(raw_xyz - target_xyz) <= np.linalg.norm(
                    converted_xyz - target_xyz
                ):
                    return raw_xyz, "robot_coord5_guess"
                return converted_xyz, "base_to_coord_guess"
            return raw_xyz, "robot_coord5_unknown"

        if SESSION.drop_target_robot is not None:
            base_drop = np.array(
                [
                    float(SESSION.drop_target_robot[0]),
                    float(SESSION.drop_target_robot[1]),
                    float(SESSION.drop_target_robot[2]),
                ],
                dtype=np.float64,
            )
            drop_coord = _base_to_coord(base_drop)
            drop_pose = {
                "x": float(drop_coord[0]),
                "y": float(drop_coord[1]),
                "z": float(drop_coord[2]) + drop_mm,
                "poscfg": poscfg,
            }
            drop_pose.update(orientation)
            drop_approach = {
                "x": float(drop_coord[0]),
                "y": float(drop_coord[1]),
                "z": float(drop_coord[2]) + approach_mm,
                "poscfg": poscfg,
            }
            drop_approach.update(orientation)
        else:
            drop_pose = _pose_from_workspace_mm(
                workspace_def, drop_target_mm[0], drop_target_mm[1], drop_mm
            )
            drop_pose["poscfg"] = poscfg
            drop_pose.update(orientation)
            drop_approach = _pose_from_workspace_mm(
                workspace_def, drop_target_mm[0], drop_target_mm[1], approach_mm
            )
            drop_approach["poscfg"] = poscfg
            drop_approach.update(orientation)
        gripper_min_width, gripper_max_width = _resolve_gripper_limits()
        summary = {"attempts": 0, "successes": 0, "details": []}

        capture: dict = {
            "timestamp": time.time(),
            "coordinate_id": ROBOT_COORDINATE_ID,
            "approach_height_mm": float(approach_mm),
            "pick_height_mm": float(pick_mm),
            "drop_height_mm": float(drop_mm),
            "grip_timeout_s": float(grip_timeout),
            "targets": targets,
            "plate_meta": plate_meta,
            "drop_target": {
                "workspace_mm": drop_target_mm,
                "coord5_pose": drop_pose,
                "coord5_approach": drop_approach,
            },
            "moves": [],
            "grip_events": [],
        }

        def _record_move(label: str, command_pose: dict) -> None:
            posture = session.position_snapshot()
            coord5_xyz, coord5_basis = _resolve_coord5_pose(
                posture, target=command_pose
            )
            coord_id = _current_coordinate_id(session)
            capture["moves"].append(
                {
                    "label": label,
                    "command_coord5": {
                        "x": float(command_pose["x"]),
                        "y": float(command_pose["y"]),
                        "z": float(command_pose["z"]),
                        "a": float(command_pose.get("a", 0.0)),
                        "b": float(command_pose.get("b", 0.0)),
                        "c": float(command_pose.get("c", 0.0)),
                    },
                    "posture_base": _posture_to_dict(posture),
                    "posture_coord5": {
                        "x": float(coord5_xyz[0]),
                        "y": float(coord5_xyz[1]),
                        "z": float(coord5_xyz[2]),
                        "a": float(posture.a),
                        "b": float(posture.b),
                        "c": float(posture.c),
                    },
                    "posture_coord5_basis": coord5_basis,
                    "posture_coordinate_id": coord_id,
                }
            )

        async def _run_sequence(robot_api: CodroidAPI) -> None:
            await robot_api.power_on()
            await robot_api.set_current_coordinate_id(ROBOT_COORDINATE_ID)
            speed_rate = max(float(SESSION.robot_speed_rate), 0.1)
            await robot_api.set_manual_move_rate(speed_rate)
            capture["robot_speed_rate"] = speed_rate
            speed_percent = int(round(speed_rate * 100))

            def _update_step(label: str) -> None:
                message = f"Plate test: {label.replace('_', ' ')} (speed {speed_percent}%)"
                _set_plate_test_state(message, running=True, label=label)

            async def _await_coord5_motion_settle(
                target: dict,
                *,
                label: str,
                timeout_s: float = PLATE_TEST_MOVE_TIMEOUT_S,
                settle_s: float = PLATE_TEST_SETTLE_S,
                motion_eps_mm: float = PLATE_TEST_MOTION_EPS_MM,
            ) -> None:
                deadline = time.monotonic() + timeout_s
                start = session.position_snapshot()
                coord5_xyz, _coord_basis = _resolve_coord5_pose(
                    start, target=target
                )
                if (
                    abs(float(coord5_xyz[0]) - float(target["x"]))
                    <= PLATE_TEST_TOLERANCE_MM
                    and abs(float(coord5_xyz[1]) - float(target["y"]))
                    <= PLATE_TEST_TOLERANCE_MM
                    and abs(float(coord5_xyz[2]) - float(target["z"]))
                    <= PLATE_TEST_Z_TOLERANCE_MM
                ):
                    return
                last = start
                moved = False
                settle_deadline: Optional[float] = None
                while time.monotonic() < deadline:
                    await robot_api.send_command_heartbeat()
                    await asyncio.sleep(0.2)
                    curr = session.position_snapshot()
                    base_delta = np.linalg.norm(
                        [
                            float(curr.x) - float(last.x),
                            float(curr.y) - float(last.y),
                            float(curr.z) - float(last.z),
                        ]
                    )
                    if base_delta > motion_eps_mm:
                        moved = True
                        settle_deadline = time.monotonic() + settle_s
                    if moved and settle_deadline is not None:
                        if base_delta > motion_eps_mm:
                            settle_deadline = time.monotonic() + settle_s
                        elif time.monotonic() >= settle_deadline:
                            return
                    last = curr
                total = np.linalg.norm(
                    [
                        float(last.x) - float(start.x),
                        float(last.y) - float(start.y),
                        float(last.z) - float(start.z),
                    ]
                )
                if not moved:
                    status = session.robot_status_snapshot()
                    LOGGER.warning(
                        "Plate test move timeout at %s: no motion detected (status=%s).",
                        label,
                        status,
                    )
                    return
                raise RuntimeError(
                    f"Plate test move timeout at {label}: motion did not settle "
                    f"(total={total:.2f} mm)."
                )

            async def _move_linear_with_hold(
                command_pose: dict,
                hold_seconds: float = 0.0,
                heartbeat_interval: float = 0.2,
                label: str = "move",
            ) -> None:
                _update_step(label)
                await robot_api.set_target_pos_type(
                    robot_api.config.target_pos_types.cpos
                )
                await robot_api.set_target_cpos(
                    robot_api.attach_coordinate_to_cpos(
                        robot_api.build_target_cpos(**command_pose),
                        coordinate_id=ROBOT_COORDINATE_ID,
                    )
                )
                await robot_api.set_robot_command(
                    robot_api.config.commands.move_target_linear
                )
                await _await_coord5_motion_settle(command_pose, label=label)
                if hold_seconds > 0:
                    deadline = time.monotonic() + hold_seconds
                    while time.monotonic() < deadline:
                        await robot_api.send_command_heartbeat()
                        await asyncio.sleep(heartbeat_interval)
                await robot_api.clear_target_position()
                await robot_api.stop_command()
                _record_move(label, command_pose)

            await _move_linear_with_hold(
                drop_approach,
                hold_seconds=PLATE_TEST_MOVE_HOLD_S,
                label="drop_approach_start",
            )
            await _move_linear_with_hold(
                drop_pose,
                hold_seconds=PLATE_TEST_MOVE_HOLD_S,
                label="drop_pose_start",
            )

            for target in targets:
                sequence = target["sequence"]
                coord_xy = target["coord5_xy"]
                plate_mm = target.get("plate_mm")
                approach_coord = np.array(
                    [coord_xy[0], coord_xy[1], approach_mm], dtype=np.float64
                )
                pick_coord = np.array(
                    [coord_xy[0], coord_xy[1], pick_mm], dtype=np.float64
                )
                approach_pose = {
                    "x": float(approach_coord[0]),
                    "y": float(approach_coord[1]),
                    "z": float(approach_coord[2]),
                    "poscfg": poscfg,
                }
                pick_pose = {
                    "x": float(pick_coord[0]),
                    "y": float(pick_coord[1]),
                    "z": float(pick_coord[2]),
                    "poscfg": poscfg,
                }
                approach_pose.update(orientation)
                pick_pose.update(orientation)
                cycle_started_at = time.monotonic()
                await _move_linear_with_hold(
                    approach_pose,
                    hold_seconds=PLATE_TEST_MOVE_HOLD_S,
                    label=f"target_{sequence}_approach",
                )
                await _move_linear_with_hold(
                    pick_pose,
                    hold_seconds=PLATE_TEST_MOVE_HOLD_S,
                    label=f"target_{sequence}_pick",
                )
                _command_gripper(
                    gripper_min_width,
                    "closed",
                    force=GRIPPER_FORCE_MIN,
                    speed=GRIPPER_SPEED_MIN,
                )
                gripped = await _await_grip_status(grip_timeout)
                capture["grip_events"].append(
                    {
                        "sequence": int(sequence),
                        "gripped": bool(gripped),
                        "timeout_s": float(grip_timeout),
                    }
                )
                summary["attempts"] += 1
                if plate_mm is not None:
                    SESSION.last_pick_mm = (plate_mm[0], plate_mm[1])

                if gripped:
                    summary["successes"] += 1
                    SESSION.robot_has_object = True
                    await _move_linear_with_hold(
                        approach_pose,
                        hold_seconds=PLATE_TEST_MOVE_HOLD_S,
                        label=f"target_{sequence}_lift",
                    )
                    await _move_linear_with_hold(
                        drop_approach,
                        hold_seconds=PLATE_TEST_MOVE_HOLD_S,
                        label=f"target_{sequence}_drop_approach",
                    )
                    await _move_linear_with_hold(
                        drop_pose,
                        hold_seconds=PLATE_TEST_MOVE_HOLD_S,
                        label=f"target_{sequence}_drop_pose",
                    )
                    _command_gripper(gripper_max_width, "opened")
                    SESSION.robot_has_object = False
                    await _move_linear_with_hold(
                        drop_approach,
                        hold_seconds=PLATE_TEST_MOVE_HOLD_S,
                        label=f"target_{sequence}_drop_retreat",
                    )
                    summary["details"].append(f"{sequence}: dropped")
                    _record_pick_cycle(True, cycle_started_at)
                else:
                    _command_gripper(gripper_max_width, "opened")
                    await _move_linear_with_hold(
                        approach_pose,
                        hold_seconds=PLATE_TEST_MOVE_HOLD_S,
                        label=f"target_{sequence}_empty_lift",
                    )
                    summary["details"].append(f"{sequence}: empty")
                    _record_pick_cycle(False, cycle_started_at)

            await _move_linear_with_hold(
                drop_approach,
                hold_seconds=PLATE_TEST_MOVE_HOLD_S,
                label="drop_approach_end",
            )
            await _move_linear_with_hold(
                drop_pose,
                hold_seconds=PLATE_TEST_MOVE_HOLD_S,
                label="drop_pose_end",
            )
            _command_gripper(gripper_max_width, "opened")
            await robot_api.power_off()

        try:
            await _with_robot_api(
                robot_uri or DEFAULT_ROBOT_URI, _run_sequence
            )
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("Plate pick & place test failed")
            message = f"Plate test failed: {exc}"
            _set_plate_test_state(message, running=False)
            return message, ""

        attempts = summary["attempts"]
        successes = summary["successes"]
        detail = ", ".join(summary["details"])
        SESSION.last_action = (
            f"Plate test complete: {successes}/{attempts} picked."
        )
        capture["summary"] = summary
        _save_plate_test_capture(capture)
        analysis = _analyze_plate_test_capture(capture)
        plate_note = (
            f"Plate markers: {plate_meta['count']} · "
            f"Reproj: {plate_meta['reprojection_error_mm']:.2f} mm"
        )
        if plate_meta.get("reference_initialized"):
            plate_note = f"{plate_note} · Plate reference set"
        message = (
            f"Plate test complete: {successes}/{attempts} picked. {plate_note}"
        )
        if detail:
            message = f"{message} ({detail})"
        _set_plate_test_state(message, running=False)
        return message, analysis

    @app.callback(
        Output("plate-test-status", "children", allow_duplicate=True),
        Input("monitor-interval", "n_intervals"),
        prevent_initial_call=True,
    )
    def _refresh_plate_test_status(_ticks: int) -> str:
        state = SESSION.plate_test_state or {}
        if state.get("running"):
            return state.get("message", "")
        return no_update

    @app.callback(
        Output("footer-gripper-status", "children"),
        Input("footer-gripper-width", "value"),
        Input("footer-gripper-force", "value"),
        Input("footer-gripper-speed", "value"),
        prevent_initial_call=True,
    )
    def _footer_gripper_actions(
        footer_width: Optional[float],
        footer_force: Optional[float],
        footer_speed: Optional[float],
    ) -> str:
        try:
            if SESSION.gripper_busy:
                return "Gripper is busy; wait for motion to finish."
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
            if trigger in {"footer-gripper-width", "footer-gripper-force", "footer-gripper-speed"}:
                width = _safe_float(
                    footer_width, GRIPPER_CLOSE_WIDTH_DEFAULT)
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
        Output("footer-gripper-width", "value", allow_duplicate=True),
        Output("footer-gripper-force", "value", allow_duplicate=True),
        Output("footer-gripper-speed", "value", allow_duplicate=True),
        Input("footer-gripper-width-decrease", "n_clicks"),
        Input("footer-gripper-width-increase", "n_clicks"),
        Input("footer-gripper-force-decrease", "n_clicks"),
        Input("footer-gripper-force-increase", "n_clicks"),
        Input("footer-gripper-speed-decrease", "n_clicks"),
        Input("footer-gripper-speed-increase", "n_clicks"),
        State("footer-gripper-width", "value"),
        State("footer-gripper-force", "value"),
        State("footer-gripper-speed", "value"),
        State("footer-gripper-width", "min"),
        State("footer-gripper-width", "max"),
        prevent_initial_call=True,
    )
    def _step_footer_gripper_controls(
        _width_dec: int,
        _width_inc: int,
        _force_dec: int,
        _force_inc: int,
        _speed_dec: int,
        _speed_inc: int,
        width: Optional[float],
        force: Optional[float],
        speed: Optional[float],
        width_min: Optional[float],
        width_max: Optional[float],
    ) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        trigger = callback_context.triggered_id
        if trigger is None:
            return no_update, no_update, no_update
        if SESSION.gripper_busy:
            return no_update, no_update, no_update

        width_value = _safe_float(width, GRIPPER_CLOSE_WIDTH_DEFAULT)
        force_value = _safe_float(force, GRIPPER_FORCE_DEFAULT)
        speed_value = _safe_float(speed, GRIPPER_SPEED_DEFAULT)

        if trigger == "footer-gripper-width-decrease":
            width_value -= 0.5
        elif trigger == "footer-gripper-width-increase":
            width_value += 0.5
        elif trigger == "footer-gripper-force-decrease":
            force_value -= 5
        elif trigger == "footer-gripper-force-increase":
            force_value += 5
        elif trigger == "footer-gripper-speed-decrease":
            speed_value -= 5
        elif trigger == "footer-gripper-speed-increase":
            speed_value += 5

        if width_min is not None:
            width_value = max(width_value, float(width_min))
        if width_max is not None:
            width_value = min(width_value, float(width_max))
        force_value = min(max(force_value, GRIPPER_FORCE_MIN), GRIPPER_FORCE_MAX)
        speed_value = min(max(speed_value, 0), 100)

        return width_value, force_value, speed_value

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
        Output("footer-gripper-width", "min", allow_duplicate=True),
        Output("footer-gripper-width", "max", allow_duplicate=True),
        Output("footer-gripper-meta", "children"),
        Output("gripper-orientation", "value"),
        Input("robot-status-interval", "n_intervals"),
        State("gripper-connection-state", "data"),
        State("footer-gripper-width", "value"),
        State("footer-gripper-force", "value"),
        prevent_initial_call=True,
    )
    def _sync_footer_gripper_values(
        _intervals: int,
        gripper_state: Optional[dict],
        current_width: Optional[float],
        current_force: Optional[float],
    ) -> Tuple[
        Optional[float],
        Optional[float],
        Optional[float],
        Optional[float],
        str,
        Optional[str],
    ]:
        if (gripper_state or {}).get("status") != "online":
            return no_update, no_update, no_update, no_update, no_update, no_update
        if SESSION.gripper is None:
            return no_update, no_update, no_update, no_update, no_update, no_update

        status_snapshot = None
        try:
            t_index = GRIPPER_DEFAULTS["t_index"]
            if SESSION.gripper.get_status_snapshot(t_index=t_index) is None:
                SESSION.gripper.start_status_stream()
            status_snapshot = SESSION.gripper.get_status_snapshot(t_index=t_index)
            if status_snapshot:
                width = status_snapshot.get("external_width")
                force = status_snapshot.get("force")
                min_width = status_snapshot.get("min_external_width")
                max_width = status_snapshot.get("max_external_width")
                busy = status_snapshot.get("busy")
                gripped = status_snapshot.get("grip_detected")
                status = status_snapshot.get("status")
                outward = status_snapshot.get("finger_orientation_outward")
                orientation = None
                if isinstance(outward, bool):
                    orientation = "outward" if outward else "inward"
                elif isinstance(outward, str):
                    cleaned = outward.strip().lower()
                    if cleaned in {"true", "1", "yes", "on"}:
                        orientation = "outward"
                    elif cleaned in {"false", "0", "no", "off"}:
                        orientation = "inward"
            else:
                width = SESSION.gripper.get_ext_width(t_index=t_index)
                force = SESSION.gripper.get_force(t_index=t_index)
                min_width = SESSION.gripper.get_min_ext_width(t_index=t_index)
                max_width = SESSION.gripper.get_max_ext_width(t_index=t_index)
                busy = SESSION.gripper.isBusy(t_index=t_index)
                gripped = SESSION.gripper.isGripped(t_index=t_index)
                status = SESSION.gripper.getStatus(t_index=t_index)
                orientation = SESSION.gripper.get_finger_orientation_label(
                    t_index=t_index
                )
        except Exception as exc:  # noqa: BLE001
            LOGGER.debug("Failed to poll gripper values: %s", exc)
            return no_update, no_update, no_update, no_update, no_update, no_update

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

        if (
            isinstance(force, (int, float))
            and force < 0
            and status_snapshot
            and isinstance(status_snapshot.get("max_force"), (int, float))
        ):
            # Avoid displaying negative force values when using status snapshots.
            force = None

        min_update: Optional[float] = no_update
        max_update: Optional[float] = no_update
        if isinstance(min_width, (int, float)) and min_width >= 0:
            if (
                SESSION.gripper_min_width is None
                or abs(min_width - SESSION.gripper_min_width) > 0.1
            ):
                min_update = float(min_width)
                SESSION.gripper_min_width = float(min_width)
        if isinstance(max_width, (int, float)) and max_width >= 0:
            if (
                SESSION.gripper_max_width is None
                or abs(max_width - SESSION.gripper_max_width) > 0.1
            ):
                max_update = float(max_width)
                SESSION.gripper_max_width = float(max_width)

        width_value = None if width_update is no_update else float(width_update)
        force_value = None if force_update is no_update else float(force_update)
        _record_gripper_sync(width_value, force_value)

        if isinstance(force, (int, float)) and force >= 0:
            SESSION.gripper_force = float(force)
        if isinstance(busy, (bool, int, float)) and busy != -2:
            SESSION.gripper_busy = bool(busy)
        if isinstance(gripped, (bool, int, float)) and gripped != -2:
            SESSION.gripper_grip_detected = bool(gripped)
        if isinstance(status, (int, float)) and status != -2:
            SESSION.gripper_status_code = int(status)
        previous_orientation = SESSION.gripper_orientation
        if orientation in {"inward", "outward"}:
            SESSION.gripper_orientation = orientation

        busy_text = (
            "unknown"
            if SESSION.gripper_busy is None
            else ("yes" if SESSION.gripper_busy else "no")
        )
        grip_text = (
            "unknown"
            if SESSION.gripper_grip_detected is None
            else ("yes" if SESSION.gripper_grip_detected else "no")
        )
        meta_text = (
            f"Orientation: {SESSION.gripper_orientation or 'unknown'}"
            f" · Busy: {busy_text}"
            f" · Grip: {grip_text}"
        )
        if isinstance(force, (int, float)) and force >= 0:
            meta_text = f"{meta_text} · Force: {float(force):.0f} N"

        orientation_update: Optional[str] = no_update
        if orientation in {"inward", "outward"} and orientation != previous_orientation:
            orientation_update = orientation

        width_min = (
            SESSION.gripper_min_width
            if SESSION.gripper_min_width is not None
            else None
        )
        width_max = (
            SESSION.gripper_max_width
            if SESSION.gripper_max_width is not None
            else None
        )
        if width_update is no_update and width_min is not None and width_max is not None:
            if isinstance(current_width, (int, float)):
                clamped = min(max(current_width, width_min), width_max)
                if abs(clamped - current_width) > 0.05:
                    width_update = float(clamped)

        return (
            width_update,
            force_update,
            min_update,
            max_update,
            meta_text,
            orientation_update,
        )

    @app.callback(
        Output("gripper-orientation-status", "children"),
        Output("footer-gripper-width", "min", allow_duplicate=True),
        Output("footer-gripper-width", "max", allow_duplicate=True),
        Input("gripper-orientation", "value"),
        State("gripper-connection-state", "data"),
        prevent_initial_call=True,
    )
    def _apply_gripper_orientation(
        orientation: Optional[str],
        gripper_state: Optional[dict],
    ) -> Tuple[str, Optional[float], Optional[float]]:
        if (gripper_state or {}).get("status") != "online":
            return "Gripper is offline.", no_update, no_update
        if SESSION.gripper is None:
            return "Gripper is unavailable.", no_update, no_update
        if orientation not in {"inward", "outward"}:
            return no_update, no_update, no_update
        if SESSION.gripper_orientation == orientation:
            return no_update, no_update, no_update
        try:
            t_index = GRIPPER_DEFAULTS["t_index"]
            result = SESSION.gripper.set_finger_orientation(
                t_index=t_index,
                orientation=orientation,
            )
            if isinstance(result, (int, float)) and result < 0:
                return "Failed to update gripper orientation.", no_update, no_update
            SESSION.gripper_orientation = orientation
            SESSION.gripper_min_width = None
            SESSION.gripper_max_width = None
            min_width = SESSION.gripper.get_min_ext_width(t_index=t_index)
            max_width = SESSION.gripper.get_max_ext_width(t_index=t_index)
            min_update = (
                float(min_width)
                if isinstance(min_width, (int, float)) and min_width >= 0
                else no_update
            )
            max_update = (
                float(max_width)
                if isinstance(max_width, (int, float)) and max_width >= 0
                else no_update
            )
            if min_update is not no_update:
                SESSION.gripper_min_width = min_update
            if max_update is not no_update:
                SESSION.gripper_max_width = max_update
            return f"Orientation set to {orientation}.", min_update, max_update
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("Failed to set gripper orientation")
            return f"Failed to set orientation: {exc}", no_update, no_update

    @app.callback(
        Output("holding-status", "children"),
        Input("robot-status-interval", "n_intervals"),
        State("gripper-connection-state", "data"),
    )
    def _update_holding_badge(
        _intervals: int, gripper_state: Optional[dict]
    ) -> str:
        status = (gripper_state or {}).get("status", "offline")
        if status != "online" or SESSION.gripper is None:
            return "Holding object: unknown (gripper offline)"
        try:
            t_index = GRIPPER_DEFAULTS["t_index"]
            is_gripped = bool(SESSION.gripper.isGripped(t_index=t_index))
            SESSION.robot_has_object = is_gripped
            return f"Holding object: {'yes' if is_gripped else 'no'}"
        except Exception as exc:  # noqa: BLE001
            LOGGER.debug("Unable to query gripper grip state: %s", exc)
            return "Holding object: unknown"

    @app.callback(
        Output("monitor-system-summary", "children"),
        Output("monitor-detection-summary", "children"),
        Output("monitor-pick-summary", "children"),
        Output("use-existing-calibration", "style"),
        Output("use-existing-workspace", "style"),
        Input("monitor-interval", "n_intervals"),
        Input("camera-interval", "n_intervals"),
        State("camera-connection-state", "data"),
        State("robot-connection-state", "data"),
        State("gripper-connection-state", "data"),
        State("robot-power-state", "data"),
        State("robot-speed-state", "data"),
        State("settings-calib-path", "value"),
    )
    def _update_monitoring_summary(
        _monitor_ticks: int,
        _camera_ticks: int,
        camera_state: Optional[dict],
        robot_state: Optional[dict],
        gripper_state: Optional[dict],
        robot_power: Optional[dict],
        robot_speed: Optional[dict],
        bundle_path_value: Optional[str],
    ) -> Tuple[list, list, list, dict, dict]:
        camera_status = (camera_state or {}).get("status", "off")
        robot_status = (robot_state or {}).get("status", "offline")
        gripper_status = (gripper_state or {}).get("status", "offline")
        calibration_state = "loaded" if SESSION.calibration_bundle is not None else "missing"
        workspace_state = "ready" if SESSION.workspace_definition is not None else "pending"
        plate_state = "ready" if SESSION.plate_calibration is not None else "pending"
        model_state = SESSION.active_model_path.name if SESSION.active_model_path else "none"
        detection_count = len(SESSION.detections)
        last_action = SESSION.last_action or "No recent actions."

        stream = SESSION.camera_stream
        stream_fps = stream.get_fps() if stream else 0.0
        stream_cfg_fps = stream.config.fps if stream else DEFAULT_FPS
        res_w, res_h = stream.config.resolution if stream else DEFAULT_RESOLUTION

        system_items = [
            dmc.Text(
                f"Camera: {camera_status} · {res_w}x{res_h} @ {stream_fps:.1f}/{stream_cfg_fps:g} FPS",
                size="sm",
            ),
            dmc.Text(
                f"Robot: {robot_status} · speed {int((robot_speed or {}).get('value', SESSION.robot_speed_rate*100))}%",
                size="sm",
            ),
            dmc.Text(
                f"Power: {('on' if (robot_power or {}).get('value') else 'off')}",
                size="sm",
            ),
            dmc.Text(f"Calibration bundle: {calibration_state}", size="sm"),
            dmc.Text(f"Workspace: {workspace_state}", size="sm"),
            dmc.Text(f"Plate calibration: {plate_state}", size="sm"),
            dmc.Text(
                f"Gripper: {gripper_status}",
                size="sm",
            ),
        ]

        detection_items = [
            dmc.Text(f"Model: {model_state}", size="sm"),
            dmc.Text(f"Detections: {detection_count}", size="sm"),
        ]
        if SESSION.last_detection_error:
            detection_items.append(
                dmc.Text(
                    f"Last error: {SESSION.last_detection_error}",
                    size="sm",
                    c="red",
                )
            )

        attempts = SESSION.pick_attempts
        successes = SESSION.pick_successes
        failures = SESSION.pick_failures
        processed = successes
        success_rate = (successes / attempts * 100.0) if attempts else None
        avg_cycle = (
            sum(SESSION.cycle_times) / len(SESSION.cycle_times)
            if SESSION.cycle_times
            else None
        )
        pick_items = [
            dmc.Text(
                f"Holding object: {'yes' if SESSION.robot_has_object else 'no'}",
                size="sm",
            ),
            dmc.Text(f"Last action: {last_action}", size="sm"),
            dmc.Text(f"Processed objects: {processed}", size="sm"),
            dmc.Text(
                f"Attempts: {attempts} · Failures: {failures}",
                size="sm",
            ),
            dmc.Text(
                f"Success rate: {success_rate:.1f}%"
                if success_rate is not None
                else "Success rate: n/a",
                size="sm",
            ),
            dmc.Text(
                f"Avg cycle: {avg_cycle:.2f}s"
                if avg_cycle is not None
                else "Avg cycle: n/a",
                size="sm",
            ),
            dmc.Text(
                f"Last cycle: {SESSION.last_cycle_s:.2f}s"
                if SESSION.last_cycle_s is not None
                else "Last cycle: n/a",
                size="sm",
            ),
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
        if SESSION.drop_target_orientation:
            pick_items.append(
                dmc.Text(
                    "Drop orientation: "
                    f"a={SESSION.drop_target_orientation[0]:.1f}, "
                    f"b={SESSION.drop_target_orientation[1]:.1f}, "
                    f"c={SESSION.drop_target_orientation[2]:.1f}",
                    size="sm",
                )
            )
        if SESSION.drop_target_robot:
            pick_items.append(
                dmc.Text(
                    "Drop robot pose: "
                    f"x={SESSION.drop_target_robot[0]:.1f}, "
                    f"y={SESSION.drop_target_robot[1]:.1f}, "
                    f"z={SESSION.drop_target_robot[2]:.1f}",
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
        Input("use-existing-calibration", "n_clicks"),
        State("settings-calib-path", "value"),
        prevent_initial_call=True,
    )
    def _handle_camera_calibration_actions(
        _calibrate: int,
        _use_existing: int,
        bundle_path: str,
    ) -> dmc.Alert:
        trigger = callback_context.triggered_id

        output_path = DEFAULT_CAPTURE_DIR
        board_path = DEFAULT_BOARD_PATH
        bundle_path = _safe_path(bundle_path, DEFAULT_BUNDLE_PATH)

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
        State("settings-marker-a", "value"),
        State("settings-marker-b", "value"),
        State("settings-marker-size", "value"),
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
        Input("camera-interval", "n_intervals"),
        prevent_initial_call=True,
    )
    def _handle_plate_actions(
        _ticks: int,
    ) -> Tuple[str, str]:
        return no_update, no_update

    @app.callback(
        Output("overlay-status", "children"),
        Output("camera-fps", "children"),
        Output("camera-calibration-metrics", "children"),
        Output("workspace-metrics", "children"),
        Output("plate-metrics", "children"),
        Output("detection-metrics", "children"),
        Output("plate-detection-state", "data"),
        Input("camera-interval", "n_intervals"),
        State("main-tabs", "value"),
        State("calibration-stepper", "active"),
        State("application-tabs", "value"),
        State("settings-marker-a", "value"),
        State("settings-marker-b", "value"),
        State("settings-marker-size", "value"),
        State("overlay-mode-select", "value"),
    )
    def _update_camera_view(
        _ticks: int,
        active_tab: Optional[str],
        calibration_step: Optional[int],
        application_tab: Optional[str],
        marker_a: Optional[float],
        marker_b: Optional[float],
        marker_size: Optional[float],
        overlay_mode: Optional[str],
    ) -> Tuple[str, str, str, str, str, str, dict]:
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
        application_tab = application_tab or "workspace"
        board_path = DEFAULT_BOARD_PATH
        plate_path = DEFAULT_PLATE_PATH
        marker_ids = (
            _safe_int(marker_a, SESSION.config.marker_ids[0]),
            _safe_int(marker_b, SESSION.config.marker_ids[1]),
        )
        marker_size_mm = _safe_float(
            marker_size, SESSION.config.marker_size_mm)

        requested_overlay = overlay_mode or "auto"
        task = _select_overlay_task(
            tab,
            calibration_step,
            application_tab,
            board_path,
            marker_ids,
            marker_size_mm,
            plate_path,
        )
        if requested_overlay != "auto":
            if requested_overlay == "calibration":
                task = OverlayTask("manual-calibration",
                                   _run_charuco_task(board_path))
            elif requested_overlay == "workspace":
                task = OverlayTask(
                    "manual-workspace",
                    _run_workspace_task(marker_ids, marker_size_mm),
                )
            elif requested_overlay == "plate":
                task = OverlayTask("manual-plate", _run_plate_task(plate_path))
            elif requested_overlay == "detection":
                task = OverlayTask("manual-detection", _run_detection_task())
            elif requested_overlay == "none":
                task = None
        overlay: Optional[OverlayResult] = None
        if task is not None:
            FRAME_PROCESSOR.submit(task, frame)
            overlay = FRAME_PROCESSOR.get_latest(task.key)

        vis = overlay.vis if overlay else frame
        status_text = overlay.status if overlay else "No overlay applied."
        metric_text = overlay.metric if overlay else "--"
        mode_label = {
            "auto": "Overlay: Auto",
            "calibration": "Overlay: Calibration",
            "workspace": "Overlay: Workspace",
            "plate": "Overlay: Plate",
            "detection": "Overlay: Detections",
            "none": "Overlay: Off",
        }.get(requested_overlay, "Overlay: Auto")
        status_text = f"{mode_label} · {status_text}"
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
                plate_state,
            )

        if tab == "applications":
            if is_new_frame:
                MJPEG_STREAM.add_frame(vis)
            overlay_fps_text = _format_fps(
                capture_fps, MJPEG_STREAM.get_fps())
            return (
                status_text,
                overlay_fps_text,
                "ChArUco corners: --",
                metric_text if application_tab == "workspace" else "Workspace: --",
                "Plate error: --",
                metric_text if application_tab == "pick-place" else "Detections: --",
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
        State("robot-connection-state", "data"),
        State("robot-uri", "value"),
        State("robot-power-state", "data"),
        prevent_initial_call=True,
    )
    async def _sync_robot_power_state(
        _intervals: int,
        robot_state: Optional[dict],
        robot_uri: Optional[str],
        current_state: Optional[dict],
    ) -> dict:
        if (robot_state or {}).get("status") != "online":
            return no_update
        now = time.time()
        if now - SESSION.robot_status_last_poll < 1.0:
            return no_update
        async def _poll(api: CodroidAPI) -> None:
            await api.read_system_data()
            await api.read_global_data()

        try:
            await _with_robot_api(robot_uri or DEFAULT_ROBOT_URI, _poll)
            SESSION.robot_status_last_poll = now
        except Exception as exc:  # noqa: BLE001
            LOGGER.debug("Robot power poll failed: %s", exc)
            return no_update

        session = _get_robot_session()
        power_on = session.robot_power_on()
        if power_on is None:
            return no_update
        desired = {"value": bool(power_on), "timestamp": time.time()}
        if current_state and current_state.get("value") == desired["value"]:
            return no_update
        return desired

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

        def _step_state(is_done: bool) -> Optional[str]:
            return "completed" if is_done else None

        stepper_children = [
            dmc.StepperStep(
                label="Point 1",
                description="Indentation 1 (origin)",
                state=_step_state(step > 1),
            ),
            dmc.StepperStep(
                label="Point 2",
                description="Indentation 2 (+X)",
                state=_step_state(step > 2),
            ),
            dmc.StepperStep(
                label="Point 3",
                description="Indentation 3 (+X/+Y)",
                state=_step_state(step > 3),
            ),
            dmc.StepperStep(
                label="Verification",
                description="Safe motion check",
                state=_step_state(verification_completed),
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
        State("plate-detection-state", "data"),
        State("robot-connection-state", "data"),
        State("robot-uri", "value"),
        prevent_initial_call=True,
    )
    async def _start_robot_calibration(
        _clicks: int,
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
            layout_path = DEFAULT_PLATE_PATH
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
            try:
                plate_result = compute_plate_calibration_from_latest(layout_path)
                SESSION.plate_reference_homography = plate_result.homography.copy()
                LOGGER.info("Captured plate reference homography for plate test.")
            except Exception as exc:  # noqa: BLE001
                LOGGER.warning(
                    "Failed to capture plate reference homography: %s", exc
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

        session = _get_robot_session()
        posture_ok, posture_message = _robot_posture_ready()
        if not posture_ok:
            _set_robot_guidance(
                posture_message,
                status="error",
                step=4,
                verification_ready=False,
            )
            return {"status": "error", "message": posture_message, "step": 4}
        _set_robot_guidance(
            "Verification move running. Keep clear of the robot.",
            status="verifying",
            step=4,
            verification_ready=False,
        )

        async def _verification_motion(api: CodroidAPI) -> None:
            await api.power_on()
            await api.set_current_coordinate_id(0)

            async def _await_motion_then_settle(
                *,
                label: str,
                timeout_s: float = VERIFICATION_MOVE_TIMEOUT_S,
                settle_s: float = VERIFICATION_SETTLE_S,
                motion_eps_mm: float = VERIFICATION_MOTION_EPS_MM,
            ) -> None:
                deadline = time.monotonic() + timeout_s
                start = session.position_snapshot()
                last = start
                moved = False
                settle_deadline: Optional[float] = None
                while time.monotonic() < deadline:
                    await api.send_command_heartbeat()
                    await asyncio.sleep(0.2)
                    curr = session.position_snapshot()
                    delta = np.linalg.norm(
                        [
                            float(curr.x) - float(last.x),
                            float(curr.y) - float(last.y),
                            float(curr.z) - float(last.z),
                        ]
                    )
                    if delta > motion_eps_mm:
                        moved = True
                        settle_deadline = time.monotonic() + settle_s
                    if moved and settle_deadline is not None:
                        if delta > motion_eps_mm:
                            settle_deadline = time.monotonic() + settle_s
                        elif time.monotonic() >= settle_deadline:
                            return
                    last = curr
                total = np.linalg.norm(
                    [
                        float(last.x) - float(start.x),
                        float(last.y) - float(start.y),
                        float(last.z) - float(start.z),
                    ]
                )
                if not moved:
                    raise RuntimeError(
                        f"Verification move timeout at {label}: no motion detected."
                    )
                raise RuntimeError(
                    f"Verification move timeout at {label}: motion did not settle "
                    f"(total={total:.2f} mm)."
                )

            async def _move_linear_with_hold(
                target: dict,
                *,
                label: str,
                hold_seconds: float = VERIFICATION_HOLD_SECONDS,
            ) -> None:
                await api.set_target_pos_type(api.config.target_pos_types.cpos)
                await api.set_target_cpos(target)
                await api.set_robot_command(api.config.commands.move_target_linear)
                await _await_motion_then_settle(label=label)
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

            await _move_linear_with_hold(_target(p3, raise_3), label="p3_raise")
            await _move_linear_with_hold(_target(p1, raise_1), label="p1_raise")
            await _move_linear_with_hold(_target(p1, z1), label="p1_down")
            await _move_linear_with_hold(_target(p1, raise_1), label="p1_raise_return")
            await _move_linear_with_hold(_target(p2, raise_2), label="p2_raise")
            await _move_linear_with_hold(_target(p2, z2), label="p2_down")
            await _move_linear_with_hold(_target(p2, raise_2), label="p2_raise_return")
            await _move_linear_with_hold(_target(p3, raise_3), label="p3_raise")
            await _move_linear_with_hold(_target(p3, z3), label="p3_down")
            await _move_linear_with_hold(_target(p3, raise_3), label="p3_raise_return")
            await _move_linear_with_hold(
                _target(p1, z1), label="p1_home", hold_seconds=0.0
            )
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
        update_title=None,
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
