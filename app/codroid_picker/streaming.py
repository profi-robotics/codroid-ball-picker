"""Streaming infrastructure for the calibration UI.

This module handles camera streaming and frame processing for the calibration
interface.
"""

from __future__ import annotations

import logging
import os
import platform
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Callable, Deque, Optional

import numpy as np

from app.codroid_picker.session import SESSION
from app.codroid_picker.utils import safe_float, safe_int
from vision.stream import DepthAIFrameStream, HostCameraFrameStream, StreamConfig

LOGGER = logging.getLogger(__name__)
_CAMERA_START_MAX_ATTEMPTS = 8
_CAMERA_START_RETRY_DELAY_S = 2.0
_RGB_MAX_FPS = float(os.getenv("CODROID_RGB_MAX_FPS", "30"))
_RGB_WITH_DEPTH_MAX_FPS = float(os.getenv("CODROID_RGB_MAX_FPS_WITH_DEPTH", "30"))
_OVERLAY_MAX_FPS = float(os.getenv("CODROID_OVERLAY_MAX_FPS", "10"))
_RETRYABLE_CAMERA_ERROR_SUBSTRINGS = (
    "no available devices",
    "device already closed or disconnected",
    "input/output error",
    "x_link",
    "xlink",
    "device likely crashed",
)


@dataclass
class OverlayResult:
    """Result of an overlay rendering operation."""
    vis: np.ndarray
    status: str
    metric: str


@dataclass
class OverlayTask:
    """Task for background overlay processing."""
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
        self._last_submit_ts = 0.0
        self._running = True
        self._thread = threading.Thread(
            target=self._worker, name="FrameProcessor", daemon=True
        )
        self._thread.start()

    def submit(self, task: OverlayTask, frame: np.ndarray) -> None:
        """Submit a frame processing task."""
        now = time.time()
        if _OVERLAY_MAX_FPS > 0 and now - self._last_submit_ts < (1.0 / _OVERLAY_MAX_FPS):
            return
        self._last_submit_ts = now
        # Drop older pending tasks and keep the newest request only.
        with self._lock:
            self._queue.clear()
            self._queue.append((task.key, frame.copy(), task.runner))

    def get_latest(self, key: str) -> Optional[OverlayResult]:
        """Get the latest processing result for a given key."""
        with self._lock:
            result = self._latest.get(key)
            if result is None:
                return None
            # Copy the image so callers cannot mutate internal state.
            return OverlayResult(result.vis.copy(), result.status, result.metric)

    def stop(self) -> None:
        """Stop the frame processor."""
        self._running = False
        if self._thread.is_alive():
            self._thread.join(timeout=1.0)

    def _worker(self) -> None:
        """Background worker thread for processing frames."""
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
                LOGGER.error("FrameProcessor error for %s: %s",
                             key, exc, exc_info=True)
                result = OverlayResult(frame, f"Processing error: {exc}", "--")
            with self._lock:
                self._latest[key] = result


class RawFrameStore:
    """In-memory store for the latest raw camera frame."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._latest: Optional[np.ndarray] = None
        self._frame_index = 0
        self._last_update_ts = 0.0

    def add_frame(self, frame: np.ndarray) -> None:
        """Store the latest raw frame."""
        with self._lock:
            self._latest = frame
            self._frame_index += 1
            self._last_update_ts = time.time()

    def get_latest_frame(self) -> Optional[np.ndarray]:
        """Return a copy of the most recent raw frame."""
        with self._lock:
            if self._latest is None:
                return None
            return self._latest.copy()

    def get_latest_frame_with_metadata(self) -> tuple[Optional[np.ndarray], int, float]:
        """Return the latest frame reference plus its monotonic index and timestamp."""
        with self._lock:
            if self._latest is None:
                return None, self._frame_index, self._last_update_ts
            return self._latest, self._frame_index, self._last_update_ts


def _format_fps(capture_fps: float) -> str:
    """Format FPS for display."""
    cap_text = f"{capture_fps:.1f}" if capture_fps > 0 else "--"
    return f"FPS — capture: {cap_text}"


# Global streaming instances
FRAME_PROCESSOR = FrameProcessor()
RAW_STREAM = RawFrameStore()
_ACTIVE_OVERLAY_TASK: Optional[OverlayTask] = None
_OVERLAY_TASK_LOCK = threading.Lock()
_STREAM_LIFECYCLE_LOCK = threading.RLock()
_CAMERA_AUTOSTART_LOCK = threading.Lock()
_CAMERA_AUTOSTART_THREAD: Optional[threading.Thread] = None


def _publish_raw_frame(frame: np.ndarray) -> None:
    """Publish a raw frame to the raw stream."""
    RAW_STREAM.add_frame(frame)


def set_active_overlay_task(task: Optional[OverlayTask]) -> None:
    """Set the currently active overlay task for camera-frame processing."""
    global _ACTIVE_OVERLAY_TASK
    with _OVERLAY_TASK_LOCK:
        _ACTIVE_OVERLAY_TASK = task


def get_active_overlay_task() -> Optional[OverlayTask]:
    """Get the currently active overlay task."""
    with _OVERLAY_TASK_LOCK:
        return _ACTIVE_OVERLAY_TASK


def process_camera_frame(frame: np.ndarray) -> None:
    """Process camera frames on arrival from the DepthAI reader thread."""
    _publish_raw_frame(frame)
    task = get_active_overlay_task()
    if task is not None:
        FRAME_PROCESSOR.submit(task, frame)


def _env_flag(name: str, default: bool) -> bool:
    """Read a boolean environment flag."""
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    return raw_value.strip().lower() not in {"0", "false", "no", "off"}


def schedule_camera_autostart(delay_s: float = 0.25) -> bool:
    """Start the configured camera from a background thread.

    The UI switch defaults to enabled, but relying on Dash's initial callback
    for camera startup is fragile while the page is mounting. Running the same
    startup path in a daemon thread keeps WebRTC fed without blocking the first
    usable page render.
    """
    global _CAMERA_AUTOSTART_THREAD
    if not _env_flag("CODROID_CAMERA_AUTOSTART", True):
        LOGGER.info("Camera autostart disabled by CODROID_CAMERA_AUTOSTART")
        return False

    with _CAMERA_AUTOSTART_LOCK:
        if (
            _CAMERA_AUTOSTART_THREAD is not None
            and _CAMERA_AUTOSTART_THREAD.is_alive()
        ):
            return False
        _CAMERA_AUTOSTART_THREAD = threading.Thread(
            target=_camera_autostart_worker,
            args=(delay_s,),
            name="CameraAutostart",
            daemon=True,
        )
        _CAMERA_AUTOSTART_THREAD.start()
        return True


def _camera_autostart_worker(delay_s: float) -> None:
    """Camera autostart worker entrypoint."""
    if delay_s > 0:
        time.sleep(delay_s)

    stream = SESSION.camera_stream
    if stream is not None and stream.is_running():
        return

    from app.settings import DEFAULT_RESOLUTION, DEFAULT_FPS, SETTINGS
    source = getattr(SETTINGS, "camera_source", "oak")
    if _normalize_camera_source(source) == "host" and platform.system() == "Darwin":
        # macOS AVFoundation permission prompts must run on the main run loop.
        # Local testing grants permission separately; app worker threads should
        # only open the already-authorized device.
        os.environ["OPENCV_AVFOUNDATION_SKIP_AUTH"] = "1"

    try:
        message = start_camera_stream(
            DEFAULT_RESOLUTION[0],
            DEFAULT_RESOLUTION[1],
            DEFAULT_FPS,
            lock_af=False,
            enable_depth_override=False,
            source=source,
            host_camera_index=int(getattr(SETTINGS, "host_camera_index", 0) or 0),
        )
        LOGGER.info("Camera autostart result: %s", message)
    except Exception as exc:  # noqa: BLE001
        LOGGER.error("Camera autostart failed: %s", exc, exc_info=True)
        SESSION.camera_last_error_message = str(exc)
        SESSION.camera_last_error_key = "Camera startup error: {error}"
        SESSION.camera_last_error_params = {"error": str(exc)}
        SESSION.camera_last_error_status = "error"


def _is_retryable_camera_error(exc: Exception) -> bool:
    """Check whether camera startup should be retried for this error."""
    message = str(exc).lower()
    return any(token in message for token in _RETRYABLE_CAMERA_ERROR_SUBSTRINGS)


def _normalize_camera_source(source: Optional[str]) -> str:
    """Return a supported camera source key."""
    normalized = (source or "").strip().lower()
    if normalized in {"host", "webcam", "opencv", "local"}:
        return "host"
    return "oak"


def start_camera_stream(
    width: Optional[float],
    height: Optional[float],
    fps: Optional[float],
    lock_af: bool = False,
    frame_hook: Optional[Callable] = None,
    enable_depth_override: Optional[bool] = None,
    *,
    source: Optional[str] = None,
    host_camera_index: Optional[int] = None,
) -> str:
    """Start the camera stream with the specified configuration."""
    with _STREAM_LIFECYCLE_LOCK:
        stop_camera_stream()

        # Get default values
        from app.settings import DEFAULT_RESOLUTION, DEFAULT_FPS, SETTINGS
        resolution = (safe_int(width, DEFAULT_RESOLUTION[0]), safe_int(
            height, DEFAULT_RESOLUTION[1]))
        fps_value = safe_float(fps, DEFAULT_FPS)
        camera_source = _normalize_camera_source(
            source or getattr(SETTINGS, "camera_source", "oak")
        )
        if enable_depth_override is None:
            enable_depth = bool(getattr(SETTINGS, "camera_enable_depth", False))
        else:
            enable_depth = bool(enable_depth_override)
        if camera_source == "host":
            enable_depth = False
        fast_depth = bool(getattr(SETTINGS, "camera_fast_depth", True))
        requested_fps = fps_value
        fps_cap = _RGB_WITH_DEPTH_MAX_FPS if enable_depth else _RGB_MAX_FPS
        if fps_cap > 0:
            fps_value = min(fps_value, fps_cap)
            if fps_value < requested_fps:
                LOGGER.info(
                    "RGB FPS capped for transport stability: requested %.1f, using %.1f",
                    requested_fps,
                    fps_value,
                )

        # Default to in-thread frame processing so stream cadence follows camera cadence.
        hook = frame_hook if frame_hook is not None else process_camera_frame

        last_error: Optional[Exception] = None
        for attempt in range(1, _CAMERA_START_MAX_ATTEMPTS + 1):
            try:
                config = StreamConfig(
                    resolution=resolution,
                    fps=fps_value,
                    lock_af=lock_af and camera_source == "oak",
                    enable_depth=enable_depth,
                    fast_depth=fast_depth,
                )
                if camera_source == "host":
                    stream = HostCameraFrameStream(
                        config,
                        frame_hook=hook,
                        camera_index=safe_int(
                            host_camera_index,
                            int(getattr(SETTINGS, "host_camera_index", 0) or 0),
                        ),
                    )
                else:
                    stream = DepthAIFrameStream(
                        config,
                        frame_hook=hook,
                    )
                stream.start()
                SESSION.camera_stream = stream

                deadline = time.monotonic() + 2.5
                while time.monotonic() < deadline:
                    if stream.status() == "live":
                        break
                    if stream.status() == "retrying":
                        break
                    time.sleep(0.05)

                device_id = stream.device_id() or "unknown"
                if stream.status() == "retrying":
                    last_error = stream.last_error() or "camera unavailable"
                    return (
                        f"Camera retrying ({resolution[0]}x{resolution[1]} @ "
                        f"{fps_value:.1f} FPS): {last_error}"
                    )
                device_label = "Host camera" if camera_source == "host" else "MXID"
                return (
                    f"Camera stream running ({resolution[0]}x{resolution[1]} @ "
                    f"{fps_value:.1f} FPS, {device_label} {device_id})."
                )
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                retryable = _is_retryable_camera_error(exc)
                final_attempt = attempt >= _CAMERA_START_MAX_ATTEMPTS
                LOGGER.error(
                    "Failed to start camera stream (attempt %s/%s): %s",
                    attempt,
                    _CAMERA_START_MAX_ATTEMPTS,
                    exc,
                )
                if final_attempt or not retryable:
                    break
                LOGGER.warning(
                    "Retrying camera start in %.1fs after transient error...",
                    _CAMERA_START_RETRY_DELAY_S,
                )
                stop_camera_stream(wait_for_release=True)
                time.sleep(_CAMERA_START_RETRY_DELAY_S)

        if last_error is None:
            raise RuntimeError("Camera stream failed to start for an unknown reason.")
        # Re-raise the exception so the callback can handle it properly
        raise last_error


def stop_camera_stream(wait_for_release: bool = False) -> None:
    """Stop the current camera stream.

    Args:
        wait_for_release: If True, wait briefly for USB device to be released.
            Use this during hot-reload cleanup to avoid device conflicts.
    """
    with _STREAM_LIFECYCLE_LOCK:
        try:
            if SESSION.camera_stream is not None:
                SESSION.camera_stream.stop()
                SESSION.camera_stream = None
                if wait_for_release:
                    # Allow time for USB device to be fully released
                    time.sleep(0.5)
        except Exception as exc:
            LOGGER.error("Error stopping camera stream: %s", exc)


def get_current_frame():
    """Get the current frame from the stream."""
    if SESSION.camera_stream is None:
        return None
    frame, _depth = SESSION.camera_stream.get_latest()
    return frame


def is_stream_active() -> bool:
    """Check if the camera stream is currently active."""
    return SESSION.camera_stream is not None


def get_stream_stats() -> dict:
    """Get streaming statistics."""
    if SESSION.camera_stream is None:
        return {
            "capture_fps": 0.0,
            "formatted_fps": _format_fps(0.0),
            "frame_count": 0,
            "status": "stopped"
        }

    capture_fps = SESSION.camera_stream.get_fps()
    return {
        "capture_fps": capture_fps,
        "formatted_fps": _format_fps(capture_fps),
        "frame_count": 0,
        "status": "running"
    }
