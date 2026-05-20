"""Streaming infrastructure for the calibration UI.

This module handles camera streaming, video feed management, and frame processing
for the calibration interface.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Deque, Optional

import cv2
import numpy as np

from app.codroid_picker.session import SESSION
from app.codroid_picker.utils import safe_float, safe_int
from vision.stream import DepthAIFrameStream, StreamConfig

LOGGER = logging.getLogger(__name__)
_CAMERA_START_MAX_ATTEMPTS = 8
_CAMERA_START_RETRY_DELAY_S = 2.0
_RGB_MAX_FPS = float(os.getenv("CODROID_RGB_MAX_FPS", "30"))
_RGB_WITH_DEPTH_MAX_FPS = float(os.getenv("CODROID_RGB_MAX_FPS_WITH_DEPTH", "15"))
_MJPEG_MAX_FPS = float(os.getenv("CODROID_MJPEG_MAX_FPS", "30"))
_OVERLAY_MAX_FPS = float(os.getenv("CODROID_OVERLAY_MAX_FPS", "5"))
_OVERLAY_EMPTY_HOLD_S = float(os.getenv("CODROID_OVERLAY_EMPTY_HOLD_S", "1.0"))
AUTOFOCUS_SETTLE_S = float(os.getenv("CODROID_AUTOFOCUS_SETTLE_S", "1.5"))
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
    primitives: list[dict[str, Any]] = field(default_factory=list)
    has_targets: bool = False


@dataclass
class OverlayTask:
    """Task for background overlay processing."""
    key: str
    runner: Callable[[np.ndarray], OverlayResult]


@dataclass(frozen=True)
class OverlayTaskSnapshot:
    """Active overlay task plus the generation it belongs to."""
    task: OverlayTask
    generation: int


@dataclass
class OverlayState:
    """Browser-rendered overlay state for a task."""
    key: str
    generation: int
    frame_width: int
    frame_height: int
    updated_at: float
    status: str
    metric: str
    primitives: list[dict[str, Any]]
    has_targets: bool
    stale: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe overlay state payload."""
        return {
            "key": self.key,
            "generation": self.generation,
            "frame_width": self.frame_width,
            "frame_height": self.frame_height,
            "updated_at": self.updated_at,
            "status": self.status,
            "metric": self.metric,
            "primitives": self.primitives,
            "has_targets": self.has_targets,
            "stale": self.stale,
        }


class MotionJPEGStream:
    """Manager for Motion JPEG video streaming for fullscreen video playback."""

    def __init__(self, frame_rate: float = 30.0, min_push_interval_s: Optional[float] = None):
        self.frame_rate = frame_rate
        self.frame_queue: list[bytes] = []
        self.lock = threading.Lock()
        self.max_queue_size = 3
        self.frame_count = 0
        if min_push_interval_s is None:
            max_fps = _MJPEG_MAX_FPS if _MJPEG_MAX_FPS > 0 else 30.0
            min_push_interval_s = 1.0 / max_fps
        self.min_push_interval_s = min_push_interval_s
        self._last_push_ts = 0.0
        self._fps = 0.0
        self._last_fps_ts = 0.0
        self._frame_index = 0

    def add_frame(self, frame: np.ndarray) -> None:
        """Add a frame to the stream queue, dropping frames if pushing too fast."""
        now = time.monotonic()
        with self.lock:
            if now - self._last_push_ts < self.min_push_interval_s:
                return
            # Reserve the stream slot before JPEG encoding. Multiple producers
            # can call add_frame concurrently, and checking this outside the
            # lock lets them all pass the rate limit at once.
            self._last_push_ts = now

        success, buffer = cv2.imencode(".jpg", frame)
        if not success:
            return

        with self.lock:
            self.frame_queue.append(buffer.tobytes())
            self.frame_count += 1
            self._frame_index += 1
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

    def get_frame_with_index(self) -> tuple[Optional[bytes], int]:
        """Get the latest frame bytes and monotonic frame index."""
        with self.lock:
            if self.frame_queue:
                return self.frame_queue[-1], self._frame_index
            return None, self._frame_index

    def get_fps(self) -> float:
        """Get the current FPS of the stream."""
        with self.lock:
            return self._fps


class FrameProcessor:
    """Background processor that keeps only the newest request and result."""

    def __init__(self, max_queue: int = 2) -> None:
        self._queue: Deque[tuple[str, int, np.ndarray,
                                 Callable[[np.ndarray], OverlayResult]]]
        self._queue = deque(maxlen=max_queue)
        self._lock = threading.Lock()
        self._latest: dict[str, OverlayState] = {}
        self._last_frame_size: tuple[int, int] | None = None
        self._busy = False
        self._last_submit_ts = 0.0
        self._min_submit_interval_s = (
            1.0 / _OVERLAY_MAX_FPS if _OVERLAY_MAX_FPS > 0 else 0.0
        )
        self._running = True
        self._thread = threading.Thread(
            target=self._worker, name="FrameProcessor", daemon=True
        )
        self._thread.start()

    def submit(self, task: OverlayTaskSnapshot, frame: np.ndarray) -> bool:
        """Submit a frame processing task when the worker can accept one."""
        now = time.monotonic()
        height, width = frame.shape[:2]
        with self._lock:
            self._last_frame_size = (int(width), int(height))
            if self._busy:
                return False
            if now - self._last_submit_ts < self._min_submit_interval_s:
                return False
            self._busy = True
            self._last_submit_ts = now
            self._queue.clear()
            self._queue.append(
                (task.task.key, task.generation, frame.copy(), task.task.runner)
            )
        return True

    def get_latest(self, key: str) -> Optional[OverlayResult]:
        """Get the latest processing result for a given key."""
        with self._lock:
            state = self._latest.get(key)
            if state is None:
                return None
            return OverlayResult(
                vis=np.zeros((1, 1, 3), dtype=np.uint8),
                status=state.status,
                metric=state.metric,
                primitives=[dict(item) for item in state.primitives],
                has_targets=state.has_targets,
            )

    def get_latest_state(self, key: str) -> Optional[OverlayState]:
        """Return the latest overlay state for a key."""
        with self._lock:
            state = self._latest.get(key)
            if state is None:
                return None
            return OverlayState(
                key=state.key,
                generation=state.generation,
                frame_width=state.frame_width,
                frame_height=state.frame_height,
                updated_at=state.updated_at,
                status=state.status,
                metric=state.metric,
                primitives=[dict(item) for item in state.primitives],
                has_targets=state.has_targets,
                stale=state.stale,
            )

    def get_active_state_payload(self) -> dict[str, Any]:
        """Return the current active overlay state payload."""
        snapshot = get_active_overlay_task()
        if snapshot is None:
            return _empty_overlay_payload()
        state = self.get_latest_state(snapshot.task.key)
        if state is None or state.generation != snapshot.generation:
            payload = _empty_overlay_payload(snapshot.task.key, snapshot.generation)
            payload["status"] = "Overlay processing..."
            payload["metric"] = "--"
            with self._lock:
                frame_size = self._last_frame_size
            if frame_size is not None:
                payload["frame_width"], payload["frame_height"] = frame_size
            return payload
        payload = state.to_dict()
        payload["stale"] = (time.time() - state.updated_at) > _OVERLAY_EMPTY_HOLD_S
        return payload

    def clear(self) -> None:
        """Clear all overlay states and pending work."""
        with self._lock:
            self._queue.clear()
            self._latest.clear()
            self._busy = False

    def store_latest(self, key: str, generation: int, frame: np.ndarray,
                     result: OverlayResult) -> None:
        """Store the latest overlay state and metrics."""
        height, width = frame.shape[:2]
        state = OverlayState(
            key=key,
            generation=generation,
            frame_width=int(width),
            frame_height=int(height),
            updated_at=time.time(),
            status=result.status,
            metric=result.metric,
            primitives=[dict(item) for item in result.primitives],
            has_targets=bool(result.has_targets),
        )
        with self._lock:
            self._last_frame_size = (int(width), int(height))
            self._latest[key] = state

    def stop(self) -> None:
        """Stop the frame processor."""
        self._running = False
        if self._thread.is_alive():
            self._thread.join(timeout=1.0)

    def _worker(self) -> None:
        """Background worker thread for processing frames.

        After computing the overlay, this worker stores JSON-ready overlay
        primitives. MJPEG streaming remains raw and independent.
        """
        while self._running:
            task: Optional[tuple[str, int, np.ndarray,
                                 Callable[[np.ndarray], OverlayResult]]] = None
            with self._lock:
                if self._queue:
                    # Pop newest to drop stale work.
                    task = self._queue.pop()
            if task is None:
                time.sleep(0.003)
                continue
            key, generation, frame, runner = task
            try:
                result = runner(frame)
            except Exception as exc:  # noqa: BLE001
                LOGGER.error("FrameProcessor error for %s: %s",
                             key, exc, exc_info=True)
                result = OverlayResult(frame, f"Processing error: {exc}", "--")
            if not is_active_overlay_task(key, generation):
                with self._lock:
                    self._busy = False
                continue
            self.store_latest(key, generation, frame, result)
            with self._lock:
                self._busy = False


class RawFrameStore:
    """In-memory store for the latest raw camera frame."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._latest: Optional[np.ndarray] = None

    def add_frame(self, frame: np.ndarray) -> None:
        """Store the latest raw frame."""
        with self._lock:
            self._latest = frame.copy()

    def get_latest_frame(self) -> Optional[np.ndarray]:
        """Return a copy of the most recent raw frame."""
        with self._lock:
            if self._latest is None:
                return None
            return self._latest.copy()


def _format_fps(capture_fps: float, stream_fps: float) -> str:
    """Format FPS for display."""
    cap_text = f"{capture_fps:.1f}" if capture_fps > 0 else "--"
    stream_text = f"{stream_fps:.1f}" if stream_fps > 0 else "--"
    return f"FPS — capture: {cap_text}, stream: {stream_text}"


def _empty_overlay_payload(
    key: Optional[str] = None,
    generation: Optional[int] = None,
) -> dict[str, Any]:
    """Return the stable empty overlay JSON shape."""
    return {
        "key": key,
        "generation": generation,
        "frame_width": 0,
        "frame_height": 0,
        "updated_at": 0.0,
        "status": "No overlay applied.",
        "metric": "--",
        "primitives": [],
        "has_targets": False,
        "stale": False,
    }


# Global streaming instances
MJPEG_STREAM = MotionJPEGStream()
FRAME_PROCESSOR = FrameProcessor()
RAW_STREAM = RawFrameStore()
_ACTIVE_OVERLAY_TASK: Optional[OverlayTask] = None
_ACTIVE_OVERLAY_GENERATION = 0
_OVERLAY_TASK_LOCK = threading.Lock()
_STREAM_LIFECYCLE_LOCK = threading.RLock()


def _publish_raw_frame(frame: np.ndarray) -> None:
    """Publish a raw frame to the raw stream."""
    RAW_STREAM.add_frame(frame)


def set_active_overlay_task(task: Optional[OverlayTask]) -> None:
    """Set the currently active overlay task for camera-frame processing."""
    global _ACTIVE_OVERLAY_TASK, _ACTIVE_OVERLAY_GENERATION
    with _OVERLAY_TASK_LOCK:
        old_key = None if _ACTIVE_OVERLAY_TASK is None else _ACTIVE_OVERLAY_TASK.key
        new_key = None if task is None else task.key
        if old_key != new_key:
            _ACTIVE_OVERLAY_GENERATION += 1
            FRAME_PROCESSOR.clear()
        _ACTIVE_OVERLAY_TASK = task


def get_active_overlay_task() -> Optional[OverlayTaskSnapshot]:
    """Get the currently active overlay task snapshot."""
    with _OVERLAY_TASK_LOCK:
        if _ACTIVE_OVERLAY_TASK is None:
            return None
        return OverlayTaskSnapshot(_ACTIVE_OVERLAY_TASK, _ACTIVE_OVERLAY_GENERATION)


def is_active_overlay_task(key: str, generation: int) -> bool:
    """Return whether an overlay result still belongs to the active task."""
    with _OVERLAY_TASK_LOCK:
        return (
            _ACTIVE_OVERLAY_TASK is not None
            and _ACTIVE_OVERLAY_TASK.key == key
            and _ACTIVE_OVERLAY_GENERATION == generation
        )


def process_camera_frame(frame: np.ndarray) -> None:
    """Process camera frames on arrival from the DepthAI reader thread."""
    _publish_raw_frame(frame)
    task = get_active_overlay_task()
    if task is None:
        MJPEG_STREAM.add_frame(frame)
        return
    FRAME_PROCESSOR.submit(task, frame)
    MJPEG_STREAM.add_frame(frame)


def _is_retryable_camera_error(exc: Exception) -> bool:
    """Check whether camera startup should be retried for this error."""
    message = str(exc).lower()
    return any(token in message for token in _RETRYABLE_CAMERA_ERROR_SUBSTRINGS)


def start_camera_stream(width: Optional[float], height: Optional[float], fps: Optional[float],
                        lock_af: bool = False, frame_hook: Optional[Callable] = None,
                        enable_depth_override: Optional[bool] = None) -> str:
    """Start the camera stream with the specified configuration."""
    with _STREAM_LIFECYCLE_LOCK:
        stop_camera_stream()

        # Get default values
        from app.settings import DEFAULT_RESOLUTION, DEFAULT_FPS, SETTINGS
        resolution = (safe_int(width, DEFAULT_RESOLUTION[0]), safe_int(
            height, DEFAULT_RESOLUTION[1]))
        fps_value = safe_float(fps, DEFAULT_FPS)
        if enable_depth_override is None:
            enable_depth = bool(getattr(SETTINGS, "camera_enable_depth", False))
        else:
            enable_depth = bool(enable_depth_override)
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
                # Create and start stream
                stream = DepthAIFrameStream(
                    StreamConfig(
                        resolution=resolution,
                        fps=fps_value,
                        lock_af=False,
                        enable_depth=enable_depth,
                        fast_depth=fast_depth,
                    ),
                    frame_hook=hook,
                )
                stream.start()
                SESSION.camera_stream = stream
                focus_suffix = ""
                if lock_af:
                    focus_position = stream.refocus_and_lock(AUTOFOCUS_SETTLE_S)
                    focus_suffix = f" Focus locked at lens position {focus_position}."

                device_id = stream.device_id() or "unknown"
                return (
                    f"Camera stream running ({resolution[0]}x{resolution[1]} @ "
                    f"{fps_value:.1f} FPS, MXID {device_id}).{focus_suffix}"
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
            "stream_fps": 0.0,
            "formatted_fps": _format_fps(0.0, 0.0),
            "frame_count": 0,
            "status": "stopped"
        }

    return {
        "capture_fps": 0.0,  # Would need to be implemented in DepthAIFrameStream
        "stream_fps": 0.0,   # Would need to be implemented in DepthAIFrameStream
        "formatted_fps": _format_fps(0.0, 0.0),
        "frame_count": 0,    # Would need to be implemented in DepthAIFrameStream
        "status": "running"
    }


def create_video_feed_generator():
    """Create a generator for the video feed endpoint."""
    def generate():
        boundary = b'--frame\r\n'
        frame_count = 0
        last_frame_index = -1
        while True:
            frame_data, frame_index = MJPEG_STREAM.get_frame_with_index()

            if not frame_data or frame_index == last_frame_index:
                time.sleep(0.005)
                continue

            last_frame_index = frame_index
            frame_count += 1
            if frame_count == 1:
                LOGGER.debug(
                    "/video_feed serving first frame, size=%s bytes", len(frame_data))

            yield boundary
            yield b'Content-Type: image/jpeg\r\n'
            yield b'Content-Length: ' + str(len(frame_data)).encode() + b'\r\n\r\n'
            yield frame_data
            yield b'\r\n'

    return generate()
