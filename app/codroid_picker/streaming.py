"""Streaming infrastructure for the calibration UI.

This module handles camera streaming, video feed management, and frame processing
for the calibration interface.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Callable, Deque, Optional

import cv2
import numpy as np

from app.codroid_picker.session import SESSION
from app.codroid_picker.utils import safe_float, safe_int
from vision.stream import DepthAIFrameStream, StreamConfig

LOGGER = logging.getLogger(__name__)


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
        self._frame_index = 0

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
            self._frame_index += 1
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
        """Submit a frame processing task."""
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
        """Background worker thread for processing frames.

        After computing the overlay, this worker pushes the result directly
        to MJPEG_STREAM, decoupling video streaming from Dash callback timing.
        """
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
            # Push directly to MJPEG stream - decouples video from Dash callbacks
            MJPEG_STREAM.add_frame(result.vis)


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


# Global streaming instances
MJPEG_STREAM = MotionJPEGStream()
FRAME_PROCESSOR = FrameProcessor()
RAW_STREAM = RawFrameStore()
_ACTIVE_OVERLAY_TASK: Optional[OverlayTask] = None
_OVERLAY_TASK_LOCK = threading.Lock()
_STREAM_LIFECYCLE_LOCK = threading.RLock()


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
    if task is None:
        MJPEG_STREAM.add_frame(frame)
        return
    FRAME_PROCESSOR.submit(task, frame)


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

        # Default to in-thread frame processing so stream cadence follows camera cadence.
        hook = frame_hook if frame_hook is not None else process_camera_frame

        try:
            # Create and start stream
            stream = DepthAIFrameStream(
                StreamConfig(
                    resolution=resolution,
                    fps=fps_value,
                    lock_af=lock_af,
                    enable_depth=enable_depth,
                    fast_depth=fast_depth,
                ),
                frame_hook=hook,
            )
            stream.start()
            SESSION.camera_stream = stream

            device_id = stream.device_id() or "unknown"
            return f"Camera stream running ({resolution[0]}x{resolution[1]} @ {fps_value:.1f} FPS, MXID {device_id})."

        except Exception as exc:
            LOGGER.error("Failed to start camera stream: %s", exc)
            # Re-raise the exception so the callback can handle it properly
            raise


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
