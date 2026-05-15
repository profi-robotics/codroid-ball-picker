"""Generic OpenCV camera adapter for host, USB, V4L2, and pipeline sources."""

from __future__ import annotations

import logging
import os
import platform
import threading
import time
from contextlib import suppress
from typing import Any, Optional, Union

import cv2
import numpy as np


OpenCVCameraSource = Union[int, str]


class OpenCVCameraStream:
    """Background OpenCV reader for generic camera sources."""

    def __init__(
        self,
        config: Any,
        frame_hook: Any = None,
        *,
        camera_source: OpenCVCameraSource = 0,
    ) -> None:
        self.config = config
        self.camera_source = _normalize_camera_source(camera_source)
        self._frame_hook = frame_hook
        self._capture: Any = None
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._lock = threading.Lock()
        self._state_lock = threading.Lock()
        self._last_frame: Optional[np.ndarray] = None
        self._fps = 0.0
        self._last_ts = 0.0
        self._status = "stopped"
        self._last_error_message: Optional[str] = None
        self._retry_attempt = 0
        self._retry_max_backoff_s = float(os.getenv("CODROID_CAMERA_RETRY_MAX_BACKOFF_S", "30"))

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._status = "starting"
        self._thread = threading.Thread(
            target=self._reader_loop,
            name=f"OpenCVCameraStream-{self.device_id()}",
            daemon=True,
        )
        self._thread.start()
        logging.info("OpenCV camera reader started for source %s", self.camera_source)

    def stop(self) -> None:
        if not self._running:
            return
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        with self._state_lock:
            self._close_capture()
        self._thread = None
        with self._lock:
            self._last_frame = None
        self._status = "stopped"
        self._last_error_message = None
        self._retry_attempt = 0

    def is_running(self) -> bool:
        return self._running

    def status(self) -> str:
        return self._status

    def last_error(self) -> Optional[str]:
        return self._last_error_message

    def device_id(self) -> str:
        return f"opencv:{self.camera_source}"

    def get_latest(self) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        with self._lock:
            if self._last_frame is None:
                return None, None
            return self._last_frame.copy(), None

    def get_fps(self) -> float:
        return self._fps

    def set_focus_lock(self, lock: bool, lens_position: Optional[int] = None) -> None:
        """Keep the camera callback API compatible; generic focus control is optional."""

    def _reader_loop(self) -> None:
        while self._running:
            try:
                capture = self._capture
                if capture is None:
                    self._open_capture_with_backoff()
                    continue

                ok, frame = capture.read()
                if not ok or frame is None:
                    raise RuntimeError(f"OpenCV camera {self.camera_source} did not return a frame.")

                frame = self._normalize_frame(frame)
                now = time.time()
                if self._last_ts > 0.0:
                    dt = now - self._last_ts
                    if dt > 0:
                        inst_fps = 1.0 / dt
                        self._fps = inst_fps if self._fps <= 0.0 else 0.9 * self._fps + 0.1 * inst_fps
                self._last_ts = now
                with self._lock:
                    self._last_frame = frame
                self._status = "live"
                self._last_error_message = None
                self._retry_attempt = 0
                if self._frame_hook is not None:
                    with suppress(Exception):
                        try:
                            self._frame_hook(frame, None)
                        except TypeError:
                            self._frame_hook(frame)
            except Exception as exc:  # noqa: BLE001
                self._last_error_message = str(exc)
                self._status = "retrying"
                logging.warning("OpenCV camera reader error: %s", exc)
                with self._state_lock:
                    self._close_capture()
                time.sleep(0.05)

    def _open_capture_with_backoff(self) -> None:
        if not self._running:
            return
        try:
            self._open_capture()
            self._status = "live"
            self._last_error_message = None
            self._retry_attempt = 0
        except Exception as exc:  # noqa: BLE001
            self._last_error_message = str(exc)
            self._status = "retrying"
            self._retry_attempt += 1
            delay = min(
                self._retry_max_backoff_s,
                1.0 if self._retry_attempt <= 1 else 3.0 * (2 ** (self._retry_attempt - 2)),
            )
            logging.warning(
                "OpenCV camera %s unavailable; retrying in %.1fs: %s",
                self.camera_source,
                delay,
                exc,
            )
            time.sleep(delay)

    def _open_capture(self) -> None:
        with self._state_lock:
            source, backend = _capture_source_and_backend(self.camera_source)
            capture = cv2.VideoCapture(source, backend)
            if not capture.isOpened():
                capture.release()
                raise RuntimeError(f"OpenCV camera {self.camera_source} unavailable.")
            width, height = self.config.resolution
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, float(width))
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, float(height))
            capture.set(cv2.CAP_PROP_FPS, float(self.config.fps))
            self._capture = capture

    def _close_capture(self) -> None:
        if self._capture is not None:
            self._capture.release()
        self._capture = None

    @staticmethod
    def _normalize_frame(frame: np.ndarray) -> np.ndarray:
        if frame.ndim == 2:
            return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        if frame.ndim == 3 and frame.shape[2] == 4:
            return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        return frame.copy()


def create_host_stream(
    config: Any,
    frame_hook: Any = None,
    *,
    camera_index: int = 0,
    camera_source: Union[int, str, None] = None,
) -> OpenCVCameraStream:
    """Create a generic OpenCV camera stream.

    ``camera_source`` may be an integer index, a ``/dev/video*`` path, a stable
    ``/dev/v4l/by-id/...`` path, or a GStreamer pipeline prefixed with ``gst:``.
    """

    return OpenCVCameraStream(
        config,
        frame_hook=frame_hook,
        camera_source=camera_source if camera_source is not None else camera_index,
    )


def _normalize_camera_source(source: OpenCVCameraSource) -> OpenCVCameraSource:
    if isinstance(source, int):
        return source
    stripped = source.strip()
    if stripped == "":
        return 0
    try:
        return int(stripped)
    except ValueError:
        return stripped


def _capture_source_and_backend(source: OpenCVCameraSource) -> tuple[OpenCVCameraSource, int]:
    if isinstance(source, str) and source.startswith("gst:"):
        return source.removeprefix("gst:"), cv2.CAP_GSTREAMER
    return source, _host_camera_backend()


def _host_camera_backend() -> int:
    if platform.system() == "Darwin":
        os.environ.setdefault("OPENCV_AVFOUNDATION_SKIP_AUTH", "1")
        return cv2.CAP_AVFOUNDATION
    if platform.system() == "Linux":
        return cv2.CAP_V4L2
    return cv2.CAP_ANY
