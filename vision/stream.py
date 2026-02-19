"""
Live DepthAI camera streaming helpers for the Dash calibration UI.

This module keeps the most recent RGB/depth frames in memory so the web UI can
poll without blocking on device queues.
"""

from __future__ import annotations

from contextlib import ExitStack, suppress
from dataclasses import dataclass
import logging
import threading
import time
from typing import Callable, Optional, Tuple

import cv2
import depthai as dai
import numpy as np

from .capture import build_pipeline, resolve_camera_sockets


@dataclass
class StreamConfig:
    resolution: tuple[int, int] = (1280, 720)
    fps: float = 30.0
    queue_size: int = 1
    enable_depth: bool = True
    fast_depth: bool = True
    lock_af: bool = False


class DepthAIFrameStream:
    """Background reader that stores the latest RGB/depth frames and can publish frames via a hook."""

    def __init__(self, config: StreamConfig, frame_hook: Optional[Callable[[np.ndarray], None]] = None) -> None:
        self.config = config
        self._stack: Optional[ExitStack] = None
        self._pipeline: Optional[dai.Pipeline] = None
        self._device: Optional[dai.Device] = None
        self._device_id: Optional[str] = None
        self._color_queue: Optional[dai.DataOutputQueue] = None
        self._depth_queue: Optional[dai.DataOutputQueue] = None
        self._control_queue: Optional[dai.DataInputQueue] = None
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._lock = threading.Lock()
        self._state_lock = threading.RLock()
        self._last_frame: Optional[np.ndarray] = None
        self._last_depth: Optional[np.ndarray] = None
        self._last_lens_pos: Optional[int] = None
        self._fps = 0.0
        self._last_ts = 0.0
        self._frame_hook = frame_hook
        self._consecutive_errors = 0
        self._last_error_ts = 0.0

    def start(self) -> None:
        with self._state_lock:
            if self._running:
                return
            if self._thread is not None and self._thread.is_alive():
                raise RuntimeError(
                    "Cannot start stream while previous reader thread is still shutting down."
                )
        try:
            self._running = True
            self._open_pipeline()
            self._thread = threading.Thread(
                target=self._reader_loop, name="DepthAIFrameStream", daemon=True)
            self._thread.start()
            logging.info("Reader thread started")
        except Exception as exc:  # noqa: BLE001
            logging.error(
                "Failed to start DepthAIFrameStream: %s", exc, exc_info=True)
            self._running = False
            self._close_pipeline()
            raise

    def stop(self) -> None:
        with self._state_lock:
            if not self._running:
                return
            self._running = False
            thread = self._thread

        if thread is not None:
            # DepthAI queue objects are native-backed; only close them after the
            # reader thread fully exits to avoid use-after-close crashes.
            thread.join(timeout=5.0)
            if thread.is_alive():
                raise RuntimeError(
                    "Timed out waiting for camera reader thread shutdown; refusing to close pipeline unsafely."
                )

        with self._state_lock:
            self._close_pipeline()
        self._thread = None
        with self._lock:
            self._last_frame = None
            self._last_depth = None
            self._last_lens_pos = None
        self._consecutive_errors = 0
        self._last_error_ts = 0.0

    def is_running(self) -> bool:
        return self._running

    def device_id(self) -> Optional[str]:
        with self._state_lock:
            return self._device_id

    def get_latest(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        with self._lock:
            if self._last_frame is None:
                return None, None
            frame = self._last_frame.copy()
            depth = None if self._last_depth is None else self._last_depth.copy()
        return frame, depth

    def get_fps(self) -> float:
        return self._fps

    def set_frame_hook(self, hook: Optional[Callable[[np.ndarray], None]]) -> None:
        self._frame_hook = hook

    def _reader_loop(self) -> None:
        while self._running:
            try:
                color_queue = self._color_queue
                if color_queue is None:
                    time.sleep(0.01)
                    continue
                packet = color_queue.tryGet()
                if packet is None:
                    time.sleep(0.001)
                    continue

                # Drain any backlog so we always keep the most recent frame.
                while True:
                    next_packet = self._color_queue.tryGet()
                    if next_packet is None:
                        break
                    packet = next_packet

                frame = packet.getCvFrame()
                if frame is None:
                    time.sleep(0.001)
                    continue
                # NV12 transport can arrive as a single-plane YUV image; convert to BGR.
                if frame.ndim == 2:
                    height = int(packet.getHeight())
                    width = int(packet.getWidth())
                    if frame.shape == (height + (height // 2), width):
                        frame = cv2.cvtColor(frame, cv2.COLOR_YUV2BGR_NV12)
                    else:
                        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                frame = frame.copy()
                with suppress(Exception):
                    lens = packet.getLensPosition()
                    with self._lock:
                        self._last_lens_pos = int(lens)
                depth_frame = None
                if self._depth_queue is not None:
                    while True:
                        depth_packet = self._depth_queue.tryGet()
                        if depth_packet is None:
                            break
                        depth_frame = depth_packet.getFrame().copy()
                now = time.time()
                if self._last_ts > 0.0:
                    dt = now - self._last_ts
                    if dt > 0:
                        inst_fps = 1.0 / dt
                        self._fps = 0.9 * self._fps + 0.1 * inst_fps
                self._last_ts = now
                with self._lock:
                    self._last_frame = frame
                    if depth_frame is not None:
                        self._last_depth = depth_frame
                if self._frame_hook is not None:
                    with suppress(Exception):
                        self._frame_hook(frame)
            except Exception as exc:  # noqa: BLE001
                logging.error(
                    "DepthAIFrameStream reader error: %s", exc, exc_info=True)
                now = time.time()
                if now - self._last_error_ts < 2.0:
                    self._consecutive_errors += 1
                else:
                    self._consecutive_errors = 1
                self._last_error_ts = now
                if self._consecutive_errors >= 3:
                    logging.warning(
                        "Restarting camera pipeline after repeated errors.")
                    self._consecutive_errors = 0
                    self._restart_pipeline()
                time.sleep(0.05)

    def _open_pipeline(self) -> None:
        with self._state_lock:
            self._stack = ExitStack()
            pipeline = self._stack.enter_context(dai.Pipeline())
            logging.info("Created pipeline, calling getDefaultDevice")
            device = pipeline.getDefaultDevice()
            logging.info("Got device: %s", device)
            if device is None:
                raise RuntimeError(
                    "getDefaultDevice() returned None; no camera connected")
            usb_speed_fn = getattr(device, "getUsbSpeed", None)
            if callable(usb_speed_fn):
                with suppress(Exception):
                    logging.info("DepthAI USB speed: %s", usb_speed_fn())
            sockets = resolve_camera_sockets(device)
            logging.info("Resolved sockets: %s", sockets)
            color_out, depth_out, control_in = build_pipeline(
                pipeline,
                sockets,
                self.config.resolution,
                self.config.fps,
                enable_depth=self.config.enable_depth,
                fast_depth=self.config.fast_depth,
                lock_af=self.config.lock_af,
            )
            logging.info("Built pipeline, creating queues")
            self._color_queue = color_out.createOutputQueue(
                maxSize=self.config.queue_size, blocking=False)
            if self.config.enable_depth and depth_out is not None:
                self._depth_queue = depth_out.createOutputQueue(
                    maxSize=self.config.queue_size, blocking=False)
            if control_in is not None:
                try:
                    self._control_queue = control_in.createInputQueue(
                        maxSize=1, blocking=False
                    )
                except Exception as exc:  # noqa: BLE001
                    logging.warning(
                        "Camera control input unavailable; focus lock changes will restart stream: %s",
                        exc,
                    )
            logging.info("Created queues, starting pipeline")
            pipeline.start()
            logging.info("Pipeline started")

            self._pipeline = pipeline
            self._device = device
            self._device_id = self._resolve_device_id(device)

    def _close_pipeline(self) -> None:
        if self._pipeline is not None:
            self._pipeline.stop()
        if self._stack is not None:
            self._stack.close()
        self._stack = None
        self._pipeline = None
        self._device = None
        self._device_id = None
        self._color_queue = None
        self._depth_queue = None
        self._control_queue = None

    def _restart_pipeline(self) -> None:
        with self._state_lock:
            if not self._running:
                return
            self._close_pipeline()
            time.sleep(0.5)
            try:
                self._open_pipeline()
            except Exception as exc:  # noqa: BLE001
                logging.error(
                    "Camera pipeline restart failed: %s", exc, exc_info=True
                )

    def set_focus_lock(self, lock: bool, lens_position: Optional[int] = None) -> None:
        """Dynamically switch AF lock without restarting the stream."""

        if self._control_queue is None:
            raise RuntimeError("Camera control input not available on this device.")

        ctrl = dai.CameraControl()
        if lock:
            pos = lens_position
            if pos is None:
                with self._lock:
                    pos = self._last_lens_pos
            if pos is None:
                pos = 130
            with suppress(Exception):
                ctrl.setAutoFocusMode(dai.CameraControl.AutoFocusMode.OFF)
            if hasattr(ctrl, "setAutoFocusEnable"):
                with suppress(Exception):
                    ctrl.setAutoFocusEnable(False)
            with suppress(Exception):
                ctrl.setManualFocus(int(pos))
        else:
            with suppress(Exception):
                ctrl.setAutoFocusMode(
                    dai.CameraControl.AutoFocusMode.CONTINUOUS_VIDEO
                )
            if hasattr(ctrl, "setAutoFocusEnable"):
                with suppress(Exception):
                    ctrl.setAutoFocusEnable(True)
        self._control_queue.send(ctrl)

    @staticmethod
    def _resolve_device_id(device: dai.Device) -> Optional[str]:
        """Resolve device MXID once to avoid cross-thread device RPC calls."""
        mx_id_fn = getattr(device, "getMxId", None)
        if mx_id_fn is not None:
            try:
                mx_id = mx_id_fn()  # type: ignore[operator]
            except (RuntimeError, TypeError):
                mx_id = None
            if mx_id:
                return str(mx_id)
        fallback = getattr(device, "mxid", None)
        return str(fallback) if fallback is not None else None
