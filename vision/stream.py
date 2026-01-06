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


class DepthAIFrameStream:
    """Background reader that stores the latest RGB/depth frames and can publish frames via a hook."""

    def __init__(self, config: StreamConfig, frame_hook: Optional[Callable[[np.ndarray], None]] = None) -> None:
        self.config = config
        self._stack: Optional[ExitStack] = None
        self._pipeline: Optional[dai.Pipeline] = None
        self._device: Optional[dai.Device] = None
        self._color_queue: Optional[dai.DataOutputQueue] = None
        self._depth_queue: Optional[dai.DataOutputQueue] = None
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._lock = threading.Lock()
        self._last_frame: Optional[np.ndarray] = None
        self._last_depth: Optional[np.ndarray] = None
        self._fps = 0.0
        self._last_ts = 0.0
        self._frame_hook = frame_hook

    def start(self) -> None:
        if self._running:
            return
        try:
            self._stack = ExitStack()
            pipeline = self._stack.enter_context(dai.Pipeline())
            logging.info("Created pipeline, calling getDefaultDevice")
            device = pipeline.getDefaultDevice()
            logging.info("Got device: %s", device)
            if device is None:
                raise RuntimeError(
                    "getDefaultDevice() returned None; no camera connected")
            sockets = resolve_camera_sockets(device)
            logging.info("Resolved sockets: %s", sockets)
            color_out, depth_out = build_pipeline(
                pipeline,
                sockets,
                self.config.resolution,
                self.config.fps,
                enable_depth=self.config.enable_depth,
                fast_depth=self.config.fast_depth,
            )
            logging.info("Built pipeline, creating queues")
            self._color_queue = color_out.createOutputQueue(
                maxSize=self.config.queue_size, blocking=False)
            if self.config.enable_depth and depth_out is not None:
                self._depth_queue = depth_out.createOutputQueue(
                    maxSize=self.config.queue_size, blocking=False)
            logging.info("Created queues, starting pipeline")
            pipeline.start()
            logging.info("Pipeline started")

            self._pipeline = pipeline
            self._device = device
            self._running = True
            self._thread = threading.Thread(
                target=self._reader_loop, name="DepthAIFrameStream", daemon=True)
            self._thread.start()
            logging.info("Reader thread started")
        except Exception as exc:  # noqa: BLE001
            logging.error(
                "Failed to start DepthAIFrameStream: %s", exc, exc_info=True)
            self._running = False
            if self._stack is not None:
                self._stack.close()
                self._stack = None
            raise

    def stop(self) -> None:
        if not self._running:
            return
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        if self._pipeline is not None:
            self._pipeline.stop()
        if self._stack is not None:
            self._stack.close()
        self._stack = None
        self._pipeline = None
        self._device = None
        self._color_queue = None
        self._depth_queue = None
        self._thread = None
        with self._lock:
            self._last_frame = None
            self._last_depth = None

    def is_running(self) -> bool:
        return self._running

    def device_id(self) -> Optional[str]:
        if self._device is None:
            return None
        mx_id_fn = getattr(self._device, "getMxId", None)
        if mx_id_fn is not None:
            try:
                mx_id = mx_id_fn()  # type: ignore[operator]
            except (RuntimeError, TypeError):
                mx_id = None
            if mx_id:
                return str(mx_id)
        fallback = getattr(self._device, "mxid", None)
        return str(fallback) if fallback is not None else None

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
        assert self._color_queue is not None
        while self._running:
            try:
                packet = self._color_queue.tryGet()
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
                depth_frame = None
                if self._depth_queue is not None:
                    while True:
                        depth_packet = self._depth_queue.tryGet()
                        if depth_packet is None:
                            break
                        depth_frame = depth_packet.getFrame()
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
            except (RuntimeError, OSError) as exc:
                logging.error(
                    "DepthAIFrameStream reader error: %s", exc, exc_info=True)
                time.sleep(0.02)
