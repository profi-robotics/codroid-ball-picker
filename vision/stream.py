"""
Live DepthAI camera streaming helpers for the Dash calibration UI.

This module keeps the most recent RGB/depth frames in memory so the web UI can
poll without blocking on device queues.
"""

from __future__ import annotations

from contextlib import ExitStack
from dataclasses import dataclass
import threading
import time
from typing import Optional, Tuple

import depthai as dai
import numpy as np

from .capture import build_pipeline, resolve_camera_sockets


@dataclass
class StreamConfig:
    resolution: tuple[int, int] = (1280, 720)
    fps: float = 30.0


class DepthAIFrameStream:
    """Background reader that stores the latest RGB/depth frames."""

    def __init__(self, config: StreamConfig) -> None:
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

    def start(self) -> None:
        if self._running:
            return
        self._stack = ExitStack()
        pipeline = self._stack.enter_context(dai.Pipeline())
        device = pipeline.getDefaultDevice()
        sockets = resolve_camera_sockets(device)
        color_out, depth_out = build_pipeline(
            pipeline,
            sockets,
            self.config.resolution,
            self.config.fps,
        )
        self._color_queue = color_out.createOutputQueue(maxSize=2, blocking=True)
        self._depth_queue = depth_out.createOutputQueue(maxSize=2, blocking=False)
        pipeline.start()

        self._pipeline = pipeline
        self._device = device
        self._running = True
        self._thread = threading.Thread(target=self._reader_loop, name="DepthAIFrameStream", daemon=True)
        self._thread.start()

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
        if callable(mx_id_fn):
            return mx_id_fn()
        return None

    def get_latest(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        with self._lock:
            if self._last_frame is None:
                return None, None
            frame = self._last_frame.copy()
            depth = None if self._last_depth is None else self._last_depth.copy()
        return frame, depth

    def _reader_loop(self) -> None:
        assert self._color_queue is not None
        while self._running:
            try:
                packet = self._color_queue.get()
                frame = packet.getCvFrame()
                depth_frame = None
                if self._depth_queue is not None:
                    depth_packet = self._depth_queue.tryGet()
                    if depth_packet is not None:
                        depth_frame = depth_packet.getFrame()
                with self._lock:
                    self._last_frame = frame
                    if depth_frame is not None:
                        self._last_depth = depth_frame
            except Exception:
                time.sleep(0.02)
