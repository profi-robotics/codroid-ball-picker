"""Newest-wins frame buffers."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class FrameSnapshot:
    frame: Optional[np.ndarray]
    depth: Optional[np.ndarray]
    frame_id: int
    timestamp: float


class RawFrameBuffer:
    """Thread-safe latest raw frame store."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._frame: Optional[np.ndarray] = None
        self._depth: Optional[np.ndarray] = None
        self._frame_id = 0
        self._timestamp = 0.0

    def put(self, frame: np.ndarray, depth: Optional[np.ndarray] = None) -> int:
        with self._lock:
            self._frame = frame.copy()
            self._depth = None if depth is None else depth.copy()
            self._frame_id += 1
            self._timestamp = time.time()
            return self._frame_id

    def latest(self, *, copy: bool = True) -> FrameSnapshot:
        with self._lock:
            frame = self._frame
            depth = self._depth
            if copy:
                frame = None if frame is None else frame.copy()
                depth = None if depth is None else depth.copy()
            return FrameSnapshot(frame, depth, self._frame_id, self._timestamp)

    def age_s(self) -> Optional[float]:
        with self._lock:
            if self._timestamp <= 0.0:
                return None
            return max(time.time() - self._timestamp, 0.0)

