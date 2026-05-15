"""Deterministic fake hardware adapters for tests and no-hardware demos."""

from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass
from typing import Any, Optional

import cv2
import numpy as np


@dataclass
class FakeStreamConfig:
    resolution: tuple[int, int] = (640, 360)
    fps: float = 15.0
    enable_depth: bool = False
    fast_depth: bool = True
    lock_af: bool = False
    frame_drop_every: int = 0


class FakeCameraStream:
    """Simple synthetic frame producer."""

    def __init__(self, config: FakeStreamConfig, frame_hook: Any = None) -> None:
        self.config = config
        self._frame_hook = frame_hook
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._last_frame: Optional[np.ndarray] = None
        self._fps = 0.0
        self._last_ts = 0.0
        self._status = "stopped"
        self.generated_frames = 0
        self.dropped_frames = 0
        self.restart_count = 0

    def start(self) -> None:
        if self._running:
            return
        if self._thread is None and self.generated_frames > 0:
            self.restart_count += 1
        self._running = True
        self._status = "live"
        self._thread = threading.Thread(target=self._loop, name="FakeCameraStream", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        self._thread = None
        self._status = "stopped"

    def is_running(self) -> bool:
        return self._running

    def status(self) -> str:
        return self._status

    def last_error(self) -> Optional[str]:
        return None

    def device_id(self) -> Optional[str]:
        return "fake:camera"

    def get_latest(self) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        with self._lock:
            if self._last_frame is None:
                return None, None
            return self._last_frame.copy(), None

    def get_fps(self) -> float:
        return self._fps

    def set_focus_lock(self, lock: bool, lens_position: Optional[int] = None) -> None:
        self.config.lock_af = lock

    def _loop(self) -> None:
        period = 1.0 / max(float(self.config.fps), 1.0)
        index = 0
        while self._running:
            if self.config.frame_drop_every > 0 and index > 0 and index % self.config.frame_drop_every == 0:
                self.dropped_frames += 1
                index += 1
                time.sleep(period)
                continue
            frame = self._make_frame(index)
            now = time.time()
            if self._last_ts > 0.0:
                dt = now - self._last_ts
                if dt > 0:
                    inst = 1.0 / dt
                    self._fps = inst if self._fps <= 0 else (0.85 * self._fps + 0.15 * inst)
            self._last_ts = now
            with self._lock:
                self._last_frame = frame
            if self._frame_hook is not None:
                self._frame_hook(frame)
            self.generated_frames += 1
            index += 1
            time.sleep(period)

    def _make_frame(self, index: int) -> np.ndarray:
        width, height = self.config.resolution
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[:, :] = (24, 30, 38)
        x = int((math.sin(index / 12.0) * 0.35 + 0.5) * width)
        y = int((math.cos(index / 16.0) * 0.25 + 0.5) * height)
        cv2.circle(frame, (x, y), max(min(width, height) // 16, 12), (0, 140, 255), -1)
        cv2.putText(frame, "Codroid fake camera", (24, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (230, 230, 230), 2)
        return frame


class FakeRobotClient:
    def __init__(self) -> None:
        self.uri: Optional[str] = None
        self.connected = False
        self.power_on = False
        self.speed_rate = 0.25
        self.pose = {"x": 0.0, "y": 0.0, "z": 100.0, "a": 0.0, "b": 0.0, "c": 0.0}
        self.coordinate_id = 5
        self.coordinate_frame: dict[str, float] = {}
        self.coordinate_calibration_calls: list[dict[str, Any]] = []
        self.di_state: dict[str, int] = {}
        self.raw_status: dict[str, Any] = {}
        self.pose_available = True
        self.clear_error_calls: list[str] = []

    async def connect(self, uri: str) -> None:
        self.uri = uri
        self.connected = True

    async def close(self) -> None:
        self.connected = False

    async def acquire(self, uri: str) -> None:
        await self.connect(uri)

    async def release(self, uri: str, *, power_off: bool = True) -> dict[str, Any]:
        if power_off:
            self.power_on = False
        self.connected = False
        return {"mode": "released", "released_at": time.time()}

    async def power(self, on: bool) -> None:
        self.power_on = bool(on)

    async def stop(self) -> None:
        return None

    async def set_speed(self, rate: float) -> None:
        self.speed_rate = min(max(float(rate), 0.0), 1.0)

    async def jog(self, *, axis: str, direction: int, hold_s: float, mode: str = "standard") -> None:
        key = axis.lower()
        if mode == "rescue" and key.startswith("j") and key[1:].isdigit():
            self.pose["c"] += float(direction)
            return
        if key in self.pose:
            self.pose[key] += float(direction) * (5.0 if key in {"x", "y", "z"} else 2.0)

    async def move(self, pose: dict[str, float], *, coordinate_id: int, timeout_s: float) -> None:
        self.coordinate_id = int(coordinate_id)
        self.pose.update({key: float(value) for key, value in pose.items() if key in self.pose})

    async def set_coordinate_id(self, coordinate_id: int, *, await_ack: bool = False, timeout_s: float = 5.0) -> None:
        self.coordinate_id = int(coordinate_id)

    async def calibrate_coordinate(
        self,
        points: list[dict[str, float]],
        *,
        coordinate_id: int,
        timeout_s: float,
    ) -> dict[str, Any]:
        if len(points) != 3:
            raise ValueError("Coordinate calibration requires exactly three points.")
        self.coordinate_id = int(coordinate_id)
        origin = points[0]
        self.coordinate_frame = {
            "id": float(coordinate_id),
            "x": float(origin.get("x", 0.0)),
            "y": float(origin.get("y", 0.0)),
            "z": float(origin.get("z", 0.0)),
            "a": float(origin.get("a", 0.0)),
            "b": float(origin.get("b", 0.0)),
            "c": float(origin.get("c", 0.0)),
        }
        self.coordinate_calibration_calls.append(
            {
                "coordinate_id": int(coordinate_id),
                "timeout_s": float(timeout_s),
                "points": [dict(point) for point in points],
                "frame": dict(self.coordinate_frame),
            }
        )
        return dict(self.coordinate_frame)

    async def clear_error(self, error_type: str) -> None:
        self.clear_error_calls.append(error_type)
        self.raw_status = {}
        return None

    def snapshot(self) -> dict[str, Any]:
        return {
            "connected": self.connected,
            "power": self.power_on,
            "speed_rate": self.speed_rate,
            "pose": dict(self.pose) if self.pose_available else None,
            "uri": self.uri,
            "di_state": dict(self.di_state),
            "coordinate_id": self.coordinate_id,
            "coordinate_frame": dict(self.coordinate_frame),
            "raw_status": dict(self.raw_status),
        }


class FakeGripperClient:
    def __init__(self) -> None:
        self.connected = False
        self.ip: Optional[str] = None
        self.gripper_type = "twofg7"
        self.width_mm = 80.0
        self.force = 20.0
        self.speed = 50.0
        self.orientation = "inward"
        self.holding = False
        self.busy = False
        self.fail_next_close = False

    def connect(self, *, ip: str, gripper_type: str, t_index: int = 0) -> dict[str, Any]:
        self.connected = True
        self.ip = ip
        self.gripper_type = "twofg7" if gripper_type == "auto" else gripper_type
        return self.telemetry()

    def disconnect(self) -> None:
        self.connected = False

    def open(self, **kwargs: Any) -> str:
        self.width_mm = float(kwargs.get("width_mm") or 80.0)
        self.holding = False
        return "Gripper opened."

    def close(self, **kwargs: Any) -> str:
        self.width_mm = float(kwargs.get("width_mm") or 20.0)
        self.force = float(kwargs.get("force") or self.force)
        self.speed = float(kwargs.get("speed") or self.speed)
        self.holding = not self.fail_next_close
        self.fail_next_close = False
        return "Gripper closed."

    def set_orientation(self, orientation: str) -> str:
        self.orientation = orientation
        return f"Orientation set to {orientation}."

    def telemetry(self) -> dict[str, Any]:
        return {
            "status": "online" if self.connected else "offline",
            "ip": self.ip,
            "gripper_type": self.gripper_type,
            "display_name": "Fake 2FG7",
            "width_mm": self.width_mm,
            "force": self.force,
            "speed": self.speed,
            "orientation": self.orientation,
            "holding": self.holding,
            "busy": self.busy,
            "capabilities": {"width": True, "force": True, "speed": True, "orientation": True},
        }
