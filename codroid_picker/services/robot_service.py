"""Robot service wrapping real or fake robot adapters."""

from __future__ import annotations

import asyncio
import contextlib
import time
from typing import Any, Optional

from codroid_picker.config import AppSettings
from codroid_picker.errors import HardwareUnavailableError
from codroid_picker.core.robot_errors import detect_robot_error
from codroid_picker.hardware.fakes import FakeRobotClient
from codroid_picker.hardware.interfaces import RobotClient
from codroid_picker.state.store import StateStore


class RobotService:
    def __init__(self, settings: AppSettings, state: StateStore, client: Optional[RobotClient] = None) -> None:
        self.settings = settings
        self.state = state
        self._client: RobotClient = client or self._make_client()
        self._lock = asyncio.Lock()
        self._poll_task: asyncio.Task[None] | None = None

    async def connect(self, uri: Optional[str] = None) -> str:
        uri = uri or self.settings.default_robot_uri
        async with self._lock:
            await self._client.connect(uri)
            self._publish(status="online", message=f"Robot connected to {uri}.")
            return f"Robot connected to {uri}."

    async def acquire(self, uri: Optional[str] = None) -> str:
        uri = uri or self.settings.default_robot_uri
        async with self._lock:
            await self._client.acquire(uri)
            self._publish(status="online", control_mode="acquired", message="Robot control acquired.")
            return "Robot control acquired."

    async def release(self, uri: Optional[str] = None) -> str:
        uri = uri or self.settings.default_robot_uri
        async with self._lock:
            await self._client.release(uri, power_off=True)
            self._publish(status="offline", control_mode="released", power=False, message="Robot control released.")
            return "Robot control released."

    async def power(self, on: bool) -> str:
        async with self._lock:
            await self._client.power(on)
            self._publish(power=on, message=f"Robot power {'on' if on else 'off'}.")
            return f"Robot power {'on' if on else 'off'}."

    async def stop(self) -> str:
        async with self._lock:
            await self._client.stop()
            self.state.request_stop("Robot stop command sent.")
            self._publish(message="Robot stop command sent.")
            return "Robot stop command sent."

    async def speed(self, rate: float) -> str:
        rate = min(max(float(rate), 0.0), 1.0)
        async with self._lock:
            if not self._client.snapshot().get("connected"):
                raise HardwareUnavailableError("Robot is not connected.")
            await self._client.set_speed(rate)
            self._publish(speed_rate=rate, message=f"Robot speed set to {rate:.0%}.")
            return f"Robot speed set to {rate:.0%}."

    async def jog(self, *, axis: str, direction: int, hold_s: float = 0.25, mode: str = "standard") -> str:
        async with self._lock:
            await self._client.jog(axis=axis, direction=int(direction), hold_s=hold_s, mode=mode)
            self._publish(message=f"Jogged {axis} {direction}.")
            return f"Jogged {axis}."

    async def move(self, pose: dict[str, float], *, coordinate_id: int, timeout_s: float) -> str:
        async with self._lock:
            await self._client.move(pose, coordinate_id=coordinate_id, timeout_s=timeout_s)
            self._publish(message="Robot move command completed.")
            return "Robot move command completed."

    async def set_coordinate_id(self, coordinate_id: int, *, await_ack: bool = False, timeout_s: float = 5.0) -> str:
        async with self._lock:
            await self._client.set_coordinate_id(coordinate_id, await_ack=await_ack, timeout_s=timeout_s)
            self._publish(coordinate_id=int(coordinate_id), message=f"Robot coordinate ID set to {coordinate_id}.")
            return f"Robot coordinate ID set to {coordinate_id}."

    async def calibrate_coordinate(
        self,
        points: list[dict[str, float]],
        *,
        coordinate_id: int,
        timeout_s: float,
    ) -> dict[str, Any]:
        async with self._lock:
            frame = await self._client.calibrate_coordinate(
                points,
                coordinate_id=coordinate_id,
                timeout_s=timeout_s,
            )
            self._publish(
                coordinate_id=int(coordinate_id),
                coordinate_frame=frame,
                message=f"Robot coordinate ID {coordinate_id} calibrated.",
            )
            return frame

    async def shutdown(self) -> None:
        await self.stop_polling()
        await self._client.close()

    def poll(self) -> None:
        self._publish()

    def start_polling(self) -> None:
        if self._poll_task is None or self._poll_task.done():
            self._poll_task = asyncio.create_task(self._poll_loop())

    async def stop_polling(self) -> None:
        if self._poll_task is None:
            return
        self._poll_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await self._poll_task
        self._poll_task = None

    async def ack_error(self, error_type: str) -> str:
        async with self._lock:
            await self._client.clear_error(error_type)
            self._publish(active_error=None, error_message=None, message=f"Acknowledged {error_type}.")
            return "Robot error acknowledged."

    async def _poll_loop(self) -> None:
        interval_s = max(float(self.settings.robot_poll_interval_ms) / 1000.0, 0.1)
        while True:
            await asyncio.sleep(interval_s)
            async with self._lock:
                self._publish()

    def _make_client(self) -> RobotClient:
        if self.settings.use_fake_hardware:
            return FakeRobotClient()
        try:
            from codroid_picker.hardware.robot_codroid import CodroidRobotClient

            return CodroidRobotClient()
        except Exception:
            return FakeRobotClient()

    def _publish(self, **overrides: Any) -> None:
        snap = self._client.snapshot()
        active_error, error_message, rescue_mode = detect_robot_error(dict(snap.get("raw_status") or snap))
        pose = snap.get("pose")
        now = time.time()
        robot_state = self.state.domain("robot")
        fallback_status = str(overrides.pop("status", "offline"))
        coordinate_id_override = overrides.pop("coordinate_id", None)
        coordinate_frame_override = overrides.pop("coordinate_frame", None)
        coordinate_id = snap.get("coordinate_id")
        if coordinate_id is None:
            coordinate_id = coordinate_id_override if coordinate_id_override is not None else robot_state.get("coordinate_id", self.settings.robot_coordinate_id)
        coordinate_frame = snap.get("coordinate_frame") or coordinate_frame_override or robot_state.get("coordinate_frame", {})
        speed_rate = snap.get("speed_rate")
        if speed_rate is None:
            speed_rate = robot_state.get("speed_rate", 0.25)
        connected = bool(snap.get("connected"))
        last_seen = now if connected and pose is not None else robot_state.get("last_seen_at")
        stale = pose is None
        if isinstance(last_seen, (int, float)) and now - float(last_seen) > float(self.settings.robot_posture_max_age_s):
            stale = True
        update = {
            "status": "online" if snap.get("connected") else fallback_status,
            "uri": snap.get("uri") or self.settings.default_robot_uri,
            "power": snap.get("power"),
            "speed_rate": float(speed_rate),
            "coordinate_id": int(coordinate_id),
            "coordinate_frame": dict(coordinate_frame),
            "posture": pose,
            "stale": stale,
            "last_seen_at": last_seen,
            "poll_age_s": None if last_seen is None else max(now - float(last_seen), 0.0),
            "di_state": {str(k): int(v) for k, v in (snap.get("di_state") or {}).items()},
            "active_error": active_error,
            "error_message": error_message,
            "rescue_mode": rescue_mode,
        }
        update.update(overrides)
        self.state.update_domain("robot", **update)
