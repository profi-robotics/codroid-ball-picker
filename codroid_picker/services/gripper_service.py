"""Gripper service wrapping OnRobot or fake adapters."""

from __future__ import annotations

import anyio
from typing import Any, Optional

from codroid_picker.config import AppSettings
from codroid_picker.hardware.fakes import FakeGripperClient
from codroid_picker.hardware.interfaces import GripperClient
from codroid_picker.state.store import StateStore


class GripperService:
    def __init__(self, settings: AppSettings, state: StateStore, client: Optional[GripperClient] = None) -> None:
        self.settings = settings
        self.state = state
        self._client: GripperClient = client or self._make_client()

    async def connect(self, *, ip: Optional[str] = None, gripper_type: Optional[str] = None, t_index: int = 0) -> str:
        ip = ip or self.settings.gripper_ip or "192.168.101.95"
        gripper_type = gripper_type or self.settings.gripper_type
        telemetry = await anyio.to_thread.run_sync(
            lambda: self._client.connect(ip=ip, gripper_type=gripper_type, t_index=t_index)
        )
        self._publish(telemetry, message=f"{telemetry.get('display_name', 'Gripper')} connected.")
        return "Gripper connected."

    async def disconnect(self) -> str:
        await anyio.to_thread.run_sync(self._client.disconnect)
        self.state.update_domain("gripper", status="offline", message="Gripper disconnected.")
        return "Gripper disconnected."

    async def open(self, **kwargs: Any) -> str:
        message = await anyio.to_thread.run_sync(lambda: self._client.open(**kwargs))
        self._publish(await anyio.to_thread.run_sync(self._client.telemetry), message=message)
        return message

    async def close(self, **kwargs: Any) -> str:
        message = await anyio.to_thread.run_sync(lambda: self._client.close(**kwargs))
        self._publish(await anyio.to_thread.run_sync(self._client.telemetry), message=message)
        return message

    async def orientation(self, orientation: str) -> str:
        message = await anyio.to_thread.run_sync(lambda: self._client.set_orientation(orientation))
        self._publish(await anyio.to_thread.run_sync(self._client.telemetry), message=message)
        return message

    def poll(self) -> None:
        try:
            self._publish(self._client.telemetry())
        except Exception:
            return

    def _make_client(self) -> GripperClient:
        if self.settings.use_fake_hardware:
            return FakeGripperClient()
        try:
            from codroid_picker.hardware.gripper_onrobot import OnRobotGripperClient

            return OnRobotGripperClient()
        except Exception:
            return FakeGripperClient()

    def _publish(self, telemetry: dict[str, Any], **overrides: Any) -> None:
        self.state.update_domain(
            "gripper",
            status=telemetry.get("status", "offline"),
            ip=telemetry.get("ip"),
            gripper_type=telemetry.get("gripper_type", self.settings.gripper_type),
            display_name=telemetry.get("display_name", "Gripper"),
            capabilities=telemetry.get("capabilities", {}),
            width_mm=telemetry.get("width_mm"),
            force=telemetry.get("force"),
            speed=telemetry.get("speed"),
            vacuum=telemetry.get("vacuum"),
            orientation=telemetry.get("orientation"),
            holding=telemetry.get("holding"),
            busy=telemetry.get("busy"),
            **overrides,
        )

