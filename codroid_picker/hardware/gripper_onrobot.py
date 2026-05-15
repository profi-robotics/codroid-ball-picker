"""OnRobot gripper adapter facade."""

from __future__ import annotations

from typing import Any, Optional


class OnRobotGripperClient:
    """Adapter over the existing normalized gripper adapters."""

    def __init__(self) -> None:
        self._adapter: Optional[Any] = None
        self._client: Optional[Any] = None
        self._t_index = 0
        self._ip: Optional[str] = None
        self._key = "auto"

    def connect(self, *, ip: str, gripper_type: str, t_index: int = 0) -> dict[str, Any]:
        from onrobot.device import Device  # type: ignore[import-untyped]
        from onrobot.detection import detect_gripper  # type: ignore[import-untyped]
        from onrobot.sg import SG  # type: ignore[import-untyped]
        from onrobot.twofg import TWOFG  # type: ignore[import-untyped]
        from onrobot.vgc10 import VG  # type: ignore[import-untyped]

        from codroid_picker.core.gripper_adapters import (
            SoftGripperAdapter,
            TwoFG7Adapter,
            VGC10Adapter,
        )

        device = Device(ip)
        self._t_index = int(t_index)
        self._ip = ip
        if gripper_type == "auto":
            detected = detect_gripper(device, t_index=self._t_index, create_client=True)
            self._client = detected.client
            self._key = detected.key
        elif gripper_type == "sg":
            self._client = SG(device)
            self._key = "sg"
        elif gripper_type == "vgc10":
            self._client = VG(device)
            self._key = "vgc10"
        else:
            self._client = TWOFG(device)
            self._key = "twofg7"

        if self._key == "sg":
            self._adapter = SoftGripperAdapter(self._client)
        elif self._key == "vgc10":
            self._adapter = VGC10Adapter(self._client)
        else:
            self._adapter = TwoFG7Adapter(self._client)
        self._adapter.ensure_ready(t_index=self._t_index)
        return self.telemetry()

    def disconnect(self) -> None:
        self._adapter = None
        self._client = None

    def open(self, **kwargs: Any) -> str:
        if self._adapter is None:
            raise RuntimeError("Gripper is not connected.")
        width = kwargs.get("width_mm")
        if width is None:
            _min_width, max_width = self._adapter.width_limits(t_index=self._t_index)
            width = max_width
        result = self._adapter.command(float(width), "open", t_index=self._t_index, speed=kwargs.get("speed"), force=kwargs.get("force"))
        return str(result)

    def close(self, **kwargs: Any) -> str:
        if self._adapter is None:
            raise RuntimeError("Gripper is not connected.")
        width = float(kwargs.get("width_mm") or self._adapter.width_limits(t_index=self._t_index)[0])
        result = self._adapter.command(
            width,
            "close",
            t_index=self._t_index,
            force=kwargs.get("force"),
            speed=kwargs.get("speed"),
            gentle=kwargs.get("gentle"),
            vacuum_setpoint=kwargs.get("vacuum_setpoint"),
            vacuum_threshold=kwargs.get("vacuum_threshold"),
        )
        return str(result)

    def set_orientation(self, orientation: str) -> str:
        if self._client is None:
            raise RuntimeError("Gripper is not connected.")
        setter = getattr(self._client, "set_finger_orientation", None)
        if callable(setter):
            setter(outward=orientation == "outward", t_index=self._t_index)
        return f"Orientation set to {orientation}."

    def telemetry(self) -> dict[str, Any]:
        if self._adapter is None:
            return {"status": "offline", "gripper_type": self._key, "ip": self._ip}
        data = self._adapter.read_telemetry(t_index=self._t_index)
        return {
            "status": "online",
            "ip": self._ip,
            "gripper_type": self._key,
            "display_name": self._adapter.display_name,
            "capabilities": self._adapter.to_state().get("capabilities", {}),
            "width_mm": data.get("width") or data.get("min_width"),
            "force": data.get("force"),
            "busy": data.get("busy"),
            "holding": data.get("gripped"),
            "orientation": data.get("orientation"),
        }
