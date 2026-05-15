"""codroid-api robot adapter."""

from __future__ import annotations

from typing import Any, Optional, cast

CLEAR_ERROR_METHODS: dict[str, tuple[str, ...]] = {
    "emergency": ("clear_emergency_stop", "clear_robot_error", "reset_error", "clear_alarm"),
    "emergency_stop": ("clear_emergency_stop", "clear_robot_error", "reset_error", "clear_alarm"),
    "overspeed": ("clear_overspeed", "clear_robot_error", "reset_error", "clear_alarm"),
    "joint_protection": ("clear_joint_protection", "clear_robot_error", "reset_error", "clear_alarm"),
    "wrong_tool": ("clear_tool_error", "clear_wrong_tool", "clear_robot_error", "reset_error", "clear_alarm"),
    "drag": ("clear_drag_not_allowed", "clear_robot_error", "reset_error", "clear_alarm"),
    "drag_not_allowed": ("clear_drag_not_allowed", "clear_robot_error", "reset_error", "clear_alarm"),
}


class CodroidRobotClient:
    """Thin adapter over ``codroid-api`` RobotSession."""

    def __init__(self) -> None:
        from codroid_api import RobotSession  # type: ignore[import-untyped]

        self._session = RobotSession()
        self._uri: Optional[str] = None

    async def connect(self, uri: str) -> None:
        self._uri = uri
        await self._session.connect(uri)

    async def close(self) -> None:
        await self._session.close()

    async def acquire(self, uri: str) -> None:
        self._uri = uri
        await self._session.acquire_control(uri, force_reconnect=False)

    async def release(self, uri: str, *, power_off: bool = True) -> dict[str, Any]:
        self._uri = uri
        result = await self._session.release_control(uri, power_off=power_off)
        return dict(result)

    async def power(self, on: bool) -> None:
        async def _apply(api: Any) -> None:
            if on:
                await api.power_on()
            else:
                await api.power_off()

        await self._session.run_with_robot(self._uri or "", _apply)

    async def stop(self) -> None:
        async def _stop(api: Any) -> None:
            await api.stop_command()

        await self._session.run_with_robot(self._uri or "", _stop)

    async def set_speed(self, rate: float) -> None:
        async def _speed(api: Any) -> None:
            await api.set_speed_multiplier(rate)

        await self._session.run_with_robot(self._uri or "", _speed)

    async def jog(self, *, axis: str, direction: int, hold_s: float, mode: str = "standard") -> None:
        key = axis.lower()
        axis_map = {"x": 1, "y": 2, "z": 3, "a": 4, "b": 5, "c": 6}
        axis_id: int | None
        if mode == "rescue" and key.startswith("j") and key[1:].isdigit():
            axis_id = int(key[1:])
        else:
            axis_id = axis_map.get(key)
        if axis_id is None:
            raise ValueError(f"Unsupported jog axis: {axis}")

        async def _jog(api: Any) -> None:
            jog_modes = getattr(api.config, "jog_modes", None)
            jog_mode = getattr(jog_modes, "joint", None) if mode == "rescue" else getattr(jog_modes, "tcp", None)
            if jog_mode is None:
                jog_mode = 0 if mode == "rescue" else 1
            await api.hold_jog(jog_mode, axis_id, int(direction), hold_seconds=hold_s)

        await self._session.run_with_robot(self._uri or "", _jog)

    async def move(self, pose: dict[str, float], *, coordinate_id: int, timeout_s: float) -> None:
        async def _move(api: Any) -> None:
            target = api.build_target_cpos(
                x=float(pose["x"]),
                y=float(pose["y"]),
                z=float(pose["z"]),
                a=float(pose.get("a", 0.0)),
                b=float(pose.get("b", 0.0)),
                c=float(pose.get("c", 0.0)),
            )
            await api.move_to_cartesian_target_linear(
                api.attach_coordinate_to_cpos(target, coordinate_id=int(coordinate_id))
            )

        await self._session.run_with_robot(self._uri or "", _move)

    async def set_coordinate_id(self, coordinate_id: int, *, await_ack: bool = False, timeout_s: float = 5.0) -> None:
        async def _set(api: Any) -> None:
            await api.set_current_coordinate_id(
                int(coordinate_id),
                await_ack=await_ack,
                timeout=timeout_s,
            )

        await self._session.run_with_robot(self._uri or "", _set)

    async def calibrate_coordinate(
        self,
        points: list[dict[str, float]],
        *,
        coordinate_id: int,
        timeout_s: float,
    ) -> dict[str, Any]:
        async def _calibrate(api: Any) -> dict[str, Any]:
            cpos_points = [
                api.build_target_cpos(
                    x=float(point["x"]),
                    y=float(point["y"]),
                    z=float(point["z"]),
                    a=float(point.get("a", 0.0)),
                    b=float(point.get("b", 0.0)),
                    c=float(point.get("c", 0.0)),
                )
                for point in points
            ]
            frame = await api.coordinate_calibration(
                cpos_points,
                coordinate_id=int(coordinate_id),
                timeout=timeout_s,
                set_active_coordinate=True,
            )
            if not frame:
                raise RuntimeError("Robot controller returned an empty coordinate calibration frame.")
            frame = self._normalize_coordinate_frame(frame)
            await api.change_coordinate_parameter(int(coordinate_id), frame, timeout=timeout_s)
            await api.set_current_coordinate_id(int(coordinate_id))
            return dict(frame)

        return cast(dict[str, Any], await self._session.run_with_robot(self._uri or "", _calibrate))

    def snapshot(self) -> dict[str, Any]:
        pose = self._session.position_snapshot()
        status = self._session.robot_status_snapshot()
        return {
            "connected": self._session.is_connected(self._uri or ""),
            "power": self._session.robot_power_on(),
            "speed_rate": None,
            "pose": {
                "x": pose.x,
                "y": pose.y,
                "z": pose.z,
                "a": pose.a,
                "b": pose.b,
                "c": pose.c,
            } if self._session.posture_seen() else None,
            "uri": self._uri,
            "di_state": self._session.flange_button_states(),
            "coordinate_id": self._coordinate_id_from_status(status),
            "coordinate_frame": self._session.coordinate_frame_snapshot(),
            "raw_status": status,
        }

    async def clear_error(self, error_type: str) -> None:
        async def _clear(api: Any) -> None:
            method_names = CLEAR_ERROR_METHODS.get(error_type, ("clear_robot_error", "reset_error", "clear_alarm"))
            for name in method_names:
                method = getattr(api, name, None)
                if callable(method):
                    await method()
                    return
            raise AttributeError(f"Robot API does not expose a clear method for {error_type}.")

        await self._session.run_with_robot(self._uri or "", _clear)

    @staticmethod
    def _coordinate_id_from_status(status: dict[str, Any]) -> Optional[int]:
        coord_id = status.get("CoordinateId")
        if coord_id is None:
            return None
        try:
            return int(coord_id)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _normalize_coordinate_frame(frame: dict[str, Any]) -> dict[str, Any]:
        normalized = dict(frame)

        def _axis(*keys: str) -> float:
            for key in keys:
                value = normalized.get(key)
                if isinstance(value, (int, float)):
                    return float(value)
                if isinstance(value, str):
                    try:
                        return float(value)
                    except ValueError:
                        continue
            return 0.0

        normalized["x"] = _axis("x")
        normalized["y"] = _axis("y")
        normalized["z"] = _axis("z")
        normalized["a"] = _axis("a", "rx", "roll")
        normalized["b"] = _axis("b", "ry", "pitch")
        normalized["c"] = _axis("c", "rz", "yaw")
        return normalized
