from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest import mock

from app.codroid_picker.robot_callbacks.connection import (
    _control_state_payload,
    _release_robot_ownership,
)
from app.codroid_picker.robot_utils import _robot_control_released


class RobotControlReleaseTests(unittest.IsolatedAsyncioTestCase):
    def test_control_state_payload_normalizes_mode(self) -> None:
        released = _control_state_payload(
            mode="released",
            message="released",
            released_at=123.4,
        )
        self.assertEqual(released["mode"], "released")
        self.assertEqual(released["released_at"], 123.4)
        self.assertEqual(released["message"], "released")

        acquired = _control_state_payload(mode="other", released_at=999.0)
        self.assertEqual(acquired["mode"], "acquired")
        self.assertEqual(acquired["released_at"], 0.0)

    def test_robot_control_released_uses_state_payload(self) -> None:
        self.assertTrue(_robot_control_released({"mode": "released"}))
        self.assertFalse(_robot_control_released({"mode": "acquired"}))
        self.assertFalse(_robot_control_released({"mode": "invalid"}))

    async def test_release_robot_ownership_stops_power_off_and_closes(self) -> None:
        calls: list[str] = []

        class _FakeSession:
            async def release_control(self, uri: str, power_off: bool, wait_closed_s: float) -> dict:
                calls.append(f"release_control:{uri}:{power_off}:{wait_closed_s}")
                return {"mode": "released", "released_at": 123.0, "last_error": None}

        class _FakeModule:
            def _get_robot_session(self) -> _FakeSession:
                return _FakeSession()

        fake_registry = SimpleNamespace(active_program=object(), stop=lambda: calls.append("registry_stop"))

        with mock.patch("app.codroid_picker.programs.get_registry", return_value=fake_registry):
            result = await _release_robot_ownership(_FakeModule(), "ws://robot/")

        self.assertEqual(
            calls,
            ["registry_stop", "release_control:ws://robot/:True:2.0"],
        )
        self.assertEqual(result["mode"], "released")

    async def test_release_robot_ownership_closes_even_when_api_fails(self) -> None:
        calls: list[str] = []

        class _FakeSession:
            async def release_control(self, _uri: str, power_off: bool, wait_closed_s: float) -> dict:
                calls.append(f"release_control:{power_off}:{wait_closed_s}")
                raise RuntimeError("release failed")

        class _FakeModule:
            def _get_robot_session(self) -> _FakeSession:
                return _FakeSession()

        fake_registry = SimpleNamespace(active_program=None, stop=lambda: calls.append("registry_stop"))

        with mock.patch("app.codroid_picker.programs.get_registry", return_value=fake_registry):
            with self.assertRaisesRegex(RuntimeError, "release failed"):
                await _release_robot_ownership(_FakeModule(), "ws://robot/")

        self.assertEqual(calls, ["release_control:True:2.0"])


if __name__ == "__main__":
    unittest.main()
