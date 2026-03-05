from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest import mock

from app.codroid_picker import robot_utils


class RobotUtilsControlTests(unittest.IsolatedAsyncioTestCase):
    def test_robot_control_mode_prefers_valid_control_state(self) -> None:
        self.assertEqual(robot_utils._robot_control_mode({"mode": "released"}), "released")
        self.assertEqual(robot_utils._robot_control_mode({"mode": "acquired"}), "acquired")
        self.assertEqual(robot_utils._robot_control_mode({"mode": "invalid"}), "acquired")

    def test_robot_control_mode_falls_back_to_registry_session(self) -> None:
        fake_registry = SimpleNamespace(session=SimpleNamespace(robot_control_mode="released"))
        with mock.patch.object(robot_utils, "registry", return_value=fake_registry):
            self.assertEqual(robot_utils._robot_control_mode(None), "released")

    async def test_probe_robot_uses_strict_mode_by_default(self) -> None:
        calls: list[tuple[str, bool]] = []

        class _FakeRobotSession:
            async def probe(self, uri: str, *, require_acquired: bool = False) -> dict:
                calls.append((uri, require_acquired))
                return {"host": "h", "port": 1}

        fake_registry = SimpleNamespace(
            get_or_create_robot_session=lambda: _FakeRobotSession(),
            default_robot_uri="ws://default/",
        )
        with mock.patch.object(robot_utils, "registry", return_value=fake_registry):
            result = await robot_utils._probe_robot("")

        self.assertEqual(result["host"], "h")
        self.assertEqual(calls, [("ws://default/", True)])

    async def test_with_robot_api_uses_strict_mode_by_default(self) -> None:
        calls: list[tuple[str, bool]] = []

        class _FakeRobotSession:
            async def run_with_robot(self, uri: str, action, *, require_acquired: bool = False):
                calls.append((uri, require_acquired))
                await action(SimpleNamespace())

        fake_registry = SimpleNamespace(
            get_or_create_robot_session=lambda: _FakeRobotSession(),
            default_robot_uri="ws://default/",
        )

        async def _noop(_api) -> None:
            return None

        with mock.patch.object(robot_utils, "registry", return_value=fake_registry):
            await robot_utils._with_robot_api("", _noop)

        self.assertEqual(calls, [("ws://default/", True)])


if __name__ == "__main__":
    unittest.main()
