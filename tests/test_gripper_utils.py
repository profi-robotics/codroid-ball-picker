"""Tests for gripper command behavior."""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from app.codroid_picker import gripper_utils


class _FakeGripper:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def isConnected(self, t_index: int) -> bool:  # noqa: N802
        return t_index == 0

    def grip(self, **kwargs) -> None:
        self.calls.append(kwargs)

    def isBusy(self, t_index: int) -> bool:  # noqa: N802
        return False


class GripperUtilsTests(unittest.TestCase):
    def test_command_gripper_does_not_wait_for_grip_detection(self) -> None:
        gripper = _FakeGripper()
        session = SimpleNamespace(gripper=gripper, last_action=None)
        fake_registry = SimpleNamespace(
            session=session,
            gripper_defaults={"t_index": 0},
        )

        with (
            patch(
                "app.codroid_picker.gripper_utils.registry",
                return_value=fake_registry,
            ),
            patch(
                "app.codroid_picker.gripper_utils._resolve_gripper_limits",
                return_value=(0.0, 100.0),
            ),
        ):
            result = gripper_utils._command_gripper(30.0, "opened")

        self.assertIn("Gripper opened to 30.0 mm", result)
        self.assertEqual(gripper.calls[0]["f_wait"], False)


if __name__ == "__main__":
    unittest.main()
