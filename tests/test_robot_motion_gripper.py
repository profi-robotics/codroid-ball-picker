"""Tests for robot-motion gripper command behavior."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

from app.codroid_picker import robot_motion


class _FakeGripper:
    def __init__(self, *, min_width: float = 0.0, max_width: float = 25.0) -> None:
        self.min_width = min_width
        self.max_width = max_width
        self.calls: list[dict] = []
        self.busy_polls = 0

    def isConnected(self, t_index: int) -> bool:  # noqa: N802
        return t_index == 0

    def get_status_snapshot(self, t_index: int) -> None:
        return None

    def get_min_ext_width(self, t_index: int) -> float:
        return self.min_width

    def get_max_ext_width(self, t_index: int) -> float:
        return self.max_width

    def grip(self, **kwargs) -> int:
        self.calls.append(kwargs)
        return 0

    def isBusy(self, t_index: int) -> bool:  # noqa: N802
        self.busy_polls += 1
        return False


def _fake_registry(gripper: _FakeGripper) -> SimpleNamespace:
    profile = SimpleNamespace(
        close_width_default=5.0,
        open_width_default=25.0,
        force_default=20,
        force_min=20,
        force_max=140,
    )
    session = SimpleNamespace(
        gripper=gripper,
        gripper_min_width=None,
        gripper_max_width=None,
        last_action=None,
    )
    return SimpleNamespace(
        gripper=gripper,
        gripper_defaults={"t_index": 0},
        settings=SimpleNamespace(gripper_profile=profile),
        session=session,
    )


def test_resolve_gripper_close_width_uses_device_minimum() -> None:
    gripper = _FakeGripper(min_width=0.0, max_width=25.0)
    fake_registry = _fake_registry(gripper)

    with (
        patch("app.codroid_picker.robot_motion.registry", return_value=fake_registry),
        patch("app.codroid_picker.robot_helpers.registry", return_value=fake_registry),
    ):
        assert robot_motion.resolve_gripper_close_width() == 0.0


def test_command_gripper_waits_for_motion_not_grip_detection() -> None:
    gripper = _FakeGripper(min_width=0.0, max_width=25.0)
    fake_registry = _fake_registry(gripper)

    with (
        patch("app.codroid_picker.robot_motion.registry", return_value=fake_registry),
        patch("app.codroid_picker.robot_helpers.registry", return_value=fake_registry),
    ):
        result = robot_motion.command_gripper(0.0, "closed", force=40)

    assert "Gripper closed to 0.0 mm" in result
    assert gripper.calls[0]["f_wait"] is False
    assert gripper.busy_polls > 0
