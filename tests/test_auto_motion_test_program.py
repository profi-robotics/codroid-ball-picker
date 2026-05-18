"""Tests for the auto-socket motion smoke test program."""

from __future__ import annotations

import asyncio
import types
from unittest.mock import AsyncMock, patch

from app.codroid_picker.programs.auto_motion_test import AutoMotionTestProgram
from app.codroid_picker.programs.auto_pick_place import AutoPickPlaceProgram
from app.codroid_picker.robot_motion import move_auto_socket_cartesian_target


class _FakeRobotAPI:
    def __init__(self) -> None:
        self.config = types.SimpleNamespace(host="192.168.101.100")

    def _local_source_ip_for(self, host: str) -> str:
        assert host == "192.168.101.100"
        return "192.168.101.110"

    def build_poscfg(self, *, mode: int) -> dict:
        return {"mode": mode}


class _FakeMoveRobotAPI:
    def __init__(self, session: "_FakeMovingSession") -> None:
        self.session = session
        self.calls: list[tuple[str, object]] = []
        self.config = types.SimpleNamespace(host="192.168.101.100")

    def build_target_cpos(self, **target_pose):
        self.calls.append(("build_target_cpos", target_pose))
        return dict(target_pose)

    def attach_coordinate_to_cpos(self, cpos: dict, *, coordinate_id: int) -> dict:
        self.calls.append(("attach_coordinate_to_cpos", coordinate_id))
        tagged = dict(cpos)
        tagged["coord"] = {"datavar": {"value": coordinate_id}}
        return tagged

    async def set_current_coordinate_id(self, coordinate_id: int) -> None:
        self.calls.append(("set_current_coordinate_id", coordinate_id))

    async def move_to_cartesian_target_linear(self, cpos: dict) -> None:
        self.calls.append(("move_to_cartesian_target_linear", cpos))
        self.session.current_pose = types.SimpleNamespace(
            x=cpos["x"],
            y=cpos["y"],
            z=cpos["z"],
            a=cpos["a"],
            b=cpos["b"],
            c=cpos["c"],
        )


class _FakeSession:
    def position_snapshot(self):
        return types.SimpleNamespace(
            x=-240.0,
            y=64.0,
            z=418.0,
            a=-180.0,
            b=-1.3,
            c=3.3,
            mode=0,
        )


class _FakeMovingSession:
    def __init__(self) -> None:
        self.current_pose = types.SimpleNamespace(
            x=0.0,
            y=0.0,
            z=100.0,
            a=0.0,
            b=0.0,
            c=0.0,
        )

    def posture_seen(self) -> bool:
        return True

    def position_snapshot(self):
        return self.current_pose


class _FakeMoveRegistry:
    def __init__(self, session: _FakeMovingSession) -> None:
        self.settings = types.SimpleNamespace(
            auto_move_timeout_s=1.0,
            auto_tolerance_mm=2.0,
            auto_z_tolerance_mm=3.0,
            auto_settle_s=0.0,
        )
        self.robot_coordinate_id = 5
        self._session = session

    def get_or_create_robot_session(self):
        return self._session


def test_motion_test_uses_current_pose_vertical_targets() -> None:
    targets, orientation, poscfg = AutoMotionTestProgram._build_current_pose_targets(
        _FakeRobotAPI(),
        _FakeSession(),
        coordinate_id=5,
    )

    assert [target.label for target in targets] == [
        "socket_test_z_plus",
        "socket_test_base",
    ]
    assert targets[0].z < targets[1].z
    assert targets[0].x == targets[1].x
    assert targets[0].y == targets[1].y
    assert orientation == (-180.0, -1.3, 3.3)
    assert poscfg == {"mode": 0}


def test_auto_socket_config_sets_controller_callback_host() -> None:
    config = AutoMotionTestProgram._build_auto_socket_config(
        types.SimpleNamespace(auto_socket_port=19093),
        _FakeRobotAPI(),
    )

    assert config.socket_port == 19093
    assert config.project_socket_host == "192.168.101.110"
    assert config.project_socket_port == 19093
    assert config.accept_timeout_s == 25.0
    assert config.install_project
    assert config.start_project
    assert config.connect_on_enter


def test_auto_socket_move_uses_cartesian_target_api() -> None:
    session = _FakeMovingSession()
    fake_registry = _FakeMoveRegistry(session)
    robot_api = _FakeMoveRobotAPI(session)
    target = {
        "x": 10.0,
        "y": 20.0,
        "z": 90.0,
        "a": 0.0,
        "b": 0.0,
        "c": 0.0,
    }

    with patch("app.codroid_picker.robot_motion.registry", return_value=fake_registry):
        asyncio.run(
            move_auto_socket_cartesian_target(
                robot_api,
                target,
                label="test_move",
                coordinate_id=5,
                timeout_s=1.0,
                settle_s=0.0,
            )
        )

    assert ("attach_coordinate_to_cpos", 5) in robot_api.calls
    assert ("set_current_coordinate_id", 5) in robot_api.calls
    move_call = robot_api.calls[-1]
    assert move_call[0] == "move_to_cartesian_target_linear"
    assert move_call[1]["coord"]["datavar"]["value"] == 5


def test_auto_pick_fixed_move_delegates_to_shared_auto_socket_helper() -> None:
    program = AutoPickPlaceProgram()
    settings = types.SimpleNamespace(
        auto_move_timeout_s=7.0,
        auto_tolerance_mm=2.0,
        auto_z_tolerance_mm=3.0,
        auto_settle_s=0.2,
    )
    fake_registry = types.SimpleNamespace(settings=settings)
    target = {
        "x": 10.0,
        "y": 20.0,
        "z": 30.0,
        "a": 0.0,
        "b": 0.0,
        "c": 0.0,
    }

    with (
        patch(
            "app.codroid_picker.programs.auto_pick_place.registry",
            return_value=fake_registry,
        ),
        patch(
            "app.codroid_picker.programs.auto_pick_place.move_auto_socket_cartesian_target",
            new=AsyncMock(return_value=None),
        ) as move_mock,
    ):
        asyncio.run(
            program._move_linear_with_hold(
                _FakeRobotAPI(),
                target,
                5,
                label="auto_pick_pose",
                hold_s=0.4,
            )
        )

    move_mock.assert_awaited_once()
    _, args, kwargs = move_mock.mock_calls[0]
    assert args[1] == target
    assert kwargs["label"] == "auto_pick_pose"
    assert kwargs["coordinate_id"] == 5
    assert kwargs["timeout_s"] == 7.0
    assert kwargs["hold_s"] == 0.4
