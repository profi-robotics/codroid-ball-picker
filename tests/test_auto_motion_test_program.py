"""Tests for the auto-socket motion smoke test program."""

from __future__ import annotations

import asyncio
import types
from unittest.mock import AsyncMock, patch

import numpy as np

from app.codroid_picker.programs.auto_motion_test import AutoMotionTestProgram
from app.codroid_picker.programs.auto_mode import (
    apply_full_auto_speed,
    controller_is_auto_mode,
    enter_controller_auto_mode,
    pause_auto_program_manual_mode,
    restore_active_auto_program_manual_mode,
    restore_manual_mode,
    resume_auto_program_auto_mode,
)
from app.codroid_picker.programs.auto_pick_place import AutoPickPlaceProgram
from app.codroid_picker.robot_motion import move_auto_socket_cartesian_target
from app.codroid_picker.session import SESSION


class _FakeRobotAPI:
    def __init__(self) -> None:
        self.config = types.SimpleNamespace(host="192.168.101.100")

    def _local_source_ip_for(self, host: str) -> str:
        assert host == "192.168.101.100"
        return "192.168.101.110"

    def build_poscfg(self, *, mode: int) -> dict:
        return {"mode": mode}


class _FakeAutoModeRobotAPI(_FakeRobotAPI):
    def __init__(self) -> None:
        super().__init__()
        self.calls: list[str] = []

    async def power_on(self) -> None:
        self.calls.append("power_on")

    async def set_auto_mode(self) -> None:
        self.calls.append("set_auto_mode")

    async def set_manual_move_rate(self, rate: float) -> None:
        self.calls.append(f"set_manual_move_rate:{rate}")

    async def set_manual_mode(self) -> None:
        self.calls.append("set_manual_mode")

    async def clear_target_position(self) -> None:
        self.calls.append("clear_target_position")

    async def stop_command(self) -> None:
        self.calls.append("stop_command")

    async def stop_auto_socket_project(self) -> None:
        self.calls.append("stop_auto_socket_project")


class _FakeAutoSocketBackend:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def clear_target_position(self) -> None:
        self.calls.append("clear_target_position")

    async def close(self) -> None:
        self.calls.append("close")


class _InspectingAutoModeRobotAPI(_FakeAutoModeRobotAPI):
    async def stop_command(self) -> None:
        detached = getattr(self, "_auto_socket_backend", None) is None
        self.calls.append(f"stop_command:detached={detached}")

    async def set_manual_mode(self) -> None:
        detached = getattr(self, "_auto_socket_backend", None) is None
        self.calls.append(f"set_manual_mode:detached={detached}")


class _FailingCleanupRobotAPI(_FakeAutoModeRobotAPI):
    async def clear_target_position(self) -> None:
        self.calls.append("clear_target_position")
        raise RuntimeError("clear failed")

    async def stop_command(self) -> None:
        self.calls.append("stop_command")
        raise RuntimeError("stop failed")

    async def stop_auto_socket_project(self) -> None:
        self.calls.append("stop_auto_socket_project")
        raise RuntimeError("project stop failed")


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

    def robot_status_snapshot(self):
        return {"CoordinateId": 0}


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


def test_motion_test_uses_coordinate5_vertical_targets() -> None:
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


def test_motion_test_restores_manual_when_post_auto_setup_fails() -> None:
    program = AutoMotionTestProgram()
    robot_api = _FakeAutoModeRobotAPI()
    fake_registry = types.SimpleNamespace(
        settings=types.SimpleNamespace(),
        robot_coordinate_id=5,
        get_or_create_robot_session=lambda: object(),
    )

    with (
        patch(
            "app.codroid_picker.programs.auto_motion_test.registry",
            return_value=fake_registry,
        ),
        patch.object(
            program,
            "_select_coordinate_and_wait_for_posture",
            new=AsyncMock(side_effect=[None, RuntimeError("coordinate stale")]),
        ),
        patch.object(program, "_apply_current_speed", new=AsyncMock(return_value=None)),
        patch.object(
            program,
            "_build_current_pose_targets",
            return_value=([], (0.0, 0.0, 0.0), {}),
        ),
        patch(
            "app.codroid_picker.programs.auto_mode._latest_controller_status",
            return_value={"ControllerMode": "MANUAL"},
        ),
    ):
        result = asyncio.run(program.execute(robot_api))

    assert result.status.name == "ERROR"
    assert "coordinate stale" in result.message
    assert robot_api.calls == [
        "power_on",
        "set_auto_mode",
        "set_manual_move_rate:1.0",
        "clear_target_position",
        "stop_command",
        "stop_auto_socket_project",
        "set_manual_mode",
    ]


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


def test_auto_mode_helpers_apply_full_speed_and_restore_manual() -> None:
    robot_api = _FakeAutoModeRobotAPI()
    robot_api._auto_socket_backend = object()
    active_program = types.SimpleNamespace(
        name="auto_pick_place",
        robot_api=robot_api,
    )

    with patch(
        "app.codroid_picker.programs.auto_mode._latest_controller_status",
        return_value={"ControllerMode": "MANUAL"},
    ):
        asyncio.run(apply_full_auto_speed(robot_api))
        restored = asyncio.run(
            restore_active_auto_program_manual_mode(
                active_program,
                logger=types.SimpleNamespace(
                    debug=lambda *_args: None,
                    warning=lambda *_args: None,
                ),
                context="test",
            )
        )

    assert restored
    assert robot_api._auto_socket_backend is None
    assert SESSION.robot_speed_rate == 1.0
    assert robot_api.calls == [
        "set_manual_move_rate:1.0",
        "clear_target_position",
        "stop_command",
        "stop_auto_socket_project",
        "set_manual_mode",
    ]


def test_restore_manual_detaches_socket_backend_before_controller_commands() -> None:
    robot_api = _InspectingAutoModeRobotAPI()
    backend = _FakeAutoSocketBackend()
    robot_api._auto_socket_backend = backend

    with patch(
        "app.codroid_picker.programs.auto_mode._latest_controller_status",
        return_value={"ControllerMode": "MANUAL"},
    ):
        asyncio.run(
            restore_manual_mode(
                robot_api,
                logger=types.SimpleNamespace(
                    debug=lambda *_args: None,
                    warning=lambda *_args: None,
                ),
                context="test",
            )
        )

    assert robot_api._auto_socket_backend is None
    assert backend.calls == ["clear_target_position", "close"]
    assert robot_api.calls == [
        "clear_target_position",
        "stop_command:detached=True",
        "stop_auto_socket_project",
        "set_manual_mode:detached=True",
    ]


def test_restore_manual_still_sends_manual_after_cleanup_failures() -> None:
    robot_api = _FailingCleanupRobotAPI()

    with patch(
        "app.codroid_picker.programs.auto_mode._latest_controller_status",
        return_value={"ControllerMode": "MANUAL"},
    ):
        asyncio.run(
            restore_manual_mode(
                robot_api,
                logger=types.SimpleNamespace(
                    debug=lambda *_args: None,
                    warning=lambda *_args: None,
                ),
                context="test",
            )
        )

    assert robot_api.calls == [
        "clear_target_position",
        "stop_command",
        "stop_auto_socket_project",
        "set_manual_mode",
    ]


def test_restore_manual_retries_while_controller_reports_auto() -> None:
    robot_api = _FakeAutoModeRobotAPI()

    with patch(
        "app.codroid_picker.programs.auto_mode._latest_controller_status",
        side_effect=[
            {"ControllerMode": "AUTO"},
            {"ControllerMode": "AUTO"},
            {"ControllerMode": "MANUAL"},
        ],
    ):
        asyncio.run(
            restore_manual_mode(
                robot_api,
                logger=types.SimpleNamespace(
                    debug=lambda *_args: None,
                    warning=lambda *_args: None,
                ),
                context="test",
            )
        )

    assert robot_api.calls == [
        "clear_target_position",
        "stop_command",
        "stop_auto_socket_project",
        "set_manual_mode",
        "set_manual_mode",
        "set_manual_mode",
    ]


def test_pause_auto_program_switches_manual_without_closing_socket() -> None:
    robot_api = _FakeAutoModeRobotAPI()
    backend = _FakeAutoSocketBackend()
    robot_api._auto_socket_backend = backend
    active_program = types.SimpleNamespace(
        name="auto_pick_place",
        robot_api=robot_api,
    )

    with patch(
        "app.codroid_picker.programs.auto_mode._latest_controller_status",
        return_value={"ControllerMode": "MANUAL"},
    ):
        paused = asyncio.run(
            pause_auto_program_manual_mode(
                active_program,
                logger=types.SimpleNamespace(
                    debug=lambda *_args: None,
                    warning=lambda *_args: None,
                ),
                context="test",
            )
        )

    assert paused
    assert robot_api._auto_socket_backend is backend
    assert backend.calls == ["clear_target_position"]
    assert robot_api.calls == ["set_manual_mode"]


def test_resume_auto_program_switches_back_to_auto_mode() -> None:
    robot_api = _FakeAutoModeRobotAPI()
    backend = _FakeAutoSocketBackend()
    robot_api._auto_socket_backend = backend
    active_program = types.SimpleNamespace(
        name="auto_pick_place",
        robot_api=robot_api,
    )

    resumed = asyncio.run(
        resume_auto_program_auto_mode(
            active_program,
            logger=types.SimpleNamespace(
                debug=lambda *_args: None,
                warning=lambda *_args: None,
            ),
            context="test",
        )
    )

    assert resumed
    assert robot_api._auto_socket_backend is backend
    assert robot_api.calls == [
        "set_auto_mode",
        "set_manual_move_rate:1.0",
    ]


def test_controller_is_auto_mode_reads_nested_status_values() -> None:
    assert controller_is_auto_mode({"ControllerMode": "AUTO"})
    assert controller_is_auto_mode({"nested": {"mode": {"value": 5}}})
    assert not controller_is_auto_mode({"ControllerMode": "manual"})
    assert not controller_is_auto_mode({"model": "auto-detection"})


def test_enter_controller_auto_mode_forces_full_speed() -> None:
    robot_api = _FakeAutoModeRobotAPI()

    asyncio.run(enter_controller_auto_mode(robot_api))

    assert SESSION.robot_speed_rate == 1.0
    assert robot_api.calls == [
        "set_auto_mode",
        "set_manual_move_rate:1.0",
    ]


def test_auto_socket_move_accepts_base_posture_for_coord5_target() -> None:
    session = _FakeMovingSession()
    session.current_pose = types.SimpleNamespace(
        x=-350.0,
        y=-30.0,
        z=364.0,
        a=0.0,
        b=0.0,
        c=0.0,
    )
    fake_registry = _FakeMoveRegistry(session)
    robot_api = _FakeMoveRobotAPI(session)
    target = {
        "x": 107.0,
        "y": -311.0,
        "z": -194.0,
        "a": 0.0,
        "b": 0.0,
        "c": 0.0,
    }

    async def _move_without_posture_change(_cpos):
        robot_api.calls.append(("move_to_cartesian_target_linear", _cpos))

    robot_api.move_to_cartesian_target_linear = _move_without_posture_change

    with (
        patch("app.codroid_picker.robot_motion.registry", return_value=fake_registry),
        patch(
            "app.codroid_picker.transforms._base_to_coord5",
            return_value=np.array([107.0, -311.0, -194.0]),
        ),
    ):
        asyncio.run(
            move_auto_socket_cartesian_target(
                robot_api,
                target,
                label="test_coord5_move",
                coordinate_id=5,
                timeout_s=0.1,
                settle_s=0.0,
            )
        )

    assert ("set_current_coordinate_id", 5) in robot_api.calls


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
