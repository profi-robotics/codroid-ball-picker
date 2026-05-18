"""Tests for the auto-socket motion smoke test program."""

from __future__ import annotations

import types

from app.codroid_picker.programs.auto_motion_test import AutoMotionTestProgram


class _FakeRobotAPI:
    def __init__(self) -> None:
        self.config = types.SimpleNamespace(host="192.168.101.100")

    def _local_source_ip_for(self, host: str) -> str:
        assert host == "192.168.101.100"
        return "192.168.101.110"

    def build_poscfg(self, *, mode: int) -> dict:
        return {"mode": mode}


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
