"""Tests for the app process supervisor."""

from __future__ import annotations

import sys

from run_codroid_picker_supervised import (
    SupervisorConfig,
    _child_command,
    _restart_delay,
)


def test_child_command_preserves_app_arguments() -> None:
    command = _child_command(["--host", "127.0.0.1", "--port", "8051"])

    assert command[0] == sys.executable
    assert command[-4:] == ["--host", "127.0.0.1", "--port", "8051"]
    assert command[1].endswith("run_codroid_picker.py")


def test_restart_delay_enters_cooldown_after_burst() -> None:
    config = SupervisorConfig(
        restart_delay_s=2.0,
        restart_window_s=10.0,
        max_restarts_per_window=3,
        cooldown_s=30.0,
    )
    restart_times: list[float] = []

    assert _restart_delay(restart_times, 100.0, config) == 2.0
    assert _restart_delay(restart_times, 101.0, config) == 2.0
    assert _restart_delay(restart_times, 102.0, config) == 30.0
    assert restart_times == []


def test_restart_delay_prunes_old_exits() -> None:
    config = SupervisorConfig(
        restart_delay_s=2.0,
        restart_window_s=10.0,
        max_restarts_per_window=3,
        cooldown_s=30.0,
    )
    restart_times = [80.0, 91.0]

    assert _restart_delay(restart_times, 101.5, config) == 2.0
    assert restart_times == [101.5]
