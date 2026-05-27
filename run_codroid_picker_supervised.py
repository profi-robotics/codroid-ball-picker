"""Run the Dash calibration UI under a small restart supervisor."""

from __future__ import annotations

from dataclasses import dataclass
import logging
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from typing import Sequence


LOGGER = logging.getLogger("codroid_picker_supervisor")


def _env_float(name: str, default: float) -> float:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    try:
        return float(raw_value)
    except ValueError:
        LOGGER.warning("Ignoring invalid %s=%r; using %.1f", name, raw_value, default)
        return default


def _env_int(name: str, default: int) -> int:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    try:
        return int(raw_value)
    except ValueError:
        LOGGER.warning("Ignoring invalid %s=%r; using %s", name, raw_value, default)
        return default


@dataclass(frozen=True)
class SupervisorConfig:
    """Restart policy for native crashes outside Python exception handling."""

    restart_delay_s: float = 3.0
    restart_window_s: float = 120.0
    max_restarts_per_window: int = 5
    cooldown_s: float = 60.0

    @classmethod
    def from_env(cls) -> "SupervisorConfig":
        return cls(
            restart_delay_s=max(
                0.0,
                _env_float("CODROID_SUPERVISOR_RESTART_DELAY_S", 3.0),
            ),
            restart_window_s=max(
                1.0,
                _env_float("CODROID_SUPERVISOR_RESTART_WINDOW_S", 120.0),
            ),
            max_restarts_per_window=max(
                1,
                _env_int("CODROID_SUPERVISOR_MAX_RESTARTS_PER_WINDOW", 5),
            ),
            cooldown_s=max(
                0.0,
                _env_float("CODROID_SUPERVISOR_COOLDOWN_S", 60.0),
            ),
        )


def _child_command(args: Sequence[str]) -> list[str]:
    script = Path(__file__).with_name("run_codroid_picker.py")
    return [sys.executable, str(script), *args]


def _restart_delay(
    restart_times: list[float],
    now: float,
    config: SupervisorConfig,
) -> float:
    cutoff = now - config.restart_window_s
    restart_times[:] = [timestamp for timestamp in restart_times if timestamp >= cutoff]
    restart_times.append(now)
    if len(restart_times) >= config.max_restarts_per_window:
        restart_times.clear()
        return config.cooldown_s
    return config.restart_delay_s


class AppSupervisor:
    """Restart the app subprocess when native code terminates it unexpectedly."""

    def __init__(self, command: Sequence[str], config: SupervisorConfig) -> None:
        self.command = list(command)
        self.config = config
        self._child: subprocess.Popen[bytes] | None = None
        self._stopping = False

    def run(self) -> int:
        restart_times: list[float] = []
        self._install_signal_handlers()
        while not self._stopping:
            LOGGER.info("Starting app subprocess: %s", " ".join(self.command))
            self._child = subprocess.Popen(self.command)
            return_code = self._child.wait()
            self._child = None

            if self._stopping or return_code == 0:
                return return_code

            delay = _restart_delay(restart_times, time.monotonic(), self.config)
            LOGGER.error(
                "App subprocess exited unexpectedly with code %s; restarting in %.1fs",
                return_code,
                delay,
            )
            time.sleep(delay)
        return 0

    def _install_signal_handlers(self) -> None:
        for signal_number in (signal.SIGINT, signal.SIGTERM):
            signal.signal(signal_number, self._handle_signal)

    def _handle_signal(self, signal_number: int, _frame: object) -> None:
        self._stopping = True
        child = self._child
        if child is None or child.poll() is not None:
            return
        LOGGER.info("Forwarding signal %s to app subprocess", signal_number)
        child.send_signal(signal_number)


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    args = list(sys.argv[1:] if argv is None else argv)
    supervisor = AppSupervisor(_child_command(args), SupervisorConfig.from_env())
    return supervisor.run()


if __name__ == "__main__":
    raise SystemExit(main())
