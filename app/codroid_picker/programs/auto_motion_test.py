"""Auto-mode robot motion smoke test program.

This program exercises auto-socket motion without using the camera, detector,
drop target, or gripper. It sends one small vertical move away from the current
robot pose and then returns to the starting pose.
"""

from __future__ import annotations

import asyncio
import logging
import time
from contextlib import suppress
from dataclasses import dataclass
from typing import TYPE_CHECKING, Tuple

import numpy as np

if TYPE_CHECKING:
    from codroid_api import CodroidAPI

from app.codroid_picker.app_instance import registry
from app.codroid_picker.programs.base import (
    ProgramResult,
    ProgramStatus,
    RobotProgram,
)
from app.codroid_picker.robot_motion import (
    move_auto_socket_cartesian_target,
    select_coordinate_and_wait_for_posture,
)

LOGGER = logging.getLogger(__name__)

MOTION_TEST_DELTA_Z_MM = 35.0
MOTION_TEST_REACH_TOLERANCE_MM = 3.0
MOTION_TEST_REACH_TIMEOUT_S = 10.0
MOTION_TEST_COORDINATE_SETTLE_S = 2.0
MOTION_TEST_COORDINATE_TIMEOUT_S = 5.0


@dataclass(frozen=True)
class MotionTarget:
    """One streamed target in the continuous motion test."""

    label: str
    x: float
    y: float
    z: float
    c_offset: float = 0.0
    duration_s: float = 0.25
    log: bool = False


class AutoMotionTestProgram(RobotProgram):
    """Finite auto-mode motion verification program.

    The path is intentionally small: current pose, current pose with a modest
    positive Z offset, then current pose again. This validates the socket
    project and controller auto-mode path without sweeping through the work
    area or rotating the wrist.
    """

    @property
    def name(self) -> str:
        return "auto_motion_test"

    @property
    def label(self) -> str:
        return "Auto Motion Test"

    def check_prerequisites(self) -> Tuple[bool, str]:
        """Check whether the continuous motion test can start."""
        from app.codroid_picker.session import SESSION

        session = registry().get_or_create_robot_session()
        if not session.posture_seen():
            return False, "Robot posture stream not available yet."

        settings = registry().settings
        posture = session.position_snapshot()
        age = posture.age_seconds()
        if age > settings.robot_posture_max_age_s:
            return False, f"Robot posture stale ({age:.1f}s old)."

        return True, "Ready for auto motion test."

    async def execute(self, robot_api: "CodroidAPI") -> ProgramResult:
        """Run continuous calibrated motion until the program is stopped."""
        from app.codroid_picker.session import SESSION

        settings = registry().settings
        session = registry().get_or_create_robot_session()
        coordinate_id = registry().robot_coordinate_id

        self.update_message("Auto motion test: powering on robot.")
        await robot_api.power_on()

        self.update_message(
            f"Auto motion test: selecting coordinate system {coordinate_id}."
        )
        await self._select_coordinate_and_wait_for_posture(
            robot_api,
            session,
            coordinate_id=coordinate_id,
        )
        loop_targets, orientation, poscfg = self._build_current_pose_targets(
            robot_api,
            session,
            coordinate_id=coordinate_id,
        )

        try:
            self.update_message("Auto motion test: switching controller to Auto mode.")
            await robot_api.set_auto_mode()
            await self._apply_auto_speed(robot_api)
            await self._select_coordinate_and_wait_for_posture(
                robot_api,
                session,
                coordinate_id=coordinate_id,
            )
            auto_socket_config = self._build_auto_socket_config(settings, robot_api)
            if not hasattr(robot_api, "auto_socket_mode"):
                raise RuntimeError(
                    "Installed codroid-api does not provide auto_socket_mode. "
                    "Install a codroid-api version with auto-socket support."
                )
            with suppress(Exception):
                await robot_api.stop_project()
                await asyncio.sleep(0.3)
            auto_socket_context = robot_api.auto_socket_mode(auto_socket_config)
            backend = await auto_socket_context.__aenter__()
            await self._select_coordinate_and_wait_for_posture(
                robot_api,
                session,
                coordinate_id=coordinate_id,
            )
            project_info = getattr(backend, "project_info", None)
            LOGGER.info(
                "Auto motion test: auto-socket active project=%s task=%s "
                "run_label=%s socket=%s:%s",
                getattr(project_info, "project_id", None),
                getattr(project_info, "task_id", None),
                getattr(project_info, "run_label_id", None),
                getattr(project_info, "socket_host", None),
                getattr(project_info, "socket_port", None),
            )
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("Auto motion test: failed to activate auto-socket motion")
            message = f"Auto socket motion setup failed: {exc}"
            with suppress(Exception):
                await robot_api.stop_auto_socket_project()
            with suppress(Exception):
                await robot_api.set_manual_mode()
            return ProgramResult(status=ProgramStatus.ERROR, message=message)

        moves_completed = 0
        started_at = time.monotonic()
        self.update_message("Auto motion test running in auto-socket mode.")

        exit_exc = (None, None, None)
        try:
            await self.wait_for_resume()
            SESSION.last_action = "Auto motion test running."
            for target in loop_targets:
                moves_completed += 1
                self.update_message(
                    f"Auto motion test: move {moves_completed}/{len(loop_targets)}."
                )
                await self._stream_target(
                    robot_api,
                    target,
                    orientation=orientation,
                    poscfg=poscfg,
                    coordinate_id=coordinate_id,
                )
        except BaseException as exc:
            exit_exc = (type(exc), exc, exc.__traceback__)
            raise
        finally:
            with suppress(Exception):
                await robot_api.clear_target_position()
            with suppress(Exception):
                await robot_api.stop_command()
            with suppress(Exception):
                await auto_socket_context.__aexit__(*exit_exc)
            with suppress(Exception):
                await robot_api.set_manual_mode()
            LOGGER.info(
                "Auto motion test stopped after %d moves and %.1fs.",
                moves_completed,
                time.monotonic() - started_at,
            )

        return ProgramResult(
            status=ProgramStatus.COMPLETED,
            message="Auto motion test completed.",
            data={"moves_completed": moves_completed},
        )

    @staticmethod
    def _build_auto_socket_config(settings: object, robot_api: "CodroidAPI"):
        """Build the codroid-api auto-socket config for the motion test."""
        try:
            from codroid_api import AutoSocketConfig
        except ImportError as exc:
            from types import SimpleNamespace as AutoSocketConfig

            LOGGER.warning(
                "Installed codroid-api does not export AutoSocketConfig; "
                "using a compatible config namespace: %s",
                exc,
            )

        port = int(getattr(settings, "auto_socket_port", 19093) or 19093)
        return AutoSocketConfig(
            socket_role="server",
            socket_bind_host="0.0.0.0",
            socket_port=port,
            project_socket_host=_resolve_local_socket_host(robot_api),
            project_socket_port=port,
            install_project=True,
            start_project=True,
            stop_project_on_exit=True,
            connect_on_enter=True,
            accept_timeout_s=25.0,
            read_timeout_ms=10000,
            move_speed="V100",
            move_acc="ACC100",
        )

    @staticmethod
    def _build_current_pose_targets(
        robot_api: "CodroidAPI",
        session: object,
        *,
        coordinate_id: int,
    ) -> tuple[list[MotionTarget], Tuple[float, float, float], object]:
        """Build a tiny vertical loop from the current reported pose."""
        posture = session.position_snapshot()
        z_delta = (
            -MOTION_TEST_DELTA_Z_MM
            if int(coordinate_id) != 0
            else MOTION_TEST_DELTA_Z_MM
        )
        base = MotionTarget(
            label="socket_test_base",
            x=float(posture.x),
            y=float(posture.y),
            z=float(posture.z),
            duration_s=MOTION_TEST_REACH_TIMEOUT_S,
            log=True,
        )
        raised = MotionTarget(
            label="socket_test_z_plus",
            x=base.x,
            y=base.y,
            z=base.z + z_delta,
            duration_s=MOTION_TEST_REACH_TIMEOUT_S,
            log=True,
        )
        orientation = (
            float(posture.a),
            float(posture.b),
            float(posture.c),
        )
        poscfg = robot_api.build_poscfg(mode=int(getattr(posture, "mode", 0) or 0))
        return [raised, base], orientation, poscfg

    @staticmethod
    async def _select_coordinate_and_wait_for_posture(
        robot_api: "CodroidAPI",
        session: object,
        *,
        coordinate_id: int,
    ) -> None:
        """Select the motion coordinate and wait for a fresh pose in that frame."""
        await select_coordinate_and_wait_for_posture(
            robot_api,
            session,
            coordinate_id=coordinate_id,
            timeout_s=MOTION_TEST_COORDINATE_TIMEOUT_S,
            settle_s=MOTION_TEST_COORDINATE_SETTLE_S,
        )

    async def _apply_current_speed(self, robot_api: "CodroidAPI") -> None:
        """Apply the currently selected robot speed rate before streaming."""
        from app.codroid_picker.session import SESSION

        speed_rate = float(np.clip(SESSION.robot_speed_rate, 0.05, 1.0))
        await robot_api.set_manual_move_rate(speed_rate)
        SESSION.robot_speed_rate = speed_rate
        SESSION.robot_speed_last_set = time.time()

    async def _apply_auto_speed(self, robot_api: "CodroidAPI") -> None:
        """Apply full robot speed for auto-socket motion."""
        from app.codroid_picker.session import SESSION

        await robot_api.set_manual_move_rate(1.0)
        SESSION.robot_speed_rate = 1.0
        SESSION.robot_speed_last_set = time.time()

    async def _stream_target(
        self,
        robot_api: "CodroidAPI",
        target: MotionTarget,
        *,
        orientation: Tuple[float, float, float],
        poscfg: object,
        coordinate_id: int,
    ) -> None:
        """Stream one target through the shared auto-socket movement helper."""
        await self.wait_for_resume()
        pose = self._build_target_pose(target, orientation, poscfg)
        await move_auto_socket_cartesian_target(
            robot_api,
            pose,
            label=target.label,
            coordinate_id=coordinate_id,
            timeout_s=target.duration_s,
            tolerance_mm=MOTION_TEST_REACH_TOLERANCE_MM,
            z_tolerance_mm=MOTION_TEST_REACH_TOLERANCE_MM,
            active_check=self.ensure_active,
        )

    @staticmethod
    def _build_target_pose(
        target: MotionTarget,
        orientation: Tuple[float, float, float],
        poscfg: object,
    ) -> dict:
        """Convert a motion target to a robot target pose dict."""
        return {
            "x": float(target.x),
            "y": float(target.y),
            "z": float(target.z),
            "a": float(orientation[0]),
            "b": float(orientation[1]),
            "c": float(orientation[2] + target.c_offset),
            "poscfg": poscfg,
        }


def _resolve_local_socket_host(robot_api: "CodroidAPI") -> str | None:
    """Resolve the app host IP the robot controller should connect back to."""
    config = getattr(robot_api, "config", None)
    robot_host = getattr(config, "host", None)
    resolver = getattr(robot_api, "_local_source_ip_for", None)
    if callable(resolver) and robot_host:
        try:
            return str(resolver(str(robot_host)))
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("Auto motion test: failed to resolve local socket host: %s", exc)
    return None
