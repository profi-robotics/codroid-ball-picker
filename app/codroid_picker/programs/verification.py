"""Verification robot program.

This program performs the robot calibration verification sequence,
moving the robot through the three calibration points to confirm
the coordinate system is correctly set up.
"""

from __future__ import annotations

import asyncio
import logging
import time
from contextlib import suppress
from typing import Optional, Tuple, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from codroid_api import CodroidAPI

from app.codroid_picker.programs.base import (
    RobotProgram,
    ProgramStatus,
    ProgramResult,
)
from app.codroid_picker.exceptions import OperationPaused
from app.codroid_picker.app_instance import registry
from app.codroid_picker.robot_motion import (
    ensure_operation_active,
    wait_for_operation_resume,
)

LOGGER = logging.getLogger(__name__)


def _resolve_local_socket_host(robot_api: "CodroidAPI") -> str | None:
    """Resolve the app host IP the robot controller should connect back to."""
    config = getattr(robot_api, "config", None)
    robot_host = getattr(config, "host", None)
    resolver = getattr(robot_api, "_local_source_ip_for", None)
    if callable(resolver) and robot_host:
        try:
            return str(resolver(str(robot_host)))
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("Verification: failed to resolve local socket host: %s", exc)
    return None


class VerificationProgram(RobotProgram):
    """Robot calibration verification program.

    Moves the robot through the three calibration points in a specific
    sequence to verify the coordinate system is correctly calibrated:
    1. Raise to point 3
    2. Move to point 1 (raise, down, raise)
    3. Move to point 2 (raise, down, raise)
    4. Move to point 3 (raise, down, raise)
    5. Return to point 1 (home)
    """

    @property
    def name(self) -> str:
        return "verification"

    @property
    def label(self) -> str:
        return "Verification Move"

    def check_prerequisites(self) -> Tuple[bool, str]:
        """Check if verification can run."""
        from app.codroid_picker.session import SESSION

        # Check for calibration points
        if not SESSION.robot_calibration_points:
            return False, "No calibration points available."
        if len(SESSION.robot_calibration_points) < 3:
            return False, "Verification requires three calibration points."

        # Check robot posture stream
        session = registry().get_or_create_robot_session()
        if not session.posture_seen():
            return False, "Robot posture stream not available yet."

        settings = registry().settings
        posture = session.position_snapshot()
        age = posture.age_seconds()
        if age > settings.robot_posture_max_age_s:
            return False, f"Robot posture stale ({age:.1f}s old)."

        return True, "Ready for verification."

    @staticmethod
    def _build_auto_socket_config(settings: object, robot_api: "CodroidAPI"):
        """Build the codroid-api auto-socket config for verification."""
        from codroid_api import AutoSocketConfig

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

    async def execute(self, robot_api: "CodroidAPI") -> ProgramResult:
        """Execute the verification motion sequence."""
        from app.codroid_picker.session import SESSION
        from vision import plate

        settings = registry().settings
        coordinate_id = registry().robot_coordinate_id

        # Get calibration points
        points = SESSION.robot_calibration_points[:3]
        min_z = SESSION.robot_calibration_min_z
        if min_z is None:
            min_z = min(float(pt.get("z", 0.0)) for pt in points)
            SESSION.robot_calibration_min_z = min_z

        session = registry().get_or_create_robot_session()

        # Power on and set coordinate system
        self.update_message("Verification move: powering on robot.")
        await robot_api.power_on()
        await robot_api.set_manual_mode()
        await robot_api.set_current_coordinate_id(coordinate_id)

        # Try to load plate layout for coordinate system 5 positions
        coord5_positions = None
        use_coord5 = True
        try:
            from app import settings as app_settings

            plate_layout = plate.load_plate_layout(
                app_settings.DEFAULT_PLATE_PATH)
            plate_points = plate_layout.robot_calibration_points()
            if len(plate_points) < 3:
                raise RuntimeError("Plate layout missing calibration points.")

            # Positions relative to Point 1 (origin)
            origin_mm = np.asarray(plate_points[0].center_mm, dtype=np.float64)
            coord5_positions = [
                np.array([
                    float(pt.center_mm[0] - origin_mm[0]),
                    float(pt.center_mm[1] - origin_mm[1]),
                    0.0,
                ], dtype=np.float64)
                for pt in plate_points[:3]
            ]
        except Exception as exc:
            LOGGER.warning(
                "Failed to load plate layout for verification, using base coordinates: %s",
                exc,
            )
            await robot_api.set_current_coordinate_id(0)
            use_coord5 = False

        # Helper to wait for motion and settle
        async def _await_motion_settle(label: str) -> None:
            timeout_s = settings.verification_move_timeout_s
            settle_s = settings.verification_settle_s
            motion_eps_mm = settings.verification_motion_eps_mm

            deadline = time.monotonic() + timeout_s
            start = session.position_snapshot()
            last = start
            moved = False
            settle_deadline: Optional[float] = None

            while time.monotonic() < deadline:
                ensure_operation_active(self._run_id)
                await robot_api.send_command_heartbeat()
                await asyncio.sleep(0.2)

                curr = session.position_snapshot()
                delta = np.linalg.norm([
                    float(curr.x) - float(last.x),
                    float(curr.y) - float(last.y),
                    float(curr.z) - float(last.z),
                ])

                if delta > motion_eps_mm:
                    moved = True
                    settle_deadline = time.monotonic() + settle_s

                if moved and settle_deadline is not None:
                    if delta > motion_eps_mm:
                        settle_deadline = time.monotonic() + settle_s
                    elif time.monotonic() >= settle_deadline:
                        return

                last = curr

            total = np.linalg.norm([
                float(last.x) - float(start.x),
                float(last.y) - float(start.y),
                float(last.z) - float(start.z),
            ])

            if not moved:
                raise RuntimeError(
                    f"Verification move timeout at {label}: no motion detected."
                )
            raise RuntimeError(
                f"Verification move timeout at {label}: motion did not settle "
                f"(total={total:.2f} mm)."
            )

        # Helper to execute a move with hold
        async def _move_with_hold(
            target: dict,
            *,
            label: str,
            hold_seconds: Optional[float] = None,
            coord_id: int,
        ) -> None:
            if hold_seconds is None:
                hold_seconds = settings.verification_hold_seconds

            while True:
                await wait_for_operation_resume(self._run_id)
                self.update_message(
                    f"Verification move: {label.replace('_', ' ')}",
                    label=label,
                )
                target_to_send = target
                if coord_id == coordinate_id and use_coord5:
                    posture = session.position_snapshot()
                    target_to_send = dict(target)
                    target_to_send["a"] = float(posture.a)
                    target_to_send["b"] = float(posture.b)
                    target_to_send["c"] = float(posture.c)
                self.log_target(label, target_to_send, coordinate_id=coord_id)

                await robot_api.set_target_pos_type(robot_api.config.target_pos_types.cpos)
                if coord_id == coordinate_id and use_coord5:
                    await robot_api.set_target_cpos(
                        robot_api.attach_coordinate_to_cpos(
                            target_to_send, coordinate_id=coordinate_id
                        )
                    )
                else:
                    await robot_api.set_target_cpos(target_to_send)

                await robot_api.set_robot_command(robot_api.config.commands.move_target_linear)

                try:
                    await _await_motion_settle(label)
                except OperationPaused:
                    continue

                try:
                    if hold_seconds > 0:
                        deadline = time.monotonic() + hold_seconds
                        while time.monotonic() < deadline:
                            ensure_operation_active(self._run_id)
                            await robot_api.send_command_heartbeat()
                            await asyncio.sleep(0.2)
                except OperationPaused:
                    continue

                await robot_api.clear_target_position()
                await robot_api.stop_command()
                return

        from app.codroid_picker.transforms import _clamp_z_for_coord5

        # Helper to clamp Z for the active coordinate system
        def _safe_z(value: float) -> float:
            z_value = float(value)
            if use_coord5:
                return _clamp_z_for_coord5(z_value, use_base_coords=False)
            return max(z_value, float(min_z))

        # Helper to build target pose
        def _target(
            point: dict,
            z_value: float,
            coord5_pos: Optional[np.ndarray] = None,
        ) -> dict:
            poscfg = robot_api.build_poscfg(mode=int(point.get("mode", 0)))

            if coord5_pos is not None:
                return robot_api.build_target_cpos(
                    x=float(coord5_pos[0]),
                    y=float(coord5_pos[1]),
                    z=_safe_z(z_value),
                    a=float(point.get("a", 0.0)),
                    b=float(point.get("b", 0.0)),
                    c=float(point.get("c", 0.0)),
                    poscfg=poscfg,
                )
            else:
                return robot_api.build_target_cpos(
                    x=float(point["x"]),
                    y=float(point["y"]),
                    z=_safe_z(z_value),
                    a=float(point.get("a", 0.0)),
                    b=float(point.get("b", 0.0)),
                    c=float(point.get("c", 0.0)),
                    poscfg=poscfg,
                )

        # Extract points and compute heights
        p1, p2, p3 = points
        z1 = _safe_z(p1.get("z", 0.0))
        z2 = _safe_z(p2.get("z", 0.0))
        z3 = _safe_z(p3.get("z", 0.0))
        raise_mm = settings.verification_raise_mm

        def _raise_z(z_value: float) -> float:
            if use_coord5:
                return float(z_value) - raise_mm
            return float(z_value) + raise_mm

        raise_1 = _raise_z(z1)
        raise_2 = _raise_z(z2)
        raise_3 = _raise_z(z3)

        # Determine coordinate system
        coord_id = coordinate_id if use_coord5 else 0
        c5p1 = coord5_positions[0] if coord5_positions else None
        c5p2 = coord5_positions[1] if coord5_positions else None
        c5p3 = coord5_positions[2] if coord5_positions else None

        previous_speed_rate = float(SESSION.robot_speed_rate)
        target_speed_rate = 0.6

        async def _set_speed(rate: float) -> None:
            await robot_api.set_manual_move_rate(rate)
            SESSION.robot_speed_rate = rate
            SESSION.robot_speed_last_set = time.time()

        async def _run_sequence(phase: str) -> None:
            """Run the calibrated verification target sequence."""
            self.update_message(f"Verification move: {phase} pass starting.")
            await _move_with_hold(
                _target(p3, raise_3, c5p3),
                label=f"{phase}_p3_raise",
                coord_id=coord_id,
            )
            await _move_with_hold(
                _target(p1, raise_1, c5p1),
                label=f"{phase}_p1_raise",
                coord_id=coord_id,
            )
            await _move_with_hold(
                _target(p1, z1, c5p1),
                label=f"{phase}_p1_down",
                coord_id=coord_id,
            )
            await _move_with_hold(
                _target(p1, raise_1, c5p1),
                label=f"{phase}_p1_raise_return",
                coord_id=coord_id,
            )
            await _move_with_hold(
                _target(p2, raise_2, c5p2),
                label=f"{phase}_p2_raise",
                coord_id=coord_id,
            )
            await _move_with_hold(
                _target(p2, z2, c5p2),
                label=f"{phase}_p2_down",
                coord_id=coord_id,
            )
            await _move_with_hold(
                _target(p2, raise_2, c5p2),
                label=f"{phase}_p2_raise_return",
                coord_id=coord_id,
            )
            await _move_with_hold(
                _target(p3, raise_3, c5p3),
                label=f"{phase}_p3_raise",
                coord_id=coord_id,
            )
            await _move_with_hold(
                _target(p3, z3, c5p3),
                label=f"{phase}_p3_down",
                coord_id=coord_id,
            )
            await _move_with_hold(
                _target(p3, raise_3, c5p3),
                label=f"{phase}_p3_raise_return",
                coord_id=coord_id,
            )
            await _move_with_hold(
                _target(p1, z1, c5p1),
                label=f"{phase}_p1_home",
                hold_seconds=0.0,
                coord_id=coord_id,
            )

        try:
            if abs(previous_speed_rate - target_speed_rate) > 0.001:
                self.update_message("Verification move: setting robot speed to 60%.")
                await _set_speed(target_speed_rate)

            await robot_api.set_manual_mode()
            await robot_api.set_current_coordinate_id(coord_id)
            await _run_sequence("manual")

            self.update_message(
                "Verification move: switching to Auto mode for fast pass."
            )
            await robot_api.set_auto_mode()
            await robot_api.set_current_coordinate_id(coord_id)
            auto_socket_context = robot_api.auto_socket_mode(
                self._build_auto_socket_config(settings, robot_api)
            )
            try:
                await auto_socket_context.__aenter__()
                await robot_api.set_current_coordinate_id(coord_id)
                await _run_sequence("auto")
            finally:
                with suppress(Exception):
                    await auto_socket_context.__aexit__(None, None, None)
                with suppress(Exception):
                    await robot_api.set_manual_mode()

            # Ensure coordinate system is set correctly at end
            if not use_coord5:
                await robot_api.set_current_coordinate_id(coordinate_id)
            else:
                await robot_api.set_current_coordinate_id(coord_id)

            # Mark complete
            self.finish(status="idle", message="Verification complete.")
            return ProgramResult(
                status=ProgramStatus.COMPLETED,
                message="Calibration verified in manual and auto modes.",
            )
        finally:
            with suppress(Exception):
                await robot_api.set_manual_mode()
            if abs(previous_speed_rate - float(SESSION.robot_speed_rate)) > 0.001:
                try:
                    await _set_speed(previous_speed_rate)
                except Exception as exc:  # noqa: BLE001
                    LOGGER.warning("Failed to restore robot speed: %s", exc)

    async def cleanup(self) -> None:
        """Restore gripper width after verification."""
        # Import here to avoid circular deps
        try:
            import app.codroid_picker as codroid_picker_module

            codroid_picker_module._restore_calibration_gripper_width()
        except Exception as exc:
            LOGGER.warning("Failed to restore gripper width: %s", exc)
