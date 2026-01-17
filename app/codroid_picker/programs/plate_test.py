"""Plate test program - pick & place test using calibration plate targets.

This program picks objects from calibration plate targets and drops them
at the defined drop position. It supports both segment tracking and
continuous linear move tracking modes.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import numpy as np

if TYPE_CHECKING:
    from codroid_api import CodroidAPI

from .base import ProgramResult, ProgramStatus, RobotProgram

LOGGER = logging.getLogger(__name__)


@dataclass
class PlateTestConfig:
    """Configuration for plate test execution."""

    tracking_mode: str = "move"  # "move" or "segment"
    segment_min_mm: float = 5.0
    segment_max_mm: float = 50.0
    segment_fraction: float = 0.5
    approach_height_mm: float = -30.0
    pick_height_mm: float = 0.0
    drop_height_mm: float = 0.0
    grip_timeout_s: float = 2.0
    use_base_coordinates: bool = False


class PlateTestProgram(RobotProgram):
    """Plate pick & place test program.

    This program:
    1. Moves to drop approach position and opens gripper
    2. For each calibration target on the plate:
       a. Tracks the plate and moves to target approach
       b. Descends to pick height
       c. Closes gripper and checks for grip
       d. If gripped: lifts, moves to drop, releases
       e. Returns to drop approach for next target
    3. Reports success/failure counts
    """

    def __init__(self, config: Optional[PlateTestConfig] = None) -> None:
        super().__init__()
        self._config = config or PlateTestConfig()
        self._capture: Dict[str, Any] = {}
        self._summary: Dict[str, Any] = {}

    @property
    def name(self) -> str:
        return "plate_test"

    @property
    def label(self) -> str:
        return "Plate Pick & Place Test"

    @property
    def config(self) -> PlateTestConfig:
        return self._config

    def check_prerequisites(self) -> Tuple[bool, str]:
        """Check all prerequisites for plate test."""
        import app.codroid_picker as ui

        SESSION = ui.SESSION
        GRIPPER_DEFAULTS = ui.GRIPPER_DEFAULTS

        if SESSION.workspace_definition is None:
            return False, "Workspace is not defined; compute it first."

        if SESSION.drop_target_mm is None:
            return False, "Teach a drop position first."

        if SESSION.robot_calibration_min_z is None:
            return False, "Robot calibration is missing; run robot calibration first."

        if SESSION.gripper is None:
            return False, "Gripper session is not active; enable the gripper."

        if not SESSION.gripper.isConnected(t_index=GRIPPER_DEFAULTS["t_index"]):
            return False, "Gripper is not connected; re-enable the gripper."

        frame, _depth = ui.get_latest_frame()
        if frame is None:
            return False, "Camera stream is not running."

        session = ui._get_robot_session()
        if not session.posture_seen():
            return False, "Robot posture stream not available yet."

        return True, "All prerequisites met."

    async def execute(self, robot_api: "CodroidAPI") -> ProgramResult:
        """Execute the plate pick & place test."""
        import app.codroid_picker as ui

        SESSION = ui.SESSION
        SETTINGS = ui.SETTINGS
        ROBOT_COORDINATE_ID = ui.ROBOT_COORDINATE_ID
        DEFAULT_PLATE_PATH = ui.DEFAULT_PLATE_PATH
        GRIPPER_FORCE_MIN = ui.GRIPPER_FORCE_MIN
        GRIPPER_SPEED_MIN = ui.GRIPPER_SPEED_MIN

        # Initialize state
        self._set_plate_test_state("Plate test starting...", running=True)
        self.update_message("Plate test starting...")

        # Get configuration
        approach_mm = self._config.approach_height_mm
        pick_mm = self._config.pick_height_mm
        drop_mm = self._config.drop_height_mm
        grip_timeout = max(self._config.grip_timeout_s, 0.5)
        segment_min = float(np.clip(self._config.segment_min_mm, 1.0, 200.0))
        segment_max = float(np.clip(self._config.segment_max_mm, 5.0, 500.0))
        segment_fraction = float(
            np.clip(self._config.segment_fraction, 0.1, 1.0))
        if segment_max < segment_min:
            segment_min, segment_max = segment_max, segment_min

        use_segment_tracking = self._config.tracking_mode.lower() == "segment"
        use_base_coordinates = self._config.use_base_coordinates

        workspace_def = SESSION.workspace_definition
        session = ui._get_robot_session()
        plate_test_coord_id = 0 if use_base_coordinates else ROBOT_COORDINATE_ID

        # Clamp approach heights
        if use_base_coordinates:
            if approach_mm < drop_mm:
                LOGGER.info(
                    "Clamping plate-test approach from %.3f to %.3f (base coords).",
                    approach_mm,
                    drop_mm,
                )
                approach_mm = drop_mm
        else:
            if approach_mm > drop_mm:
                LOGGER.info(
                    "Clamping plate-test approach from %.3f to %.3f (coord5).",
                    approach_mm,
                    drop_mm,
                )
                approach_mm = drop_mm

        # Resolve orientation
        if (
            SESSION.drop_target_orientation is not None
            and (
                not use_base_coordinates
                or SESSION.drop_target_coord_id in (None, 0)
            )
        ):
            orientation = {
                "a": float(SESSION.drop_target_orientation[0]),
                "b": float(SESSION.drop_target_orientation[1]),
                "c": float(SESSION.drop_target_orientation[2]),
            }
        else:
            posture = session.position_snapshot()
            orientation = {
                "a": float(posture.a),
                "b": float(posture.b),
                "c": float(posture.c),
            }

        # Resolve plate targets
        try:
            (
                targets,
                plate_meta,
                baseline_transform,
                reference_initialized,
            ) = ui._resolve_plate_test_coord5_targets(DEFAULT_PLATE_PATH)
            target_sequence_order = [int(target["sequence"])
                                     for target in targets]
            targets_by_sequence: Dict[int, dict] = {
                int(target["sequence"]): dict(target) for target in targets
            }
        except Exception as exc:
            message = f"Plate detection failed: {exc}"
            self._set_plate_test_state(message, running=False)
            return ProgramResult(status=ProgramStatus.ERROR, message=message)

        if not targets:
            message = "No calibration targets found on the plate."
            self._set_plate_test_state(message, running=False)
            return ProgramResult(status=ProgramStatus.ERROR, message=message)

        # Build poses
        drop_target_mm = SESSION.drop_target_mm
        poscfg = robot_api.build_poscfg(
            mode=int(SESSION.robot_calibration_points[0].get("mode", 0))
        )
        drop_orientation = orientation.copy()
        plate_orientation = orientation.copy()

        def _update_plate_alignment() -> float:
            plate_rotation_rad = np.arctan2(
                baseline_transform.rotation[1, 0],
                baseline_transform.rotation[0, 0],
            )
            plate_rotation_deg = np.degrees(plate_rotation_rad)
            offset_rotation_deg = float(
                plate_meta.get("offset_rotation_deg", 0.0))
            total = float(plate_rotation_deg + offset_rotation_deg)
            plate_orientation["c"] = float(orientation["c"] + total)
            LOGGER.info("Plate rotation angle: %.2f degrees",
                        plate_rotation_deg)
            return total

        plate_rotation_total = _update_plate_alignment()

        def _base_to_coord(base_xyz: np.ndarray) -> np.ndarray:
            inv_rot = baseline_transform.rotation.T
            inv_trans = -inv_rot @ baseline_transform.translation
            return inv_rot @ base_xyz + inv_trans

        def _coord_to_base(coord_xyz: np.ndarray) -> np.ndarray:
            return (
                baseline_transform.rotation @ coord_xyz
                + baseline_transform.translation
            )

        def _coord5_pose_from_base(base_pose: dict) -> dict:
            base_xyz = np.array(
                [base_pose["x"], base_pose["y"], base_pose["z"]],
                dtype=np.float64,
            )
            coord_xyz = _base_to_coord(base_xyz)
            coord_pose = dict(base_pose)
            coord_pose["x"] = float(coord_xyz[0])
            coord_pose["y"] = float(coord_xyz[1])
            coord_pose["z"] = float(coord_xyz[2])
            return coord_pose

        def _base_pose_from_coord5(coord_pose: dict) -> dict:
            coord_xyz = np.array(
                [coord_pose["x"], coord_pose["y"], coord_pose["z"]],
                dtype=np.float64,
            )
            base_xyz = _coord_to_base(coord_xyz)
            base_pose = dict(coord_pose)
            base_pose["x"] = float(base_xyz[0])
            base_pose["y"] = float(base_xyz[1])
            base_pose["z"] = float(base_xyz[2])
            return base_pose

        # Build drop poses
        if (
            SESSION.drop_target_robot is not None
            and SESSION.drop_target_coord_id == ROBOT_COORDINATE_ID
        ):
            drop_z = ui._clamp_z_for_coord5(
                float(SESSION.drop_target_robot[2]) + drop_mm,
                use_base_coords=use_base_coordinates,
            )
            approach_z = ui._clamp_z_for_coord5(
                float(SESSION.drop_target_robot[2]) + approach_mm,
                use_base_coords=use_base_coordinates,
            )
            drop_pose_coord5 = {
                "x": float(SESSION.drop_target_robot[0]),
                "y": float(SESSION.drop_target_robot[1]),
                "z": drop_z,
                "poscfg": poscfg,
            }
            drop_pose_coord5.update(drop_orientation)
            drop_approach_coord5 = {
                "x": float(SESSION.drop_target_robot[0]),
                "y": float(SESSION.drop_target_robot[1]),
                "z": approach_z,
                "poscfg": poscfg,
            }
            drop_approach_coord5.update(drop_orientation)
            drop_pose_base = _base_pose_from_coord5(drop_pose_coord5)
            drop_approach_base = _base_pose_from_coord5(drop_approach_coord5)
        else:
            if SESSION.drop_target_robot is not None:
                drop_pose_base = {
                    "x": float(SESSION.drop_target_robot[0]),
                    "y": float(SESSION.drop_target_robot[1]),
                    "z": float(SESSION.drop_target_robot[2]) + drop_mm,
                    "poscfg": poscfg,
                }
                drop_pose_base.update(drop_orientation)
                drop_approach_base = {
                    "x": float(SESSION.drop_target_robot[0]),
                    "y": float(SESSION.drop_target_robot[1]),
                    "z": float(SESSION.drop_target_robot[2]) + approach_mm,
                    "poscfg": poscfg,
                }
                drop_approach_base.update(drop_orientation)
            else:
                drop_pose_base = ui._pose_from_workspace_mm(
                    workspace_def,
                    drop_target_mm[0],
                    drop_target_mm[1],
                    drop_mm,
                )
                drop_pose_base["poscfg"] = poscfg
                drop_pose_base.update(drop_orientation)
                drop_approach_base = ui._pose_from_workspace_mm(
                    workspace_def,
                    drop_target_mm[0],
                    drop_target_mm[1],
                    approach_mm,
                )
                drop_approach_base["poscfg"] = poscfg
                drop_approach_base.update(drop_orientation)
            drop_pose_coord5 = _coord5_pose_from_base(drop_pose_base)
            drop_approach_coord5 = _coord5_pose_from_base(drop_approach_base)
            drop_pose_coord5["z"] = ui._clamp_z_for_coord5(
                drop_pose_coord5["z"], use_base_coords=use_base_coordinates
            )
            drop_approach_coord5["z"] = ui._clamp_z_for_coord5(
                drop_approach_coord5["z"], use_base_coords=use_base_coordinates
            )

        drop_pose = drop_pose_base if use_base_coordinates else drop_pose_coord5
        drop_approach = drop_approach_base if use_base_coordinates else drop_approach_coord5

        gripper_min_width, gripper_max_width = ui._resolve_gripper_limits()
        self._summary = {"attempts": 0, "successes": 0, "details": []}

        self._capture = {
            "timestamp": time.time(),
            "coordinate_id": plate_test_coord_id,
            "approach_height_mm": float(approach_mm),
            "pick_height_mm": float(pick_mm),
            "drop_height_mm": float(drop_mm),
            "grip_timeout_s": float(grip_timeout),
            "targets": [dict(target) for target in targets],
            "plate_meta": plate_meta,
            "drop_target": {
                "workspace_mm": drop_target_mm,
                "coord5_pose": drop_pose_coord5,
                "coord5_approach": drop_approach_coord5,
            },
            "moves": [],
            "grip_events": [],
        }

        # Run the pick sequence
        try:
            await self._run_pick_sequence(
                robot_api=robot_api,
                session=session,
                targets_by_sequence=targets_by_sequence,
                target_sequence_order=target_sequence_order,
                plate_test_coord_id=plate_test_coord_id,
                use_base_coordinates=use_base_coordinates,
                use_segment_tracking=use_segment_tracking,
                approach_mm=approach_mm,
                pick_mm=pick_mm,
                drop_pose=drop_pose,
                drop_approach=drop_approach,
                plate_orientation=plate_orientation,
                poscfg=poscfg,
                gripper_min_width=gripper_min_width,
                gripper_max_width=gripper_max_width,
                grip_timeout=grip_timeout,
                segment_min=segment_min,
                segment_max=segment_max,
                segment_fraction=segment_fraction,
                baseline_transform=baseline_transform,
                plate_meta=plate_meta,
                _coord_to_base=_coord_to_base,
            )
        except Exception as exc:
            LOGGER.exception("Plate pick & place test failed")
            message = f"Plate test failed: {exc}"
            self._set_plate_test_state(message, running=False)
            return ProgramResult(status=ProgramStatus.ERROR, message=message)

        # Build result
        attempts = self._summary["attempts"]
        successes = self._summary["successes"]
        detail = ", ".join(self._summary["details"])
        ui.SESSION.last_action = f"Plate test complete: {successes}/{attempts} picked."
        self._capture["summary"] = self._summary
        ui._save_plate_test_capture(self._capture)
        analysis = ui._analyze_plate_test_capture(self._capture)

        plate_note = (
            f"Plate markers: {plate_meta['count']} · "
            f"Reproj: {plate_meta['reprojection_error_mm']:.2f} mm"
        )
        if plate_meta.get("reference_initialized"):
            plate_note = f"{plate_note} · Plate reference set"
        message = f"Plate test complete: {successes}/{attempts} picked. {plate_note}"
        if detail:
            message = f"{message} ({detail})"

        self._set_plate_test_state(message, running=False)
        self.finish(status="idle", message=message)

        return ProgramResult(
            status=ProgramStatus.COMPLETED,
            message=message,
            data={"analysis": analysis, "summary": self._summary},
        )

    async def _run_pick_sequence(
        self,
        robot_api: "CodroidAPI",
        session: Any,
        targets_by_sequence: Dict[int, dict],
        target_sequence_order: List[int],
        plate_test_coord_id: int,
        use_base_coordinates: bool,
        use_segment_tracking: bool,
        approach_mm: float,
        pick_mm: float,
        drop_pose: dict,
        drop_approach: dict,
        plate_orientation: dict,
        poscfg: dict,
        gripper_min_width: float,
        gripper_max_width: float,
        grip_timeout: float,
        segment_min: float,
        segment_max: float,
        segment_fraction: float,
        baseline_transform: Any,
        plate_meta: dict,
        _coord_to_base: Any,
    ) -> None:
        """Execute the main pick sequence loop."""
        import app.codroid_picker as ui

        SETTINGS = ui.SETTINGS

        await self.wait_for_resume()
        await robot_api.power_on()
        await robot_api.set_current_coordinate_id(plate_test_coord_id)
        speed_rate = max(float(ui.SESSION.robot_speed_rate), 0.1)
        await robot_api.set_manual_move_rate(speed_rate)
        self._capture["robot_speed_rate"] = speed_rate

        def _travel_z(z_a: float, z_b: float) -> float:
            if use_base_coordinates:
                return float(max(z_a, z_b))
            return float(min(z_a, z_b))

        def _plate_pose_for_target(
            target: dict,
            z_value: float,
            *,
            use_travel_z: bool,
        ) -> dict:
            coord_xy = target.get("adjusted_xy", target["reference_xy"])
            coord = np.array(
                [coord_xy[0], coord_xy[1], z_value], dtype=np.float64)
            pose_coord5 = {
                "x": float(coord[0]),
                "y": float(coord[1]),
                "z": ui._clamp_z_for_coord5(float(coord[2]), use_base_coords=use_base_coordinates),
                "poscfg": poscfg,
            }
            pose_coord5.update(plate_orientation)
            base = _coord_to_base(coord)
            pose_base = {
                "x": float(base[0]),
                "y": float(base[1]),
                "z": float(base[2]),
                "poscfg": poscfg,
            }
            pose_base.update(plate_orientation)
            pose = pose_base if use_base_coordinates else pose_coord5
            if use_travel_z:
                travel = _travel_z(float(drop_approach["z"]), float(pose["z"]))
                pose = dict(pose)
                pose["z"] = travel
            return pose

        # Initial moves: go to drop approach, open gripper
        await self._move_linear_with_hold(
            robot_api, drop_approach, plate_test_coord_id, label="drop_approach_start"
        )
        await self._move_linear_with_hold(
            robot_api, drop_pose, plate_test_coord_id, label="return_to_drop_pose"
        )
        ui._command_gripper(gripper_max_width, "opened")
        await self._move_linear_with_hold(
            robot_api, drop_approach, plate_test_coord_id, label="drop_approach_ready"
        )

        # Pick each target
        for sequence in target_sequence_order:
            await self.wait_for_resume()
            target = targets_by_sequence.get(sequence)
            if target is None:
                raise RuntimeError(
                    f"Plate test target {sequence} missing from pose data.")

            coord_xy = target.get("adjusted_xy", target["reference_xy"])
            plate_mm = target.get("plate_mm")

            LOGGER.info(
                "Target %d: plate_mm=(%.1f, %.1f), adjusted_xy=(%.1f, %.1f)",
                sequence,
                plate_mm[0] if plate_mm else 0,
                plate_mm[1] if plate_mm else 0,
                coord_xy[0],
                coord_xy[1],
            )

            # Build pick poses
            approach_pose = _plate_pose_for_target(
                target, approach_mm, use_travel_z=False)
            pick_pose = _plate_pose_for_target(
                target, pick_mm, use_travel_z=False)
            travel_z = _travel_z(
                float(drop_approach["z"]), float(approach_pose["z"]))

            cycle_started_at = time.monotonic()

            # Move to target (either segment tracking or linear follow)
            transit_pose = _plate_pose_for_target(
                target, approach_mm, use_travel_z=True)
            await self._move_linear_with_hold(
                robot_api, transit_pose, plate_test_coord_id,
                label=f"target_{sequence}_transit_to_pick"
            )
            await self._move_linear_with_hold(
                robot_api, approach_pose, plate_test_coord_id,
                label=f"target_{sequence}_approach"
            )
            await self._move_linear_with_hold(
                robot_api, pick_pose, plate_test_coord_id,
                label=f"target_{sequence}_pick"
            )

            # Grip
            await self.wait_for_resume()
            ui._command_gripper(
                gripper_min_width,
                "closed",
                force=ui.GRIPPER_FORCE_MIN,
                speed=ui.GRIPPER_SPEED_MIN,
            )
            await self.wait_for_resume()
            gripped = await ui._await_grip_status(grip_timeout)
            self._capture["grip_events"].append({
                "sequence": int(sequence),
                "gripped": bool(gripped),
                "timeout_s": float(grip_timeout),
            })
            self._summary["attempts"] += 1
            if plate_mm is not None:
                ui.SESSION.last_pick_mm = (plate_mm[0], plate_mm[1])

            if gripped:
                self._summary["successes"] += 1
                ui.SESSION.robot_has_object = True

                # Lift, transit, drop
                lift_pose = dict(pick_pose)
                lift_pose["z"] = travel_z
                await self._move_linear_with_hold(
                    robot_api, lift_pose, plate_test_coord_id,
                    label=f"target_{sequence}_lift"
                )
                drop_travel = dict(drop_approach)
                drop_travel["z"] = travel_z
                await self._move_linear_with_hold(
                    robot_api, drop_travel, plate_test_coord_id,
                    label=f"target_{sequence}_transit_to_drop"
                )
                await self._move_linear_with_hold(
                    robot_api, drop_approach, plate_test_coord_id,
                    label=f"target_{sequence}_drop_approach"
                )
                await self._move_linear_with_hold(
                    robot_api, drop_pose, plate_test_coord_id,
                    label=f"target_{sequence}_drop_pose"
                )
                await self.wait_for_resume()
                ui._command_gripper(gripper_max_width, "opened")
                ui.SESSION.robot_has_object = False
                await self._move_linear_with_hold(
                    robot_api, drop_approach, plate_test_coord_id,
                    label=f"target_{sequence}_drop_retreat"
                )
                self._summary["details"].append(f"{sequence}: dropped")
                ui._record_pick_cycle(True, cycle_started_at)
            else:
                # Empty grip - return to drop approach
                await self.wait_for_resume()
                ui._command_gripper(gripper_max_width, "opened")
                lift_pose = dict(pick_pose)
                lift_pose["z"] = travel_z
                await self._move_linear_with_hold(
                    robot_api, lift_pose, plate_test_coord_id,
                    label=f"target_{sequence}_empty_lift"
                )
                drop_travel = dict(drop_approach)
                drop_travel["z"] = travel_z
                await self._move_linear_with_hold(
                    robot_api, drop_travel, plate_test_coord_id,
                    label=f"target_{sequence}_empty_transit_to_drop"
                )
                await self._move_linear_with_hold(
                    robot_api, drop_approach, plate_test_coord_id,
                    label=f"target_{sequence}_return_to_drop"
                )
                self._summary["details"].append(f"{sequence}: empty")
                ui._record_pick_cycle(False, cycle_started_at)

        # Final cleanup
        await self.wait_for_resume()
        ui._command_gripper(gripper_max_width, "opened")
        await robot_api.power_off()

    async def _move_linear_with_hold(
        self,
        robot_api: "CodroidAPI",
        target_pose: dict,
        coordinate_id: int,
        *,
        label: str = "move",
        hold_s: float = 0.0,
    ) -> None:
        """Execute a linear move with optional hold."""
        import app.codroid_picker as ui

        SETTINGS = ui.SETTINGS
        session = ui._get_robot_session()

        await self.wait_for_resume()
        self._update_step(label)
        ui._log_move_target(label, target_pose, coordinate_id=coordinate_id)

        await robot_api.set_target_pos_type(robot_api.config.target_pos_types.cpos)
        await robot_api.set_target_cpos(
            robot_api.attach_coordinate_to_cpos(
                robot_api.build_target_cpos(**target_pose),
                coordinate_id=coordinate_id,
            )
        )
        await robot_api.set_robot_command(robot_api.config.commands.move_target_linear)

        # Wait for motion to settle
        deadline = time.monotonic() + SETTINGS.plate_test_move_timeout_s
        settle_s = SETTINGS.plate_test_settle_s
        motion_eps = SETTINGS.plate_test_motion_eps_mm
        last_xyz = self._get_posture_xyz(session, target_pose, coordinate_id)
        settle_since: Optional[float] = None

        while time.monotonic() < deadline:
            self.ensure_active()
            await robot_api.send_command_heartbeat()
            await asyncio.sleep(0.1)

            curr_xyz = self._get_posture_xyz(
                session, target_pose, coordinate_id)
            target_xyz = np.array(
                [target_pose["x"], target_pose["y"], target_pose["z"]],
                dtype=np.float64,
            )

            if (
                abs(curr_xyz[0] - target_xyz[0]
                    ) <= SETTINGS.plate_test_tolerance_mm
                and abs(curr_xyz[1] - target_xyz[1]) <= SETTINGS.plate_test_tolerance_mm
                and abs(curr_xyz[2] - target_xyz[2]) <= SETTINGS.plate_test_z_tolerance_mm
            ):
                if settle_s <= 0:
                    break
                if settle_since is None:
                    settle_since = time.monotonic()
                elif time.monotonic() - settle_since >= settle_s:
                    break
            else:
                settle_since = None

            last_xyz = curr_xyz

        # Hold if requested
        if hold_s > 0:
            deadline = time.monotonic() + hold_s
            while time.monotonic() < deadline:
                self.ensure_active()
                await robot_api.send_command_heartbeat()
                await asyncio.sleep(0.1)

        await robot_api.clear_target_position()
        await robot_api.stop_command()
        self._record_move(label, target_pose, coordinate_id, session)

    def _get_posture_xyz(self, session: Any, target: dict, coord_id: int) -> np.ndarray:
        """Get current position in the appropriate coordinate system."""
        import app.codroid_picker as ui

        posture = session.position_snapshot()
        ROBOT_COORDINATE_ID = ui.ROBOT_COORDINATE_ID

        if coord_id == ROBOT_COORDINATE_ID:
            return np.array([posture.x, posture.y, posture.z], dtype=np.float64)
        # For base coordinates, return raw posture
        return np.array([posture.x, posture.y, posture.z], dtype=np.float64)

    def _record_move(
        self,
        label: str,
        command_pose: dict,
        coord_id: int,
        session: Any,
    ) -> None:
        """Record a move for analysis."""
        posture = session.position_snapshot()
        self._capture["moves"].append({
            "label": label,
            "command_coord5": {
                "x": float(command_pose["x"]),
                "y": float(command_pose["y"]),
                "z": float(command_pose["z"]),
                "a": float(command_pose.get("a", 0.0)),
                "b": float(command_pose.get("b", 0.0)),
                "c": float(command_pose.get("c", 0.0)),
            },
            "posture_coord5": {
                "x": float(posture.x),
                "y": float(posture.y),
                "z": float(posture.z),
                "a": float(posture.a),
                "b": float(posture.b),
                "c": float(posture.c),
            },
            "posture_coordinate_id": coord_id,
        })

    def _update_step(self, label: str) -> None:
        """Update progress message for current step."""
        import app.codroid_picker as ui

        speed_percent = int(round(ui.SESSION.robot_speed_rate * 100))
        message = f"Plate test: {label.replace('_', ' ')} (speed {speed_percent}%)"
        self._set_plate_test_state(message, running=True, label=label)
        self.update_message(message, label=label)

    def _set_plate_test_state(
        self,
        message: str,
        *,
        running: bool = True,
        label: Optional[str] = None,
    ) -> None:
        """Update the plate test state in SESSION."""
        import app.codroid_picker as ui

        ui._set_plate_test_state(message, running=running, label=label)
