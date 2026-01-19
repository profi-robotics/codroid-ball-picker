"""Robot calibration program - capture 3 poses via flange button and calibrate coord ID.

This program guides the operator to position the gripper at three calibration
points on the calibration plate, captures the robot poses when the flange
button is pressed, and sends the coordinate calibration request to the robot
controller.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple

if TYPE_CHECKING:
    from codroid_api import CodroidAPI

from app.codroid_picker.procedures import _ensure_robot_coordinate
from .base import ProgramResult, ProgramStatus, RobotProgram

LOGGER = logging.getLogger(__name__)


@dataclass
class RobotCalibrationConfig:
    """Configuration for robot calibration execution."""

    flange_timeout_s: float = 90.0
    coordinate_timeout_s: float = 10.0
    target_coordinate_id: int = 5


@dataclass
class RobotCalibrationResult:
    """Result data from robot calibration."""

    robot_points: List[Dict[str, Any]] = field(default_factory=list)
    coord_frame: Dict[str, Any] = field(default_factory=dict)
    accuracy: Dict[str, float] = field(default_factory=dict)


class RobotCalibrationProgram(RobotProgram):
    """Robot plate calibration program.

    This program:
    1. Powers on robot and selects CoordinateId=0 (base)
    2. For each of 3 calibration points:
       a. Prompts operator to position gripper at the calibration groove
       b. Waits for flange button press
       c. Captures robot pose
    3. Sends coordinate calibration request to robot controller
    4. Updates coordinate system with calibrated frame

    The operator guides the robot manually to each calibration point.
    """

    def __init__(
        self,
        plate_layout: Any,
        config: Optional[RobotCalibrationConfig] = None,
    ) -> None:
        super().__init__()
        self._plate_layout = plate_layout
        self._config = config or RobotCalibrationConfig()
        self._result = RobotCalibrationResult()
        self._baseline_initialized = False

    @property
    def name(self) -> str:
        return "robot_calibration"

    @property
    def label(self) -> str:
        return "Robot Calibration"

    @property
    def config(self) -> RobotCalibrationConfig:
        return self._config

    @property
    def result(self) -> RobotCalibrationResult:
        return self._result

    def check_prerequisites(self) -> Tuple[bool, str]:
        """Check all prerequisites for robot calibration."""
        import app.codroid_picker as ui

        # Check plate layout has calibration points
        calib_points = self._plate_layout.robot_calibration_points()
        if len(calib_points) < 3:
            return False, "Calibration plate layout is missing calibration points."

        # Check plate visibility
        plate_ok, plate_message = ui._plate_is_stable(require_baseline=False)
        if not plate_ok:
            return False, plate_message

        # Check robot posture stream
        posture_ok, posture_message = ui._robot_posture_ready()
        if not posture_ok:
            return False, posture_message

        return True, "All prerequisites met."

    async def execute(self, robot_api: "CodroidAPI") -> ProgramResult:
        """Execute the robot calibration procedure."""
        import app.codroid_picker as ui

        SESSION = ui.SESSION
        SETTINGS = ui.SETTINGS
        ROBOT_COORDINATE_ID = ui.ROBOT_COORDINATE_ID

        session = ui._get_robot_session()
        target_coord_id = self._config.target_coordinate_id

        # Override target coordinate ID from settings if configured
        if hasattr(SETTINGS, "robot_coordinate_id"):
            target_coord_id = getattr(
                SETTINGS, "robot_coordinate_id", target_coord_id)
        if ROBOT_COORDINATE_ID:
            target_coord_id = ROBOT_COORDINATE_ID

        # Initialize
        ui._clear_robot_accuracy_guidance()
        self._set_guidance(
            "Powering on robot and selecting CoordinateId=0 (base) for calibration...",
            status="running",
            step=1,
        )

        # Power on and set base coordinate system
        await robot_api.power_on()
        await self._ensure_coordinate_id(robot_api, session, 0)
        await session.reset_flange_queue()

        # Get calibration points from plate layout
        calib_points = self._plate_layout.robot_calibration_points()
        point_labels = [
            "Point 1 (Origin)",
            "Point 2 (+X)",
            "Point 3 (+X/+Y)",
        ]

        robot_points: List[Dict[str, Any]] = []

        # Capture 3 poses
        for idx in range(3):
            self._ensure_plate_stable(allow_lost=True)
            point_label = point_labels[idx] if idx < len(
                point_labels) else f"Point {idx + 1}"

            self._set_guidance(
                f"Step {idx + 1}/3: Place the gripper fingers in indentation {idx + 1}. "
                "Keep the flange vertical. When aligned, press flange button 1.",
                step=idx + 1,
                status="running",
            )
            LOGGER.info(
                "Step %d/3: Waiting for gripper to align with %s...", idx + 1, point_label)

            try:
                posture = await self._await_flange_press_with_guard(
                    session,
                    timeout_s=self._config.flange_timeout_s,
                )
            except asyncio.TimeoutError:
                self._set_guidance(
                    f"Timed out waiting for flange button at {point_label}.",
                    status="error",
                )
                return ProgramResult(
                    status=ProgramStatus.ERROR,
                    message=f"Timed out waiting for flange button 1 press at {point_label}. "
                    f"Move gripper to the groove target and press the flange button.",
                )

            posture = ui._ensure_posture_axes(
                posture, session.position_snapshot())
            LOGGER.info(
                "Captured %s. Position: X=%.1f, Y=%.1f, Z=%.1f",
                point_label,
                posture.get("x", 0.0),
                posture.get("y", 0.0),
                posture.get("z", 0.0),
            )
            robot_points.append(posture)

            if idx < 2:
                self._set_guidance(
                    f"Captured {point_label}. "
                    f"Next: place the gripper in indentation {idx + 2} and press flange button 1.",
                    step=idx + 1,
                    status="running",
                )

        # Switch to target coordinate system
        self._set_guidance(
            f"Switching to CoordinateId {target_coord_id} for calibration...",
            status="running",
        )
        await self._ensure_coordinate_id(
            robot_api, session, target_coord_id, timeout_s=5.0, require=False
        )

        # Compute and send calibration
        self._set_guidance(
            "Computing coordinate transform from 3 calibration points...",
            status="running",
        )
        LOGGER.info(
            "Computing coordinate transformation from 3 calibration points...")

        coord_frame = await self._send_calibration_request(
            robot_api,
            session,
            robot_points,
            target_coord_id,
            SETTINGS.robot_coordinate_timeout_s,
        )

        self._set_guidance(
            "Coordinate frame updated. Preparing verification step...",
            status="running",
            step=3,
        )
        LOGGER.info("Robot calibration coordinate ID %d updated.",
                    target_coord_id)

        self._result.robot_points = robot_points
        self._result.coord_frame = coord_frame

        return ProgramResult(
            status=ProgramStatus.COMPLETED,
            message="Robot calibration complete.",
            data={
                "robot_points": robot_points,
                "coord_frame": coord_frame,
            },
        )

    # -------------------------------------------------------------------------
    # Helper Methods
    # -------------------------------------------------------------------------

    def _set_guidance(
        self,
        message: str,
        *,
        step: Optional[int] = None,
        status: Optional[str] = None,
    ) -> None:
        """Update the robot calibration guidance UI."""
        import app.codroid_picker as ui

        guidance = ui.state.get_calibration_guidance_snapshot()
        ui._set_robot_guidance(
            message,
            step=step,
            status=status,
            verification_ready=False,
            modal_suppressed=guidance.get("modal_suppressed", False),
        )

    def _ensure_plate_baseline(self) -> bool:
        """Ensure plate baseline is set for movement detection."""
        import app.codroid_picker as ui

        SESSION = ui.SESSION
        PLATE_STATUS = ui.state.get_plate_status_snapshot()
        SETTINGS = ui.SETTINGS

        if SESSION.robot_calibration_baseline_px is not None:
            return True

        center = PLATE_STATUS.get("center_px")
        if center is not None:
            import numpy as np

            SESSION.robot_calibration_baseline_px = np.asarray(
                center, dtype=np.float64)
            return True

        frame, _depth = ui.get_latest_frame()
        if frame is None:
            return False

        try:
            from vision import plate

            calibration_result = plate.estimate_plate_calibration(
                frame, self._plate_layout)
        except Exception as exc:
            LOGGER.warning("Plate baseline detection failed: %s", exc)
            return False

        center_px = ui._plate_center_px(calibration_result.detections)
        if center_px is None:
            return False

        import numpy as np

        SESSION.robot_calibration_baseline_px = np.asarray(
            center_px, dtype=np.float64)
        ui.state.update_plate_status(
            count=len(calibration_result.detections),
            ok=len(calibration_result.detections) >= SETTINGS.min_plate_markers,
            stable=len(
                calibration_result.detections) >= SETTINGS.min_plate_markers,
            center_px=(float(center_px[0]), float(center_px[1])),
            movement_px=None,
            message="Plate baseline refreshed from live detection.",
        )
        return True

    def _ensure_plate_stable(self, *, allow_lost: bool = False) -> None:
        """Ensure the calibration plate is stable."""
        import app.codroid_picker as ui

        ui._ensure_plate_stable(allow_lost=allow_lost)

    async def _ensure_coordinate_id(
        self,
        robot_api: "CodroidAPI",
        session: Any,
        expected_id: int,
        timeout_s: float = 5.0,
        require: bool = True,
    ) -> None:
        """Ensure robot is using the expected coordinate system.

        Delegates to the shared _ensure_robot_coordinate helper in procedures.py.
        """
        try:
            await _ensure_robot_coordinate(
                robot_api,
                expected_id,
                timeout_s=timeout_s,
                require=require,
            )
        except RuntimeError as exc:
            # Provide calibration-specific error message
            raise RuntimeError(
                "Calibration requires the robot to be on CoordinateId=0 (base). "
                f"Current coordinate could not be verified. Switch to 0 and retry. ({exc})"
            ) from exc

    async def _await_flange_press_with_guard(
        self,
        session: Any,
        timeout_s: float,
    ) -> Dict[str, Any]:
        """Wait for flange button press with plate stability guard."""
        import app.codroid_picker as ui

        deadline = time.monotonic() + timeout_s

        while True:
            self.ensure_active()
            if ui.SESSION.robot_calibration_baseline_px is None:
                self._ensure_plate_baseline()

            plate_ok, plate_message = ui._plate_is_stable(
                require_baseline=True,
                allow_lost=True,
            )
            if not plate_ok:
                raise RuntimeError(f"Calibration canceled: {plate_message}")

            posture_ok, posture_message = ui._robot_posture_ready()
            if not posture_ok:
                raise RuntimeError(f"Calibration canceled: {posture_message}")

            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise asyncio.TimeoutError()

            try:
                return await session.next_flange_press(timeout=min(0.5, remaining))
            except asyncio.TimeoutError:
                continue

    async def _send_calibration_request(
        self,
        robot_api: "CodroidAPI",
        session: Any,
        robot_points: List[Dict[str, Any]],
        target_coord_id: int,
        timeout_s: float,
    ) -> Dict[str, Any]:
        """Send coordinate calibration request to robot controller."""
        from codroid_api import CodroidAPI as CodroidAPIClass

        # Build CPOS points
        cpos_points = [
            CodroidAPIClass.build_target_cpos(
                x=float(pt["x"]),
                y=float(pt["y"]),
                z=float(pt["z"]),
                a=float(pt["a"]),
                b=float(pt["b"]),
                c=float(pt["c"]),
            )
            for pt in robot_points
        ]

        self._set_guidance(
            "Waiting for robot controller to return the calibrated frame...",
            status="running",
        )
        LOGGER.info(
            "Requesting coordinate calibration from robot controller...")

        # Send calibration request
        calibration_request_time = time.time()
        payload_points = [
            robot_api.attach_coordinate_to_cpos(point, coordinate_id=0)
            for point in cpos_points
        ]
        calibration_request = robot_api.build_message(
            message_type="Robot",
            action="CoordinateCalibration",
            data=payload_points,
        )
        await robot_api.send_message(calibration_request)

        # Wait for calibration response
        try:
            coord_frame = await self._await_listener_frame(
                session.calibration_frame_snapshot,
                session.calibration_frame_timestamp,
                calibration_request_time,
                timeout_s,
                "coordinate calibration response",
            )
        except asyncio.TimeoutError as exc:
            raise RuntimeError(
                "Timed out waiting for coordinate calibration response."
            ) from exc

        if not coord_frame:
            raise RuntimeError(
                "Robot controller returned empty calibration frame.")

        # Normalize orientation axes
        coord_frame = self._normalize_coord_frame(coord_frame)

        # Send change request
        change_request_time = time.time()
        change_payload = {
            "id": target_coord_id,
            "x": coord_frame["x"],
            "y": coord_frame["y"],
            "z": coord_frame["z"],
            "a": coord_frame["a"],
            "b": coord_frame["b"],
            "c": coord_frame["c"],
        }
        change_request = robot_api.build_message(
            message_type="Robot",
            action="ChangeCoordinateParameter",
            data=change_payload,
        )
        await robot_api.send_message(change_request)

        # Wait for coordinate frame publish
        publish_timeout = min(timeout_s, 5.0)
        try:
            await self._await_listener_frame(
                session.coordinate_frame_snapshot,
                session.coordinate_frame_timestamp,
                change_request_time,
                publish_timeout,
                "coordinate frame publish",
            )
        except asyncio.TimeoutError:
            LOGGER.warning(
                "No RobotCoordinate publish after calibration; reselecting CoordinateId."
            )
            await robot_api.set_current_coordinate_id(target_coord_id)
            try:
                await self._await_listener_frame(
                    session.coordinate_frame_snapshot,
                    session.coordinate_frame_timestamp,
                    time.time(),
                    publish_timeout,
                    "coordinate frame publish after reselect",
                )
            except asyncio.TimeoutError:
                LOGGER.warning(
                    "Timed out waiting for coordinate frame publish after reselect."
                )

        return coord_frame

    async def _await_listener_frame(
        self,
        snapshot: Callable[[], Dict[str, Any]],
        timestamp: Callable[[], Optional[float]],
        since: float,
        timeout_s: float,
        label: str,
    ) -> Dict[str, Any]:
        """Wait for a frame update from a listener."""
        deadline = time.monotonic() + timeout_s

        while time.monotonic() < deadline:
            updated_at = timestamp()
            if updated_at is not None and updated_at >= since:
                frame = snapshot()
                if frame:
                    return frame
            await asyncio.sleep(0.1)

        raise asyncio.TimeoutError(f"Timed out waiting for {label}.")

    def _normalize_coord_frame(self, coord_frame: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize orientation axes in coordinate frame."""
        import app.codroid_picker as ui

        def _coord_axis(keys: Tuple[str, ...]) -> float:
            for key in keys:
                if key in coord_frame:
                    return ui._safe_float(coord_frame.get(key), 0.0)
            return 0.0

        coord_frame["a"] = _coord_axis(("a", "rx", "roll"))
        coord_frame["b"] = _coord_axis(("b", "ry", "pitch"))
        coord_frame["c"] = _coord_axis(("c", "rz", "yaw"))

        return coord_frame
