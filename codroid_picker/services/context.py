"""Application context construction."""

from __future__ import annotations

import asyncio
import contextlib
from dataclasses import dataclass, field
import logging

from codroid_picker.config import AppSettings, SettingsRepository
from codroid_picker.services.auto_pick_service import AutoPickService
from codroid_picker.services.calibration_service import CalibrationService
from codroid_picker.services.camera_service import CameraService
from codroid_picker.services.detection_service import DetectionService
from codroid_picker.services.gripper_service import GripperService
from codroid_picker.services.monitoring_service import MonitoringService
from codroid_picker.services.overlay_service import OverlayService
from codroid_picker.services.pick_place_service import PickPlaceService
from codroid_picker.services.plate_test_service import PlateTestService
from codroid_picker.services.robot_service import RobotService
from codroid_picker.services.safety_service import SafetyService
from codroid_picker.services.settings_service import SettingsService
from codroid_picker.services.video_service import VideoService
from codroid_picker.state.events import StatePublisher
from codroid_picker.state.store import StateStore
from codroid_picker.video.frame_buffer import RawFrameBuffer

LOGGER = logging.getLogger(__name__)


@dataclass
class AppContext:
    settings: SettingsService
    state: StateStore
    publisher: StatePublisher
    frame_buffer: RawFrameBuffer
    camera: CameraService
    video: VideoService
    detection: DetectionService
    robot: RobotService
    gripper: GripperService
    calibration: CalibrationService
    auto_pick: AutoPickService
    pick_place: PickPlaceService
    plate_test: PlateTestService
    monitoring: MonitoringService
    overlay: OverlayService
    safety: SafetyService
    startup_tasks: list[asyncio.Task[None]] = field(default_factory=list)

    async def shutdown(self) -> None:
        for task in self.startup_tasks:
            task.cancel()
        if self.startup_tasks:
            await asyncio.gather(*self.startup_tasks, return_exceptions=True)
        await self.auto_pick.shutdown()
        await self.safety.shutdown()
        await self.detection.stop_scheduler()
        self.detection.stop()
        self.camera.shutdown()
        with contextlib.suppress(Exception):
            await self.gripper.disconnect()
        await self.robot.shutdown()

    def start_hardware_autoconnect(self) -> None:
        if self.settings.current.robot_autoconnect:
            self.startup_tasks.append(asyncio.create_task(self._autoconnect_robot()))
        if self.settings.current.gripper_autoconnect:
            self.startup_tasks.append(asyncio.create_task(self._autoconnect_gripper()))

    async def _autoconnect_robot(self) -> None:
        timeout_s = max(
            float(self.settings.current.robot_autoconnect_timeout_s),
            1.0,
        )
        uri = self.settings.current.default_robot_uri
        self.state.update_domain(
            "robot",
            status="connecting",
            message=f"Connecting to robot at {uri}.",
        )
        try:
            await asyncio.wait_for(self.robot.connect(uri), timeout=timeout_s)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("Robot autoconnect failed: %s", exc)
            self.state.update_domain(
                "robot",
                status="offline",
                message=f"Robot auto-connect failed: {exc}",
            )

    async def _autoconnect_gripper(self) -> None:
        timeout_s = max(
            float(self.settings.current.gripper_autoconnect_timeout_s),
            1.0,
        )
        ip = self.settings.current.gripper_ip or "192.168.101.95"
        self.state.update_domain(
            "gripper",
            status="connecting",
            message=f"Connecting to gripper at {ip}.",
        )
        try:
            await asyncio.wait_for(
                self.gripper.connect(
                    ip=ip,
                    gripper_type=self.settings.current.gripper_type,
                ),
                timeout=timeout_s,
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("Gripper autoconnect failed: %s", exc)
            self.state.update_domain(
                "gripper",
                status="offline",
                message=f"Gripper auto-connect failed: {exc}",
            )


def build_context() -> AppContext:
    publisher = StatePublisher()
    state = StateStore(publisher)
    settings = AppSettings()
    repository = SettingsRepository()
    settings_service = SettingsService(settings, repository, state)
    frame_buffer = RawFrameBuffer()
    camera = CameraService(settings_service.current, state, frame_buffer)
    video = VideoService(settings_service.current, state, frame_buffer)
    detection = DetectionService(settings_service.current, state, frame_buffer)
    robot = RobotService(settings_service.current, state)
    gripper = GripperService(settings_service.current, state)
    calibration = CalibrationService(settings_service.current, state, frame_buffer)
    monitoring = MonitoringService(state)
    overlay = OverlayService(state)
    safety = SafetyService(settings_service.current, state, calibration)
    pick_place = PickPlaceService(
        settings_service.current,
        state,
        detection,
        robot,
        gripper,
        calibration,
    )
    plate_test = PlateTestService(
        settings_service.current,
        state,
        calibration,
        robot,
        gripper,
        pick_place,
    )
    auto_pick = AutoPickService(settings_service.current, state, detection, pick_place)
    detection.start()
    detection.set_enricher(pick_place.enrich_detection)
    detection.start_scheduler()
    robot.start_polling()
    safety.start()
    if settings.camera_autostart:
        try:
            camera.start()
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("Camera autostart failed: %s", exc)
            state.update_domain("camera", status="error", last_error=str(exc))
    context = AppContext(
        settings=settings_service,
        state=state,
        publisher=publisher,
        frame_buffer=frame_buffer,
        camera=camera,
        video=video,
        detection=detection,
        robot=robot,
        gripper=gripper,
        calibration=calibration,
        auto_pick=auto_pick,
        pick_place=pick_place,
        plate_test=plate_test,
        monitoring=monitoring,
        overlay=overlay,
        safety=safety,
    )
    context.start_hardware_autoconnect()
    return context
