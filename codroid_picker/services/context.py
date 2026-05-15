"""Application context construction."""

from __future__ import annotations

from dataclasses import dataclass
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

    async def shutdown(self) -> None:
        await self.auto_pick.shutdown()
        await self.safety.shutdown()
        await self.detection.stop_scheduler()
        self.detection.stop()
        self.camera.shutdown()
        await self.robot.shutdown()


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
    pick_place = PickPlaceService(settings_service.current, state, detection, robot, gripper, calibration)
    plate_test = PlateTestService(settings_service.current, state, calibration, robot, gripper, pick_place)
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
    return AppContext(
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
