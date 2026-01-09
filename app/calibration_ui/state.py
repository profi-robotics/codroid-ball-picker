"""State and configuration helpers for the calibration UI."""

from __future__ import annotations

import json
import os
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Deque, Optional, Tuple, TYPE_CHECKING

import numpy as np
from codroid_api import CodroidSettings, RobotSession
from onrobot.device import Device as GripperDevice
from onrobot.twofg import TWOFG as TwoFG7Gripper
from pydantic import BaseModel, ConfigDict
from vision.stream import DepthAIFrameStream

from app.orchestrator import DemoConfig
from .constants import (
    DEFAULT_CAPTURE_DIR,
    DEFAULT_DROP_MARKER_ID,
    FLANGE_BUTTON_PORT,
    FLANGE_BUTTON_PORTS,
    SETTINGS_PATH,
)

if TYPE_CHECKING:
    from vision import calibration, plate, workspace


class AppSettings(BaseModel):
    model_config = ConfigDict(extra="ignore")
    robot_host: Optional[str] = None
    robot_port: Optional[int] = None
    user_port: Optional[int] = None
    gripper_ip: Optional[str] = None
    marker_ids: Tuple[int, int] = (1, 2)
    marker_size_mm: float = 40.0
    last_model: Optional[str] = None
    drop_target: Optional[Tuple[float, float]] = None
    calibration_bundle_path: Optional[str] = None
    workspace_path: Optional[str] = None
    robot_calibration_path: Optional[str] = None


def _load_settings() -> AppSettings:
    if not SETTINGS_PATH.exists():
        return AppSettings()
    try:
        return AppSettings.model_validate_json(SETTINGS_PATH.read_text())
    except Exception:  # noqa: BLE001
        return AppSettings()


def _save_settings(settings: AppSettings) -> None:
    SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    SETTINGS_PATH.write_text(settings.model_dump_json(indent=2))


SETTINGS = _load_settings()


def _commit_settings(**updates) -> None:
    global SETTINGS
    SETTINGS = SETTINGS.model_copy(update=updates)
    _save_settings(SETTINGS)


def _load_codroid_defaults() -> dict:
    try:
        settings = CodroidSettings()
        host = settings.host
        robot_port = settings.robot_port
        ws_port = settings.ws_port
    except Exception:
        host = SETTINGS.robot_host or os.getenv("CODROID_HOST", "192.168.101.100")
        robot_port = int(SETTINGS.robot_port or os.getenv("CODROID_ROBOT_PORT", "9000"))
        ws_port = int(SETTINGS.user_port or os.getenv("CODROID_WS_PORT", "9098"))
    robot_ws_url = f"ws://{host}:{robot_port}/"
    user_ws_url = f"ws://{host}:{ws_port}/"
    return {
        "host": host,
        "robot_port": robot_port,
        "ws_port": ws_port,
        "robot_ws_url": robot_ws_url,
        "user_ws_url": user_ws_url,
    }


def _load_gripper_defaults() -> dict:
    return {
        "compute_box_ip": SETTINGS.gripper_ip or os.getenv("ONROBOT_CB_IP", "192.168.101.180"),
        "t_index": int(os.getenv("ONROBOT_T_INDEX", "0")),
    }


CODROID_DEFAULTS = _load_codroid_defaults()
DEFAULT_ROBOT_URI = CODROID_DEFAULTS["robot_ws_url"]
GRIPPER_DEFAULTS = _load_gripper_defaults()
DEFAULT_GRIPPER_IP = GRIPPER_DEFAULTS["compute_box_ip"]


@dataclass
class CaptureState:
    output_dir: Path = DEFAULT_CAPTURE_DIR
    frames: list[Path] = field(default_factory=list)
    depth_frame: Optional[np.ndarray] = None


@dataclass
class UISession:
    config: DemoConfig = field(default_factory=DemoConfig)
    camera_stream: Optional[DepthAIFrameStream] = None
    gripper_device: Optional[GripperDevice] = None
    gripper: Optional[TwoFG7Gripper] = None
    capture: CaptureState = field(default_factory=CaptureState)
    calibration_bundle: Optional["calibration.CalibrationBundle"] = None
    workspace_definition: Optional["workspace.WorkspaceDefinition"] = None
    plate_calibration: Optional["plate.PlateCalibration"] = None
    active_model_path: Optional[Path] = None
    yolo_model: Optional[object] = None
    model_loaded: bool = False
    detections: list[dict] = field(default_factory=list)
    last_detection_error: Optional[str] = None
    drop_marker_id: int = DEFAULT_DROP_MARKER_ID
    drop_target_mm: Optional[Tuple[float, float]] = None
    drop_target_orientation: Optional[Tuple[float, float, float]] = None
    drop_target_robot: Optional[Tuple[float, float, float]] = None
    plate_reference_homography: Optional[np.ndarray] = None
    robot_session: Optional[RobotSession] = None
    robot_connection_attempted: bool = False
    robot_last_connection_error: Optional[str] = None
    last_pick_mm: Optional[Tuple[float, float]] = None
    last_action: Optional[str] = None
    last_plate_test_capture: Optional[str] = None
    plate_test_state: dict = field(default_factory=dict)
    last_frame_id: int = 0
    continuous_detection: bool = False
    detection_confidence: float = 0.7
    detection_iou: float = 0.45
    detection_max: int = 10
    detection_classes: Optional[str] = None
    gripper_sync: dict = field(default_factory=dict)
    gripper_orientation: Optional[str] = None
    gripper_min_width: Optional[float] = None
    gripper_max_width: Optional[float] = None
    gripper_busy: Optional[bool] = None
    gripper_grip_detected: Optional[bool] = None
    gripper_force: Optional[float] = None
    gripper_status_code: Optional[int] = None
    robot_speed_rate: float = 0.25
    robot_speed_last_set: float = 0.0
    robot_status_last_poll: float = 0.0
    robot_calibration_points: list[dict] = field(default_factory=list)
    robot_calibration_min_z: Optional[float] = None
    robot_calibration_baseline_px: Optional[np.ndarray] = None
    robot_calibration_gripper_width: Optional[float] = None
    robot_has_object: bool = False
    pick_attempts: int = 0
    pick_successes: int = 0
    pick_failures: int = 0
    last_cycle_s: Optional[float] = None
    cycle_times: Deque[float] = field(default_factory=lambda: deque(maxlen=50))


SESSION = UISession(
    config=DemoConfig(
        robot_host=CODROID_DEFAULTS["host"],
        robot_port=CODROID_DEFAULTS["robot_port"],
        user_port=CODROID_DEFAULTS["ws_port"],
        marker_ids=SETTINGS.marker_ids,
        marker_size_mm=SETTINGS.marker_size_mm,
    )
)


def _get_robot_session() -> RobotSession:
    if SESSION.robot_session is None:
        SESSION.robot_session = RobotSession(
            CodroidSettings(),
            flange_button_port=FLANGE_BUTTON_PORT,
            flange_button_ports=FLANGE_BUTTON_PORTS,
        )
    return SESSION.robot_session


PLATE_STATUS = {
    "count": 0,
    "ok": False,
    "stable": False,
    "center_px": None,
    "movement_px": None,
    "timestamp": time.time(),
    "message": "Waiting for plate markers",
}

ROBOT_CALIBRATION_GUIDANCE = {
    "status": "idle",
    "step": 0,
    "message": "Robot calibration idle. Click 'Begin calibrating robot' to start.",
    "verification_ready": False,
    "timestamp": time.time(),
}


__all__ = [
    "AppSettings",
    "CaptureState",
    "CODROID_DEFAULTS",
    "DEFAULT_GRIPPER_IP",
    "DEFAULT_ROBOT_URI",
    "GRIPPER_DEFAULTS",
    "PLATE_STATUS",
    "ROBOT_CALIBRATION_GUIDANCE",
    "SETTINGS",
    "SESSION",
    "_commit_settings",
    "_get_robot_session",
]
