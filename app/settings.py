"""
Application configuration helpers stored separately from the UI code.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Optional, Tuple

from codroid_api import CodroidSettings
from onrobot.gripper_profiles import (
    GripperProfile,
    get_gripper_profile,
)
from pydantic_settings import BaseSettings, SettingsConfigDict

LOGGER = logging.getLogger(__name__)

DEFAULT_BOARD_PATH = Path(
    "data/markers/camera_calibration/camera_calibration_board.json")
DEFAULT_BUNDLE_PATH = Path("data/calibration_bundle.npz")
DEFAULT_WORKSPACE_PATH = Path("data/workspace_definition.npz")
DEFAULT_ROBOT_CALIBRATION_PATH = Path("data/robot_calibration.json")

_BASE_PLATE_LAYOUT_PATH = Path(
    "data/markers/robot_calibration/calibration_plate_layout.json")
_BASE_CAPTURE_DIR = Path("data/calibration")
_BASE_CAMERA_RESOLUTION: Tuple[int, int] = (1920, 1080)
_BASE_CAMERA_FPS = 30.0
_BASE_CAMERA_SOURCE = "oak"
_BASE_HOST_CAMERA_INDEX = 0
_BASE_CAMERA_ENABLE_DEPTH = False
_BASE_CAMERA_FAST_DEPTH = True
_BASE_MODEL_DIR = Path("data/models")
_BASE_DROP_MARKER_ID = 30
_BASE_CAMERA_CALIBRATION_FRAMES = 30
_BASE_CAMERA_CALIBRATION_DELAY_S = 0.1
_BASE_CAMERA_CALIBRATION_MIN_USABLE_FRAMES = 1
_BASE_AUTO_TARGET_LABEL = "orange"
_BASE_AUTO_TARGET_LABELS = ["orange", "sports ball"]
_BASE_AUTO_LABEL_PROFILES: dict[str, dict[str, float]] = {}
_BASE_AUTO_DROP_IGNORE_RADIUS_MM = 60.0
_BASE_AUTO_INTRUSION_MATCH_MM = 40.0
_BASE_AUTO_DETECTION_STALE_S = 0.3
_BASE_AUTO_IDLE_SLEEP_S = 0.25
_BASE_AUTO_APPROACH_MM = -30.0
_BASE_AUTO_PICK_MM = 0.0
_BASE_AUTO_DROP_MM = 0.0
_BASE_AUTO_ESTIMATE_WIDTH = False
_BASE_AUTO_MOVE_TIMEOUT_S = 15.0
_BASE_AUTO_SETTLE_S = 0.1
_BASE_AUTO_TOLERANCE_MM = 2.0
_BASE_AUTO_Z_TOLERANCE_MM = 3.0
_BASE_AUTO_HOLD_S = 0.5
_BASE_AUTO_CLUSTER_NEIGHBOR_MM = 35.0
_BASE_AUTO_ROTATION_CLEARANCE_MM = 8.0
_BASE_AUTO_ROTATION_STEP_DEG = 15.0
_BASE_AUTO_ROTATION_MAX_DELTA_DEG = 90.0
_BASE_AUTO_COLLISION_BODY_RADIUS_MM = 12.0
_BASE_AUTO_COLLISION_FINGER_REACH_MM = 20.0
_BASE_AUTO_SEPARATION_PUSH_MM = 28.0
_BASE_AUTO_SEPARATION_ENTRY_MM = 10.0
_BASE_AUTO_SEPARATION_MAX_ATTEMPTS = 2
_BASE_AUTO_RESCAN_DELAY_S = 0.25
_BASE_PICK_FOLLOW_TIMEOUT_S = 15.0
_BASE_PICK_FOLLOW_UPDATE_S = 0.05
_BASE_PICK_FOLLOW_SEGMENT_S = 0.1
_BASE_PICK_FOLLOW_TOLERANCE_MM = 3.0
_BASE_PICK_FOLLOW_SETTLE_S = 0.6
_BASE_RESCUE_JOG_HOLD_S = 0.35
_BASE_COORD5_MAX_Z_MM = 0.0
_BASE_ROBOT_COORDINATE_ID = 5
_BASE_CAMERA_SAFETY_ENABLED = True
_BASE_CAMERA_SAFETY_ZONE = "workspace"
_BASE_CAMERA_SAFETY_LABELS = ["person", "people"]
_BASE_CAMERA_SAFETY_CHECK_INTERVAL_MS = 250
_BASE_DETECTION_FRAME_SKIP = 3
_BASE_UI_STATUS_INTERVAL_MS = 1000
_BASE_UI_MONITOR_INTERVAL_MS = 2000

_BASE_PLATE_TEST_CAPTURE_DIR = Path("data/plate_test_runs")
_BASE_MIN_PLATE_MARKERS = 2
_BASE_PLATE_MOVEMENT_THRESHOLD_PX = 500.0
_BASE_PLATE_VISIBILITY_MAX_AGE_S = 5.0
_BASE_ROBOT_POSTURE_MAX_AGE_S = 2.0
_BASE_VERIFICATION_RAISE_MM = 300.0
_BASE_ROBOT_COORDINATE_TIMEOUT_S = 60.0
_BASE_VERIFICATION_HOLD_SECONDS = 5.0
_BASE_VERIFICATION_MOVE_TIMEOUT_S = 20.0
_BASE_VERIFICATION_TOLERANCE_MM = 2.0
_BASE_VERIFICATION_Z_TOLERANCE_MM = 2.0
_BASE_VERIFICATION_SETTLE_S = 1.5
_BASE_VERIFICATION_MOTION_EPS_MM = 0.5
_BASE_PLATE_TEST_GRIP_TIMEOUT_S = 2.5
_BASE_PLATE_TEST_MOVE_HOLD_S = 0.2
_BASE_PLATE_TEST_MOVE_TIMEOUT_S = 25.0
_BASE_PLATE_TEST_TOLERANCE_MM = 2.0
_BASE_PLATE_TEST_Z_TOLERANCE_MM = 2.0
_BASE_PLATE_TEST_SETTLE_S = 0.4
_BASE_PLATE_TEST_MOTION_EPS_MM = 0.5
_BASE_PLATE_TEST_FOLLOW_UPDATE_S = 0.05
_BASE_PLATE_TEST_FOLLOW_SEGMENT_S = 0.1
_BASE_PLATE_TEST_USE_BASE_COORDINATES = False

DEFAULT_PLATE_PATH = _BASE_PLATE_LAYOUT_PATH
DEFAULT_CAPTURE_DIR = _BASE_CAPTURE_DIR
DEFAULT_RESOLUTION = _BASE_CAMERA_RESOLUTION
DEFAULT_FPS = _BASE_CAMERA_FPS
DEFAULT_MODEL_DIR = _BASE_MODEL_DIR
DEFAULT_DROP_MARKER_ID = _BASE_DROP_MARKER_ID

DROP_TARGET_PATH = Path("data/drop_target.json")
SETTINGS_PATH = Path("data/app_settings.json")
AUTO_GRIPPER_TYPE = "auto"
SUPPORTED_APP_GRIPPER_TYPES = ("twofg7", "sg", "vgc10")
GRIPPER_PROFILE_OPTIONS = [
    {"value": AUTO_GRIPPER_TYPE, "label": "Auto detect"},
    *[
        {"value": key, "label": get_gripper_profile(key).display_name}
        for key in SUPPORTED_APP_GRIPPER_TYPES
    ],
]


class AppSettings(BaseSettings):
    """Application configuration persisted to disk and overrideable via .env."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="CODROID_",
        extra="ignore",
    )

    # Robot connection settings (host + ports)
    robot_host: Optional[str] = None
    robot_port: Optional[int] = None
    user_port: Optional[int] = None
    ui_language: str = "en"

    # Marker and tracking details that define the workspace
    marker_ids: Tuple[int, int] = (1, 2)
    marker_size_mm: float = 40.0

    # Gripper and model metadata / debug helpers
    gripper_ip: Optional[str] = None
    last_model: Optional[str] = None
    active_model: Optional[str] = None
    drop_target: Optional[Tuple[float, float]] = None
    gripper_type: str = AUTO_GRIPPER_TYPE

    # Asset paths that can be redirected via settings
    calibration_bundle_path: Path = DEFAULT_BUNDLE_PATH
    workspace_path: Path = DEFAULT_WORKSPACE_PATH
    robot_calibration_path: Path = DEFAULT_ROBOT_CALIBRATION_PATH

    plate_layout_path: Path = _BASE_PLATE_LAYOUT_PATH
    capture_dir: Path = _BASE_CAPTURE_DIR
    camera_resolution: Tuple[int, int] = _BASE_CAMERA_RESOLUTION
    camera_fps: float = _BASE_CAMERA_FPS
    camera_source: str = _BASE_CAMERA_SOURCE
    host_camera_index: int = _BASE_HOST_CAMERA_INDEX
    camera_enable_depth: bool = _BASE_CAMERA_ENABLE_DEPTH
    camera_fast_depth: bool = _BASE_CAMERA_FAST_DEPTH
    model_dir: Path = _BASE_MODEL_DIR
    drop_marker_id: int = _BASE_DROP_MARKER_ID
    plate_test_capture_dir: Path = _BASE_PLATE_TEST_CAPTURE_DIR
    min_plate_markers: int = _BASE_MIN_PLATE_MARKERS
    plate_movement_threshold_px: float = _BASE_PLATE_MOVEMENT_THRESHOLD_PX
    plate_visibility_max_age_s: float = _BASE_PLATE_VISIBILITY_MAX_AGE_S
    robot_posture_max_age_s: float = _BASE_ROBOT_POSTURE_MAX_AGE_S
    verification_raise_mm: float = _BASE_VERIFICATION_RAISE_MM
    robot_coordinate_timeout_s: float = _BASE_ROBOT_COORDINATE_TIMEOUT_S
    verification_hold_seconds: float = _BASE_VERIFICATION_HOLD_SECONDS
    verification_move_timeout_s: float = _BASE_VERIFICATION_MOVE_TIMEOUT_S
    verification_tolerance_mm: float = _BASE_VERIFICATION_TOLERANCE_MM
    verification_z_tolerance_mm: float = _BASE_VERIFICATION_Z_TOLERANCE_MM
    verification_settle_s: float = _BASE_VERIFICATION_SETTLE_S
    verification_motion_eps_mm: float = _BASE_VERIFICATION_MOTION_EPS_MM
    plate_test_grip_timeout_s: float = _BASE_PLATE_TEST_GRIP_TIMEOUT_S
    plate_test_move_hold_s: float = _BASE_PLATE_TEST_MOVE_HOLD_S
    plate_test_move_timeout_s: float = _BASE_PLATE_TEST_MOVE_TIMEOUT_S
    plate_test_tolerance_mm: float = _BASE_PLATE_TEST_TOLERANCE_MM
    plate_test_z_tolerance_mm: float = _BASE_PLATE_TEST_Z_TOLERANCE_MM
    plate_test_settle_s: float = _BASE_PLATE_TEST_SETTLE_S
    plate_test_motion_eps_mm: float = _BASE_PLATE_TEST_MOTION_EPS_MM
    plate_test_follow_update_s: float = _BASE_PLATE_TEST_FOLLOW_UPDATE_S
    plate_test_follow_segment_s: float = _BASE_PLATE_TEST_FOLLOW_SEGMENT_S
    plate_test_use_base_coordinates: bool = _BASE_PLATE_TEST_USE_BASE_COORDINATES

    # Camera calibration capture controls
    camera_calibration_frame_count: int = _BASE_CAMERA_CALIBRATION_FRAMES
    camera_calibration_frame_delay_s: float = _BASE_CAMERA_CALIBRATION_DELAY_S
    camera_calibration_min_usable_frames: int = _BASE_CAMERA_CALIBRATION_MIN_USABLE_FRAMES

    # UI tracking controls
    tracking_mode: str = "move"
    segment_min: float = 20.0
    segment_max: float = 300.0
    segment_fraction: float = 0.8

    # Pick-follow defaults used by conservative follow motion helpers
    pick_follow_timeout_s: float = _BASE_PICK_FOLLOW_TIMEOUT_S
    pick_follow_update_interval_s: float = _BASE_PICK_FOLLOW_UPDATE_S
    pick_follow_segment_hold_s: float = _BASE_PICK_FOLLOW_SEGMENT_S
    pick_follow_tolerance_mm: float = _BASE_PICK_FOLLOW_TOLERANCE_MM
    pick_follow_settle_s: float = _BASE_PICK_FOLLOW_SETTLE_S

    # Safety guards
    rescue_jog_hold_s: float = _BASE_RESCUE_JOG_HOLD_S
    coord5_max_z_mm: float = _BASE_COORD5_MAX_Z_MM
    camera_safety_enabled: bool = _BASE_CAMERA_SAFETY_ENABLED
    camera_safety_zone: str = _BASE_CAMERA_SAFETY_ZONE
    camera_safety_labels: list[str] = _BASE_CAMERA_SAFETY_LABELS
    camera_safety_check_interval_ms: int = _BASE_CAMERA_SAFETY_CHECK_INTERVAL_MS
    detection_frame_skip: int = _BASE_DETECTION_FRAME_SKIP
    ui_status_interval_ms: int = _BASE_UI_STATUS_INTERVAL_MS
    ui_monitor_interval_ms: int = _BASE_UI_MONITOR_INTERVAL_MS

    # Robot coordinate selection
    robot_coordinate_id: int = _BASE_ROBOT_COORDINATE_ID

    # Auto pick & place defaults
    auto_target_label: str = _BASE_AUTO_TARGET_LABEL
    auto_target_labels: list[str] = _BASE_AUTO_TARGET_LABELS
    auto_label_profiles: dict[str, dict[str, Any]] = _BASE_AUTO_LABEL_PROFILES
    auto_drop_ignore_radius_mm: float = _BASE_AUTO_DROP_IGNORE_RADIUS_MM
    auto_intrusion_match_mm: float = _BASE_AUTO_INTRUSION_MATCH_MM
    auto_detection_stale_s: float = _BASE_AUTO_DETECTION_STALE_S
    auto_idle_sleep_s: float = _BASE_AUTO_IDLE_SLEEP_S
    auto_approach_mm: float = _BASE_AUTO_APPROACH_MM
    auto_pick_mm: float = _BASE_AUTO_PICK_MM
    auto_drop_mm: float = _BASE_AUTO_DROP_MM
    auto_estimate_width: bool = _BASE_AUTO_ESTIMATE_WIDTH
    auto_move_timeout_s: float = _BASE_AUTO_MOVE_TIMEOUT_S
    auto_settle_s: float = _BASE_AUTO_SETTLE_S
    auto_tolerance_mm: float = _BASE_AUTO_TOLERANCE_MM
    auto_z_tolerance_mm: float = _BASE_AUTO_Z_TOLERANCE_MM
    auto_hold_s: float = _BASE_AUTO_HOLD_S
    auto_cluster_neighbor_mm: float = _BASE_AUTO_CLUSTER_NEIGHBOR_MM
    auto_rotation_clearance_mm: float = _BASE_AUTO_ROTATION_CLEARANCE_MM
    auto_rotation_step_deg: float = _BASE_AUTO_ROTATION_STEP_DEG
    auto_rotation_max_delta_deg: float = _BASE_AUTO_ROTATION_MAX_DELTA_DEG
    auto_collision_body_radius_mm: float = _BASE_AUTO_COLLISION_BODY_RADIUS_MM
    auto_collision_finger_reach_mm: float = _BASE_AUTO_COLLISION_FINGER_REACH_MM
    auto_separation_push_mm: float = _BASE_AUTO_SEPARATION_PUSH_MM
    auto_separation_entry_mm: float = _BASE_AUTO_SEPARATION_ENTRY_MM
    auto_separation_max_attempts: int = _BASE_AUTO_SEPARATION_MAX_ATTEMPTS
    auto_rescan_delay_s: float = _BASE_AUTO_RESCAN_DELAY_S

    @property
    def gripper_profile(self) -> GripperProfile:
        """Return the profile that matches the persisted gripper type."""
        return get_gripper_profile(self.gripper_type)

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls,
        init_settings,
        env_settings,
        dotenv_settings,
        file_secret_settings,
    ):
        return (
            init_settings,
            env_settings,
            cls._json_settings_source,
            dotenv_settings,
            file_secret_settings,
        )

    @classmethod
    def _json_settings_source(cls):
        if not SETTINGS_PATH.exists():
            return {}
        try:
            raw = json.loads(SETTINGS_PATH.read_text())
            if not isinstance(raw, dict):
                return {}
            # Backward-compatibility for older settings files that persisted
            # required fields (for example Path fields) as null.
            # Dropping nulls lets Pydantic apply model defaults.
            return {key: value for key, value in raw.items() if value is not None}
        except Exception as exc:
            LOGGER.warning(
                "Failed to read persisted settings from %s: %s", SETTINGS_PATH, exc)
            return {}


SETTINGS = AppSettings()


def _save_settings(settings: AppSettings) -> None:
    SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    SETTINGS_PATH.write_text(settings.model_dump_json(indent=2))


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
        host = SETTINGS.robot_host or os.getenv(
            "CODROID_HOST", "192.168.101.100")
        robot_port = int(SETTINGS.robot_port or os.getenv(
            "CODROID_ROBOT_PORT", "9000"))
        ws_port = int(SETTINGS.user_port or os.getenv(
            "CODROID_WS_PORT", "9098"))
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
    """Load OnRobot gripper default configuration."""
    return {
        "compute_box_ip": SETTINGS.gripper_ip or os.getenv("ONROBOT_CB_IP", "192.168.101.95"),
        "t_index": int(os.getenv("ONROBOT_T_INDEX", "0")),
    }


def _refresh_dynamic_defaults() -> None:
    """Update exported defaults that follow the current AppSettings values."""
    global DEFAULT_PLATE_PATH, DEFAULT_CAPTURE_DIR, DEFAULT_RESOLUTION, DEFAULT_FPS, DEFAULT_MODEL_DIR, DEFAULT_DROP_MARKER_ID
    DEFAULT_PLATE_PATH = SETTINGS.plate_layout_path
    DEFAULT_CAPTURE_DIR = SETTINGS.capture_dir
    DEFAULT_RESOLUTION = SETTINGS.camera_resolution
    DEFAULT_FPS = SETTINGS.camera_fps
    DEFAULT_MODEL_DIR = SETTINGS.model_dir
    DEFAULT_DROP_MARKER_ID = SETTINGS.drop_marker_id


def reload_defaults() -> None:
    """Recompute derived connection defaults after settings change."""
    global CODROID_DEFAULTS, DEFAULT_ROBOT_URI, GRIPPER_DEFAULTS, DEFAULT_GRIPPER_IP
    CODROID_DEFAULTS = _load_codroid_defaults()
    DEFAULT_ROBOT_URI = CODROID_DEFAULTS["robot_ws_url"]
    GRIPPER_DEFAULTS = _load_gripper_defaults()
    DEFAULT_GRIPPER_IP = GRIPPER_DEFAULTS["compute_box_ip"]
    _refresh_dynamic_defaults()


def set_asset_paths(
    *,
    calibration_bundle_path: Optional[str] = None,
    workspace_path: Optional[str] = None,
    robot_calibration_path: Optional[str] = None,
) -> None:
    """Update the stored defaults for asset paths."""
    global DEFAULT_BUNDLE_PATH, DEFAULT_WORKSPACE_PATH, DEFAULT_ROBOT_CALIBRATION_PATH
    if calibration_bundle_path:
        DEFAULT_BUNDLE_PATH = Path(calibration_bundle_path)
    if workspace_path:
        DEFAULT_WORKSPACE_PATH = Path(workspace_path)
    if robot_calibration_path:
        DEFAULT_ROBOT_CALIBRATION_PATH = Path(robot_calibration_path)


set_asset_paths(
    calibration_bundle_path=str(SETTINGS.calibration_bundle_path),
    workspace_path=str(SETTINGS.workspace_path),
    robot_calibration_path=str(SETTINGS.robot_calibration_path),
)

reload_defaults()
