"""Settings and persisted configuration for the FastAPI runtime."""

from __future__ import annotations

import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any, Optional, Tuple

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

LOGGER = logging.getLogger(__name__)

DEFAULT_BOARD_PATH = Path("data/markers/camera_calibration/camera_calibration_board.json")
DEFAULT_BUNDLE_PATH = Path("data/calibration_bundle.npz")
DEFAULT_WORKSPACE_PATH = Path("data/workspace_definition.npz")
DEFAULT_ROBOT_CALIBRATION_PATH = Path("data/robot_calibration.json")
DEFAULT_PLATE_LAYOUT_PATH = Path("data/markers/robot_calibration/calibration_plate_layout.json")
DEFAULT_CAPTURE_DIR = Path("data/calibration")
DEFAULT_MODEL_DIR = Path("data/models")
DEFAULT_DROP_TARGET_PATH = Path("data/drop_target.json")
DEFAULT_SETTINGS_PATH = Path("data/app_settings.json")

SUPPORTED_GRIPPER_TYPES = {"auto", "twofg7", "sg", "vgc10"}


class AppSettings(BaseSettings):
    """Validated effective application settings.

    Values are loaded from explicit constructor values, environment variables,
    ``data/app_settings.json``, ``.env``, and defaults in that order.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="CODROID_",
        extra="ignore",
    )

    robot_host: Optional[str] = None
    robot_port: Optional[int] = None
    user_port: Optional[int] = None
    ui_language: str = "en"

    marker_ids: Tuple[int, int] = (1, 2)
    marker_size_mm: float = 40.0
    gripper_ip: Optional[str] = None
    last_model: Optional[str] = None
    active_model: Optional[str] = None
    drop_target: Optional[Tuple[float, float]] = None
    gripper_type: str = "auto"

    calibration_bundle_path: Path = DEFAULT_BUNDLE_PATH
    workspace_path: Path = DEFAULT_WORKSPACE_PATH
    robot_calibration_path: Path = DEFAULT_ROBOT_CALIBRATION_PATH
    plate_layout_path: Path = DEFAULT_PLATE_LAYOUT_PATH
    capture_dir: Path = DEFAULT_CAPTURE_DIR
    camera_resolution: Tuple[int, int] = (1920, 1080)
    camera_fps: float = 30.0
    camera_source: str = "oak"
    host_camera_index: int = 0
    host_camera_device: Optional[str] = None
    camera_autostart: bool = False
    camera_enable_depth: bool = False
    camera_fast_depth: bool = True
    model_dir: Path = DEFAULT_MODEL_DIR
    drop_target_path: Path = DEFAULT_DROP_TARGET_PATH
    drop_marker_id: int = 30
    plate_test_capture_dir: Path = Path("data/plate_test_runs")

    min_plate_markers: int = 2
    plate_movement_threshold_px: float = 500.0
    plate_visibility_max_age_s: float = 5.0
    robot_posture_max_age_s: float = 2.0
    verification_raise_mm: float = 300.0
    robot_coordinate_timeout_s: float = 60.0
    verification_hold_seconds: float = 5.0
    verification_move_timeout_s: float = 20.0
    verification_tolerance_mm: float = 2.0
    verification_z_tolerance_mm: float = 2.0
    verification_settle_s: float = 1.5
    verification_motion_eps_mm: float = 0.5
    plate_test_grip_timeout_s: float = 2.5
    plate_test_move_hold_s: float = 0.2
    plate_test_move_timeout_s: float = 25.0
    plate_test_tolerance_mm: float = 2.0
    plate_test_z_tolerance_mm: float = 2.0
    plate_test_settle_s: float = 0.4
    plate_test_motion_eps_mm: float = 0.5
    plate_test_follow_update_s: float = 0.05
    plate_test_follow_segment_s: float = 0.1
    plate_test_use_base_coordinates: bool = False

    camera_calibration_frame_count: int = 30
    camera_calibration_frame_delay_s: float = 0.1
    camera_calibration_min_usable_frames: int = 1

    tracking_mode: str = "move"
    segment_min: float = 20.0
    segment_max: float = 300.0
    segment_fraction: float = 0.8
    pick_follow_timeout_s: float = 15.0
    pick_follow_update_interval_s: float = 0.05
    pick_follow_segment_hold_s: float = 0.1
    pick_follow_tolerance_mm: float = 3.0
    pick_follow_settle_s: float = 0.6

    rescue_jog_hold_s: float = 0.35
    coord5_max_z_mm: float = 0.0
    camera_safety_enabled: bool = True
    camera_safety_zone: str = "workspace"
    camera_safety_labels: list[str] = Field(default_factory=lambda: ["person", "people"])
    camera_safety_check_interval_ms: int = 250
    detection_frame_skip: int = 3
    ui_status_interval_ms: int = 1000
    ui_monitor_interval_ms: int = 2000
    robot_poll_interval_ms: int = 500
    robot_coordinate_id: int = 5

    auto_target_label: str = "orange"
    auto_target_labels: list[str] = Field(default_factory=lambda: ["orange", "sports ball"])
    auto_label_profiles: dict[str, dict[str, Any]] = Field(default_factory=dict)
    auto_drop_ignore_radius_mm: float = 60.0
    auto_intrusion_match_mm: float = 40.0
    auto_detection_stale_s: float = 0.3
    auto_idle_sleep_s: float = 0.25
    auto_approach_mm: float = -30.0
    auto_pick_mm: float = 0.0
    auto_drop_mm: float = 0.0
    auto_estimate_width: bool = False
    auto_move_timeout_s: float = 15.0
    auto_settle_s: float = 0.1
    auto_tolerance_mm: float = 2.0
    auto_z_tolerance_mm: float = 3.0
    auto_hold_s: float = 0.5
    auto_cluster_neighbor_mm: float = 35.0
    auto_rotation_clearance_mm: float = 8.0
    auto_rotation_step_deg: float = 15.0
    auto_rotation_max_delta_deg: float = 90.0
    auto_collision_body_radius_mm: float = 12.0
    auto_collision_finger_reach_mm: float = 20.0
    auto_separation_push_mm: float = 28.0
    auto_separation_entry_mm: float = 10.0
    auto_separation_max_attempts: int = 2
    auto_rescan_delay_s: float = 0.25

    video_max_fps: float = 15.0
    video_max_width: int = 960
    video_jpeg_quality: int = 70
    video_snapshot_width: int = 640
    detection_worker: bool = True
    use_fake_hardware: bool = False
    log_level: str = "INFO"

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: Any,
        env_settings: Any,
        dotenv_settings: Any,
        file_secret_settings: Any,
    ) -> tuple[Any, ...]:
        return (
            init_settings,
            env_settings,
            cls._json_settings_source,
            dotenv_settings,
            file_secret_settings,
        )

    @classmethod
    def _json_settings_source(cls) -> dict[str, Any]:
        settings_path = Path(os.getenv("CODROID_SETTINGS_PATH", str(DEFAULT_SETTINGS_PATH)))
        if not settings_path.exists():
            return {}
        try:
            raw = json.loads(settings_path.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("Failed to read settings %s: %s", settings_path, exc)
            return {}
        if not isinstance(raw, dict):
            return {}
        return {key: value for key, value in raw.items() if value is not None}

    @field_validator("ui_language")
    @classmethod
    def _validate_language(cls, value: str) -> str:
        normalized = (value or "en").strip().lower()
        if normalized.startswith("uk"):
            return "uk"
        return "en"

    @field_validator("camera_source")
    @classmethod
    def _validate_camera_source(cls, value: str) -> str:
        normalized = (value or "oak").strip().lower()
        if normalized in {"host", "webcam", "opencv", "local", "generic", "usb", "v4l2", "azure-kinect", "azure_kinect"}:
            return "host"
        if normalized == "fake":
            return "fake"
        return "oak"

    @field_validator("gripper_type")
    @classmethod
    def _validate_gripper_type(cls, value: str) -> str:
        normalized = (value or "auto").strip().lower()
        if normalized == "2fg7":
            normalized = "twofg7"
        return normalized if normalized in SUPPORTED_GRIPPER_TYPES else "auto"

    @field_validator("camera_safety_zone")
    @classmethod
    def _validate_safety_zone(cls, value: str) -> str:
        normalized = (value or "workspace").strip().lower()
        return normalized if normalized in {"workspace", "full_frame"} else "workspace"

    @field_validator("detection_frame_skip", "auto_separation_max_attempts")
    @classmethod
    def _positive_int(cls, value: int) -> int:
        return max(int(value), 1)

    @field_validator("video_jpeg_quality")
    @classmethod
    def _jpeg_quality(cls, value: int) -> int:
        return min(max(int(value), 30), 95)

    @field_validator("video_max_width", "video_snapshot_width")
    @classmethod
    def _video_width(cls, value: int) -> int:
        return min(max(int(value), 160), 3840)

    @field_validator("video_max_fps")
    @classmethod
    def _video_fps(cls, value: float) -> float:
        return min(max(float(value), 1.0), 60.0)

    @property
    def default_robot_uri(self) -> str:
        host = self.robot_host or os.getenv("CODROID_HOST", "192.168.101.100")
        raw_port = self.robot_port
        if raw_port is None:
            raw_port = int(os.getenv("CODROID_ROBOT_PORT", "9000"))
        port = int(raw_port)
        return f"ws://{host}:{port}/"


class SettingsRepository:
    """Atomic JSON persistence for editable settings."""

    def __init__(self, path: Optional[Path] = None) -> None:
        self.path = path or Path(os.getenv("CODROID_SETTINGS_PATH", str(DEFAULT_SETTINGS_PATH)))

    def save(self, settings: AppSettings) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = settings.model_dump_json(indent=2)
        fd, tmp_name = tempfile.mkstemp(
            prefix=f".{self.path.name}.",
            suffix=".tmp",
            dir=str(self.path.parent),
            text=True,
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp_name, self.path)
        finally:
            if os.path.exists(tmp_name):
                os.unlink(tmp_name)
