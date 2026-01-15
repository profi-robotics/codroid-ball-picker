"""
Application configuration helpers stored separately from the UI code.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Optional, Tuple

from codroid_api import CodroidSettings
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
_BASE_MODEL_DIR = Path("data/models")
_BASE_DROP_MARKER_ID = 30

DEFAULT_PLATE_PATH = _BASE_PLATE_LAYOUT_PATH
DEFAULT_CAPTURE_DIR = _BASE_CAPTURE_DIR
DEFAULT_RESOLUTION = _BASE_CAMERA_RESOLUTION
DEFAULT_FPS = _BASE_CAMERA_FPS
DEFAULT_MODEL_DIR = _BASE_MODEL_DIR
DEFAULT_DROP_MARKER_ID = _BASE_DROP_MARKER_ID

DROP_TARGET_PATH = Path("data/drop_target.json")
SETTINGS_PATH = Path("data/app_settings.json")


class AppSettings(BaseSettings):
    """Application configuration persisted to disk and overrideable via .env."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="CODROID_",
    )

    # Robot connection settings (host + ports)
    robot_host: Optional[str] = None
    robot_port: Optional[int] = None
    user_port: Optional[int] = None

    # Marker and tracking details that define the workspace
    marker_ids: Tuple[int, int] = (1, 2)
    marker_size_mm: float = 40.0

    # Gripper and model metadata / debug helpers
    gripper_ip: Optional[str] = None
    last_model: Optional[str] = None
    active_model: Optional[str] = None
    drop_target: Optional[Tuple[float, float]] = None

    # Asset paths that can be redirected via settings
    calibration_bundle_path: Path = DEFAULT_BUNDLE_PATH
    workspace_path: Path = DEFAULT_WORKSPACE_PATH
    robot_calibration_path: Path = DEFAULT_ROBOT_CALIBRATION_PATH

    plate_layout_path: Path = _BASE_PLATE_LAYOUT_PATH
    capture_dir: Path = _BASE_CAPTURE_DIR
    camera_resolution: Tuple[int, int] = _BASE_CAMERA_RESOLUTION
    camera_fps: float = _BASE_CAMERA_FPS
    model_dir: Path = _BASE_MODEL_DIR
    drop_marker_id: int = _BASE_DROP_MARKER_ID

    # UI tracking controls
    tracking_mode: str = "move"
    segment_min: float = 20.0
    segment_max: float = 300.0
    segment_fraction: float = 0.8

    @classmethod
    def settings_customise_sources(
        cls,
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
    def _json_settings_source(cls, settings):
        if not SETTINGS_PATH.exists():
            return {}
        try:
            return json.loads(SETTINGS_PATH.read_text())
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
        "compute_box_ip": SETTINGS.gripper_ip or os.getenv("ONROBOT_CB_IP", "192.168.101.180"),
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
