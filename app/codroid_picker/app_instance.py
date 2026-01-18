"""Application instance and registry for the calibration UI.

This module provides a central registry for shared application state,
eliminating circular imports between callback modules and the main UI.
All callback modules should import from here instead of late-importing
codroid_picker.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from dash import Dash
    from codroid_api import RobotSession
    from onrobot.twofg import TWOFG

LOGGER = logging.getLogger(__name__)


class AppRegistry:
    """Singleton registry for shared application state.

    This eliminates circular imports by providing a single source of truth
    for accessing the Dash app, robot session, gripper, settings, streams,
    and path constants.

    Usage:
        from app.codroid_picker.app_instance import registry

        # Access properties
        session = registry().session
        settings = registry().settings

        # Check initialization
        if registry().is_initialized:
            ...
    """

    _instance: Optional["AppRegistry"] = None

    def __new__(cls) -> "AppRegistry":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
            cls._instance._fully_initialized = False
        return cls._instance

    def __init__(self) -> None:
        if self._initialized:
            return
        self._initialized = True
        self._fully_initialized = False

        # Dash app
        self._app: Optional[Dash] = None

        # Core objects
        self._session: Any = None
        self._settings: Any = None
        self._robot_session: Optional[RobotSession] = None
        self._gripper: Optional[TWOFG] = None
        self._logger: logging.Logger = LOGGER

        # Constants
        self._gripper_defaults: dict = {}
        self._robot_coordinate_id: int = 0
        self._default_robot_uri: str = ""

        # Path constants
        self._drop_target_path: Optional[Path] = None
        self._bundle_path: Optional[Path] = None
        self._workspace_path: Optional[Path] = None
        self._robot_calibration_path: Optional[Path] = None
        self._model_dir: Optional[Path] = None

        # Streams
        self._mjpeg_stream: Any = None
        self._raw_stream: Any = None

    # -------------------------------------------------------------------------
    # Initialization
    # -------------------------------------------------------------------------

    def initialize(
        self,
        *,
        session: Any,
        settings: Any,
        gripper_defaults: dict,
        robot_coordinate_id: int,
        default_robot_uri: str,
        drop_target_path: Path,
        bundle_path: Path,
        workspace_path: Path,
        robot_calibration_path: Path,
        model_dir: Path,
        mjpeg_stream: Any,
        raw_stream: Any,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        """Initialize the registry with all required dependencies.

        This should be called once during application startup.

        Args:
            session: The SESSION object from codroid_picker.session.
            settings: The SETTINGS object from app.settings.
            gripper_defaults: Dict of gripper default values.
            robot_coordinate_id: Coordinate ID for calibrated operations.
            default_robot_uri: Default robot URI string.
            drop_target_path: Path to drop target JSON.
            bundle_path: Path to calibration bundle.
            workspace_path: Path to workspace definition.
            robot_calibration_path: Path to robot calibration JSON.
            model_dir: Path to model directory.
            mjpeg_stream: MotionJPEGStream instance for overlayed frames.
            raw_stream: MotionJPEGStream instance for raw frames.
            logger: Optional logger instance.
        """
        self._session = session
        self._settings = settings
        self._gripper_defaults = gripper_defaults
        self._robot_coordinate_id = robot_coordinate_id
        self._default_robot_uri = default_robot_uri
        self._drop_target_path = drop_target_path
        self._bundle_path = bundle_path
        self._workspace_path = workspace_path
        self._robot_calibration_path = robot_calibration_path
        self._model_dir = model_dir
        self._mjpeg_stream = mjpeg_stream
        self._raw_stream = raw_stream
        if logger:
            self._logger = logger
        self._fully_initialized = True

    @property
    def is_initialized(self) -> bool:
        """Check if the registry has been fully initialized."""
        return self._fully_initialized

    def _check_initialized(self) -> None:
        """Raise if registry is not initialized."""
        if not self._fully_initialized:
            raise RuntimeError(
                "AppRegistry not initialized. Call registry().initialize() first."
            )

    # -------------------------------------------------------------------------
    # Dash App
    # -------------------------------------------------------------------------

    @property
    def app(self) -> Optional["Dash"]:
        """Get the Dash application instance."""
        return self._app

    def set_app(self, app: "Dash") -> None:
        """Set the Dash application instance."""
        self._app = app

    # -------------------------------------------------------------------------
    # Core Objects
    # -------------------------------------------------------------------------

    @property
    def session(self) -> Any:
        """Get the SESSION object."""
        self._check_initialized()
        return self._session

    @property
    def settings(self) -> Any:
        """Get the SETTINGS object."""
        self._check_initialized()
        return self._settings

    @property
    def logger(self) -> logging.Logger:
        """Get the logger instance."""
        return self._logger

    # -------------------------------------------------------------------------
    # Robot Session
    # -------------------------------------------------------------------------

    @property
    def robot_session(self) -> Optional["RobotSession"]:
        """Get the robot session instance."""
        return self._robot_session

    def set_robot_session(self, session: "RobotSession") -> None:
        """Set the robot session instance."""
        self._robot_session = session

    def get_or_create_robot_session(self) -> "RobotSession":
        """Get or create the robot session.

        This lazy-initializes the session if not already set,
        using defaults from settings.
        """
        if self._robot_session is None:
            from codroid_api import (
                CodroidSettings,
                RobotSession,
                DEFAULT_FLANGE_BUTTON_PORT,
                DEFAULT_FLANGE_BUTTON_PORTS,
            )

            self._robot_session = RobotSession(
                CodroidSettings(),
                flange_button_port=DEFAULT_FLANGE_BUTTON_PORT,
                flange_button_ports=DEFAULT_FLANGE_BUTTON_PORTS,
            )
        return self._robot_session

    # -------------------------------------------------------------------------
    # Gripper
    # -------------------------------------------------------------------------

    @property
    def gripper(self) -> Optional["TWOFG"]:
        """Get the gripper instance."""
        return self._gripper

    def set_gripper(self, gripper: Optional["TWOFG"]) -> None:
        """Set the gripper instance."""
        self._gripper = gripper

    # -------------------------------------------------------------------------
    # Constants
    # -------------------------------------------------------------------------

    @property
    def gripper_defaults(self) -> dict:
        """Get gripper default configuration."""
        self._check_initialized()
        return self._gripper_defaults

    @property
    def robot_coordinate_id(self) -> int:
        """Get the robot coordinate ID for calibrated operations."""
        self._check_initialized()
        return self._robot_coordinate_id

    @property
    def default_robot_uri(self) -> str:
        """Get the default robot URI."""
        self._check_initialized()
        return self._default_robot_uri

    # -------------------------------------------------------------------------
    # Path Constants
    # -------------------------------------------------------------------------

    @property
    def drop_target_path(self) -> Path:
        """Get the drop target JSON path."""
        self._check_initialized()
        return self._drop_target_path

    @property
    def bundle_path(self) -> Path:
        """Get the calibration bundle path."""
        self._check_initialized()
        return self._bundle_path

    @property
    def workspace_path(self) -> Path:
        """Get the workspace definition path."""
        self._check_initialized()
        return self._workspace_path

    @property
    def robot_calibration_path(self) -> Path:
        """Get the robot calibration JSON path."""
        self._check_initialized()
        return self._robot_calibration_path

    @property
    def model_dir(self) -> Path:
        """Get the model directory path."""
        self._check_initialized()
        return self._model_dir

    # -------------------------------------------------------------------------
    # Streams
    # -------------------------------------------------------------------------

    @property
    def mjpeg_stream(self) -> Any:
        """Get the MJPEG stream for overlayed frames."""
        self._check_initialized()
        return self._mjpeg_stream

    @property
    def raw_stream(self) -> Any:
        """Get the raw frame stream."""
        self._check_initialized()
        return self._raw_stream


# Module-level singleton accessor
def registry() -> AppRegistry:
    """Get the application registry singleton.

    This is the primary way to access shared state across the application.

    Returns:
        The AppRegistry singleton instance.

    Example:
        from app.codroid_picker.app_instance import registry

        session = registry().session
        settings = registry().settings
    """
    return AppRegistry()

