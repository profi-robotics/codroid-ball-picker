"""
Dash calibration UI for the Codroid bring-up workflow.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

from dash import Dash, Input, Output, State, no_update

import app.settings as app_settings
from app.codroid_picker import bootstrap
from app.codroid_picker.camera_callbacks import register_camera_callbacks
from app.codroid_picker.gpu_utils import (
    format_gpu_status,
)
from app.codroid_picker.layout import build_layout
from app.codroid_picker.models import (
    _model_options,
    activate_model,
)
from app.codroid_picker.persistence import (
    _restore_persisted_state,
)
from app.codroid_picker.robot_callbacks import register_robot_callbacks
from app.codroid_picker.session import SESSION, GripperDevice  # noqa: F401 - used via globals()
# Import functions that were extracted to modules but are still accessed via codroid_picker_module
from app.codroid_picker.streaming import (
    MJPEG_STREAM,
    RAW_STREAM,
)
from app.codroid_picker.ui_callbacks import register_ui_callbacks
from app.codroid_picker.video_routes import register_video_routes

# noqa: F401 - The following imports are used dynamically via globals()
from app.codroid_picker.procedures import (  # noqa: F401
    _get_robot_session,
    _current_coordinate_id,
    _robot_to_workspace_mm,
    _robot_calibration_path,
    _robot_posture_ready,
    _ensure_coordinate5_active,
    _set_robot_guidance,
    _clear_robot_accuracy_guidance,
    _load_robot_calibration,
    _acknowledge_robot_error,
)
from app.codroid_picker.error_detection import (  # noqa: F401
    _detect_emergency_stop_status,
    _detect_overspeed_status,
    _detect_joint_protection_status,
    _detect_wrong_tool_status,
    _detect_drag_not_allowed_status,
    _detect_rescue_mode_status,
    _activate_robot_error,
    _error_snapshot,
    _update_robot_error_state,
)
from codroid_api import (  # noqa: F401
    CodroidAPI,
    RobotSession,
    DEFAULT_FLANGE_BUTTON_PORTS,
)
from app.codroid_picker.utils.safe_types import (  # noqa: F401
    safe_int as _safe_int,
    safe_float as _safe_float,
    safe_path as _safe_path,
)
from app.codroid_picker.session import TwoFG7Gripper  # noqa: F401
from app.codroid_picker.robot_utils import (  # noqa: F401
    _probe_robot,
    _with_robot_api,
    _ensure_robot_coord5_for_uri,
    _log_move_target,
)
from app.codroid_picker.transforms import (  # noqa: F401
    _coord5_to_workspace_mm,
    _pose_from_coord5_mm,
)
from app.codroid_picker.persistence import (  # noqa: F401
    _save_drop_target,
)
from app.codroid_picker.state import (  # noqa: F401
    _begin_operation,
    _finish_operation,
    _operation_snapshot,
    _pause_operation,
    _update_operation_message,
    resume_operation as _resume_operation,
)
from app.codroid_picker.calibration_menu import (  # noqa: F401
    _calibration_data_complete,
    _calibration_metadata_entries,
    _reset_calibration_assets,
    _camera_calibration_path,
    _refresh_calibration_menu_visibility,
)
from app.codroid_picker.gripper_utils import (  # noqa: F401
    refresh_gripper_constants,
    _record_gripper_sync,
    _recent_gripper_sync,
    _restore_calibration_gripper_width,
    _resolve_gripper_limits,
    _command_gripper,
)
from app.codroid_picker.models import (  # noqa: F401
    save_uploaded_model,
    _list_capture_frames,
)
from app.codroid_picker.camera_calibration import (  # noqa: F401
    get_latest_frame,
    load_calibration_bundle,
    load_workspace_definition,
    compute_workspace_from_latest,
    _clear_workspace_definition,
    capture_color_frame,
    capture_depth_snapshot,
    compute_calibration_bundle,
)
from app.codroid_picker.auto_operation import (  # noqa: F401
    _format_detection_list,
    _run_detection,
)
from app.codroid_picker.overlay_tasks import (  # noqa: F401
    _select_overlay_task,
    _run_charuco_task,
    _run_workspace_task,
    _run_plate_task,
    _run_detection_task,
    _run_drop_marker_task,
)
from app.settings import (  # noqa: F401
    _save_settings,
    _commit_settings,
    set_asset_paths,
    reload_defaults,
)

LOGGER = logging.getLogger(__name__)

# Make SESSION and LOGGER available as module attributes for robot callbacks
# These need to be assigned after imports to avoid circular import issues


try:
    import pynvml
    NVML_AVAILABLE = True
except ImportError:
    NVML_AVAILABLE = False

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

DEFAULT_BOARD_PATH = app_settings.DEFAULT_BOARD_PATH
DEFAULT_PLATE_PATH = app_settings.DEFAULT_PLATE_PATH
DEFAULT_BUNDLE_PATH = app_settings.DEFAULT_BUNDLE_PATH
DEFAULT_WORKSPACE_PATH = app_settings.DEFAULT_WORKSPACE_PATH
DEFAULT_ROBOT_CALIBRATION_PATH = app_settings.DEFAULT_ROBOT_CALIBRATION_PATH
DEFAULT_CAPTURE_DIR = app_settings.DEFAULT_CAPTURE_DIR
DEFAULT_RESOLUTION = app_settings.DEFAULT_RESOLUTION
DEFAULT_FPS = app_settings.DEFAULT_FPS
DEFAULT_MODEL_DIR = app_settings.DEFAULT_MODEL_DIR
DEFAULT_DROP_MARKER_ID = app_settings.DEFAULT_DROP_MARKER_ID
DROP_TARGET_PATH = app_settings.DROP_TARGET_PATH
SETTINGS = app_settings.SETTINGS
CODROID_DEFAULTS = app_settings.CODROID_DEFAULTS
DEFAULT_ROBOT_URI = app_settings.DEFAULT_ROBOT_URI
GRIPPER_DEFAULTS = app_settings.GRIPPER_DEFAULTS
DEFAULT_GRIPPER_IP = app_settings.DEFAULT_GRIPPER_IP
GRIPPER_PROFILE_OPTIONS = app_settings.GRIPPER_PROFILE_OPTIONS
GRIPPER_PROFILE_KEYS = {option["value"] for option in GRIPPER_PROFILE_OPTIONS}
GRIPPER_PROFILE = SETTINGS.gripper_profile
GRIPPER_FORCE_MIN = GRIPPER_PROFILE.force_min
GRIPPER_FORCE_MAX = GRIPPER_PROFILE.force_max
GRIPPER_FORCE_DEFAULT = GRIPPER_PROFILE.force_default
GRIPPER_SPEED_DEFAULT = GRIPPER_PROFILE.speed_default
GRIPPER_SPEED_MIN = GRIPPER_PROFILE.speed_min
GRIPPER_CALIBRATION_WIDTH_MM = GRIPPER_PROFILE.calibration_width
GRIPPER_OPEN_WIDTH_DEFAULT = GRIPPER_PROFILE.open_width_default
GRIPPER_CLOSE_WIDTH_DEFAULT = GRIPPER_PROFILE.close_width_default
ROBOT_COORDINATE_ID = SETTINGS.robot_coordinate_id


ROOT_DIR = Path(__file__).resolve().parent.parent
ASSETS_DIR = ROOT_DIR / "assets"

# Guard to prevent double initialization during hot-reload
_INITIALIZED = False


def _initialize_app() -> None:
    """Initialize the app dependencies (called once per process)."""
    global _INITIALIZED
    if _INITIALIZED:
        return
    _INITIALIZED = True

    # Initialize all submodules with their dependencies
    bootstrap.initialize_modules(
        # Core state objects
        session=SESSION,
        settings=SETTINGS,
        gripper_defaults=GRIPPER_DEFAULTS,
        robot_coordinate_id=ROBOT_COORDINATE_ID,
        # Path constants
        drop_target_path=DROP_TARGET_PATH,
        bundle_path=DEFAULT_BUNDLE_PATH,
        workspace_path=DEFAULT_WORKSPACE_PATH,
        robot_calibration_path=DEFAULT_ROBOT_CALIBRATION_PATH,
        model_dir=DEFAULT_MODEL_DIR,
        default_robot_uri=DEFAULT_ROBOT_URI,
        # Stream references
        mjpeg_stream=MJPEG_STREAM,
        raw_stream=RAW_STREAM,
        # Logger
        logger=LOGGER,
        # Component builder dependencies
        gpu_status_fn=format_gpu_status,
        model_options_fn=_model_options,
    )

    # Restore persisted state on module load
    _restore_persisted_state()


def reset_initialization() -> None:
    """Reset the initialization flag for hot-reload.

    This allows the app to be re-initialized after cleanup.
    Called by the cleanup handler in run_codroid_picker.py.
    """
    global _INITIALIZED
    _INITIALIZED = False


# Run initialization when module loads (guarded against double execution)
_initialize_app()

# Make SESSION and LOGGER available as module attributes for robot callbacks
# This allows callbacks to access them via codroid_picker_module.SESSION/LOGGER
# Must be after imports to avoid circular dependency issues
current_module = sys.modules[__name__]

# Expose commonly accessed variables and functions as module attributes
# This supports the lazy import pattern used in robot callbacks
globals_to_expose = [
    # Core objects
    'SESSION', 'LOGGER', 'SETTINGS',

    # Constants
    'ROBOT_COORDINATE_ID', 'DEFAULT_ROBOT_URI', 'DEFAULT_FPS', 'DEFAULT_RESOLUTION',
    'GRIPPER_FORCE_DEFAULT', 'GRIPPER_FORCE_MAX', 'GRIPPER_SPEED_DEFAULT',
    'GRIPPER_OPEN_WIDTH_DEFAULT', 'GRIPPER_CLOSE_WIDTH_DEFAULT',
    'DEFAULT_FLANGE_BUTTON_PORTS', 'GRIPPER_PROFILE_KEYS',
    'CODROID_DEFAULTS', 'DEFAULT_GRIPPER_IP', 'GRIPPER_DEFAULTS',
    'no_update',  # From Dash

    # Path constants
    'DEFAULT_BUNDLE_PATH', 'DEFAULT_WORKSPACE_PATH', 'DEFAULT_ROBOT_CALIBRATION_PATH',
    'DEFAULT_MODEL_DIR',

    # Helper functions
    '_safe_float', '_safe_int', '_safe_path', '_model_options',
    '_probe_robot', '_with_robot_api', '_get_robot_session',
    '_current_coordinate_id', '_ensure_robot_coord5_for_uri',
    '_coord5_to_workspace_mm', '_robot_to_workspace_mm',
    '_pose_from_coord5_mm',
    '_save_drop_target', '_save_settings', '_commit_settings',
    '_begin_operation', '_finish_operation', 'get_latest_frame',
    '_log_move_target', '_ensure_coordinate5_active',
    '_set_robot_guidance', '_clear_robot_accuracy_guidance',
    '_load_robot_calibration', '_acknowledge_robot_error',
    '_pause_operation', '_update_operation_message', '_resume_operation',

    # Calibration functions
    '_calibration_data_complete', '_calibration_metadata_entries',
    '_reset_calibration_assets', '_camera_calibration_path',
    '_robot_calibration_path', '_refresh_calibration_menu_visibility',
    'refresh_gripper_constants', '_record_gripper_sync', '_recent_gripper_sync',
    '_restore_calibration_gripper_width', '_resolve_gripper_limits',
    '_command_gripper',
    '_operation_snapshot', '_robot_posture_ready',

    # Model and GPU functions
    'activate_model', 'save_uploaded_model', 'format_gpu_status',

    # Workspace functions
    'load_calibration_bundle', 'load_workspace_definition',
    'compute_workspace_from_latest', '_clear_workspace_definition',
    '_list_capture_frames', 'capture_color_frame', 'capture_depth_snapshot',
    'compute_calibration_bundle',

    # Robot state functions
    '_detect_emergency_stop_status', '_detect_overspeed_status',
    '_detect_joint_protection_status', '_detect_wrong_tool_status',
    '_detect_drag_not_allowed_status', '_detect_rescue_mode_status',
    '_activate_robot_error', '_error_snapshot', '_update_robot_error_state',

    # Auto operation functions
    '_format_detection_list', '_run_detection',

    # Overlay tasks
    '_select_overlay_task', '_run_charuco_task', '_run_workspace_task',
    '_run_plate_task', '_run_detection_task', '_run_drop_marker_task',

    # API classes
    'CodroidAPI', 'RobotSession', 'GripperDevice', 'TwoFG7Gripper',

    # Settings functions
    'set_asset_paths', 'reload_defaults',
]

# Dynamically expose globals as module attributes
missing_globals = []
for name in globals_to_expose:
    if name in globals():
        setattr(current_module, name, globals()[name])
    else:
        missing_globals.append(name)

# Log missing globals for debugging
if missing_globals:
    print(
        f"WARNING: Missing globals not exposed as module attributes: {missing_globals}")


def register_callbacks(app: Dash) -> None:
    # Register callbacks from extracted modules
    register_robot_callbacks(app)
    register_camera_callbacks(app)
    register_ui_callbacks(app)

    # Client-side toggle for the Setup drawer (faster UI response).
    app.clientside_callback(
        """
        function(openClicks, closeClicks, opened) {
            const ctx = dash_clientside.callback_context;
            if (!ctx.triggered || ctx.triggered.length === 0) {
                return opened;
            }
            const trigger = ctx.triggered[0].prop_id.split('.')[0];
            if (trigger === "open-setup") {
                return true;
            }
            if (trigger === "setup-close") {
                return false;
            }
            // Fallback: compare counts.
            const o = openClicks || 0;
            const c = closeClicks || 0;
            if (o > c) return true;
            if (c > o) return false;
            return opened;
        }
        """,
        Output("setup-drawer", "opened"),
        Input("open-setup", "n_clicks"),
        Input("setup-close", "n_clicks"),
        State("setup-drawer", "opened"),
        prevent_initial_call=False,
    )

    # Auto-refresh camera image (enabled only when camera is live)
    # (Client refresh removed; MJPEG stream handles updates)


def create_app() -> Dash:
    app = Dash(
        __name__,
        assets_folder=str(ASSETS_DIR),
        suppress_callback_exceptions=True,
        update_title=None,
    )
    app.title = "Estun Codroid Pick-and-Place"
    app._favicon = "favicon.ico"
    app.layout = build_layout()
    register_callbacks(app)

    # Register Flask video streaming routes
    register_video_routes(app)

    return app


app = create_app()
server = app.server


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    app.run(debug=True)
