"""Robot callbacks package - organized by functional area.

This package contains robot-related Dash callbacks split into logical modules:
- auto_operation: Auto pick & place operation start/stop and state sync
- connection: Robot/gripper connection and power management
- error_handling: Error modal updates and response handlers
- rescue_mode: Manual robot recovery operations
- calibration: Robot calibration workflow and verification
- monitoring: Robot operation state monitoring and control
- state_sync: Robot state display and synchronization (position, errors, rescue mode)
- controls: Robot speed controls and safety stop
- gripper: Gripper status synchronization and orientation control
- detection: YOLO model detection parameters and display
- workspace: Workspace calibration and marker detection
- pick_place: Manual pick and place operations
- plate_test: Plate pick & place test
"""

from dash import Dash

from .auto_operation import register_auto_operation_callbacks
from .calibration import register_calibration_callbacks
from .connection import register_connection_callbacks
from .controls import register_controls_callbacks
from .detection import register_detection_callbacks
from .error_handling import register_error_handling_callbacks
from .gripper import register_gripper_callbacks
from .monitoring import register_monitoring_callbacks
from .pick_place import register_pick_place_callbacks
from .plate_test import register_plate_test_callbacks
from .rescue_mode import register_rescue_mode_callbacks
from .state_sync import register_state_sync_callbacks
from .workspace import register_workspace_callbacks


def register_robot_callbacks(app: Dash) -> None:
    """Register all robot-related callbacks with the app."""
    # Guard against duplicate registration
    if hasattr(app, '_robot_callbacks_registered'):
        return
    app._robot_callbacks_registered = True

    register_auto_operation_callbacks(app)
    register_connection_callbacks(app)
    register_error_handling_callbacks(app)
    register_rescue_mode_callbacks(app)
    register_calibration_callbacks(app)
    register_monitoring_callbacks(app)
    register_state_sync_callbacks(app)
    register_controls_callbacks(app)
    register_gripper_callbacks(app)
    register_detection_callbacks(app)
    register_workspace_callbacks(app)
    register_pick_place_callbacks(app)
    register_plate_test_callbacks(app)


__all__ = [
    "register_robot_callbacks",
    "register_auto_operation_callbacks",
    "register_connection_callbacks",
    "register_error_handling_callbacks",
    "register_rescue_mode_callbacks",
    "register_calibration_callbacks",
    "register_monitoring_callbacks",
    "register_state_sync_callbacks",
    "register_controls_callbacks",
    "register_gripper_callbacks",
    "register_detection_callbacks",
    "register_workspace_callbacks",
    "register_pick_place_callbacks",
    "register_plate_test_callbacks",
]
