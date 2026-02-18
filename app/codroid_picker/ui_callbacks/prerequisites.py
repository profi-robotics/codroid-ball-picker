"""Prerequisites callback for tracking system readiness state.

This module provides a callback that monitors various system prerequisites
(camera calibration, workspace definition, robot calibration, etc.) and
exposes them via a dcc.Store for other callbacks to consume.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from dash import Input, Output

from app.codroid_picker.app_instance import registry

if TYPE_CHECKING:
    from dash import Dash


def register_prerequisites_callbacks(app: "Dash") -> None:
    """Register the prerequisites state update callback."""
    @app.callback(
        Output("prerequisites-state", "data"),
        Input("monitor-interval", "n_intervals"),
        Input("camera-connection-state", "data"),
        Input("gripper-connection-state", "data"),
        Input("robot-connection-state", "data"),
        Input("robot-control-state", "data"),
        Input("model-status", "children"),
        Input("workspace-action-status", "children"),
        prevent_initial_call=True,
    )
    def _update_prerequisites(
        _monitor: int,
        camera_state: Optional[dict],
        gripper_state: Optional[dict],
        robot_state: Optional[dict],
        robot_control_state: Optional[dict],
        _model_status: Optional[str],
        _workspace_status: Optional[str],
    ) -> dict:
        """Update prerequisites state based on current system status.

        This callback periodically checks the current state of various
        system prerequisites and returns a dict indicating which are ready.
        Other callbacks can use this state to enable/disable features.
        """
        import app.codroid_picker as codroid_picker_module

        session = registry().session
        camera_live = (camera_state or {}).get("status") == "live"
        robot_connected = (
            (robot_state or {}).get("status") == "online"
            and not codroid_picker_module._robot_control_released(robot_control_state)
        )

        result = {
            "camera_live": camera_live,
            "camera_calibrated": session.calibration_bundle is not None,
            "workspace_defined": session.workspace_definition is not None,
            "robot_calibrated": session.robot_calibration_min_z is not None,
            "robot_connected": robot_connected,
            "model_loaded": session.model_loaded,
            "gripper_ready": (gripper_state or {}).get("status") == "online",
            "drop_target_defined": session.drop_target_mm is not None,
        }

        # LOGGER.info(f"Prerequisites callback fired, returning: {result}")

        return result
