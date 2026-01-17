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
        Input("camera-interval", "n_intervals"),
        Input("monitor-interval", "n_intervals"),
        Input("gripper-connection-state", "data"),
    )
    def _update_prerequisites(
        _n: int,
        _monitor: int,
        gripper_state: Optional[dict],
    ) -> dict:
        """Update prerequisites state based on current system status.

        This callback periodically checks the current state of various
        system prerequisites and returns a dict indicating which are ready.
        Other callbacks can use this state to enable/disable features.
        """
        session = registry().session
        return {
            "camera_calibrated": session.calibration_bundle is not None,
            "workspace_defined": session.workspace_definition is not None,
            "robot_calibrated": session.robot_calibration_min_z is not None,
            "model_loaded": session.model_loaded,
            "gripper_ready": (gripper_state or {}).get("status") == "online",
            "drop_target_defined": session.drop_target_mm is not None,
        }
