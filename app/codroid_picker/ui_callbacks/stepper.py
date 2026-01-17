"""Calibration stepper callbacks.

This module contains Dash callbacks for:
- Calibration stepper navigation (prev/next)
- Calibration menu open/close state
- Calibration metadata display
"""

from typing import Any, Optional, Tuple

import dash_mantine_components as dmc
from dash import Dash, Input, Output, State, callback_context, no_update


def register_stepper_callbacks(app: Dash) -> None:
    """Register calibration stepper callbacks with the Dash app."""

    @app.callback(
        Output("calibration-stepper", "active"),
        Input("step-prev", "n_clicks"),
        Input("step-next", "n_clicks"),
        Input("use-existing-calibration", "n_clicks"),
        State("calibration-stepper", "active"),
    )
    def _update_stepper(
        prev_clicks: int,
        next_clicks: int,
        use_saved_calib: int,
        active: Optional[int],
    ) -> int:
        """Handle stepper navigation prev/next and skip to robot calibration."""
        current = active or 0
        trigger = callback_context.triggered_id
        if trigger == "step-next":
            return min(current + 1, 1)
        if trigger == "step-prev":
            return max(current - 1, 0)
        if trigger == "use-existing-calibration":
            return 1  # Advance to robot calibration step
        return current

    @app.callback(
        Output("calibration-stepper-body", "style"),
        Output("calibration-loaded-body", "style"),
        Output("calibration-nav-actions", "style"),
        Input("calibration-menu-open-store", "data"),
    )
    def _toggle_calibration_menu(
        opened: bool,
    ) -> Tuple[dict[str, str], dict[str, str], dict[str, str]]:
        """Toggle visibility of calibration stepper vs loaded summary."""
        stepper_style = {} if opened else {"display": "none"}
        loaded_style = {} if not opened else {"display": "none"}
        return stepper_style, loaded_style, stepper_style

    @app.callback(
        Output("calibration-menu-open-store", "data"),
        Input("calibration-reset-button", "n_clicks"),
        Input("calibration-progress-store", "data"),
        Input("robot-calibration-state", "data"),
        State("calibration-menu-open-store", "data"),
    )
    def _sync_calibration_menu_state(
        reset_clicks: Optional[int],
        _progress: dict,
        _robot_state: dict,
        current_open: Optional[bool],
    ) -> bool | None:
        """Sync calibration menu open state based on calibration completeness."""
        import app.codroid_picker as codroid_picker_module

        trigger = callback_context.triggered_id
        if trigger == "calibration-reset-button" and reset_clicks:
            codroid_picker_module._reset_calibration_assets()
            return True
        desired = not codroid_picker_module._calibration_data_complete()
        if current_open is None or desired == current_open:
            return no_update
        return desired

    @app.callback(
        Output("calibration-metadata", "children"),
        Input("calibration-menu-open-store", "data"),
        Input("calibration-progress-store", "data"),
        Input("robot-calibration-state", "data"),
    )
    def _update_calibration_metadata(
        opened: Optional[bool],
        _progress: dict,
        _robot_state: dict,
    ) -> list[Any] | str:
        """Update calibration metadata display when calibration is complete."""
        import app.codroid_picker as codroid_picker_module

        if not codroid_picker_module._calibration_data_complete():
            return ""
        entries = codroid_picker_module._calibration_metadata_entries()
        if not entries:
            return dmc.Text("Calibration metadata unavailable.", size="xs", c="dimmed")
        return [dmc.Text(entry, size="xs", c="dimmed") for entry in entries]
