"""Calibration stepper callbacks.

This module contains Dash callbacks for:
- Calibration stepper navigation (prev/next)
- Calibration menu open/close state
- Calibration metadata display
"""

from typing import Any, Optional, Tuple

import dash_mantine_components as dmc
from dash import Dash, Input, Output, State, callback_context, no_update
from app.codroid_picker.i18n import t


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
        Input("calibration-ui-interval", "n_intervals"),
        State("calibration-menu-open-store", "data"),
    )
    def _toggle_calibration_menu(
        _ticks: int,
        opened: bool,
    ) -> Tuple[dict[str, str], dict[str, str], dict[str, str]]:
        """Toggle visibility of calibration stepper vs loaded summary."""
        stepper_style = {} if opened else {"display": "none"}
        loaded_style = {} if not opened else {"display": "none"}
        return stepper_style, loaded_style, stepper_style

    @app.callback(
        Output("calibration-menu-open-store", "data", allow_duplicate=True),
        Input("calibration-progress-store", "data"),
        Input("robot-calibration-state", "data"),
        State("calibration-menu-open-store", "data"),
        prevent_initial_call=True,
    )
    def _sync_calibration_menu_state(
        _progress: dict,
        _robot_state: dict,
        current_open: Optional[bool],
    ) -> bool | None:
        """Sync calibration menu open state based on calibration completeness."""
        import app.codroid_picker as codroid_picker_module

        desired = not codroid_picker_module._calibration_data_complete()
        if current_open is None or desired == current_open:
            return no_update
        return desired

    @app.callback(
        Output("calibration-menu-open-store", "data", allow_duplicate=True),
        Input("calibration-reset-button", "n_clicks"),
        prevent_initial_call=True,
    )
    def _reset_calibration_menu(_reset_clicks: Optional[int]) -> bool:
        """Reset calibration assets and reopen the calibration menu."""
        import app.codroid_picker as codroid_picker_module

        if not _reset_clicks:
            return no_update
        codroid_picker_module._reset_calibration_assets()
        return True

    @app.callback(
        Output("calibration-metadata", "children"),
        Input("calibration-ui-interval", "n_intervals"),
        State("calibration-menu-open-store", "data"),
        State("calibration-progress-store", "data"),
        State("robot-calibration-state", "data"),
    )
    def _update_calibration_metadata(
        _ticks: int,
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
            return dmc.Text(t("Calibration metadata unavailable."), size="xs", c="dimmed")
        return [dmc.Text(entry, size="xs", c="dimmed") for entry in entries]
