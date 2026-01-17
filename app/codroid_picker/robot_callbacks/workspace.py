"""Workspace calibration callbacks - marker detection and workspace definition."""

from typing import Optional

from dash import callback_context, Dash, Input, no_update, Output, State


def register_workspace_callbacks(app: Dash) -> None:
    """Register workspace-related callbacks."""

    @app.callback(
        Output("workspace-action-status", "children"),
        Input("compute-workspace", "n_clicks"),
        Input("use-existing-workspace", "n_clicks"),
        Input("clear-workspace", "n_clicks"),
        State("settings-marker-a", "value"),
        State("settings-marker-b", "value"),
        State("settings-marker-size", "value"),
        prevent_initial_call=True,
    )
    def _handle_workspace_action(
        _clicks: int,
        _use_existing: int,
        _clear: int,
        marker_a: Optional[float],
        marker_b: Optional[float],
        marker_size: Optional[float],
    ) -> str:
        """Handle workspace compute/load/clear actions."""
        import app.codroid_picker as codroid_picker_module

        SESSION = codroid_picker_module.SESSION
        _safe_int = codroid_picker_module._safe_int
        _safe_float = codroid_picker_module._safe_float
        DEFAULT_BUNDLE_PATH = codroid_picker_module.DEFAULT_BUNDLE_PATH
        DEFAULT_WORKSPACE_PATH = codroid_picker_module.DEFAULT_WORKSPACE_PATH
        load_calibration_bundle = codroid_picker_module.load_calibration_bundle
        load_workspace_definition = codroid_picker_module.load_workspace_definition
        compute_workspace_from_latest = codroid_picker_module.compute_workspace_from_latest
        _clear_workspace_definition = codroid_picker_module._clear_workspace_definition

        marker_ids = (
            _safe_int(marker_a, SESSION.marker_ids[0]),
            _safe_int(marker_b, SESSION.marker_ids[1]),
        )
        marker_size_mm = _safe_float(
            marker_size, SESSION.marker_size_mm)
        try:
            # Load calibration bundle from disk if not already in memory
            if SESSION.calibration_bundle is None and DEFAULT_BUNDLE_PATH.exists():
                load_calibration_bundle(DEFAULT_BUNDLE_PATH)
            trigger = callback_context.triggered_id
            if trigger == "clear-workspace":
                _clear_workspace_definition(DEFAULT_WORKSPACE_PATH)
                return "Workspace definition cleared; robot movement unrestricted."
            if trigger == "use-existing-workspace":
                ws = load_workspace_definition(DEFAULT_WORKSPACE_PATH)
                return (
                    f"Workspace loaded from {DEFAULT_WORKSPACE_PATH}: "
                    f"{ws.width_mm:.1f} x {ws.height_mm:.1f} mm"
                )
            ws = compute_workspace_from_latest(marker_ids, marker_size_mm)
            return (
                f"Workspace computed and saved ({DEFAULT_WORKSPACE_PATH}): "
                f"{ws.width_mm:.1f} x {ws.height_mm:.1f} mm"
            )
        except Exception as exc:
            return f"Workspace error: {exc}"

    @app.callback(
        Output("plate-action-status", "children"),
        Input("camera-interval", "n_intervals"),
        prevent_initial_call=True,
    )
    def _handle_plate_actions(
        _ticks: int,
    ) -> str:
        """Handle continuous plate detection updates (currently no-op)."""
        return no_update
