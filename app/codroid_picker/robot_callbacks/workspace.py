"""Workspace calibration callbacks - marker detection and workspace definition."""

from typing import Optional

from dash import callback_context, Dash, Input, Output, State
from app.codroid_picker.i18n import t


def register_workspace_callbacks(app: Dash) -> None:
    """Register workspace-related callbacks."""

    @app.callback(
        Output("compute-workspace", "disabled"),
        Output("compute-workspace", "title"),
        Input("application-ui-interval", "n_intervals"),
        State("camera-connection-state", "data"),
        State("prerequisites-state", "data"),
        prevent_initial_call=False,
    )
    def _update_compute_workspace_state(
        _ticks: int,
        camera_state: Optional[dict],
        prerequisites: Optional[dict],
    ) -> tuple[bool, str]:
        """Enable compute workspace only when camera stream and calibration are ready."""
        camera_live = (camera_state or {}).get("status") == "live"
        camera_calibrated = bool(
            (prerequisites or {}).get("camera_calibrated", False)
        )
        if not camera_live:
            return True, t("Enable live camera stream to compute workspace.")
        if not camera_calibrated:
            return True, t(
                "Load or run camera calibration before computing workspace."
            )
        return False, ""

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
                return t("Workspace definition cleared; robot movement unrestricted.")
            if trigger == "use-existing-workspace":
                ws = load_workspace_definition(DEFAULT_WORKSPACE_PATH)
                return (
                    t(
                        "Workspace loaded from {path}: {w:.1f} x {h:.1f} mm",
                        path=DEFAULT_WORKSPACE_PATH,
                        w=ws.width_mm,
                        h=ws.height_mm,
                    )
                )
            ws = compute_workspace_from_latest(marker_ids, marker_size_mm)
            return (
                t(
                    "Workspace computed and saved ({path}): {w:.1f} x {h:.1f} mm",
                    path=DEFAULT_WORKSPACE_PATH,
                    w=ws.width_mm,
                    h=ws.height_mm,
                )
            )
        except Exception as exc:
            return t("Workspace error: {error}", error=exc)

    # Note: _handle_plate_actions callback removed - it was a no-op returning no_update
