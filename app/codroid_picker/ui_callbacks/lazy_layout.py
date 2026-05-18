"""Lazy rendering callbacks for tab content."""

from __future__ import annotations

from typing import Optional

from dash import Dash, Input, Output, State, callback_context, no_update

from app.codroid_picker.i18n import t

from app.codroid_picker.components import (
    build_application_tab,
    build_calibration_tab,
    build_monitoring_tab,
)


def _application_value(state: Optional[dict]) -> str:
    """Return the stored application subtab value."""
    value = (state or {}).get("value")
    return value if value in {"pick-place", "workspace"} else "pick-place"


def _calibration_step(state: Optional[dict]) -> int:
    """Return the stored calibration step."""
    try:
        return max(0, min(1, int((state or {}).get("active", 0) or 0)))
    except (TypeError, ValueError):
        return 0


def render_main_tab(
    active_tab: Optional[str],
    application_state: Optional[dict] = None,
    calibration_state: Optional[dict] = None,
) -> tuple[object, bool, bool, bool]:
    """Build active tab content and interval disabled states."""
    tab = active_tab or "applications"
    if tab == "monitoring":
        return build_monitoring_tab(), True, True, False
    if tab == "calibration":
        return build_calibration_tab(_calibration_step(calibration_state)), True, False, True
    return build_application_tab(_application_value(application_state)), False, True, True


def sidebar_layout_state(collapsed: bool) -> tuple[dict, dict, str]:
    """Return sidebar style, camera span, and toggle label for a visibility state."""
    if collapsed:
        return (
            {"display": "none"},
            {"base": 12, "md": 12, "lg": 12},
            t("Show controls"),
        )
    return (
        {},
        {"base": 12, "md": 8, "lg": 9},
        t("Hide controls"),
    )


def register_lazy_layout_callbacks(app: Dash) -> None:
    """Render tab content only after the user selects the tab."""
    if hasattr(app, "_lazy_layout_callbacks_registered"):
        return
    app._lazy_layout_callbacks_registered = True

    @app.callback(
        Output("main-tab-content", "children"),
        Output("application-ui-interval", "disabled"),
        Output("calibration-ui-interval", "disabled"),
        Output("monitoring-graph-interval", "disabled"),
        Input("main-tabs", "value"),
        State("application-tab-state", "data"),
        State("calibration-step-state", "data"),
        prevent_initial_call=False,
    )
    def _render_active_main_tab(
        active_tab: str,
        application_state: Optional[dict],
        calibration_state: Optional[dict],
    ) -> tuple[object, bool, bool, bool]:
        return render_main_tab(active_tab, application_state, calibration_state)

    @app.callback(
        Output("application-tab-state", "data"),
        Input("application-tabs", "value"),
        State("application-tab-state", "data"),
        prevent_initial_call=True,
    )
    def _remember_application_tab(value: Optional[str], current: Optional[dict]) -> dict:
        normalized = value if value in {"pick-place", "workspace"} else "pick-place"
        desired = {"value": normalized}
        if current == desired:
            return no_update
        return desired

    @app.callback(
        Output("calibration-step-state", "data"),
        Input("calibration-stepper", "active"),
        State("calibration-step-state", "data"),
        prevent_initial_call=True,
    )
    def _remember_calibration_step(active: Optional[int], current: Optional[dict]) -> dict:
        desired = {"active": _calibration_step({"active": active})}
        if current == desired:
            return no_update
        return desired

    @app.callback(
        Output("sidebar-visibility", "data"),
        Output("tabs-sidebar", "style"),
        Output("camera-grid-column", "span"),
        Output("sidebar-toggle", "children"),
        Input("sidebar-toggle", "n_clicks"),
        State("sidebar-visibility", "data"),
        prevent_initial_call=False,
    )
    def _toggle_sidebar_visibility(
        n_clicks: Optional[int],
        state: Optional[dict],
    ) -> tuple[dict, dict, dict, str]:
        """Collapse the left control rail while keeping camera controls visible."""
        collapsed = bool((state or {}).get("collapsed", False))
        if callback_context.triggered_id == "sidebar-toggle" and n_clicks:
            collapsed = not collapsed
        sidebar_style, camera_span, label = sidebar_layout_state(collapsed)
        return {"collapsed": collapsed}, sidebar_style, camera_span, label
