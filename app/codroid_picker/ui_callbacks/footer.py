"""Footer visibility callbacks.

This module contains Dash callbacks for:
- Footer visibility toggle
- Footer class and style synchronization
"""

from typing import Optional, Tuple

from dash import Dash, Input, Output, State, callback_context


def register_footer_callbacks(app: Dash) -> None:
    """Register footer-related callbacks with the Dash app."""

    @app.callback(
        Output("footer-visibility", "data"),
        Output("app-footer", "className"),
        Output("footer-toggle", "children"),
        Output("app-main", "className"),
        Output("footer-body", "style"),
        Input("footer-toggle", "n_clicks"),
        Input("robot-operation-state", "data"),
        State("footer-visibility", "data"),
    )
    def _toggle_footer_visibility(
        n_clicks: Optional[int],
        operation_state: Optional[dict],
        state: Optional[dict],
    ) -> Tuple[dict, str, str, str, dict]:
        """Toggle footer visibility and update related UI classes."""

        def _main_class(collapsed: bool) -> str:
            base = "app-main footer-collapsed" if collapsed else "app-main footer-expanded"
            status = (operation_state or {}).get("status", "idle")
            if status == "running":
                return f"{base} operation-running"
            if status in {"paused", "stopping"}:
                return f"{base} operation-paused"
            return base

        collapsed = bool((state or {}).get("collapsed", False))
        trigger = callback_context.triggered_id
        if not n_clicks or trigger != "footer-toggle":
            # honor current state without toggling on first render or non-toggle updates
            class_name = "app-footer is-collapsed" if collapsed else "app-footer"
            label = "Show footer" if collapsed else "Hide footer"
            main_class = _main_class(collapsed)
            style = {"display": "none"} if collapsed else {}
            return state or {"collapsed": False}, class_name, label, main_class, style

        new_collapsed = not collapsed
        class_name = "app-footer is-collapsed" if new_collapsed else "app-footer"
        label = "Show footer" if new_collapsed else "Hide footer"
        main_class = _main_class(new_collapsed)
        style = {"display": "none"} if new_collapsed else {}
        return {"collapsed": new_collapsed}, class_name, label, main_class, style
