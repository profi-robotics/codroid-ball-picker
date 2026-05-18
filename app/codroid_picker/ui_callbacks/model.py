"""Model and GPU status callbacks.

This module contains Dash callbacks for:
- Model upload and activation
- GPU status display
"""

from pathlib import Path
from typing import Optional, Tuple

from dash import Dash, Input, Output, State, callback_context, no_update


def register_model_callbacks(app: Dash) -> None:
    """Register model-related callbacks with the Dash app."""

    @app.callback(
        Output("model-status", "children"),
        Output("model-select", "data"),
        Output("model-select", "value"),
        Input("model-upload", "contents"),
        Input("load-model", "n_clicks"),
        State("model-upload", "filename"),
        State("model-select", "value"),
        prevent_initial_call=True,
    )
    def _handle_model_actions(
        contents: Optional[str],
        _load_clicks: int,
        filename: Optional[str],
        selected: Optional[str],
    ) -> Tuple[str, list, Optional[str]]:
        """Handle model upload and load actions."""
        import app.codroid_picker as codroid_picker_module

        trigger = callback_context.triggered_id
        options = codroid_picker_module._model_options()
        if trigger == "model-upload" and contents:
            try:
                model_path = codroid_picker_module.save_uploaded_model(
                    contents,
                    filename or "model.bin",
                    codroid_picker_module.DEFAULT_MODEL_DIR,
                )
                status = codroid_picker_module.activate_model(model_path)
                options = codroid_picker_module._model_options()
                return status, options, str(model_path)
            except Exception as exc:
                return f"Model upload failed: {exc}", options, selected
        if trigger == "load-model":
            if selected:
                try:
                    status = codroid_picker_module.activate_model(
                        Path(selected))
                except Exception as exc:
                    status = f"Model load failed: {exc}"
                return status, options, selected
            status = codroid_picker_module.activate_model(None)
            return status, options, None
        return no_update, options, no_update

    @app.callback(
        Output("model-load-state", "data"),
        Input("hardware-init", "n_intervals"),
        prevent_initial_call=False,
    )
    def _auto_load_model_after_first_paint(_n: int) -> dict:
        """Auto-load the configured model after the first visible UI paint."""
        if (_n or 0) <= 0:
            return no_update
        import app.codroid_picker as codroid_picker_module

        return codroid_picker_module.auto_load_configured_model()

    @app.callback(
        Output("model-status", "children", allow_duplicate=True),
        Output("model-select", "data", allow_duplicate=True),
        Output("model-select", "value", allow_duplicate=True),
        Input("application-ui-interval", "n_intervals"),
        State("model-load-state", "data"),
        prevent_initial_call=True,
    )
    def _refresh_model_status(
        n_intervals: int,
        _model_load_state: Optional[dict],
    ) -> Tuple[str, list, Optional[str]]:
        """Reflect post-paint model auto-load in the lazy application tab."""
        if (n_intervals or 0) <= 0:
            return no_update, no_update, no_update
        import app.codroid_picker as codroid_picker_module

        options = codroid_picker_module._model_options()
        active_path = codroid_picker_module.SESSION.active_model_path
        if codroid_picker_module.SESSION.model_loaded and active_path:
            return f"Model loaded: {active_path.name}.", options, str(active_path)
        if active_path:
            error = codroid_picker_module.SESSION.last_detection_error
            if error:
                return f"Model ready but not loaded: {error}", options, str(active_path)
            return f"Model selected: {active_path.name}.", options, str(active_path)
        return "No model selected.", options, None

    @app.callback(
        Output("gpu-status-display", "children"),
        Input("application-ui-interval", "n_intervals"),
        prevent_initial_call=True,
    )
    def _update_gpu_status(_n: int) -> list:
        """Update GPU status display periodically."""
        import app.codroid_picker as codroid_picker_module

        if (_n or 0) <= 0:
            return no_update
        return codroid_picker_module.format_gpu_status()
