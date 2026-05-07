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
        Output("auto-target-labels", "data"),
        Output("auto-target-labels", "value"),
        Output("settings-auto-target-labels", "data"),
        Output("settings-auto-target-labels", "value"),
        Output("model-benchmark-status", "children"),
        Input("model-upload", "contents"),
        Input("load-model", "n_clicks"),
        Input("benchmark-model", "n_clicks"),
        State("model-upload", "filename"),
        State("model-select", "value"),
        State("auto-target-labels", "value"),
        prevent_initial_call=True,
    )
    def _handle_model_actions(
        contents: Optional[str],
        _load_clicks: int,
        _benchmark_clicks: int,
        filename: Optional[str],
        selected: Optional[str],
        selected_labels: Optional[list[str]],
    ) -> Tuple[str, list, Optional[str], list, list, list, list, str]:
        """Handle model upload and load actions."""
        import app.codroid_picker as codroid_picker_module
        from app.codroid_picker.models import resolve_selected_pick_labels

        trigger = callback_context.triggered_id
        options = codroid_picker_module._model_options()
        label_options = codroid_picker_module.model_label_options()
        labels = [option["value"] for option in label_options]
        resolved_labels = resolve_selected_pick_labels(labels, selected_labels)
        benchmark_status = codroid_picker_module.format_model_benchmark()

        def _payload(
            status: str,
            model_value: Optional[str],
            benchmark: Optional[dict] = None,
        ) -> Tuple[str, list, Optional[str], list, list, list, list, str]:
            status_text = status
            benchmark_text = codroid_picker_module.format_model_benchmark(benchmark)
            refreshed_label_options = codroid_picker_module.model_label_options()
            refreshed_labels = [option["value"] for option in refreshed_label_options]
            refreshed_selected = resolve_selected_pick_labels(
                refreshed_labels, selected_labels
            )
            return (
                status_text,
                codroid_picker_module._model_options(),
                model_value,
                refreshed_label_options,
                refreshed_selected,
                refreshed_label_options,
                refreshed_selected,
                benchmark_text,
            )

        if trigger == "model-upload" and contents:
            try:
                model_path = codroid_picker_module.save_uploaded_model(
                    contents,
                    filename or "model.bin",
                    codroid_picker_module.DEFAULT_MODEL_DIR,
                )
                status = codroid_picker_module.activate_model(model_path)
                benchmark = codroid_picker_module.run_model_benchmark()
                return _payload(status, str(model_path), benchmark)
            except Exception as exc:
                return (
                    f"Model upload failed: {exc}",
                    options,
                    selected,
                    label_options,
                    resolved_labels,
                    label_options,
                    resolved_labels,
                    benchmark_status,
                )
        if trigger == "load-model":
            if selected:
                try:
                    status = codroid_picker_module.activate_model(
                        Path(selected))
                    benchmark = codroid_picker_module.run_model_benchmark()
                except Exception as exc:
                    status = f"Model load failed: {exc}"
                    benchmark = None
                return _payload(status, selected, benchmark)
            status = codroid_picker_module.activate_model(None)
            return _payload(status, None)
        if trigger == "benchmark-model":
            benchmark = codroid_picker_module.run_model_benchmark()
            return (
                no_update,
                options,
                no_update,
                label_options,
                resolved_labels,
                label_options,
                resolved_labels,
                codroid_picker_module.format_model_benchmark(benchmark),
            )
        return (
            no_update,
            options,
            no_update,
            label_options,
            resolved_labels,
            label_options,
            resolved_labels,
            benchmark_status,
        )

    @app.callback(
        Output("gpu-status-display", "children"),
        Input("monitor-interval", "n_intervals"),
        prevent_initial_call=True,
    )
    def _update_gpu_status(_n: int) -> list:
        """Update GPU status display periodically."""
        import app.codroid_picker as codroid_picker_module

        if (_n or 0) <= 0:
            return no_update
        return codroid_picker_module.format_gpu_status()
