"""Detection callbacks - YOLO label selection and display."""

from typing import Optional, Tuple

import dash_mantine_components as dmc
from dash import ALL, Dash, Input, Output, State, callback_context, html, no_update
from app.codroid_picker.i18n import t


def _normalize_pick_config(raw_config: Optional[list]) -> list[dict]:
    """Normalize persisted pick label thresholds for UI and inference."""
    config = []
    seen = set()
    for item in raw_config or []:
        if not isinstance(item, dict):
            continue
        label = str(item.get("label") or "").strip()
        if not label or label.lower() in seen:
            continue
        try:
            confidence = float(item.get("confidence", 0.1))
        except (TypeError, ValueError):
            confidence = 0.1
        confidence = max(0.0, min(1.0, confidence))
        config.append({"label": label, "confidence": confidence})
        seen.add(label.lower())
    if not config:
        config.append({"label": "orange", "confidence": 0.1})
    return config


def _sync_pick_config_to_session(config: list[dict]) -> None:
    """Make the pick label UI config available to detection and auto-pick."""
    import app.codroid_picker as codroid_picker_module

    normalized = _normalize_pick_config(config)
    labels = [item["label"] for item in normalized]
    codroid_picker_module.SESSION.pick_label_confidences = {
        item["label"]: item["confidence"] for item in normalized
    }
    codroid_picker_module.SESSION.auto_config.update(
        {
            "target_label": labels[0] if labels else "",
            "target_labels": labels,
        }
    )
    codroid_picker_module.SESSION.detections = []
    codroid_picker_module.SESSION.auto_detection_updated_at = 0.0


def _pick_label_table_children(config: list[dict]) -> list:
    """Build table content for selected pick labels."""
    rows = [
        html.Thead(
            [
                html.Tr(
                    [
                        html.Th(t("Label")),
                        html.Th(t("Confidence")),
                        html.Th(""),
                    ]
                )
            ]
        )
    ]
    body_rows = []
    for item in _normalize_pick_config(config):
        label = item["label"]
        body_rows.append(
            html.Tr(
                [
                    html.Td(label),
                    html.Td(
                        dmc.NumberInput(
                            id={"type": "pick-label-confidence", "label": label},
                            value=item["confidence"],
                            min=0.0,
                            max=1.0,
                            step=0.05,
                            w=130,
                        )
                    ),
                    html.Td(
                        dmc.Button(
                            t("Remove"),
                            id={"type": "pick-label-remove", "label": label},
                            variant="subtle",
                            color="red",
                            size="xs",
                        )
                    ),
                ]
            )
        )
    rows.append(html.Tbody(body_rows))
    return rows


def _model_label_options() -> list[dict]:
    """Build select options from the active model's labels."""
    import app.codroid_picker as codroid_picker_module

    labels = codroid_picker_module.SESSION.model_labels or []
    return [{"label": label, "value": label} for label in labels]


def register_detection_callbacks(app: Dash) -> None:
    """Register detection-related callbacks."""

    @app.callback(
        Output("pick-label-add-select", "data"),
        Input("model-load-state", "data"),
        Input("application-ui-interval", "n_intervals"),
        prevent_initial_call=True,
    )
    def _populate_pick_label_options(
        _model_load_state: Optional[dict],
        _n: int,
    ) -> list[dict]:
        """Populate labels exposed by the active model."""
        return _model_label_options()

    @app.callback(
        Output("pick-label-config-state", "data"),
        Output("pick-label-config-list", "children"),
        Input("pick-label-add", "n_clicks"),
        Input({"type": "pick-label-remove", "label": ALL}, "n_clicks"),
        Input({"type": "pick-label-confidence", "label": ALL}, "value"),
        State("pick-label-add-select", "value"),
        State("pick-label-add-confidence", "value"),
        State("pick-label-config-state", "data"),
        State({"type": "pick-label-confidence", "label": ALL}, "id"),
        prevent_initial_call=True,
    )
    def _update_pick_label_config(
        _add_clicks: Optional[int],
        _remove_clicks: list,
        confidence_values: list,
        selected_label: Optional[str],
        add_confidence: Optional[float],
        current_config: Optional[list],
        confidence_ids: list,
    ) -> Tuple[list[dict], list]:
        """Add, remove, and edit pick labels."""
        config = _normalize_pick_config(current_config)
        triggered = callback_context.triggered_id

        if triggered == "pick-label-add" and selected_label:
            label = selected_label.strip()
            try:
                confidence = float(add_confidence)
            except (TypeError, ValueError):
                confidence = 0.1
            confidence = max(0.0, min(1.0, confidence))
            config = [
                item
                for item in config
                if item["label"].lower() != label.lower()
            ]
            config.append({"label": label, "confidence": confidence})
        elif (
            isinstance(triggered, dict)
            and triggered.get("type") == "pick-label-remove"
        ):
            label = str(triggered.get("label") or "")
            config = [item for item in config if item["label"] != label]
        else:
            by_label = {item["label"]: item for item in config}
            for control_id, value in zip(
                confidence_ids or [],
                confidence_values or [],
            ):
                label = str((control_id or {}).get("label") or "")
                if label not in by_label:
                    continue
                try:
                    confidence = float(value)
                except (TypeError, ValueError):
                    confidence = by_label[label]["confidence"]
                by_label[label]["confidence"] = max(0.0, min(1.0, confidence))
            config = list(by_label.values())

        config = _normalize_pick_config(config)
        _sync_pick_config_to_session(config)
        return config, _pick_label_table_children(config)

    @app.callback(
        Output("detection-list", "children", allow_duplicate=True),
        Output("pick-detection-select", "data", allow_duplicate=True),
        Input("application-ui-interval", "n_intervals"),
        State("application-tab-state", "data"),
        prevent_initial_call=True,
    )
    def _update_detection_display(
        _n: int,
        application_tab_state: Optional[dict],
    ) -> Tuple[list, list]:
        """Update detection display from continuous detection results.

        Detection updates are produced automatically when a model is loaded;
        if no model is loaded skip updating the display.
        """
        import app.codroid_picker as codroid_picker_module

        application_tab = (application_tab_state or {}).get("value") or "pick-place"
        if application_tab != "pick-place":
            return no_update, no_update

        SESSION = codroid_picker_module.SESSION
        _format_detection_list = codroid_picker_module._format_detection_list

        if not SESSION.model_loaded:
            return no_update, no_update

        detections = SESSION.detections or []
        detection_list = _format_detection_list(detections)

        options = []
        for idx, det in enumerate(detections):
            label = det.get("label", t("object"))
            conf = det.get("confidence", 0.0)
            options.append(
                {"value": str(idx), "label": f"{idx + 1}: {label} ({conf:.2f})"})

        return detection_list, options
