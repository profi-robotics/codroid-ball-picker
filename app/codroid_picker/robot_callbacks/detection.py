"""Detection callbacks - YOLO model detection parameters and display."""

from typing import Optional, Tuple

from dash import Dash, Input, Output, State, no_update
from app.codroid_picker.i18n import t


def register_detection_callbacks(app: Dash) -> None:
    """Register detection-related callbacks."""

    @app.callback(
        Output("detection-confidence", "value"),
        Input("detection-confidence", "value"),
        Input("detection-iou", "value"),
        Input("detection-max", "value"),
        Input("detection-classes", "value"),
        State("detection-confidence", "value"),
        prevent_initial_call=True,
    )
    def _update_detection_parameters(
        confidence: Optional[float],
        iou: Optional[float],
        max_detections: Optional[float],
        class_filter: Optional[str],
        current_confidence: Optional[float],
    ) -> float:
        """Store detection parameters when they change."""
        import app.codroid_picker as codroid_picker_module

        SESSION = codroid_picker_module.SESSION
        _safe_float = codroid_picker_module._safe_float
        _safe_int = codroid_picker_module._safe_int
        normalize_confidence = codroid_picker_module._normalize_detection_confidence
        normalize_iou = codroid_picker_module._normalize_detection_iou
        normalize_max = codroid_picker_module._normalize_max_detections

        fallback_confidence = _safe_float(current_confidence, SESSION.detection_confidence)
        SESSION.detection_confidence = normalize_confidence(
            _safe_float(confidence, fallback_confidence),
            fallback_confidence,
        )
        SESSION.detection_iou = normalize_iou(_safe_float(iou, 0.45))
        SESSION.detection_max = normalize_max(_safe_int(max_detections, 10))
        SESSION.detection_classes = class_filter
        SESSION.detections = []
        SESSION.auto_detection_updated_at = 0.0
        return SESSION.detection_confidence

    @app.callback(
        Output("detection-status", "children"),
        Output("detection-list", "children", allow_duplicate=True),
        Output("pick-detection-select", "data", allow_duplicate=True),
        Output("pick-detection-select", "value"),
        Input("run-detection", "n_clicks"),
        State("detection-confidence", "value"),
        State("detection-iou", "value"),
        State("detection-max", "value"),
        State("detection-classes", "value"),
        prevent_initial_call=True,
    )
    def _run_detection_action(
        _clicks: int,
        confidence: Optional[float],
        iou: Optional[float],
        max_detections: Optional[float],
        class_filter: Optional[str],
    ) -> Tuple[str, list, list, Optional[str]]:
        """Run object detection on the current camera frame."""
        import app.codroid_picker as codroid_picker_module

        SESSION = codroid_picker_module.SESSION
        get_latest_frame = codroid_picker_module.get_latest_frame
        _safe_float = codroid_picker_module._safe_float
        _safe_int = codroid_picker_module._safe_int
        normalize_confidence = codroid_picker_module._normalize_detection_confidence
        normalize_iou = codroid_picker_module._normalize_detection_iou
        normalize_max = codroid_picker_module._normalize_max_detections
        _run_detection = codroid_picker_module._run_detection
        _format_detection_list = codroid_picker_module._format_detection_list

        frame, _depth = get_latest_frame()
        if frame is None:
            return (
                t("Camera stream is not running."),
                _format_detection_list([]),
                [],
                None,
            )
        conf_value = normalize_confidence(_safe_float(confidence, 0.4))
        iou_value = normalize_iou(_safe_float(iou, 0.45))
        max_value = normalize_max(_safe_int(max_detections, 10))
        try:
            detections = _run_detection(
                frame, conf_value, iou_value, max_value, class_filter)
            SESSION.detections = detections
            SESSION.last_detection_error = None
            SESSION.last_action = t("Detection run completed.")
        except Exception as exc:
            SESSION.detections = []
            SESSION.last_detection_error = str(exc)
            return (
                t("Detection failed: {error}", error=exc),
                _format_detection_list([]),
                [],
                None,
            )

        options = []
        for idx, det in enumerate(detections):
            label = det.get("label", t("object"))
            options.append({"value": str(idx), "label": f"{idx + 1}: {label}"})
        selected = options[0]["value"] if options else None
        return (
            t("Detection complete: {count} object(s).", count=len(detections)),
            _format_detection_list(detections),
            options,
            selected,
        )

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
