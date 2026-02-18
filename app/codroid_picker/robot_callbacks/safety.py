"""Camera safety callbacks for automatic pause/resume."""

from __future__ import annotations

import logging

from dash import Dash, Input, Output, no_update

from app.codroid_picker.programs import (
    get_registry,
    pause_program,
    resume_program,
)
from app.codroid_picker.i18n import t
from app.codroid_picker.safety_guard import (
    desired_operation_action,
    evaluate_safety_state,
    normalize_safety_labels,
)

LOGGER = logging.getLogger(__name__)


def register_safety_callbacks(app: Dash) -> None:
    """Register periodic camera safety callback."""

    @app.callback(
        Output("camera-safety-state", "data"),
        Input("camera-safety-interval", "n_intervals"),
        prevent_initial_call=True,
    )
    def _run_camera_safety(
        _ticks: int,
    ) -> dict:
        import app.codroid_picker as codroid_picker_module
        from app.codroid_picker.auto_operation import _run_detection
        from app.codroid_picker.session import SESSION

        state = codroid_picker_module._operation_snapshot()
        status = state.get("status", "idle")
        if status not in {"running", "paused"}:
            return {"blocked": False, "matches": 0, "action": "none", "active": False}

        safety_cfg = SESSION.safety_config or {}
        enabled = bool(safety_cfg.get("enabled", True))
        if not enabled:
            return {"blocked": False, "matches": 0, "action": "none", "active": True}

        if not SESSION.model_loaded or SESSION.yolo_model is None:
            return no_update

        frame, _depth = codroid_picker_module.get_latest_frame()
        if frame is None:
            return no_update

        labels = normalize_safety_labels(safety_cfg.get("labels"))
        if not labels:
            return {"blocked": False, "matches": 0, "action": "none", "active": True}

        zone = str(safety_cfg.get("zone", "workspace")).strip().lower()

        try:
            detections = _run_detection(
                frame,
                SESSION.detection_confidence,
                SESSION.detection_iou,
                SESSION.detection_max,
                SESSION.detection_classes,
                crop_to_workspace=(zone != "full_frame"),
            )
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("Camera safety detection failed: %s", exc)
            return {"blocked": False, "matches": 0, "action": "none", "error": str(exc)}

        blocked, matches = evaluate_safety_state(
            detections,
            enabled=enabled,
            labels=labels,
            zone=zone,
            workspace_def=SESSION.workspace_definition,
        )

        action = desired_operation_action(
            status=status,
            pause_reason=state.get("pause_reason"),
            blocked=blocked,
        )

        registry = get_registry()
        if action == "pause":
            if registry.active_program is not None:
                pause_program("person_detected")
            else:
                codroid_picker_module._pause_operation(
                    "person_detected",
                    t("Paused (person detected)."),
                )
        elif action == "resume":
            if registry.active_program is not None:
                resume_program()
            else:
                codroid_picker_module._resume_operation(t("Resumed (area clear)."))

        return {
            "blocked": blocked,
            "matches": len(matches),
            "action": action,
            "active": True,
            "zone": zone,
            "labels": labels,
        }
