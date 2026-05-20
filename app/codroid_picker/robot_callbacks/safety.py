"""Camera safety callbacks for automatic pause/resume."""

from __future__ import annotations

import logging
import time
from typing import Optional

import cv2
import numpy as np
from dash import Dash, Input, Output, no_update

from app.codroid_picker.programs import (
    get_registry,
    pause_program,
    resume_program,
)
from app.codroid_picker.i18n import t
from app.codroid_picker.safety_guard import (
    desired_operation_action,
    evaluate_heuristic_intrusion,
    evaluate_safety_state,
    normalize_safety_labels,
    normalize_safety_mode,
    preprocess_gray,
)
from app.codroid_picker.streaming import RAW_STREAM

LOGGER = logging.getLogger(__name__)
def _workspace_roi_mask(frame: np.ndarray, zone: str, workspace_def: Optional[object]) -> Optional[np.ndarray]:
    """Build ROI mask for heuristic monitoring."""
    mode = str(zone or "workspace").strip().lower()
    if mode == "full_frame":
        return None
    if workspace_def is None:
        return np.zeros(frame.shape[:2], dtype=np.uint8)
    try:
        from app.codroid_picker.overlays import _workspace_corners_px

        corners = _workspace_corners_px(workspace_def).astype(np.int32)
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [corners], 255)
        return mask
    except Exception:  # noqa: BLE001
        return np.zeros(frame.shape[:2], dtype=np.uint8)


def _robot_ignore_mask(frame: np.ndarray, workspace_def: Optional[object]) -> Optional[np.ndarray]:
    """Build a coarse mask around projected tool center to reduce false positives."""
    if workspace_def is None:
        return None
    try:
        from app.codroid_picker.procedures import (
            _current_coordinate_id,
            _get_robot_session,
            _robot_to_workspace_mm,
        )
        from app.codroid_picker.transforms import _coord5_to_workspace_mm
        from app.codroid_picker.session import SESSION

        robot_session = _get_robot_session()
        if not robot_session.posture_seen():
            return None
        pos = robot_session.position_snapshot()
        coord_id = _current_coordinate_id(robot_session)
        if coord_id == 5:
            workspace_xy = _coord5_to_workspace_mm(
                workspace_def, (float(pos.x), float(pos.y))
            )
        else:
            workspace_xyz = np.array([pos.x, pos.y, pos.z], dtype=np.float64)
            workspace_xy = _robot_to_workspace_mm(workspace_def, workspace_xyz)
        if workspace_xy is None or SESSION.calibration_bundle is None:
            return None
        scale = float(SESSION.calibration_bundle.pixel_to_mm)
        if scale <= 0:
            return None
        cam_xyz = workspace_def.workspace_to_camera(
            float(workspace_xy[0]), float(workspace_xy[1]), 0.0
        )
        center = (
            int(round(float(cam_xyz[0]) / scale)),
            int(round(float(cam_xyz[1]) / scale)),
        )
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        cv2.circle(mask, center, 70, 255, -1, lineType=cv2.LINE_AA)
        return mask
    except Exception:  # noqa: BLE001
        return None


def register_safety_callbacks(app: Dash) -> None:
    """Register periodic camera safety callback."""

    @app.callback(
        Output("camera-safety-interval", "disabled", allow_duplicate=True),
        Input("camera-connection-state", "data"),
        prevent_initial_call=True,
    )
    def _toggle_camera_safety_interval(camera_state: Optional[dict]) -> bool:
        import app.codroid_picker as codroid_picker_module

        mode = normalize_safety_mode(
            codroid_picker_module.SETTINGS.camera_safety_mode,
            codroid_picker_module.SETTINGS.camera_safety_enabled,
        )
        if mode == "off":
            return True
        return (camera_state or {}).get("status") != "live"

    @app.callback(
        Output("camera-safety-state", "data"),
        Input("camera-safety-interval", "n_intervals"),
        prevent_initial_call=True,
    )
    def _run_camera_safety(
        _ticks: int,
    ) -> dict:
        import app.codroid_picker as codroid_picker_module
        from app.codroid_picker.auto_operation import _run_detection_serialized
        from app.codroid_picker.session import SESSION

        state = codroid_picker_module._operation_snapshot()
        status = state.get("status", "idle")
        motion_active = status in {"running", "paused"}
        runtime = SESSION.safety_runtime
        if not motion_active:
            runtime["baseline_gray"] = None
            runtime["baseline_ts"] = 0.0
            runtime["clear_streak"] = 0
            runtime["last_frame_shape"] = None
            runtime["last_motion_active"] = False
            return {"blocked": False, "matches": 0, "action": "none", "active": False}

        safety_cfg = SESSION.safety_config or {}
        mode = normalize_safety_mode(
            safety_cfg.get("mode"),
            safety_cfg.get("enabled"),
        )
        if mode == "off":
            return {
                "blocked": False,
                "matches": 0,
                "action": "none",
                "active": True,
                "mode": mode,
            }

        frame = RAW_STREAM.get_latest_frame()
        if frame is None:
            return no_update

        zone = str(safety_cfg.get("zone", "workspace")).strip().lower()
        blocked = False
        matches: list[dict] = []
        metrics: dict = {}

        if mode == "model":
            labels = normalize_safety_labels(safety_cfg.get("labels"))
            if not labels:
                return {
                    "blocked": False,
                    "matches": 0,
                    "action": "none",
                    "active": True,
                    "mode": mode,
                }
            if SESSION.yolo_model is None:
                runtime["clear_streak"] = 0
                return {
                    "blocked": False,
                    "matches": 0,
                    "action": "none",
                    "active": False,
                    "mode": mode,
                    "reason": "model_not_loaded",
                }
            detections = SESSION.detections or []
            stale_s = float(getattr(codroid_picker_module.SETTINGS, "auto_detection_stale_s", 0.3))
            detections_age_s = time.monotonic() - float(SESSION.auto_detection_updated_at or 0.0)
            need_refresh = (
                not detections
                or detections_age_s > stale_s
            )
            if need_refresh:
                detections = _run_detection_serialized(
                    frame,
                    SESSION.detection_confidence,
                    SESSION.detection_iou,
                    SESSION.detection_max,
                    SESSION.detection_classes,
                    crop_to_workspace=(zone != "full_frame"),
                    apply_pick_filter=False,
                )
                SESSION.detections = detections
                SESSION.auto_detection_updated_at = time.monotonic()
            blocked, matches = evaluate_safety_state(
                detections,
                enabled=True,
                labels=labels,
                zone=zone,
                workspace_def=SESSION.workspace_definition,
            )
            metrics["detections_age_s"] = max(0.0, detections_age_s)
        else:
            threshold = float(safety_cfg.get("heuristic_threshold", 0.045))
            min_area_ratio = float(safety_cfg.get("heuristic_min_area_ratio", 0.01))
            baseline_refresh_ms = int(safety_cfg.get("baseline_refresh_ms", 1200))
            clear_frames = int(safety_cfg.get("clear_frames", 3))
            clear_frames = max(clear_frames, 1)
            now = time.monotonic()

            if not runtime.get("last_motion_active", False):
                runtime["baseline_gray"] = None
                runtime["clear_streak"] = 0
            runtime["last_motion_active"] = True

            roi_mask = _workspace_roi_mask(frame, zone, SESSION.workspace_definition)
            ignore_mask = _robot_ignore_mask(frame, SESSION.workspace_definition)
            baseline_gray = runtime.get("baseline_gray")
            frame_shape = tuple(frame.shape[:2])

            if (
                baseline_gray is None
                or runtime.get("last_frame_shape") != frame_shape
            ):
                runtime["baseline_gray"] = preprocess_gray(frame)
                runtime["baseline_ts"] = now
                runtime["last_frame_shape"] = frame_shape
                return {
                    "blocked": False,
                    "matches": 0,
                    "action": "none",
                    "active": True,
                    "mode": mode,
                    "warming_up": True,
                }

            blocked_raw, metrics = evaluate_heuristic_intrusion(
                frame,
                baseline_gray,
                roi_mask=roi_mask,
                ignore_mask=ignore_mask,
                threshold=threshold,
                min_area_ratio=min_area_ratio,
            )

            if blocked_raw:
                runtime["clear_streak"] = 0
                blocked = True
            else:
                runtime["clear_streak"] = int(runtime.get("clear_streak", 0)) + 1
                blocked = runtime["clear_streak"] < clear_frames
                if (
                    runtime["clear_streak"] >= clear_frames
                    and (now - float(runtime.get("baseline_ts") or 0.0)) * 1000.0
                    >= float(baseline_refresh_ms)
                ):
                    runtime["baseline_gray"] = preprocess_gray(frame)
                    runtime["baseline_ts"] = now

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
            "mode": mode,
            "labels": normalize_safety_labels(safety_cfg.get("labels")),
            "metrics": metrics,
        }
