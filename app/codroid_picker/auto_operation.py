"""
Auto pick-and-place operation logic.

This module provides functions for running detection, target filtering,
and helper utilities for the auto pick-and-place program.
"""

from __future__ import annotations

import asyncio
import logging
import os
import threading
import time
from typing import Callable, Optional, Tuple

import numpy as np

import dash_mantine_components as dmc
from vision import workspace
from app.codroid_picker.session import SESSION
import app.settings as app_settings
from app.codroid_picker.i18n import t
from app.codroid_picker.state import (
    _operation_snapshot,
)
from app.codroid_picker.state import resume_operation as _resume_operation
from app.codroid_picker.exceptions import OperationAbort, OperationPaused

LOGGER = logging.getLogger(__name__)
_DETECTION_LOCK = threading.Lock()
_LOG_DETECTION_THROUGHPUT = os.getenv(
    "CODROID_LOG_DETECTION_THROUGHPUT", ""
).strip().lower() in {"1", "true", "yes", "on"}

SETTINGS = app_settings.SETTINGS
ROBOT_COORDINATE_ID = SETTINGS.robot_coordinate_id


def _normalize_unit_interval(value: float, fallback: float) -> float:
    """Return a finite value clamped to the unit interval."""
    try:
        fallback_value = float(fallback)
    except (TypeError, ValueError):
        fallback_value = 0.0
    if not np.isfinite(fallback_value):
        fallback_value = 0.0
    try:
        normalized = float(value)
    except (TypeError, ValueError):
        normalized = fallback_value
    if not np.isfinite(normalized):
        normalized = fallback_value
    return float(np.clip(normalized, 0.0, 1.0))


def _normalize_detection_confidence(value: float, fallback: float = 0.4) -> float:
    """Normalize YOLO confidence to the range accepted by Ultralytics."""
    return _normalize_unit_interval(value, fallback)


def _normalize_detection_iou(value: float, fallback: float = 0.45) -> float:
    """Normalize YOLO IoU to the range accepted by Ultralytics."""
    return _normalize_unit_interval(value, fallback)


def _normalize_max_detections(value: int, fallback: int = 10) -> int:
    """Normalize max detection count to a positive integer."""
    try:
        normalized = int(value)
    except (TypeError, ValueError):
        normalized = int(fallback)
    return max(1, normalized)


def _pick_label_confidences() -> dict[str, float]:
    """Return normalized per-label pick confidence thresholds."""
    thresholds: dict[str, float] = {}
    raw = getattr(SESSION, "pick_label_confidences", {}) or {}
    if isinstance(raw, dict):
        items = raw.items()
    else:
        items = []
    for label, confidence in items:
        normalized_label = str(label or "").strip()
        if not normalized_label:
            continue
        thresholds[normalized_label] = _normalize_detection_confidence(
            confidence, 0.1
        )
    return thresholds


def _effective_detection_confidence(confidence: float) -> float:
    """Use the lowest selected pick threshold so low-confidence labels can pass."""
    base_confidence = _normalize_detection_confidence(confidence)
    thresholds = _pick_label_confidences()
    if not thresholds:
        return base_confidence
    return min(base_confidence, min(thresholds.values()))


def _passes_pick_label_filter(
    label: str,
    confidence: float,
    thresholds: dict[str, float],
) -> bool:
    """Return whether a detection is selected for picking."""
    if not thresholds:
        return True
    normalized_label = str(label or "").strip().lower()
    for target_label, threshold in thresholds.items():
        if normalized_label == target_label.strip().lower():
            return confidence >= threshold
    return False


def _pick_label_class_ids(
    model: object,
    thresholds: dict[str, float],
) -> Optional[list[int]]:
    """Map selected pick labels to YOLO class IDs when model names are available."""
    if not thresholds:
        return None
    wanted = {label.strip().lower() for label in thresholds if label.strip()}
    names = getattr(model, "names", None)
    class_ids: list[int] = []
    if isinstance(names, dict):
        for class_id, label in names.items():
            if str(label).strip().lower() not in wanted:
                continue
            try:
                class_ids.append(int(class_id))
            except (TypeError, ValueError):
                continue
    elif isinstance(names, (list, tuple)):
        for class_id, label in enumerate(names):
            if str(label).strip().lower() in wanted:
                class_ids.append(class_id)
    return sorted(class_ids) or None


# ---------------------------------------------------------------------------
# Lazy imports to avoid circular dependencies
# ---------------------------------------------------------------------------

# Import from procedures
from app.codroid_picker.procedures import (
    _get_robot_session,
    _current_coordinate_id,
    _workspace_pixel_to_mm,
    _robot_to_workspace_mm,
    _pixel_to_robot_base_xyz,
)

# Import from robot_motion
from app.codroid_picker.robot_motion import (
    _operation_status,
    wait_for_operation_resume as _wait_for_operation_resume,
)

# Import from overlays
from app.codroid_picker.overlays import _workspace_corners_px

# Import from transforms
from app.codroid_picker.transforms import (
    _pixel_to_plate_mm,
    _plate_to_robot_transform,
    _base_to_coord5,
    _workspace_to_coord5_mm,
    _pose_from_workspace_mm,
)


def get_latest_frame():
    """Get the latest frame from the camera stream."""
    # Import here to avoid circular dependency
    import app.codroid_picker as codroid_picker_module
    return codroid_picker_module.get_latest_frame()



def _estimate_detection_width_mm(detection, workspace_def):
    """Estimate detection width in mm based on bounding box and workspace scale.

    Args:
        detection: Detection object with bbox attribute [x1, y1, x2, y2] in pixels
        workspace_def: Workspace definition with px_per_mm scale factor

    Returns:
        Estimated width in mm, or None if cannot be determined
    """
    if detection is None or workspace_def is None:
        return None

    try:
        bbox = detection.bbox if hasattr(
            detection, 'bbox') else detection.get('bbox')
        if bbox is None or len(bbox) < 4:
            return None

        # Get width in pixels from bounding box
        width_px = abs(bbox[2] - bbox[0])
        height_px = abs(bbox[3] - bbox[1])

        # Use the smaller dimension as the width estimate (for round objects)
        size_px = min(width_px, height_px)

        # Convert to mm using workspace scale
        px_per_mm = getattr(workspace_def, 'px_per_mm', None)
        if px_per_mm is None:
            px_per_mm = workspace_def.get('px_per_mm') if isinstance(
                workspace_def, dict) else None

        if px_per_mm is None or px_per_mm <= 0:
            return None

        return size_px / px_per_mm
    except Exception:
        return None



# ---------------------------------------------------------------------------
# Detection functions
# ---------------------------------------------------------------------------


def _run_detection(
    frame: np.ndarray,
    confidence: float,
    iou: float,
    max_detections: int,
    class_filter: Optional[str],
    *,
    crop_to_workspace: bool = True,
    apply_pick_filter: bool = True,
) -> list[dict]:
    """Run YOLO detection on a frame."""
    if SESSION.yolo_model is None:
        raise RuntimeError(
            "No YOLO model loaded. Upload a model or install ultralytics.")
    confidence = _normalize_detection_confidence(confidence)
    pick_thresholds = _pick_label_confidences() if apply_pick_filter else {}
    if apply_pick_filter:
        confidence = _effective_detection_confidence(confidence)
    iou = _normalize_detection_iou(iou)
    max_detections = _normalize_max_detections(max_detections)
    classes = None
    if class_filter:
        classes = [int(val.strip())
                   for val in class_filter.split(",") if val.strip().isdigit()]
    elif apply_pick_filter:
        classes = _pick_label_class_ids(SESSION.yolo_model, pick_thresholds)

    # Crop to workspace region only when both workspace and calibration are available.
    crop_offset_x = 0
    crop_offset_y = 0
    detection_frame = frame

    if (
        crop_to_workspace
        and
        SESSION.workspace_definition is not None
        and SESSION.calibration_bundle is not None
    ):
        try:
            corners = _workspace_corners_px(SESSION.workspace_definition)
            # Get bounding box of workspace region
            x_min = int(np.min(corners[:, 0]))
            y_min = int(np.min(corners[:, 1]))
            x_max = int(np.max(corners[:, 0]))
            y_max = int(np.max(corners[:, 1]))

            # Clamp to frame bounds
            x_min = max(0, x_min)
            y_min = max(0, y_min)
            x_max = min(frame.shape[1], x_max)
            y_max = min(frame.shape[0], y_max)

            # Crop frame to workspace region
            detection_frame = frame[y_min:y_max, x_min:x_max]
            crop_offset_x = x_min
            crop_offset_y = y_min
        except Exception:
            # Fall back to full frame when workspace crop cannot be computed.
            pass

    results = SESSION.yolo_model.predict(
        detection_frame,
        conf=confidence,
        iou=iou,
        max_det=max_detections,
        classes=classes,
        verbose=False,
    )
    detections: list[dict] = []
    if not results:
        return detections
    for result in results:
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            continue
        xyxy = boxes.xyxy.cpu().numpy() if hasattr(boxes.xyxy, "cpu") else boxes.xyxy
        confs = boxes.conf.cpu().numpy() if hasattr(boxes.conf, "cpu") else boxes.conf
        clss = boxes.cls.cpu().numpy() if hasattr(boxes.cls, "cpu") else boxes.cls
        for box, conf, cls_idx in zip(xyxy, confs, clss):
            # Adjust coordinates back to full frame
            x1, y1, x2, y2 = [float(val) for val in box]
            x1 += crop_offset_x
            y1 += crop_offset_y
            x2 += crop_offset_x
            y2 += crop_offset_y
            center = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
            label = str(int(cls_idx))
            if hasattr(SESSION.yolo_model, "names"):
                label = SESSION.yolo_model.names.get(int(cls_idx), label)
            confidence_value = float(conf)
            if not _passes_pick_label_filter(
                label,
                confidence_value,
                pick_thresholds,
            ):
                continue
            detection = {
                "bbox": (x1, y1, x2, y2),
                "center": center,
                "label": label,
                "confidence": confidence_value,
            }
            if SESSION.workspace_definition is not None:
                try:
                    detection["workspace_mm"] = _workspace_pixel_to_mm(
                        SESSION.workspace_definition, center)
                except Exception:
                    detection["workspace_mm"] = None
            detections.append(detection)
    return detections


def _run_detection_serialized(
    frame: np.ndarray,
    confidence: float,
    iou: float,
    max_detections: int,
    class_filter: Optional[str],
    *,
    crop_to_workspace: bool = True,
    apply_pick_filter: bool = True,
) -> list[dict]:
    """Run detection with a non-blocking process-wide lock.

    If another consumer is currently running inference, reuse the latest
    cached detections to avoid concurrent YOLO calls that hurt FPS.
    """
    if not _DETECTION_LOCK.acquire(blocking=False):
        return SESSION.detections or []
    try:
        detections = _run_detection(
            frame,
            confidence,
            iou,
            max_detections,
            class_filter,
            crop_to_workspace=crop_to_workspace,
            apply_pick_filter=apply_pick_filter,
        )
        SESSION.detection_inference_total += 1
        now = time.monotonic()
        if (
            _LOG_DETECTION_THROUGHPUT
            and now - float(SESSION.detection_inference_last_log_ts) >= 30.0
        ):
            LOGGER.debug(
                "Detection throughput: total_inference=%d stale_age=%.3fs",
                SESSION.detection_inference_total,
                max(0.0, now - float(SESSION.auto_detection_updated_at or 0.0)),
            )
            SESSION.detection_inference_last_log_ts = now
        return detections
    finally:
        _DETECTION_LOCK.release()


def _format_detection_list(detections: list[dict]) -> list:
    """Format detections for display in the UI."""
    if not detections:
        return [dmc.Text(t("No detections yet."), size="sm", c="dimmed")]
    rows = []
    for idx, det in enumerate(detections, start=1):
        label = det.get("label", "object")
        conf = det.get("confidence", 0.0)
        coord_text = "c5: --"
        coord5_mm = _detection_coord5_mm(det, SESSION.workspace_definition)
        if coord5_mm is not None:
            coord_text = f"c5 {coord5_mm[0]:.1f}, {coord5_mm[1]:.1f}"
        rows.append(
            dmc.Text(f"{idx}. {label} ({conf:.2f}) @ {coord_text}", size="sm"))
    return rows


def _refresh_follow_detections() -> list[dict]:
    """Refresh detections during follow mode."""
    frame, _depth = get_latest_frame()
    if frame is None:
        return SESSION.detections or []
    if SESSION.yolo_model is None:
        return SESSION.detections or []
    try:
        detections = _run_detection_serialized(
            frame,
            SESSION.detection_confidence,
            SESSION.detection_iou,
            SESSION.detection_max,
            SESSION.detection_classes,
        )
        SESSION.detections = detections
        SESSION.last_detection_error = None
        return detections
    except Exception as exc:  # noqa: BLE001
        SESSION.last_detection_error = str(exc)
        LOGGER.warning("Follow detection update failed: %s", exc)
        return SESSION.detections or []


# ---------------------------------------------------------------------------
# Workspace helpers
# ---------------------------------------------------------------------------


def _workspace_contains(
    workspace_def: workspace.WorkspaceDefinition,
    workspace_xy: Tuple[float, float],
) -> bool:
    """Check if a point is within the workspace bounds."""
    return (
        0.0 <= workspace_xy[0] <= workspace_def.width_mm
        and 0.0 <= workspace_xy[1] <= workspace_def.height_mm
    )


# ---------------------------------------------------------------------------
# Detection refresh and filtering
# ---------------------------------------------------------------------------


def _auto_refresh_detections(frame: np.ndarray) -> list[dict]:
    """Refresh detections for auto mode."""
    detections = _run_detection_serialized(
        frame,
        SESSION.detection_confidence,
        SESSION.detection_iou,
        SESSION.detection_max,
        SESSION.detection_classes,
    )
    SESSION.detections = detections
    SESSION.last_detection_error = None
    SESSION.auto_detection_updated_at = time.monotonic()
    return detections


def _auto_filter_targets(
    detections: list[dict],
    workspace_def: workspace.WorkspaceDefinition,
    *,
    target_label: str,
    drop_target_mm: Optional[Tuple[float, float]],
    drop_ignore_radius_mm: float,
) -> list[dict]:
    """Filter detections to valid pick targets."""
    label_norm = target_label.strip().lower()
    # Log what labels are available
    available_labels = set(str(det.get("label", "")).strip().lower()
                           for det in detections if det.get("label"))
    LOGGER.info(
        f"_auto_filter_targets: Filtering {len(detections)} detections, target_label='{target_label}', available_labels={available_labels}")
    targets = []
    for i, det in enumerate(detections):
        if label_norm:
            det_label = str(det.get("label", "")).strip().lower()
            if det_label != label_norm:
                LOGGER.info(
                    f"_auto_filter_targets: Det {i} skipped - label mismatch ('{det_label}' != '{label_norm}')")
                continue
        workspace_mm = _detection_workspace_mm(det, workspace_def)
        if workspace_mm is None:
            LOGGER.info(
                f"_auto_filter_targets: Det {i} skipped - no workspace coordinates")
            continue
        if (
            drop_target_mm is not None
            and drop_ignore_radius_mm > 0.0
            and float(
                np.linalg.norm(
                    np.array(workspace_mm) - np.array(drop_target_mm)
                )
            )
            <= drop_ignore_radius_mm
        ):
            dist = float(np.linalg.norm(
                np.array(workspace_mm) - np.array(drop_target_mm)))
            LOGGER.info(
                f"_auto_filter_targets: Det {i} skipped - too close to drop target (dist={dist:.1f}mm, threshold={drop_ignore_radius_mm}mm)")
            continue
        LOGGER.info(
            f"_auto_filter_targets: Det {i} accepted - workspace={workspace_mm}")
        targets.append(det)
    LOGGER.info(
        f"_auto_filter_targets: Result: {len(targets)} targets from {len(detections)} detections")
    return targets


def _auto_select_nearest_target(
    detections: list[dict],
    workspace_def: workspace.WorkspaceDefinition,
) -> Optional[dict]:
    """Select the nearest target to the robot's current position."""
    if not detections:
        return None
    session = _get_robot_session()
    if session.posture_seen():
        pos = session.position_snapshot()
        coord_id = _current_coordinate_id(session)
        robot_xy = None
        if coord_id == ROBOT_COORDINATE_ID:
            robot_xy = (float(pos.x), float(pos.y))
        if robot_xy is None:
            try:
                robot_xy = _robot_to_workspace_mm(
                    workspace_def,
                    np.array([pos.x, pos.y, pos.z], dtype=float),
                )
            except Exception:
                robot_xy = None
        if robot_xy is not None:
            best_det = None
            best_dist = float("inf")
            for det in detections:
                if coord_id == ROBOT_COORDINATE_ID:
                    det_xy = _detection_coord5_mm(det, workspace_def)
                else:
                    det_xy = _detection_workspace_mm(det, workspace_def)
                if det_xy is None:
                    continue
                dist = float(
                    np.linalg.norm(
                        np.array(det_xy) - np.array(robot_xy)
                    )
                )
                if dist < best_dist:
                    best_dist = dist
                    best_det = det
            if best_det is not None:
                return best_det
    return detections[0]


def _auto_intrusion_present(
    detections: list[dict],
    workspace_def: workspace.WorkspaceDefinition,
    *,
    target_xy: Optional[Tuple[float, float]],
    drop_target_mm: Optional[Tuple[float, float]],
    drop_ignore_radius_mm: float,
    target_match_mm: float,
) -> bool:
    """Check if there's an intrusion in the workspace (unexpected object)."""
    for det in detections:
        workspace_xy = _detection_workspace_mm(det, workspace_def)
        if workspace_xy is None:
            continue
        if not _workspace_contains(workspace_def, workspace_xy):
            continue
        if (
            drop_target_mm is not None
            and drop_ignore_radius_mm > 0.0
            and float(
                np.linalg.norm(
                    np.array(workspace_xy) - np.array(drop_target_mm)
                )
            )
            <= drop_ignore_radius_mm
        ):
            continue
        if target_xy is not None:
            if (
                float(
                    np.linalg.norm(
                        np.array(workspace_xy) - np.array(target_xy)
                    )
                )
                <= target_match_mm
            ):
                continue
        return True
    return False


# ---------------------------------------------------------------------------
# Detection coordinate extraction
# ---------------------------------------------------------------------------


def _detection_workspace_mm(
    detection: dict,
    workspace_def: workspace.WorkspaceDefinition,
) -> Optional[Tuple[float, float]]:
    """Get workspace (mm) coordinates for a detection."""
    workspace_mm = detection.get("workspace_mm")
    if workspace_mm is not None:
        return (float(workspace_mm[0]), float(workspace_mm[1]))
    center = detection.get("center")
    if center is None:
        return None
    try:
        workspace_mm = _workspace_pixel_to_mm(workspace_def, center)
    except Exception:  # noqa: BLE001
        return None
    return (float(workspace_mm[0]), float(workspace_mm[1]))


def _try_set_coord5_from_base(detection: dict, base_xyz: np.ndarray) -> Optional[Tuple[float, float]]:
    """Attempt to map a robot base XYZ to coord5 and annotate the detection.

    Returns the coord5 (x,y) tuple if mapping succeeds, otherwise None.
    """
    coord5_xyz = _base_to_coord5(base_xyz)
    if coord5_xyz is not None:
        coord5_xy = (float(coord5_xyz[0]), float(coord5_xyz[1]))
        detection["coord5_mm"] = coord5_xy
        detection["robot_base_mm"] = (
            float(base_xyz[0]),
            float(base_xyz[1]),
            float(base_xyz[2]),
        )
        return coord5_xy
    return None


def _detection_coord5_mm(
    detection: dict,
    workspace_def: Optional[workspace.WorkspaceDefinition],
) -> Optional[Tuple[float, float]]:
    """Get coord5 (mm) coordinates for a detection."""
    coord5_mm = detection.get("coord5_mm")
    if coord5_mm is not None:
        return (float(coord5_mm[0]), float(coord5_mm[1]))
    center = detection.get("center")
    if center is None:
        return None
    plate_xy = _pixel_to_plate_mm((float(center[0]), float(center[1])))
    plate_transform = _plate_to_robot_transform()
    if plate_xy is not None and plate_transform is not None:
        transform, origin_mm = plate_transform
        plate_rel = np.array(
            [
                float(plate_xy[0] - origin_mm[0]),
                float(plate_xy[1] - origin_mm[1]),
                0.0,
            ],
            dtype=np.float64,
        )
        base_xyz = transform.rotation @ plate_rel + transform.translation
        coord5_xy = _try_set_coord5_from_base(detection, base_xyz)
        if coord5_xy is not None:
            return coord5_xy
    base_xyz = _pixel_to_robot_base_xyz((float(center[0]), float(center[1])))
    if base_xyz is not None:
        coord5_xy = _try_set_coord5_from_base(detection, base_xyz)
        if coord5_xy is not None:
            return coord5_xy
    if workspace_def is None:
        return None
    workspace_mm = _detection_workspace_mm(detection, workspace_def)
    if workspace_mm is None:
        return None
    coord5_xy = _workspace_to_coord5_mm(workspace_def, workspace_mm)
    if coord5_xy is None:
        return None
    detection["coord5_mm"] = coord5_xy
    return coord5_xy


# ---------------------------------------------------------------------------
# Detection selection and pose building
# ---------------------------------------------------------------------------


def _select_follow_detection(
    detections: list[dict],
    last_workspace_mm: Optional[Tuple[float, float]],
    fallback: Optional[dict],
) -> Optional[dict]:
    """Select the best detection for follow mode based on proximity."""
    if not detections:
        return fallback
    if last_workspace_mm is not None:
        best_detection = None
        best_distance = float("inf")
        for detection in detections:
            workspace_mm = detection.get("workspace_mm")
            if workspace_mm is None:
                continue
            dist = float(
                np.linalg.norm(
                    np.array(workspace_mm[:2]) - np.array(last_workspace_mm)
                )
            )
            if dist < best_distance:
                best_distance = dist
                best_detection = detection
        if best_detection is not None:
            return best_detection
    return fallback or detections[0]


def _build_follow_pose_fetcher(
    workspace_def: workspace.WorkspaceDefinition,
    initial_detection: dict,
    z_offset_mm: float,
) -> Callable[[], Optional[dict]]:
    """Build a pose fetcher function for follow mode."""
    last_detection = initial_detection
    last_workspace_mm = _detection_workspace_mm(
        initial_detection, workspace_def)

    def _fetch() -> Optional[dict]:
        nonlocal last_detection, last_workspace_mm
        detections = _refresh_follow_detections()
        detection = _select_follow_detection(
            detections, last_workspace_mm, last_detection
        )
        if detection is None:
            return None
        last_detection = detection
        workspace_mm = _detection_workspace_mm(detection, workspace_def)
        if workspace_mm is None:
            workspace_mm = last_workspace_mm
        if workspace_mm is None:
            return None
        last_workspace_mm = workspace_mm
        return _pose_from_workspace_mm(
            workspace_def, workspace_mm[0], workspace_mm[1], z_offset_mm
        )

    return _fetch


# ---------------------------------------------------------------------------
# Auto pick-and-place loop
# ---------------------------------------------------------------------------


async def _auto_wait_for_clear(
    run_id: int,
    *,
    workspace_def: workspace.WorkspaceDefinition,
    target_xy: Optional[Tuple[float, float]],
    drop_target_mm: Optional[Tuple[float, float]],
    drop_ignore_radius_mm: float,
) -> None:
    """Wait for workspace to be clear of intrusions."""
    while True:
        status = _operation_status(run_id)
        if status in {"stopping", "stopped", "idle", "superseded"}:
            raise OperationAbort("Program stopped.")
        snapshot = _operation_snapshot()
        if snapshot.get("pause_reason") != "workspace_intrusion":
            await _wait_for_operation_resume(run_id)
            return
        frame, _depth = get_latest_frame()
        detections = SESSION.detections or []
        if frame is not None and (
            time.monotonic() - SESSION.auto_detection_updated_at
            > SETTINGS.auto_detection_stale_s
        ):
            try:
                detections = await asyncio.to_thread(
                    _auto_refresh_detections, frame
                )
            except Exception as exc:  # noqa: BLE001
                SESSION.last_detection_error = str(exc)
                LOGGER.warning("Auto detection refresh failed: %s", exc)
        if not _auto_intrusion_present(
            detections,
            workspace_def,
            target_xy=target_xy,
            drop_target_mm=drop_target_mm,
            drop_ignore_radius_mm=drop_ignore_radius_mm,
            target_match_mm=SETTINGS.auto_intrusion_match_mm,
        ):
            _resume_operation("Workspace clear. Resuming.")
            return
        await asyncio.sleep(SETTINGS.auto_idle_sleep_s)
