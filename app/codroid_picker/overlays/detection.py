"""Detection and monitoring overlay drawing functions."""

from __future__ import annotations

import logging
import time
from typing import Tuple

import cv2
import numpy as np

from app.codroid_picker.session import SESSION
from app.settings import SETTINGS
from .drawing import _draw_label

LOGGER = logging.getLogger(__name__)


def _refresh_detections(frame: np.ndarray) -> None:
    """Refresh detections at a throttled cadence."""
    if not SESSION.model_loaded or SESSION.yolo_model is None:
        return

    now = time.monotonic()
    if now - SESSION.auto_detection_updated_at < SETTINGS.auto_detection_stale_s:
        return

    from app.codroid_picker.auto_operation import _run_detection_serialized

    try:
        detections = _run_detection_serialized(
            frame,
            SESSION.detection_confidence,
            SESSION.detection_iou,
            SESSION.detection_max,
            SESSION.detection_classes,
        )
        if detections:
            SESSION.detections = detections
            SESSION.last_detection_error = None
    except Exception as exc:  # noqa: BLE001
        SESSION.last_detection_error = str(exc)
        LOGGER.debug("Detection refresh failed: %s", exc)
    finally:
        SESSION.auto_detection_updated_at = now


def draw_detection_overlay(frame: np.ndarray) -> Tuple[np.ndarray, str, str]:
    """Draw object detection overlay."""
    vis = frame.copy()

    _refresh_detections(frame)

    if not SESSION.detections:
        status = "No objects detected."
        return vis, status, "Detections: 0"

    try:
        count = len(SESSION.detections)

        for detection in SESSION.detections:
            # Extract detection data
            box = detection.get("bbox") or detection.get("box")
            confidence = detection.get("confidence", 0.0)
            class_name = detection.get("label") or detection.get(
                "class_name", "unknown"
            )

            if box is not None:
                x1, y1, x2, y2 = map(int, box)

                # Draw bounding box
                cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # Draw label with confidence
                label = f"{class_name}: {confidence:.2f}"
                label_size = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                cv2.rectangle(vis, (x1, y1 - label_size[1] - 10),
                              (x1 + label_size[0], y1), (0, 255, 0), -1)
                cv2.putText(vis, label, (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        status = f"Objects detected: {count}"
        metric = f"Detections: {count}"

    except Exception as exc:
        status = f"Detection overlay error: {exc}"
        metric = "Detections: error"

    return vis, status, metric


def draw_drop_marker_overlay(frame: np.ndarray, marker_id: int) -> Tuple[np.ndarray, str, str]:
    """Draw drop target marker detection overlay."""
    vis = frame.copy()

    try:
        # Detect the drop marker
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Use ArUco detection for drop marker
        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        detector_params = cv2.aruco.DetectorParameters()
        detector_params.detectInvertedMarker = True
        detector = cv2.aruco.ArucoDetector(dictionary, detector_params)

        corners, ids, _ = detector.detectMarkers(gray)

        if ids is not None and marker_id in ids.flatten():
            # Find the index of our marker
            marker_idx = np.where(ids.flatten() == marker_id)[0][0]
            marker_corners = corners[marker_idx]

            # Draw the marker
            cv2.polylines(vis, [marker_corners.astype(int)],
                          True, (0, 165, 255), 2)

            # Draw center point
            center = np.mean(marker_corners, axis=0).astype(int).flatten()
            cv2.circle(vis, tuple(center), 5, (0, 165, 255), -1)

            # Draw marker ID
            _draw_label(vis, f"Drop #{marker_id}", tuple(
                center + [10, -10]), (0, 165, 255))

            status = f"Drop marker #{marker_id} detected"
            metric = f"Drop marker: found"
        else:
            status = f"Drop marker #{marker_id} not detected"
            metric = f"Drop marker: missing"

    except Exception as exc:
        status = f"Drop marker detection error: {exc}"
        metric = "Drop marker: error"

    return vis, status, metric


def draw_monitor_overlay(frame: np.ndarray, marker_id: int) -> Tuple[np.ndarray, str, str, str]:
    """Draw combined detection and drop marker overlay for monitoring."""
    vis, status, detection_metric = draw_detection_overlay(frame)
    vis, drop_status, drop_metric = draw_drop_marker_overlay(vis, marker_id)
    status = f"{status} | {drop_status}"
    return vis, status, detection_metric, drop_metric
