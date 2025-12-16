"""
Marker detection utilities for defining the dynamic workspace using two fiducials.

The code uses OpenCV ArUco (AprilTag-compatible) detection to find two markers
that define the logical workspace rectangle:
- Marker A: top-left, maps to workspace (0, 0)
- Marker B: bottom-right, maps to workspace (W, H)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import cv2
import numpy as np


@dataclass
class MarkerDetection:
    marker_id: int
    corners: np.ndarray  # shape (4, 2)
    center: Tuple[float, float]


@dataclass
class WorkspaceMarkers:
    marker_a: MarkerDetection
    marker_b: MarkerDetection


def _get_dictionary(dictionary_name: str) -> cv2.aruco_Dictionary:
    dictionary_id = getattr(cv2.aruco, dictionary_name)
    return cv2.aruco.getPredefinedDictionary(dictionary_id)


def detect_markers(
    image: np.ndarray,
    marker_ids: Tuple[int, int],
    dictionary_name: str = "DICT_4X4_50",
) -> WorkspaceMarkers:
    """
    Detect the top-left (A) and bottom-right (B) markers within an RGB image.

    Args:
        image: Input RGB or BGR frame.
        marker_ids: Tuple of two marker IDs representing (A, B).
        dictionary_name: ArUco dictionary name to use for detection.

    Returns:
        WorkspaceMarkers containing detections for A and B.
    """

    if image is None or image.size == 0:
        raise ValueError("Empty image provided for marker detection.")

    dictionary = _get_dictionary(dictionary_name)
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(dictionary, parameters)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    corners_list, ids, _ = detector.detectMarkers(gray)
    if ids is None or len(ids) == 0:
        raise RuntimeError("No markers detected. Ensure markers are in frame and high-contrast.")

    detections: Dict[int, MarkerDetection] = {}
    for marker_corners, marker_id in zip(corners_list, ids.flatten()):
        pts = marker_corners.reshape((4, 2))
        center = tuple(np.mean(pts, axis=0).tolist())
        detections[int(marker_id)] = MarkerDetection(marker_id=int(marker_id), corners=pts, center=center)

    missing = [marker for marker in marker_ids if marker not in detections]
    if missing:
        raise RuntimeError(f"Missing required markers: {missing}")

    return WorkspaceMarkers(marker_a=detections[marker_ids[0]], marker_b=detections[marker_ids[1]])
