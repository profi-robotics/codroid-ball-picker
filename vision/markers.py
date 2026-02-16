"""
Marker detection utilities for defining the dynamic workspace using two fiducials.

The code uses OpenCV ArUco (AprilTag-compatible) detection to find two markers
that define the logical workspace rectangle:
- Marker A: top-left, maps to workspace (0, 0)
- Marker B: bottom-right, maps to workspace (W, H)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Tuple

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


@dataclass
class WorkspaceMarkerDetectionResult:
    """Compatibility container for workspace marker detection overlays."""

    detections: Dict[int, MarkerDetection]
    marker_ids: Tuple[int, int]
    marker_size_mm: float


def _get_dictionary(dictionary_name: str) -> cv2.aruco_Dictionary:
    dictionary_id = getattr(cv2.aruco, dictionary_name)
    return cv2.aruco.getPredefinedDictionary(dictionary_id)


def _detect_all_markers(
    image: np.ndarray,
    dictionary_name: str = "DICT_4X4_50",
) -> Dict[int, MarkerDetection]:
    if image is None or image.size == 0:
        raise ValueError("Empty image provided for marker detection.")

    dictionary = _get_dictionary(dictionary_name)
    parameters = cv2.aruco.DetectorParameters()
    parameters.detectInvertedMarker = True

    # Optimize detection for small markers (20mm) by using smaller adaptive threshold windows
    # This prevents the threshold window from being too large and "breaking" small markers
    # Smaller minimum for small markers (default: 3)
    parameters.adaptiveThreshWinSizeMin = 3
    # Reasonable maximum (default: 23)
    parameters.adaptiveThreshWinSizeMax = 23
    # More granular steps (default: 10)
    parameters.adaptiveThreshWinSizeStep = 4

    # Lower the minimum marker perimeter rate to detect smaller markers more reliably
    # This is the ratio of marker perimeter to image perimeter
    parameters.minMarkerPerimeterRate = 0.01  # More lenient (default: 0.03)

    # Improve corner refinement for more accurate detection
    parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    # Small window for precision (default: 5)
    parameters.cornerRefinementWinSize = 3

    # Relax polygon approximation to better handle slightly distorted markers
    # More lenient (default: 0.03)
    parameters.polygonalApproxAccuracyRate = 0.05

    detector = cv2.aruco.ArucoDetector(dictionary, parameters)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    corners_list, ids, _ = detector.detectMarkers(gray)
    if ids is None or len(ids) == 0:
        raise RuntimeError(
            "No markers detected. Ensure markers are in frame and high-contrast.")

    detections: Dict[int, MarkerDetection] = {}
    for marker_corners, marker_id in zip(corners_list, ids.flatten()):
        pts = marker_corners.reshape((4, 2))
        center = tuple(np.mean(pts, axis=0).tolist())
        detections[int(marker_id)] = MarkerDetection(
            marker_id=int(marker_id), corners=pts, center=center)

    return detections


def detect_all_markers(
    image: np.ndarray,
    dictionary_name: str = "DICT_4X4_50",
) -> Dict[int, MarkerDetection]:
    """Detect all ArUco markers in the frame and return a mapping keyed by ID."""

    return _detect_all_markers(image, dictionary_name=dictionary_name)


def detect_markers_map(
    image: np.ndarray,
    marker_ids: Iterable[int],
    dictionary_name: str = "DICT_4X4_50",
) -> Dict[int, MarkerDetection]:
    """Detect a specific set of markers and return a mapping keyed by ID."""

    detections = _detect_all_markers(image, dictionary_name=dictionary_name)
    requested = tuple(marker_ids)
    missing = [marker for marker in requested if marker not in detections]
    if missing:
        raise RuntimeError(f"Missing required markers: {missing}")

    return {marker_id: detections[marker_id] for marker_id in requested}


def workspace_markers_from_detections(
    detections: Dict[int, MarkerDetection],
    marker_ids: Tuple[int, int],
) -> WorkspaceMarkers:
    missing = [marker for marker in marker_ids if marker not in detections]
    if missing:
        raise RuntimeError(f"Missing required markers: {missing}")

    return WorkspaceMarkers(marker_a=detections[marker_ids[0]], marker_b=detections[marker_ids[1]])


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

    detections = detect_markers_map(
        image, marker_ids, dictionary_name=dictionary_name)
    return workspace_markers_from_detections(detections, marker_ids)


def detect_workspace_markers(
    image: np.ndarray,
    marker_ids: Tuple[int, int],
    marker_size_mm: float,
    dictionary_name: str = "DICT_4X4_50",
) -> WorkspaceMarkerDetectionResult:
    """Detect workspace markers while allowing partial/empty results.

    This compatibility API is used by the live workspace overlay, which should
    continue rendering even when one or both markers are not currently visible.
    """

    try:
        all_detections = _detect_all_markers(
            image, dictionary_name=dictionary_name)
    except RuntimeError:
        all_detections = {}

    requested = {
        marker_id: all_detections[marker_id]
        for marker_id in marker_ids
        if marker_id in all_detections
    }
    return WorkspaceMarkerDetectionResult(
        detections=requested,
        marker_ids=marker_ids,
        marker_size_mm=marker_size_mm,
    )
