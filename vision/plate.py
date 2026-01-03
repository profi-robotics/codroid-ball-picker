"""
Calibration plate utilities for mapping pixels to real-world millimetres.

The plate layout is defined in data/markers/robot_calibration/calibration_plate_layout.json and
uses ArUco markers to establish a homography between image pixels and plate
coordinates, plus optional calibration point coordinates for robot touch targets.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import cv2
import numpy as np

from .markers import MarkerDetection, detect_all_markers


@dataclass
class PlateMarker:
    marker_id: int
    center_mm: np.ndarray
    role: str | None = None
    sequence: int | None = None


@dataclass
class CalibrationPoint:
    sequence: int
    center_mm: np.ndarray


@dataclass
class CalibrationPlate:
    marker_size_mm: float
    plate_size_mm: Tuple[float, float]
    dictionary: str
    markers: Dict[int, PlateMarker]
    calibration_points: List[CalibrationPoint]
    coordinate_system: str

    def robot_calibration_markers(self) -> List[PlateMarker]:
        robots = [marker for marker in self.markers.values() if marker.role == "robot_calibration"]
        return sorted(robots, key=lambda marker: marker.sequence or 0)

    def robot_calibration_points(self) -> List[CalibrationPoint]:
        return sorted(self.calibration_points, key=lambda point: point.sequence)


@dataclass
class PlateCalibration:
    homography: np.ndarray
    reprojection_error_mm: float
    detections: Dict[int, MarkerDetection]
    plate: CalibrationPlate


def load_plate_layout(layout_path: Path) -> CalibrationPlate:
    data = json.loads(layout_path.read_text(encoding="utf-8"))
    markers: Dict[int, PlateMarker] = {}
    for entry in data["markers"]:
        marker_id = int(entry["id"])
        center_mm = np.asarray(entry["center_mm"], dtype=np.float64)
        markers[marker_id] = PlateMarker(
            marker_id=marker_id,
            center_mm=center_mm,
            role=entry.get("role"),
            sequence=entry.get("sequence"),
        )

    calibration_points: List[CalibrationPoint] = []
    for entry in data.get("calibration_points", []):
        calibration_points.append(
            CalibrationPoint(
                sequence=int(entry["sequence"]),
                center_mm=np.asarray(entry["center_mm"], dtype=np.float64),
            )
        )

    return CalibrationPlate(
        marker_size_mm=float(data["marker_size_mm"]),
        plate_size_mm=(float(data["plate_size_mm"][0]), float(data["plate_size_mm"][1])),
        dictionary=data.get("dictionary", "DICT_4X4_50"),
        markers=markers,
        calibration_points=calibration_points,
        coordinate_system=data.get("coordinate_system", "origin at plate top-left, +x right, +y down"),
    )


def marker_corners_mm(center_mm: np.ndarray, marker_size_mm: float) -> np.ndarray:
    half = marker_size_mm / 2.0
    return np.array(
        [
            [center_mm[0] - half, center_mm[1] - half],
            [center_mm[0] + half, center_mm[1] - half],
            [center_mm[0] + half, center_mm[1] + half],
            [center_mm[0] - half, center_mm[1] + half],
        ],
        dtype=np.float64,
    )


def collect_correspondences(
    detections: Dict[int, MarkerDetection],
    plate: CalibrationPlate,
) -> Tuple[np.ndarray, np.ndarray]:
    image_points: List[np.ndarray] = []
    plate_points: List[np.ndarray] = []

    for marker_id, detection in detections.items():
        marker = plate.markers.get(marker_id)
        if marker is None:
            continue
        image_points.append(detection.corners.astype(np.float64))
        plate_points.append(marker_corners_mm(marker.center_mm, plate.marker_size_mm))

    if not image_points:
        raise RuntimeError("No calibration plate markers found in the detections.")

    return np.vstack(image_points), np.vstack(plate_points)


def estimate_plate_calibration(
    image: np.ndarray,
    plate: CalibrationPlate,
    required_ids: Iterable[int] | None = None,
) -> PlateCalibration:
    detections_all = detect_all_markers(image, dictionary_name=plate.dictionary)
    detections = {marker_id: det for marker_id, det in detections_all.items() if marker_id in plate.markers}
    required = tuple(required_ids or ())
    missing = [marker_id for marker_id in required if marker_id not in detections]
    if missing:
        raise RuntimeError(f"Missing required calibration plate markers: {missing}")

    image_points, plate_points = collect_correspondences(detections, plate)
    if len(image_points) < 4:
        raise RuntimeError("At least one full marker is required to estimate a plate homography.")

    homography, _mask = cv2.findHomography(image_points, plate_points, method=cv2.RANSAC)
    if homography is None:
        raise RuntimeError("Failed to compute homography from calibration plate markers.")

    projected = cv2.perspectiveTransform(image_points.reshape(-1, 1, 2), homography).reshape(-1, 2)
    reprojection_error = float(np.mean(np.linalg.norm(projected - plate_points, axis=1)))

    return PlateCalibration(
        homography=homography,
        reprojection_error_mm=reprojection_error,
        detections=detections,
        plate=plate,
    )


def estimate_pixel_scale_from_markers(
    detections: Dict[int, MarkerDetection],
    plate: CalibrationPlate,
    marker_ids: Tuple[int, int],
) -> float:
    if marker_ids[0] not in detections or marker_ids[1] not in detections:
        raise RuntimeError("Required markers missing for pixel scale estimation.")

    px_distance = np.linalg.norm(
        np.asarray(detections[marker_ids[0]].center) - np.asarray(detections[marker_ids[1]].center)
    )
    if px_distance <= 0:
        raise RuntimeError("Marker pixel distance must be positive for scale estimation.")

    mm_distance = np.linalg.norm(
        plate.markers[marker_ids[0]].center_mm - plate.markers[marker_ids[1]].center_mm
    )
    if mm_distance <= 0:
        raise RuntimeError("Marker spacing must be positive for scale estimation.")

    return float(mm_distance / px_distance)


def image_points_to_plate_mm(homography: np.ndarray, points: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 1, 2)
    mapped = cv2.perspectiveTransform(pts, homography).reshape(-1, 2)
    return mapped
