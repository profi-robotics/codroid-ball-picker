"""
Camera calibration and workspace plane estimation utilities for the Codroid + OAK-D setup.

This module focuses on producing artifacts that downstream mapping and robot
control code can consume without additional transformation logic:
- Camera intrinsics and distortion coefficients
- Workspace plane representation (normal + distance)
- Pixel-to-metric scale factors along the workspace plane

All outputs are serialisable to JSON/YAML so they can be persisted on disk and
reused without recalibration when the hardware setup is unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import cv2
import numpy as np

from .aruco_board import ArucoBoardSpec


@dataclass
class CameraCalibration:
    """Container for intrinsic camera parameters."""

    camera_matrix: np.ndarray
    distortion_coeffs: np.ndarray

    def to_dict(self) -> Dict[str, List[List[float]]]:
        return {
            "camera_matrix": self.camera_matrix.tolist(),
            "distortion_coeffs": self.distortion_coeffs.tolist(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, List[List[float]]]) -> "CameraCalibration":
        return cls(
            camera_matrix=np.asarray(data["camera_matrix"], dtype=np.float64),
            distortion_coeffs=np.asarray(data["distortion_coeffs"], dtype=np.float64),
        )


@dataclass
class WorkspacePlane:
    """Represents a planar workspace in camera coordinates."""

    normal: np.ndarray
    offset: float

    def to_dict(self) -> Dict[str, List[float]]:
        return {
            "normal": self.normal.tolist(),
            "offset": float(self.offset),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, List[float]]) -> "WorkspacePlane":
        return cls(normal=np.asarray(data["normal"], dtype=np.float64), offset=float(data["offset"]))


@dataclass
class CalibrationBundle:
    """Full calibration payload saved to disk."""

    camera: CameraCalibration
    workspace_plane: WorkspacePlane
    pixel_to_mm: float
    plate_homography: np.ndarray | None = None
    plate_layout_path: str | None = None

    def to_dict(self) -> Dict[str, object]:
        data: Dict[str, object] = {
            "camera": self.camera.to_dict(),
            "workspace_plane": self.workspace_plane.to_dict(),
            "pixel_to_mm": float(self.pixel_to_mm),
        }
        if self.plate_homography is not None:
            data["plate_homography"] = self.plate_homography.tolist()
        if self.plate_layout_path:
            data["plate_layout_path"] = self.plate_layout_path
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, object]) -> "CalibrationBundle":
        plate_homography = data.get("plate_homography")
        if plate_homography is not None:
            plate_homography = np.asarray(plate_homography, dtype=np.float64)
        plate_layout_path = data.get("plate_layout_path")
        return cls(
            camera=CameraCalibration.from_dict(data["camera"]),
            workspace_plane=WorkspacePlane.from_dict(data["workspace_plane"]),
            pixel_to_mm=float(data["pixel_to_mm"]),
            plate_homography=plate_homography,
            plate_layout_path=str(plate_layout_path) if plate_layout_path else None,
        )


def calibrate_camera(image_paths: Iterable[Path], board_size: Tuple[int, int], square_size_mm: float) -> CameraCalibration:
    """
    Calibrate the OAK-D camera using a chessboard pattern.

    Args:
        image_paths: Iterable of captured chessboard images.
        board_size: Chessboard inner corner pattern size, e.g. (7, 6).
        square_size_mm: Physical size of a chessboard square in millimetres.

    Returns:
        CameraCalibration with computed intrinsics.
    """

    # Prepare object points based on the known square size.
    objp = np.zeros((board_size[0] * board_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0 : board_size[0], 0 : board_size[1]].T.reshape(-1, 2)
    objp *= square_size_mm

    objpoints: List[np.ndarray] = []
    imgpoints: List[np.ndarray] = []

    for path in image_paths:
        image = cv2.imread(str(path))
        if image is None:
            raise FileNotFoundError(f"Calibration image not found: {path}")

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, board_size, None)
        if not ret:
            continue

        criteria = (
            cv2.TermCriteria_EPS + cv2.TermCriteria_MAX_ITER,
            30,
            0.001,
        )
        corners_subpix = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

        objpoints.append(objp)
        imgpoints.append(corners_subpix)

    if not objpoints:
        raise RuntimeError("No chessboard detections found. Check calibration images and board_size.")

    ret, camera_matrix, dist_coeffs, _, _ = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    if not ret:
        raise RuntimeError("OpenCV calibration failed to converge.")

    return CameraCalibration(camera_matrix=camera_matrix, distortion_coeffs=dist_coeffs)


def calibrate_camera_aruco(image_paths: Iterable[Path], board_spec: ArucoBoardSpec) -> CameraCalibration:
    """
    Calibrate the camera using an ArUco grid board.

    Args:
        image_paths: Iterable of captured ArUco board images.
        board_spec: Board specification describing marker layout in millimetres.

    Returns:
        CameraCalibration with computed intrinsics.
    """

    def _detect_markers(gray: np.ndarray) -> Dict[int, np.ndarray]:
        dictionary_id = getattr(cv2.aruco, board_spec.dictionary)
        dictionary = cv2.aruco.getPredefinedDictionary(dictionary_id)
        parameters = cv2.aruco.DetectorParameters()
        detector = cv2.aruco.ArucoDetector(dictionary, parameters)
        corners_list, ids, _ = detector.detectMarkers(gray)
        if ids is None or len(ids) == 0:
            return {}
        detections: Dict[int, np.ndarray] = {}
        for marker_corners, marker_id in zip(corners_list, ids.flatten()):
            detections[int(marker_id)] = marker_corners.reshape((4, 2))
        return detections

    board_corners = board_spec.marker_corners()
    objpoints: List[np.ndarray] = []
    imgpoints: List[np.ndarray] = []
    image_size = None

    for path in image_paths:
        image = cv2.imread(str(path))
        if image is None:
            raise FileNotFoundError(f"Calibration image not found: {path}")

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if image_size is None:
            image_size = (gray.shape[1], gray.shape[0])
        detections = _detect_markers(gray)
        if not detections:
            continue

        obj = []
        img = []
        for marker_id, img_corners in detections.items():
            board_marker = board_corners.get(marker_id)
            if board_marker is None:
                continue
            obj.extend(board_marker.tolist())
            img.extend(img_corners.tolist())

        if len(obj) < 4:
            continue
        objpoints.append(np.asarray(obj, dtype=np.float32))
        imgpoints.append(np.asarray(img, dtype=np.float32))

    if not objpoints or image_size is None:
        raise RuntimeError("No ArUco board detections found. Check calibration images and board layout.")

    ret, camera_matrix, dist_coeffs, _, _ = cv2.calibrateCamera(objpoints, imgpoints, image_size, None, None)
    if not ret:
        raise RuntimeError("OpenCV calibration failed to converge.")

    return CameraCalibration(camera_matrix=camera_matrix, distortion_coeffs=dist_coeffs)


def estimate_workspace_plane(depth_frame: np.ndarray, intrinsics: CameraCalibration, max_points: int = 5000) -> WorkspacePlane:
    """
    Estimate the workspace plane equation using a depth frame.

    A simple RANSAC plane fit is applied to a subset of depth points to keep the
    routine fast enough to run during demo initialisation.
    """

    if depth_frame.ndim != 2:
        raise ValueError("Depth frame must be a single-channel array of depths in metres.")

    # Filter out invalid/zero depths.
    valid_mask = np.isfinite(depth_frame) & (depth_frame > 0)
    valid_indices = np.argwhere(valid_mask)
    if valid_indices.size == 0:
        raise RuntimeError("No valid depth pixels available for plane estimation.")

    sample_count = min(max_points, len(valid_indices))
    choice_idx = np.random.choice(len(valid_indices), size=sample_count, replace=False)
    uv = valid_indices[choice_idx]
    v = uv[:, 0]
    u = uv[:, 1]
    z = depth_frame[v, u]

    fx = intrinsics.camera_matrix[0, 0]
    fy = intrinsics.camera_matrix[1, 1]
    cx = intrinsics.camera_matrix[0, 2]
    cy = intrinsics.camera_matrix[1, 2]

    # Back-project sample points into 3D camera coordinates.
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    pts = np.stack([x, y, z], axis=1)

    # Fit plane via SVD on centred points.
    centroid = np.mean(pts, axis=0)
    demeaned = pts - centroid
    _, _, vh = np.linalg.svd(demeaned)
    normal = vh[-1]
    normal /= np.linalg.norm(normal)
    offset = -np.dot(normal, centroid)

    return WorkspacePlane(normal=normal, offset=offset)


def estimate_pixel_to_mm(marker_pixel_distance: float, marker_real_distance_mm: float) -> float:
    """Compute a pixel-to-mm scale factor from a known marker separation."""

    if marker_pixel_distance <= 0:
        raise ValueError("Marker pixel distance must be positive.")
    if marker_real_distance_mm <= 0:
        raise ValueError("Marker real distance must be positive.")

    return marker_real_distance_mm / marker_pixel_distance


def chessboard_pixel_scale(image_path: Path, board_size: Tuple[int, int], square_size_mm: float) -> float:
    """
    Estimate mm-per-pixel from a single chessboard image by averaging square edge lengths in pixels.
    """

    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Calibration image not found: {image_path}")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, board_size, None)
    if not ret:
        raise RuntimeError(f"Chessboard not detected in {image_path}")

    # Use refined corners for more accurate spacing.
    criteria = (cv2.TermCriteria_EPS + cv2.TermCriteria_MAX_ITER, 30, 0.001)
    corners_subpix = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria).reshape(-1, 2)

    # Compute average horizontal and vertical pixel spacing between adjacent corners.
    cols = board_size[0]
    rows = board_size[1]
    horiz_distances = []
    vert_distances = []
    for r in range(rows):
        for c in range(cols - 1):
            idx = r * cols + c
            horiz_distances.append(np.linalg.norm(corners_subpix[idx + 1] - corners_subpix[idx]))
    for r in range(rows - 1):
        for c in range(cols):
            idx = r * cols + c
            vert_distances.append(np.linalg.norm(corners_subpix[idx + cols] - corners_subpix[idx]))

    pixel_per_square = float(np.mean(horiz_distances + vert_distances))
    if pixel_per_square <= 0:
        raise RuntimeError("Invalid chessboard pixel spacing computed.")

    return square_size_mm / pixel_per_square


def aruco_board_pixel_scale(image_path: Path, board_spec: ArucoBoardSpec) -> float:
    """
    Estimate mm-per-pixel from a single ArUco board image by averaging marker edge lengths.
    """

    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Calibration image not found: {image_path}")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    dictionary_id = getattr(cv2.aruco, board_spec.dictionary)
    dictionary = cv2.aruco.getPredefinedDictionary(dictionary_id)
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(dictionary, parameters)
    corners_list, ids, _ = detector.detectMarkers(gray)
    if ids is None or len(ids) == 0:
        raise RuntimeError(f"No ArUco markers detected in {image_path}")

    edge_lengths = []
    board_corners = board_spec.marker_corners()
    for marker_corners, marker_id in zip(corners_list, ids.flatten()):
        if int(marker_id) not in board_corners:
            continue
        pts = marker_corners.reshape((4, 2))
        edge_lengths.append(np.linalg.norm(pts[1] - pts[0]))
        edge_lengths.append(np.linalg.norm(pts[2] - pts[1]))

    if not edge_lengths:
        raise RuntimeError("No board markers matched the layout for pixel scale estimation.")

    pixel_per_marker = float(np.mean(edge_lengths))
    if pixel_per_marker <= 0:
        raise RuntimeError("Invalid ArUco pixel spacing computed.")

    return board_spec.marker_length_mm / pixel_per_marker


def save_calibration(bundle: CalibrationBundle, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **bundle.to_dict())


def load_calibration(path: Path) -> CalibrationBundle:
    data = np.load(path, allow_pickle=True)
    def _unwrap(value: object) -> object:
        if isinstance(value, np.ndarray) and value.shape == () and value.dtype == object:
            return value.item()
        return value

    # np.load wraps nested structures; unwrap them back to dict for constructors.
    camera_dict = _unwrap(data["camera"])
    workspace_dict = _unwrap(data["workspace_plane"])
    plate_homography = None
    if "plate_homography" in data.files:
        plate_homography = _unwrap(data["plate_homography"])
        if plate_homography is not None:
            plate_homography = np.asarray(plate_homography, dtype=np.float64)
    plate_layout_path = None
    if "plate_layout_path" in data.files:
        plate_layout_path = _unwrap(data["plate_layout_path"])
    return CalibrationBundle(
        camera=CameraCalibration.from_dict(camera_dict),
        workspace_plane=WorkspacePlane.from_dict(workspace_dict),
        pixel_to_mm=float(data["pixel_to_mm"].item()),
        plate_homography=plate_homography,
        plate_layout_path=str(plate_layout_path) if plate_layout_path else None,
    )
