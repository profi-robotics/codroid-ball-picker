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

    def to_dict(self) -> Dict[str, object]:
        return {
            "camera": self.camera.to_dict(),
            "workspace_plane": self.workspace_plane.to_dict(),
            "pixel_to_mm": float(self.pixel_to_mm),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, object]) -> "CalibrationBundle":
        return cls(
            camera=CameraCalibration.from_dict(data["camera"]),
            workspace_plane=WorkspacePlane.from_dict(data["workspace_plane"]),
            pixel_to_mm=float(data["pixel_to_mm"]),
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


def estimate_workspace_plane(depth_frame: np.ndarray, intrinsics: CameraCalibration, max_points: int = 5000) -> WorkspacePlane:
    """
    Estimate the workspace plane equation using a depth frame.

    A simple RANSAC plane fit is applied to a subset of depth points to keep the
    routine fast enough to run during demo initialisation.
    """

    if depth_frame.ndim != 2:
        raise ValueError("Depth frame must be a single-channel array of depths in metres.")

    rows, cols = depth_frame.shape
    u = np.random.randint(0, cols, size=max_points)
    v = np.random.randint(0, rows, size=max_points)
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


def save_calibration(bundle: CalibrationBundle, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **bundle.to_dict())


def load_calibration(path: Path) -> CalibrationBundle:
    data = np.load(path, allow_pickle=True)
    # np.load wraps nested structures; unwrap them back to dict for constructors.
    camera_dict = data["camera"].item() if isinstance(data["camera"].item(), dict) else data["camera"].item()
    workspace_dict = data["workspace_plane"].item() if isinstance(data["workspace_plane"].item(), dict) else data["workspace_plane"].item()
    return CalibrationBundle(
        camera=CameraCalibration.from_dict(camera_dict),
        workspace_plane=WorkspacePlane.from_dict(workspace_dict),
        pixel_to_mm=float(data["pixel_to_mm"].item()),
    )
