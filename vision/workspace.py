"""
Workspace computation helpers.

The workspace is defined in logical 2D coordinates using two detected markers.
This module maps between:
- Marker pixel positions → workspace dimensions (mm)
- Workspace coordinates → camera plane coordinates
- Camera coordinates → robot base coordinates using a stored extrinsic transform
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

from .calibration import CalibrationBundle
from .markers import WorkspaceMarkers
from .markers import MarkerDetection
from .plate import CalibrationPlate


@dataclass
class RigidTransform:
    rotation: np.ndarray  # 3x3
    translation: np.ndarray  # 3x1

    def as_matrix(self) -> np.ndarray:
        mat = np.eye(4)
        mat[:3, :3] = self.rotation
        mat[:3, 3] = self.translation.reshape(3)
        return mat

    @classmethod
    def from_matrix(cls, mat: np.ndarray) -> "RigidTransform":
        return cls(rotation=mat[:3, :3], translation=mat[:3, 3])

    def to_dict(self) -> Dict[str, list]:
        return {"rotation": self.rotation.tolist(), "translation": self.translation.tolist()}

    @classmethod
    def from_dict(cls, data: Dict[str, list]) -> "RigidTransform":
        return cls(rotation=np.asarray(data["rotation"], dtype=np.float64), translation=np.asarray(data["translation"], dtype=np.float64))


@dataclass
class WorkspaceDefinition:
    width_mm: float
    height_mm: float
    origin_camera: np.ndarray
    x_axis_camera: np.ndarray
    y_axis_camera: np.ndarray
    camera_to_robot: RigidTransform

    def to_dict(self) -> Dict[str, object]:
        return {
            "width_mm": float(self.width_mm),
            "height_mm": float(self.height_mm),
            "origin_camera": self.origin_camera.tolist(),
            "x_axis_camera": self.x_axis_camera.tolist(),
            "y_axis_camera": self.y_axis_camera.tolist(),
            "camera_to_robot": self.camera_to_robot.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, object]) -> "WorkspaceDefinition":
        return cls(
            width_mm=float(data["width_mm"]),
            height_mm=float(data["height_mm"]),
            origin_camera=np.asarray(data["origin_camera"], dtype=np.float64),
            x_axis_camera=np.asarray(data["x_axis_camera"], dtype=np.float64),
            y_axis_camera=np.asarray(data["y_axis_camera"], dtype=np.float64),
            camera_to_robot=RigidTransform.from_dict(data["camera_to_robot"]),
        )

    def workspace_to_camera(self, x_mm: float, y_mm: float, z_offset_mm: float = 0.0) -> np.ndarray:
        point = self.origin_camera + \
            (x_mm * self.x_axis_camera) + (y_mm * self.y_axis_camera)
        point += z_offset_mm * np.cross(self.x_axis_camera, self.y_axis_camera)
        return point

    def workspace_to_robot(self, x_mm: float, y_mm: float, z_offset_mm: float = 0.0) -> np.ndarray:
        camera_point = self.workspace_to_camera(x_mm, y_mm, z_offset_mm)
        camera_h = np.hstack([camera_point, 1.0])
        robot_point_h = self.camera_to_robot.as_matrix() @ camera_h
        return robot_point_h[:3]


def compute_workspace(
    markers: WorkspaceMarkers,
    calibration: CalibrationBundle,
    marker_size_mm: float,
    camera_to_robot: RigidTransform,
    scale_multiplier: float = 1.0,
) -> WorkspaceDefinition:
    """
    Build a workspace representation given two diagonal corner markers.

    The workspace is a rectangle defined by two markers placed at opposite corners
    (diagonal). The function computes:
    1. The scale from marker size
    2. The workspace bounding rectangle from diagonal corners
    3. Axis-aligned coordinate system with origin at marker_a corner
    
    Scale is estimated from the detected marker edge length (more reliable than
    a static pixel_to_mm when the camera standoff changes) and falls back to the
    calibration bundle if marker-derived scale is unavailable.
    """

    marker_a = markers.marker_a
    marker_b = markers.marker_b

    def _edge_scales(marker: MarkerDetection) -> Tuple[float, float]:
        # Assume OpenCV ordering: corners[0]=top-left, 1=top-right, 2=bottom-right, 3=bottom-left
        x_len = np.linalg.norm(marker.corners[1] - marker.corners[0])
        y_len = np.linalg.norm(marker.corners[2] - marker.corners[1])
        x_scale = marker_size_mm / x_len if x_len > 0 else 0.0
        y_scale = marker_size_mm / y_len if y_len > 0 else 0.0
        return x_scale, y_scale

    ax_scale, ay_scale = _edge_scales(marker_a)
    bx_scale, by_scale = _edge_scales(marker_b)
    mm_per_px_x = np.mean([s for s in (ax_scale, bx_scale) if s > 0]) if any(
        s > 0 for s in (ax_scale, bx_scale)) else 0.0
    mm_per_px_y = np.mean([s for s in (ay_scale, by_scale) if s > 0]) if any(
        s > 0 for s in (ay_scale, by_scale)) else 0.0

    if mm_per_px_x == 0 or mm_per_px_y == 0:
        fallback = calibration.pixel_to_mm if calibration.pixel_to_mm > 0 else marker_size_mm / max(
            1e-6, np.linalg.norm(marker_a.corners[1] - marker_a.corners[0])
        )
        mm_per_px_x = mm_per_px_x or fallback
        mm_per_px_y = mm_per_px_y or fallback

    # Compute workspace as a rectangle with diagonal corners at marker_a and marker_b
    scale_avg = (mm_per_px_x + mm_per_px_y) / 2.0
    scale_avg *= scale_multiplier
    
    # Get the diagonal vector from marker_a to marker_b in pixel space
    diagonal_px = np.asarray(marker_b.center) - np.asarray(marker_a.center)
    
    # Compute workspace axes aligned with the marker orientations (table surface orientation)
    # Use marker_a's edge vectors to define the workspace coordinate frame
    # Marker corners: [0]=top-left, [1]=top-right, [2]=bottom-right, [3]=bottom-left
    
    # X-axis: direction from left edge to right edge of marker_a (normalized)
    x_axis_px = (marker_a.corners[1] - marker_a.corners[0])  # top-right - top-left
    x_axis_px_len = np.linalg.norm(x_axis_px)
    if x_axis_px_len > 0:
        x_axis_px = x_axis_px / x_axis_px_len
    else:
        x_axis_px = np.array([1.0, 0.0])
    
    # Y-axis: direction from top edge to bottom edge of marker_a (normalized)
    y_axis_px = (marker_a.corners[3] - marker_a.corners[0])  # bottom-left - top-left
    y_axis_px_len = np.linalg.norm(y_axis_px)
    if y_axis_px_len > 0:
        y_axis_px = y_axis_px / y_axis_px_len
    else:
        y_axis_px = np.array([0.0, 1.0])
    
    # Ensure orthogonality: recompute y as perpendicular to x
    y_axis_px = y_axis_px - (np.dot(y_axis_px, x_axis_px) * x_axis_px)
    y_axis_px_len = np.linalg.norm(y_axis_px)
    if y_axis_px_len > 0:
        y_axis_px = y_axis_px / y_axis_px_len
    else:
        # Fallback: rotate x by 90 degrees
        y_axis_px = np.array([-x_axis_px[1], x_axis_px[0]])
    
    # Project diagonal vector onto the rotated axes to get width and height
    width_px = np.dot(diagonal_px, x_axis_px)
    height_px = np.dot(diagonal_px, y_axis_px)
    
    # Convert to mm
    width_mm = abs(width_px * scale_avg)
    height_mm = abs(height_px * scale_avg)
    
    # Convert pixel axes to mm space (normalized directions in 3D)
    x_dir = np.array([x_axis_px[0], x_axis_px[1], 0.0])
    y_dir = np.array([y_axis_px[0], y_axis_px[1], 0.0])
    
    # Origin at marker_a position in camera coordinates (mm)
    origin = np.array([marker_a.center[0] * scale_avg,
                      marker_a.center[1] * scale_avg, 0.0])

    return WorkspaceDefinition(
        width_mm=float(width_mm),
        height_mm=float(height_mm),
        origin_camera=origin,
        x_axis_camera=x_dir,
        y_axis_camera=y_dir,
        camera_to_robot=camera_to_robot,
    )


def compute_workspace_from_plate(
    plate: CalibrationPlate,
    marker_ids: Tuple[int, int],
    camera_to_robot: RigidTransform,
    scale_multiplier: float = 1.0,
) -> WorkspaceDefinition:
    """
    Build a workspace definition directly from a calibration plate layout.

    This uses the plate's known marker spacing to define workspace dimensions,
    and treats the plate coordinate system as the camera plane.
    """

    marker_a = plate.markers.get(marker_ids[0])
    marker_b = plate.markers.get(marker_ids[1])
    if marker_a is None or marker_b is None:
        raise RuntimeError(
            "Calibration plate layout is missing workspace marker definitions.")

    delta = marker_b.center_mm - marker_a.center_mm
    width_mm = abs(float(delta[0])) * scale_multiplier
    height_mm = abs(float(delta[1])) * scale_multiplier
    x_sign = 1.0 if delta[0] >= 0 else -1.0
    y_sign = 1.0 if delta[1] >= 0 else -1.0

    origin = np.array([marker_a.center_mm[0], marker_a.center_mm[1], 0.0])
    x_axis = np.array([x_sign * scale_multiplier, 0.0, 0.0])
    y_axis = np.array([0.0, y_sign * scale_multiplier, 0.0])

    return WorkspaceDefinition(
        width_mm=width_mm,
        height_mm=height_mm,
        origin_camera=origin,
        x_axis_camera=x_axis,
        y_axis_camera=y_axis,
        camera_to_robot=camera_to_robot,
    )


def smooth_workspace(prev: WorkspaceDefinition, curr: WorkspaceDefinition, alpha: float) -> WorkspaceDefinition:
    """Exponential smoothing of workspace dimensions and origin to reduce jitter."""

    width = (1 - alpha) * prev.width_mm + alpha * curr.width_mm
    height = (1 - alpha) * prev.height_mm + alpha * curr.height_mm
    origin = (1 - alpha) * prev.origin_camera + alpha * curr.origin_camera
    x_axis = (1 - alpha) * prev.x_axis_camera + alpha * curr.x_axis_camera
    y_axis = (1 - alpha) * prev.y_axis_camera + alpha * curr.y_axis_camera
    return WorkspaceDefinition(
        width_mm=float(width),
        height_mm=float(height),
        origin_camera=origin,
        x_axis_camera=x_axis,
        y_axis_camera=y_axis,
        camera_to_robot=curr.camera_to_robot,
    )


def workspace_center(workspace: WorkspaceDefinition) -> Tuple[float, float]:
    return workspace.width_mm / 2.0, workspace.height_mm / 2.0


def bottom_left(workspace: WorkspaceDefinition) -> Tuple[float, float]:
    return 0.0, workspace.height_mm


def save_workspace(definition: WorkspaceDefinition, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **definition.to_dict())


def load_workspace(path: Path) -> WorkspaceDefinition:
    data = np.load(path, allow_pickle=True)

    def _unwrap(value: object) -> object:
        if isinstance(value, np.ndarray) and value.shape == () and value.dtype == object:
            return value.item()
        return value

    return WorkspaceDefinition.from_dict(
        {
            "width_mm": float(_unwrap(data["width_mm"])),
            "height_mm": float(_unwrap(data["height_mm"])),
            "origin_camera": _unwrap(data["origin_camera"]),
            "x_axis_camera": _unwrap(data["x_axis_camera"]),
            "y_axis_camera": _unwrap(data["y_axis_camera"]),
            "camera_to_robot": _unwrap(data["camera_to_robot"]),
        }
    )
