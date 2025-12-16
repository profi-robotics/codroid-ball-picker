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
from typing import Dict, Tuple

import numpy as np

from .calibration import CalibrationBundle
from .markers import WorkspaceMarkers
from .markers import MarkerDetection


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

    def workspace_to_camera(self, x_mm: float, y_mm: float, z_offset_mm: float = 0.0) -> np.ndarray:
        point = self.origin_camera + (x_mm * self.x_axis_camera) + (y_mm * self.y_axis_camera)
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
    Build a workspace representation given marker detections and calibration data.

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
    mm_per_px_x = np.mean([s for s in (ax_scale, bx_scale) if s > 0]) if any(s > 0 for s in (ax_scale, bx_scale)) else 0.0
    mm_per_px_y = np.mean([s for s in (ay_scale, by_scale) if s > 0]) if any(s > 0 for s in (ay_scale, by_scale)) else 0.0

    if mm_per_px_x == 0 or mm_per_px_y == 0:
        fallback = calibration.pixel_to_mm if calibration.pixel_to_mm > 0 else marker_size_mm / max(
            1e-6, np.linalg.norm(marker_a.corners[1] - marker_a.corners[0])
        )
        mm_per_px_x = mm_per_px_x or fallback
        mm_per_px_y = mm_per_px_y or fallback

    # Use axis-aligned projection in pixel space but mix both scales to reduce skew impact.
    delta = np.asarray(marker_b.center) - np.asarray(marker_a.center)
    width_px = abs(delta[0])
    height_px = abs(delta[1])
    # Blend scales to mitigate axis misalignment from rotation.
    scale_avg = (mm_per_px_x + mm_per_px_y) / 2.0
    scale_avg *= scale_multiplier
    width_mm = width_px * scale_avg
    height_mm = height_px * scale_avg

    x_dir = np.array([1.0, 0.0, 0.0]) * scale_avg
    y_dir = np.array([0.0, 1.0, 0.0]) * scale_avg
    origin = np.array([marker_a.center[0] * scale_avg, marker_a.center[1] * scale_avg, 0.0])

    return WorkspaceDefinition(
        width_mm=float(width_mm),
        height_mm=float(height_mm),
        origin_camera=origin,
        x_axis_camera=x_dir,
        y_axis_camera=y_dir,
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
