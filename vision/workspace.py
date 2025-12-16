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


def compute_workspace(markers: WorkspaceMarkers, calibration: CalibrationBundle, marker_size_mm: float, camera_to_robot: RigidTransform) -> WorkspaceDefinition:
    """
    Build a workspace representation given marker detections and calibration data.

    The pixel distance between markers combined with the known physical marker
    size approximates the metric scale along the table plane.
    """

    marker_a = markers.marker_a.center
    marker_b = markers.marker_b.center
    pixel_distance = np.linalg.norm(np.asarray(marker_b) - np.asarray(marker_a))
    pixel_to_mm = calibration.pixel_to_mm
    if pixel_to_mm == 0:
        # Fallback to estimating scale from marker size if pixel_to_mm was not measured.
        pixel_to_mm = marker_size_mm / np.linalg.norm(markers.marker_a.corners[1] - markers.marker_a.corners[0])

    width_mm = (marker_b[0] - marker_a[0]) * pixel_to_mm
    height_mm = (marker_b[1] - marker_a[1]) * pixel_to_mm

    # Build camera-plane basis vectors.
    x_dir = np.array([1.0, 0.0, 0.0]) * pixel_to_mm
    y_dir = np.array([0.0, 1.0, 0.0]) * pixel_to_mm
    origin = np.array([marker_a[0] * pixel_to_mm, marker_a[1] * pixel_to_mm, 0.0])

    return WorkspaceDefinition(
        width_mm=width_mm,
        height_mm=height_mm,
        origin_camera=origin,
        x_axis_camera=x_dir,
        y_axis_camera=y_dir,
        camera_to_robot=camera_to_robot,
    )


def workspace_center(workspace: WorkspaceDefinition) -> Tuple[float, float]:
    return workspace.width_mm / 2.0, workspace.height_mm / 2.0


def bottom_left(workspace: WorkspaceDefinition) -> Tuple[float, float]:
    return 0.0, workspace.height_mm
