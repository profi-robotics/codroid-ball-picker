"""Small transform helpers that are independent of the web layer."""

from __future__ import annotations

from typing import Optional

import numpy as np

from vision import workspace


def clamp_coord5_z(z: float, max_z_mm: float) -> float:
    """Clamp Coord5 Z so positive-down moves cannot exceed the safe floor."""
    z_value = float(z)
    return min(z_value, float(max_z_mm))


def default_transform() -> workspace.RigidTransform:
    """Return the identity camera-to-robot transform used before robot calibration."""
    return workspace.RigidTransform(
        rotation=np.eye(3, dtype=np.float64),
        translation=np.zeros(3, dtype=np.float64),
    )


def rigid_transform(from_pts: np.ndarray, to_pts: np.ndarray) -> workspace.RigidTransform:
    """Compute the best-fit rigid transform from source points to target points."""
    source = np.asarray(from_pts, dtype=np.float64)
    target = np.asarray(to_pts, dtype=np.float64)
    if source.shape != target.shape:
        raise ValueError("Point clouds must have the same shape.")
    if source.shape[0] < 3:
        raise ValueError("At least 3 points are required for a rigid transform")
    centroid_source = np.mean(source, axis=0)
    centroid_target = np.mean(target, axis=0)
    centered_source = source - centroid_source
    centered_target = target - centroid_target
    h_mat = centered_source.T @ centered_target
    u_mat, _singular_values, vt_mat = np.linalg.svd(h_mat)
    rotation = vt_mat.T @ u_mat.T
    if np.linalg.det(rotation) < 0:
        vt_mat[-1, :] *= -1
        rotation = vt_mat.T @ u_mat.T
    translation = centroid_target - rotation @ centroid_source
    return workspace.RigidTransform(rotation=rotation, translation=translation)


def workspace_pixel_to_mm(
    workspace_def: workspace.WorkspaceDefinition,
    pixel_xy: tuple[float, float],
    *,
    pixel_to_mm: Optional[float] = None,
) -> tuple[float, float]:
    """Project a pixel location into workspace millimeters."""
    if pixel_to_mm is not None and pixel_to_mm > 0:
        camera_xyz = np.array(
            [float(pixel_xy[0]) * pixel_to_mm, float(pixel_xy[1]) * pixel_to_mm, 0.0],
            dtype=np.float64,
        )
        return project_camera_xyz_to_workspace(workspace_def, camera_xyz)

    scale = float(workspace_def.x_axis_camera[0])
    if abs(scale) < 1e-9:
        raise RuntimeError("Workspace scale is zero; recompute workspace markers.")
    origin_px_x = float(workspace_def.origin_camera[0]) / scale
    origin_px_y = float(workspace_def.origin_camera[1]) / scale
    return (float(pixel_xy[0] - origin_px_x) * scale, float(pixel_xy[1] - origin_px_y) * scale)


def project_camera_xyz_to_workspace(
    workspace_def: workspace.WorkspaceDefinition,
    camera_xyz: np.ndarray,
) -> tuple[float, float]:
    """Project a camera-space 3D point into workspace XY millimeters."""
    basis = np.column_stack([workspace_def.x_axis_camera, workspace_def.y_axis_camera])
    coeffs, *_unused = np.linalg.lstsq(basis, np.asarray(camera_xyz, dtype=np.float64) - workspace_def.origin_camera, rcond=None)
    return float(coeffs[0]), float(coeffs[1])


def robot_to_workspace_mm(
    workspace_def: workspace.WorkspaceDefinition,
    robot_xyz: np.ndarray,
) -> tuple[float, float]:
    """Map robot XYZ back to workspace XY millimeters using the workspace transform."""
    robot_h = np.hstack([np.asarray(robot_xyz, dtype=np.float64), 1.0])
    camera_h = np.linalg.inv(workspace_def.camera_to_robot.as_matrix()) @ robot_h
    return project_camera_xyz_to_workspace(workspace_def, camera_h[:3])


def pose_from_xy(
    xy_mm: tuple[float, float],
    *,
    z_mm: float,
    a: float = 0.0,
    b: float = 0.0,
    c: float = 0.0,
    max_z_mm: Optional[float] = None,
) -> dict[str, float]:
    """Build a robot pose in the active XY coordinate frame."""
    z_value = float(z_mm)
    if max_z_mm is not None:
        z_value = clamp_coord5_z(z_value, max_z_mm)
    return {
        "x": float(xy_mm[0]),
        "y": float(xy_mm[1]),
        "z": z_value,
        "a": float(a),
        "b": float(b),
        "c": float(c),
    }
