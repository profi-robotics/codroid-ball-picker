"""Pick and place helper functions.

This module contains helper functions for pick and place operations,
including detection selection, pose computation, and gripper width estimation.
These helpers are used by the _handle_pick_place callback and the
ManualPickPlaceProgram.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple

import numpy as np

if TYPE_CHECKING:
    from vision import workspace

LOGGER = logging.getLogger(__name__)


def estimate_object_width_mm(
    detection: Dict[str, Any],
    workspace_def: "workspace.WorkspaceDefinition",
    *,
    enabled: bool = True,
) -> Optional[float]:
    """Estimate object width in mm from detection bounding box.

    Args:
        detection: Detection dict with bbox field.
        workspace_def: Workspace definition for scale computation.
        enabled: Whether width estimation is enabled.

    Returns:
        Estimated width in mm, or None if cannot estimate.
    """
    if not enabled or workspace_def is None:
        return None

    bbox = detection.get("bbox")
    if not bbox:
        return None

    x1, y1, x2, y2 = bbox
    width_px = max(abs(x2 - x1), abs(y2 - y1))
    if width_px <= 0:
        return None

    x_scale = np.linalg.norm(workspace_def.x_axis_camera)
    y_scale = np.linalg.norm(workspace_def.y_axis_camera)
    denom = (x_scale + y_scale) / 2.0
    if denom <= 0:
        return None

    mm_per_px = 1.0 / denom
    return width_px * mm_per_px


def select_closest_detection(
    detections: List[Dict[str, Any]],
    reference_xy: Tuple[float, float],
    coord5_fn: Optional[Callable[[Dict[str, Any]],
                                 Optional[Tuple[float, float]]]] = None,
) -> Dict[str, Any]:
    """Select the detection closest to a reference point.

    Args:
        detections: List of detection dicts.
        reference_xy: Reference XY coordinates to measure distance from.
        coord5_fn: Optional function to get coord5 XY for a detection.

    Returns:
        The closest detection dict.

    Raises:
        ValueError: If no detections available.
    """
    if not detections:
        raise ValueError("No detections available.")

    best_det = detections[0]
    best_dist = float("inf")

    for det in detections:
        if coord5_fn is not None:
            det_xy = coord5_fn(det)
        else:
            det_xy = det.get("workspace_mm")

        if det_xy is None:
            continue

        dist = np.linalg.norm(np.array(det_xy[:2]) - np.array(reference_xy))
        if dist < best_dist:
            best_dist = dist
            best_det = det

    return best_det


def resolve_gripper_widths(
    gripper: Any,
    gripper_defaults: Dict[str, Any],
    *,
    estimate_width: Optional[float] = None,
    open_default: float = 50.0,
    close_default: float = 10.0,
) -> Tuple[float, float]:
    """Resolve gripper open and close widths.

    Args:
        gripper: Gripper instance (may be None).
        gripper_defaults: Dict with t_index and other defaults.
        estimate_width: Optional estimated object width in mm.
        open_default: Default open width if gripper unavailable.
        close_default: Default close width if gripper unavailable.

    Returns:
        Tuple of (close_width, open_width) in mm.
    """
    close_val = close_default
    open_val = open_default

    try:
        from app.codroid_picker.session import SESSION

        adapter = SESSION.gripper_adapter
    except Exception:  # noqa: BLE001
        adapter = None

    if adapter is not None:
        try:
            close_val, open_val = adapter.width_limits(
                t_index=gripper_defaults.get("t_index", 0)
            )
        except Exception as e:  # noqa: BLE001
            LOGGER.warning(f"Error getting dynamic gripper limits: {e}")
    elif gripper is not None and gripper.isConnected(t_index=gripper_defaults.get("t_index", 0)):
        try:
            max_width = gripper.get_max_ext_width(
                t_index=gripper_defaults["t_index"])
            min_width = gripper.get_min_ext_width(
                t_index=gripper_defaults["t_index"])

            if isinstance(max_width, (int, float)) and max_width > 0:
                open_val = max_width
                LOGGER.debug(f"Using dynamic gripper max width: {open_val} mm")

            if isinstance(min_width, (int, float)) and min_width > 0:
                close_val = min_width
                LOGGER.debug(
                    f"Using dynamic gripper min width: {close_val} mm")

        except Exception as e:
            LOGGER.warning(f"Error getting dynamic gripper limits: {e}")

    # Adjust for estimated object width
    if estimate_width is not None:
        close_val = max(close_val, estimate_width * 0.9)
        open_val = max(open_val, estimate_width + 5.0)

    return close_val, open_val


def build_pick_poses(
    coord5_xy: Tuple[float, float],
    orientation: Dict[str, float],
    approach_mm: float,
    pick_mm: float,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Build approach and pick poses from coord5 coordinates.

    Args:
        coord5_xy: XY coordinates in coord5 frame.
        orientation: Dict with a, b, c orientation values.
        approach_mm: Z offset for approach position.
        pick_mm: Z offset for pick position.

    Returns:
        Tuple of (approach_pose, pick_pose) dicts.
    """
    a = float(orientation.get("a", 0.0))
    b = float(orientation.get("b", 0.0))
    c = float(orientation.get("c", 0.0))

    approach_pose = _build_pose(coord5_xy[0], coord5_xy[1], approach_mm, a, b, c)
    pick_pose = _build_pose(coord5_xy[0], coord5_xy[1], pick_mm, a, b, c)
    return approach_pose, pick_pose


def _build_pose(
    x: float,
    y: float,
    z: float,
    a: float = 0.0,
    b: float = 0.0,
    c: float = 0.0,
) -> Dict[str, float]:
    """Build a pose dictionary from coordinates and orientation.

    This is the canonical pose builder used by all pick/place pose functions.
    """
    return {
        "x": float(x),
        "y": float(y),
        "z": float(z),
        "a": float(a),
        "b": float(b),
        "c": float(c),
    }


def build_drop_poses(
    drop_robot_xyz: Tuple[float, float, float],
    orientation: Optional[Tuple[float, float, float]],
    approach_mm: float,
    drop_mm: float,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Build approach and drop poses from stored drop target.

    Args:
        drop_robot_xyz: Drop target robot coordinates (x, y, z).
        orientation: Optional orientation (a, b, c) tuple.
        approach_mm: Z offset for approach position.
        drop_mm: Z offset for drop position.

    Returns:
        Tuple of (approach_pose, drop_pose) dicts.
    """
    orient = orientation or (0.0, 0.0, 0.0)
    base_z = float(drop_robot_xyz[2])

    approach_pose = _build_pose(
        drop_robot_xyz[0], drop_robot_xyz[1], base_z + approach_mm,
        orient[0], orient[1], orient[2]
    )
    drop_pose = _build_pose(
        drop_robot_xyz[0], drop_robot_xyz[1], base_z + drop_mm,
        orient[0], orient[1], orient[2]
    )
    return approach_pose, drop_pose


class PickPlaceContext:
    """Context for a pick/place operation with all resolved parameters.

    This class holds all the resolved state needed for a pick/place operation,
    including detection, poses, gripper widths, and coordination IDs.
    """

    def __init__(
        self,
        *,
        detection: Optional[Dict[str, Any]] = None,
        coord5_xy: Optional[Tuple[float, float]] = None,
        workspace_mm: Optional[Tuple[float, float]] = None,
        approach_mm: float = -30.0,
        pick_mm: float = 0.0,
        drop_mm: float = 0.0,
        orientation: Optional[Dict[str, float]] = None,
        gripper_close_width: float = 10.0,
        gripper_open_width: float = 50.0,
        drop_robot_xyz: Optional[Tuple[float, float, float]] = None,
        drop_orientation: Optional[Tuple[float, float, float]] = None,
        coordinate_id: int = 5,
    ) -> None:
        self.detection = detection
        self.coord5_xy = coord5_xy
        self.workspace_mm = workspace_mm
        self.approach_mm = approach_mm
        self.pick_mm = pick_mm
        self.drop_mm = drop_mm
        self.orientation = orientation or {"a": 0.0, "b": 0.0, "c": 0.0}
        self.gripper_close_width = gripper_close_width
        self.gripper_open_width = gripper_open_width
        self.drop_robot_xyz = drop_robot_xyz
        self.drop_orientation = drop_orientation
        self.coordinate_id = coordinate_id

    def get_pick_approach_pose(self) -> Optional[Dict[str, Any]]:
        """Get the pick approach pose."""
        if self.coord5_xy is None:
            return None
        approach, _ = build_pick_poses(
            self.coord5_xy,
            self.orientation,
            self.approach_mm,
            self.pick_mm,
        )
        return approach

    def get_pick_pose(self) -> Optional[Dict[str, Any]]:
        """Get the pick pose."""
        if self.coord5_xy is None:
            return None
        _, pick = build_pick_poses(
            self.coord5_xy,
            self.orientation,
            self.approach_mm,
            self.pick_mm,
        )
        return pick

    def get_drop_approach_pose(self) -> Optional[Dict[str, Any]]:
        """Get the drop approach pose."""
        if self.drop_robot_xyz is None:
            return None
        approach, _ = build_drop_poses(
            self.drop_robot_xyz,
            self.drop_orientation,
            self.approach_mm,
            self.drop_mm,
        )
        return approach

    def get_drop_pose(self) -> Optional[Dict[str, Any]]:
        """Get the drop pose."""
        if self.drop_robot_xyz is None:
            return None
        _, drop = build_drop_poses(
            self.drop_robot_xyz,
            self.drop_orientation,
            self.approach_mm,
            self.drop_mm,
        )
        return drop
