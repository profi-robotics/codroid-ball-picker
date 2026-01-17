"""
Plate test target resolution utilities.

This module provides functions for resolving calibration plate test targets
and computing plate-to-robot coordinate transformations.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np

import app.settings as app_settings
from app.codroid_picker import state
from app.codroid_picker.camera_calibration import compute_plate_calibration_from_latest
from app.codroid_picker.procedures import _pixel_to_workspace_mm, _rigid_transform, _rigid_transform_2d
from app.codroid_picker.session import SESSION
from vision import plate, workspace

LOGGER = logging.getLogger(__name__)

SETTINGS = app_settings.SETTINGS


def _get_plate_status() -> dict:
    """Get the current plate status snapshot."""
    return state.get_plate_status_snapshot()


def _resolve_plate_test_targets(
    layout_path: Path,
    workspace_def: workspace.WorkspaceDefinition,
) -> Tuple[list[dict], dict]:
    """Resolve plate test targets for workspace-based testing."""
    calibration_result = compute_plate_calibration_from_latest(layout_path)
    plate_points = calibration_result.plate.robot_calibration_points()
    if len(plate_points) < 3:
        raise RuntimeError("Calibration plate layout is missing targets 1-3.")
    plate_xy = np.array(
        [[pt.center_mm[0], pt.center_mm[1]] for pt in plate_points],
        dtype=np.float64,
    ).reshape(-1, 1, 2)
    homography = calibration_result.homography
    if homography is None:
        raise RuntimeError("Plate homography unavailable.")
    inv_h = np.linalg.inv(homography)
    pixel_xy = cv2.perspectiveTransform(plate_xy, inv_h).reshape(-1, 2)
    targets = []
    for point, px in zip(plate_points, pixel_xy):
        workspace_xy = _pixel_to_workspace_mm(workspace_def, px)
        sequence = point.sequence if point.sequence is not None else len(
            targets) + 1
        targets.append(
            {
                "sequence": int(sequence),
                "workspace_mm": workspace_xy,
                "plate_mm": (float(point.center_mm[0]), float(point.center_mm[1])),
                "pixel": (float(px[0]), float(px[1])),
            }
        )
    state.update_plate_status(
        ok=len(calibration_result.detections) >= SETTINGS.min_plate_markers,
        stable=len(calibration_result.detections) >= SETTINGS.min_plate_markers,
        message=f"Plate markers detected: {len(calibration_result.detections)}",
        count=len(calibration_result.detections),
        center_px=None,
        movement_px=None,
    )
    return targets, {
        "count": len(calibration_result.detections),
        "reprojection_error_mm": calibration_result.reprojection_error_mm,
    }


def _resolve_plate_test_coord5_targets(
    layout_path: Path,
) -> Tuple[list[dict], dict, workspace.RigidTransform, bool]:
    """Resolve plate test targets for coord5-based testing."""
    if not SESSION.robot_calibration_points or len(SESSION.robot_calibration_points) < 3:
        raise RuntimeError(
            "Robot calibration points are missing; re-run robot calibration."
        )

    calibration_result = compute_plate_calibration_from_latest(layout_path)
    plate_layout = calibration_result.plate
    plate_points = plate_layout.robot_calibration_points()
    if len(plate_points) < 3:
        raise RuntimeError("Calibration plate layout is missing targets 1-3.")

    reference_initialized = False
    if SESSION.plate_reference_homography is None:
        bundle = SESSION.calibration_bundle
        if bundle is not None and bundle.plate_homography is not None:
            SESSION.plate_reference_homography = bundle.plate_homography.copy()
        else:
            SESSION.plate_reference_homography = calibration_result.homography.copy()
            reference_initialized = True

    origin_mm = np.asarray(plate_points[0].center_mm, dtype=np.float64)
    marker_base: list[np.ndarray] = []
    marker_curr: list[np.ndarray] = []
    marker_info: dict[int, dict] = {}
    reference_h = SESSION.plate_reference_homography
    for marker_id, detection in calibration_result.detections.items():
        marker = plate_layout.markers.get(marker_id)
        if marker is None:
            continue
        base_pt = marker.center_mm - origin_mm
        curr_pt = plate.image_points_to_plate_mm(
            reference_h, np.asarray([detection.center], dtype=np.float64)
        )[0]
        marker_base.append(np.asarray(base_pt, dtype=np.float64))
        marker_curr.append(np.asarray(curr_pt - origin_mm, dtype=np.float64))
        marker_info[int(marker_id)] = {
            "plate_mm": (
                float(marker.center_mm[0]),
                float(marker.center_mm[1]),
            ),
            "base_xy": (float(base_pt[0]), float(base_pt[1])),
        }

    if len(marker_base) < 2:
        raise RuntimeError(
            "Not enough plate markers detected for offset tracking.")

    r2, t2 = _rigid_transform_2d(
        np.vstack(marker_base),
        np.vstack(marker_curr),
    )

    offset_rotation_rad = float(np.arctan2(r2[1, 0], r2[0, 0]))
    offset_rotation_deg = float(np.degrees(offset_rotation_rad))
    # Debug: log plate offset tracking transformation
    LOGGER.info("Plate offset tracking: detected %d markers", len(marker_base))
    LOGGER.info("Plate offset r2 (rotation):\n%s", r2)
    LOGGER.info("Plate offset t2 (translation): %s", t2)
    LOGGER.info("Plate offset rotation: %.2f degrees", offset_rotation_deg)

    robot_pts = np.asarray(
        [
            [
                float(pt["x"]),
                float(pt["y"]),
                float(pt["z"]),
            ]
            for pt in SESSION.robot_calibration_points[:3]
        ],
        dtype=np.float64,
    )
    baseline_transform = _rigid_transform(
        np.asarray(
            [
                [
                    float(pt.center_mm[0] - origin_mm[0]),
                    float(pt.center_mm[1] - origin_mm[1]),
                    0.0,
                ]
                for pt in plate_points[:3]
            ],
            dtype=np.float64,
        ),
        robot_pts,
    )

    # Debug logging for coordinate transformation
    LOGGER.info("Baseline transform rotation:\n%s",
                baseline_transform.rotation)
    LOGGER.info("Baseline transform translation: %s",
                baseline_transform.translation)
    LOGGER.info(
        "Plate points (relative to P1): %s",
        [
            (pt.center_mm[0] - origin_mm[0], pt.center_mm[1] - origin_mm[1])
            for pt in plate_points[:3]
        ],
    )
    LOGGER.info(
        "Robot calibration points: %s",
        [(pt["x"], pt["y"], pt["z"])
         for pt in SESSION.robot_calibration_points[:3]],
    )

    targets = []
    for point in plate_points:
        base_xy = np.array(
            [
                float(point.center_mm[0] - origin_mm[0]),
                float(point.center_mm[1] - origin_mm[1]),
            ],
            dtype=np.float64,
        )
        adj_xy = (r2 @ base_xy) + t2
        sequence = point.sequence if point.sequence is not None else len(
            targets) + 1
        targets.append(
            {
                "sequence": int(sequence),
                "reference_xy": (float(base_xy[0]), float(base_xy[1])),
                "adjusted_xy": (float(adj_xy[0]), float(adj_xy[1])),
                "plate_mm": (float(point.center_mm[0]), float(point.center_mm[1])),
            }
        )

    for marker_id, meta in marker_info.items():
        base_xy = np.asarray(meta["base_xy"], dtype=np.float64)
        adj_xy = (r2 @ base_xy) + t2
        meta["reference_xy"] = (float(base_xy[0]), float(base_xy[1]))
        meta["adjusted_xy"] = (float(adj_xy[0]), float(adj_xy[1]))

    state.update_plate_status(
        ok=len(calibration_result.detections) >= SETTINGS.min_plate_markers,
        stable=len(calibration_result.detections) >= SETTINGS.min_plate_markers,
        message=f"Plate markers detected: {len(calibration_result.detections)}",
        count=len(calibration_result.detections),
        center_px=None,
        movement_px=None,
    )

    return (
        targets,
        {
            "count": len(calibration_result.detections),
            "reprojection_error_mm": calibration_result.reprojection_error_mm,
            "reference_initialized": reference_initialized,
            "markers_coord5": marker_info,
            "offset_rotation_deg": offset_rotation_deg,
        },
        baseline_transform,
        reference_initialized,
    )
