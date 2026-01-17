"""Low-level drawing helper functions for overlays."""

from __future__ import annotations

import logging
from typing import Tuple

import cv2
import numpy as np

from app.codroid_picker.session import SESSION
from vision import workspace


def _draw_label(frame: np.ndarray, text: str, origin: Tuple[int, int], color: Tuple[int, int, int]) -> None:
    """Draw text label on the frame."""
    cv2.putText(frame, text, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


def _draw_dotted_polyline(
    frame: np.ndarray,
    points: np.ndarray,
    color: Tuple[int, int, int] = (0, 215, 255),
    thickness: int = 2,
    gap: int = 10,
) -> None:
    """Draw a dotted polyline on the frame."""
    for idx in range(len(points)):
        start = points[idx]
        end = points[(idx + 1) % len(points)]
        vec = end - start
        length = float(np.linalg.norm(vec))
        if length == 0:
            continue
        direction = vec / length
        dist = 0.0
        while dist < length:
            segment_start = start + direction * dist
            dist = min(length, dist + gap)
            segment_end = start + direction * dist
            cv2.line(
                frame,
                tuple(np.round(segment_start).astype(int)),
                tuple(np.round(segment_end).astype(int)),
                color,
                thickness,
                lineType=cv2.LINE_AA,
            )
            dist += gap


def _workspace_corners_px(workspace_def: workspace.WorkspaceDefinition) -> np.ndarray:
    """Convert workspace definition to pixel coordinates for overlay drawing."""
    # Workspace definition stores everything in mm in camera coordinates
    # The axes are normalized direction vectors
    # We need calibration to convert mm back to pixels
    if SESSION.calibration_bundle is None:
        raise RuntimeError("Calibration bundle required to display workspace")

    scale = SESSION.calibration_bundle.pixel_to_mm
    if scale == 0:
        raise RuntimeError(
            "Workspace scale is zero; recompute workspace markers.")

    # Convert origin from mm to pixels
    origin_px = np.array([workspace_def.origin_camera[0],
                         workspace_def.origin_camera[1]]) / scale

    # Convert width and height from mm to pixels
    width_px = workspace_def.width_mm / scale
    height_px = workspace_def.height_mm / scale

    # Extract 2D axes (they are already normalized directions in mm-space)
    x_dir = np.array([workspace_def.x_axis_camera[0],
                      workspace_def.x_axis_camera[1]])
    y_dir = np.array([workspace_def.y_axis_camera[0],
                      workspace_def.y_axis_camera[1]])

    # Compute workspace corners in pixels
    corners = np.array([
        origin_px,
        origin_px + x_dir * width_px,
        origin_px + x_dir * width_px + y_dir * height_px,
        origin_px + y_dir * height_px,
    ])

    return corners


def _apply_workspace_overlay(frame: np.ndarray) -> None:
    """Apply workspace boundary overlay to the frame."""
    if SESSION.workspace_definition is None:
        return
    try:
        corners_px = _workspace_corners_px(SESSION.workspace_definition)
    except Exception as exc:
        logging.debug("Workspace overlay skipped: %s", exc)
        return
    _draw_dotted_polyline(frame, corners_px)
