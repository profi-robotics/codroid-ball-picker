"""Calibration menu and path utilities.

Provides functions for managing calibration artifacts, metadata display,
and menu visibility state.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING

from app.codroid_picker.app_instance import registry

if TYPE_CHECKING:
    pass

LOGGER = logging.getLogger(__name__)


def _camera_calibration_path() -> Path:
    """Return the currently configured camera calibration bundle path."""
    return registry().bundle_path


def _robot_calibration_path_value() -> Path:
    """Return the currently configured robot calibration path."""
    return registry().robot_calibration_path


def _calibration_metadata_entries() -> list[str]:
    """Produce human-readable metadata about existing calibration artifacts."""
    from app.codroid_picker import state

    session = registry().session
    bundle_path = registry().bundle_path
    robot_cal_path = registry().robot_calibration_path
    entries: list[str] = []
    bundle = session.calibration_bundle
    if bundle is not None:
        pixel_scale = bundle.pixel_to_mm
        entries.append(f"Pixel scale: {pixel_scale:.3f} mm/px")
        cam = bundle.camera.camera_matrix
        entries.append(
            f"Camera focal: fx={cam[0, 0]:.1f}, fy={cam[1, 1]:.1f} px"
        )
        plane = bundle.workspace_plane
        entries.append(
            f"Workspace plane offset: {plane.offset:.1f} mm · normal="
            f"{tuple(float(v) for v in plane.normal.round(3))}"
        )
    if bundle_path.exists():
        entries.append(
            f"Camera bundle saved: "
            f"{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(bundle_path.stat().st_mtime))}"
        )
    if session.robot_calibration_points:
        entries.append(
            f"Robot calibration points: {len(session.robot_calibration_points)}"
        )
    if session.robot_calibration_min_z is not None:
        entries.append(
            f"Robot min Z: {session.robot_calibration_min_z:.1f} mm"
        )
    if robot_cal_path.exists():
        entries.append(
            f"Robot calibration saved: "
            f"{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(robot_cal_path.stat().st_mtime))}"
        )
    accuracy = state.get_calibration_guidance_snapshot().get("accuracy")
    if accuracy:
        max_err = accuracy.get("max_xy_error")
        mean_err = accuracy.get("mean_xy_error")
        if max_err is not None and mean_err is not None:
            entries.append(
                f"Calibration accuracy max/mean XY error: "
                f"{max_err:.2f}/{mean_err:.2f} mm"
            )
    return entries


def _calibration_data_complete() -> bool:
    """Check whether both camera and robot calibration artifacts exist."""
    return _camera_calibration_path().exists() and _robot_calibration_path_value().exists()


def _refresh_calibration_menu_visibility() -> bool:
    """Update the session flag that controls whether the calibration stepper is visible."""
    session = registry().session
    session.calibration_menu_open = not _calibration_data_complete()
    return session.calibration_menu_open


def _reset_calibration_assets() -> None:
    """Delete saved calibration files and clear cached session state."""
    session = registry().session
    for path in (_camera_calibration_path(), _robot_calibration_path_value()):
        try:
            path.unlink()
        except FileNotFoundError:
            pass
    session.calibration_bundle = None
    session.plate_calibration = None
    session.robot_calibration_points = []
    session.robot_calibration_min_z = None
    session.plate_reference_homography = None
    session.plate_to_robot_transform = None
    session.plate_origin_mm = None
    session.robot_calibration_accuracy = None
    session.robot_calibration_gripper_width = None
    _refresh_calibration_menu_visibility()
