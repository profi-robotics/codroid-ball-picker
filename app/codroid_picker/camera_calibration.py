"""
Camera calibration utilities.

This module provides functions for camera calibration using ChArUco boards,
workspace detection, depth frame capture, and calibration bundle management.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

from vision import aruco_board, calibration, markers, plate, workspace
from app.settings import _commit_settings
from app.codroid_picker.session import SESSION
import app.settings as app_settings
from app.codroid_picker.models import _frame_index, _update_capture_state
from app.codroid_picker.transforms import _default_transform

LOGGER = logging.getLogger(__name__)

DEFAULT_WORKSPACE_PATH = app_settings.DEFAULT_WORKSPACE_PATH


def get_latest_frame() -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Get the latest frame and depth from the camera stream."""
    if SESSION.camera_stream is None or not SESSION.camera_stream.is_running():
        return None, None
    return SESSION.camera_stream.get_latest()


def capture_color_frame(output_dir: Path) -> Path:
    """Capture a color frame from the camera stream and save it to disk."""
    # Retry logic: frame might not be available immediately, especially
    # right after a stream restart on slower Jetson devices.
    max_retries = 30
    retry_delay_s = 0.1
    for attempt in range(max_retries):
        frame, _depth = get_latest_frame()
        if frame is not None and frame.size > 0:
            break
        if attempt < max_retries - 1:
            time.sleep(retry_delay_s)

    if frame is None or frame.size == 0:
        raise RuntimeError(
            f"Camera stream not providing frames after {max_retries} attempts.")

    output_dir.mkdir(parents=True, exist_ok=True)
    frames = _update_capture_state(output_dir)
    index = _frame_index(frames[-1]) + 1 if frames else 1
    path = output_dir / f"calibration_color_{index:02}.png"

    success = cv2.imwrite(str(path), frame)
    if not success:
        raise RuntimeError(f"Failed to write frame to {path}.")

    # Verify by reading back. Retry a few times to avoid transient FS races.
    verify = None
    for _ in range(5):
        verify = cv2.imread(str(path))
        if verify is not None:
            break
        time.sleep(0.05)
    if verify is None:
        raise RuntimeError(f"Failed to read back saved frame from {path}.")

    SESSION.capture.frames.append(path)
    return path


def capture_depth_snapshot(output_dir: Path) -> Path:
    """Capture a depth frame from the camera stream and save it to disk."""
    depth = None
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        _frame, depth = get_latest_frame()
        if depth is not None:
            break
        time.sleep(0.05)
    if depth is None:
        raise RuntimeError(
            "No depth frame available from the camera stream. "
            "Unable to acquire depth data after waiting 5 seconds."
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "calibration_depth_frame.npz"
    np.savez_compressed(path, depth=depth.astype(np.float32))
    SESSION.capture.depth_frame = depth
    return path


def load_depth_snapshot(output_dir: Path) -> Optional[np.ndarray]:
    """Load a depth snapshot from disk."""
    path = output_dir / "calibration_depth_frame.npz"
    if not path.exists():
        return None
    data = np.load(path, allow_pickle=True)
    depth = data.get("depth")
    if depth is None:
        return None
    return depth.astype(np.float32)


def _clear_calibration_outputs(output_dir: Path) -> None:
    """Remove calibration output files from a directory."""
    if not output_dir.exists():
        return
    for pattern in ("calibration_color_*.png", "calibration_depth_frame.npz"):
        for path in output_dir.glob(pattern):
            try:
                path.unlink()
            except OSError:
                logging.warning(f"[CALIBRATE] Unable to remove {path}")


def _charuco_detection_counts(
    image_paths: list[Path], board_path: Path
) -> Tuple[list[int], int]:
    """Count ChArUco corner detections in each image."""
    spec = aruco_board.load_charuco_board_spec(board_path)
    board = spec.create_board()

    # Get dictionary for initial marker detection
    dictionary_id = getattr(cv2.aruco, spec.dictionary)
    dictionary = cv2.aruco.getPredefinedDictionary(dictionary_id)
    detector_params = cv2.aruco.DetectorParameters()
    detector_params.detectInvertedMarker = True  # Match live overlay behavior

    counts: list[int] = []
    usable_frames = 0
    for path in image_paths:
        image = cv2.imread(str(path))
        if image is None:
            counts.append(0)
            continue
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # First detect markers
        detector = cv2.aruco.ArucoDetector(dictionary, detector_params)
        marker_corners, marker_ids, _rejected = detector.detectMarkers(gray)

        # Then detect ChArUco corners if markers were found
        corner_count = 0
        if marker_ids is not None and len(marker_ids) > 0:
            charuco_params = cv2.aruco.CharucoParameters()
            charuco_detector = cv2.aruco.CharucoDetector(
                board, charuco_params, detector_params
            )
            charuco_corners, charuco_ids, _, _ = charuco_detector.detectBoard(
                gray)
            corner_count = 0 if charuco_ids is None else int(len(charuco_ids))

            if charuco_ids is not None and charuco_corners is not None:
                obj, img = board.matchImagePoints(charuco_corners, charuco_ids)
                if obj is not None and img is not None and len(obj) >= 4:
                    usable_frames += 1

        counts.append(corner_count)

    return counts, usable_frames


def compute_calibration_bundle(
    board_path: Path, bundle_path: Path, output_dir: Path
) -> str:
    """Compute and save a camera calibration bundle."""
    if not board_path.exists():
        raise FileNotFoundError(
            f"ChArUco board layout not found: {board_path}")
    frames = _update_capture_state(output_dir)
    if not frames:
        raise RuntimeError(
            "No calibration frames found. Capture frames before calibrating."
        )
    depth_frame = (
        SESSION.capture.depth_frame
        if SESSION.capture.depth_frame is not None
        else load_depth_snapshot(output_dir)
    )
    if depth_frame is None:
        raise RuntimeError(
            "No depth snapshot found. Capture a depth frame before calibrating."
        )
    board_spec = aruco_board.load_charuco_board_spec(board_path)
    cam = calibration.calibrate_camera_charuco(frames, board_spec)
    try:
        pixel_to_mm = calibration.charuco_board_pixel_scale(
            frames[0], board_spec)
    except Exception as exc:  # fallback to nominal pixel spacing
        LOGGER.warning(
            "Falling back to nominal pixel scale while calibrating: %s", exc
        )
        pixel_to_mm = board_spec.marker_length_mm
    plane = calibration.estimate_workspace_plane(depth_frame, cam)
    bundle = calibration.CalibrationBundle(
        camera=cam,
        workspace_plane=plane,
        pixel_to_mm=pixel_to_mm,
    )
    calibration.save_calibration(bundle, bundle_path)
    SESSION.calibration_bundle = bundle
    _commit_settings(calibration_bundle_path=str(bundle_path))
    return f"Calibration bundle saved to {bundle_path}."


def load_calibration_bundle(bundle_path: Path) -> str:
    """Load a calibration bundle from disk."""
    if not bundle_path.exists():
        raise FileNotFoundError(f"Calibration bundle not found: {bundle_path}")
    SESSION.calibration_bundle = calibration.load_calibration(bundle_path)
    _commit_settings(calibration_bundle_path=str(bundle_path))
    return f"Loaded calibration bundle from {bundle_path}."


def compute_workspace_from_latest(
    marker_ids: Tuple[int, int], marker_size_mm: float
) -> workspace.WorkspaceDefinition:
    """Compute workspace definition from the latest camera frame."""
    frame, _depth = get_latest_frame()
    if frame is None:
        raise RuntimeError("Camera stream is not providing frames yet.")
    if SESSION.calibration_bundle is None:
        raise RuntimeError("Calibration bundle is not loaded.")
    detections = markers.detect_markers(frame, marker_ids)
    ws = workspace.compute_workspace(
        detections,
        SESSION.calibration_bundle,
        marker_size_mm=marker_size_mm,
        camera_to_robot=_default_transform(),
    )
    SESSION.workspace_definition = ws
    save_workspace_definition(ws, DEFAULT_WORKSPACE_PATH)
    return ws


def save_workspace_definition(
    workspace_def: workspace.WorkspaceDefinition,
    workspace_path: Path = DEFAULT_WORKSPACE_PATH,
) -> str:
    """Save a workspace definition to disk."""
    workspace.save_workspace(workspace_def, workspace_path)
    _commit_settings(workspace_path=str(workspace_path))
    return f"Workspace saved to {workspace_path}."


def load_workspace_definition(
    workspace_path: Path = DEFAULT_WORKSPACE_PATH,
) -> workspace.WorkspaceDefinition:
    """Load a workspace definition from disk."""
    if not workspace_path.exists():
        raise FileNotFoundError(
            f"Workspace definition not found: {workspace_path}")
    SESSION.workspace_definition = workspace.load_workspace(workspace_path)
    _commit_settings(workspace_path=str(workspace_path))
    return SESSION.workspace_definition


def _clear_workspace_definition(path: Path = DEFAULT_WORKSPACE_PATH) -> None:
    """Drop any cached workspace definition without deleting the saved file."""
    SESSION.workspace_definition = None


def compute_plate_calibration_from_latest(layout_path: Path) -> plate.PlateCalibration:
    """Compute plate calibration from the latest camera frame."""
    frame, _depth = get_latest_frame()
    if frame is None:
        raise RuntimeError("Camera stream is not providing frames yet.")
    if not layout_path.exists():
        raise FileNotFoundError(
            f"Calibration plate layout not found: {layout_path}")
    layout = plate.load_plate_layout(layout_path)
    result = plate.estimate_plate_calibration(frame, layout)
    SESSION.plate_calibration = result
    return result


def attach_plate_calibration(bundle_path: Path, layout_path: Path) -> str:
    """Attach plate calibration to the calibration bundle."""
    if SESSION.calibration_bundle is None:
        raise RuntimeError("Calibration bundle is not loaded.")
    if SESSION.plate_calibration is None:
        raise RuntimeError("Calibration plate has not been detected yet.")
    updated = calibration.CalibrationBundle(
        camera=SESSION.calibration_bundle.camera,
        workspace_plane=SESSION.calibration_bundle.workspace_plane,
        pixel_to_mm=SESSION.calibration_bundle.pixel_to_mm,
        plate_homography=SESSION.plate_calibration.homography,
        plate_layout_path=str(layout_path),
    )
    calibration.save_calibration(updated, bundle_path)
    SESSION.calibration_bundle = updated
    return f"Updated bundle saved to {bundle_path}."
