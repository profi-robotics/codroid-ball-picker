"""
Shared helpers for capturing ChArUco board images and depth frames from an OAK-D.
"""

from __future__ import annotations

from contextlib import ExitStack
from dataclasses import dataclass
import os
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import depthai as dai
import numpy as np
import logging

LOGGER = logging.getLogger(__name__)

DEFAULT_RESOLUTION: tuple[int, int] = (1920, 1080)
DEFAULT_CAPTURE_COUNT = 15
DEFAULT_FPS = 30.0
DEFAULT_DEPTH_MAX_FPS = 10.0


def _resolve_color_output_type() -> dai.ImgFrame.Type:
    """Resolve color stream transport format from env."""
    raw_value = os.getenv("CODROID_COLOR_OUTPUT_FORMAT", "nv12").strip().lower()
    if raw_value in {"bgr", "bgr888i"}:
        return dai.ImgFrame.Type.BGR888i
    return dai.ImgFrame.Type.NV12


def _resolve_depth_max_fps() -> float:
    """Resolve depth FPS cap from env, falling back to a stable default."""
    raw_value = os.getenv("CODROID_DEPTH_MAX_FPS", str(DEFAULT_DEPTH_MAX_FPS))
    try:
        value = float(raw_value)
    except (TypeError, ValueError):
        return DEFAULT_DEPTH_MAX_FPS
    if value <= 0:
        return DEFAULT_DEPTH_MAX_FPS
    return value


@dataclass
class CaptureConfig:
    output_dir: Path = Path("data/calibration")
    count: int = DEFAULT_CAPTURE_COUNT
    preview: bool = False
    resolution: tuple[int, int] = DEFAULT_RESOLUTION
    fps: float = DEFAULT_FPS


@dataclass
class CameraSockets:
    color: dai.CameraBoardSocket
    left: dai.CameraBoardSocket
    right: dai.CameraBoardSocket


def resolve_camera_sockets(device: dai.Device) -> CameraSockets:
    """Detect which board sockets expose RGB + stereo sensors."""

    features = device.getConnectedCameraFeatures()
    color_socket: dai.CameraBoardSocket | None = None
    mono_sockets: List[dai.CameraBoardSocket] = []

    for feature in features:
        sensor_type = getattr(feature, "sensorType", None) or getattr(
            feature, "type", None)
        if sensor_type == dai.CameraSensorType.COLOR and color_socket is None:
            color_socket = feature.socket
        elif sensor_type == dai.CameraSensorType.MONO:
            mono_sockets.append(feature.socket)

    if color_socket is None:
        color_socket = getattr(dai.CameraBoardSocket, "CAM_A", None) or getattr(
            dai.CameraBoardSocket, "RGB", None)
    if len(mono_sockets) < 2:
        # Fall back to standard CAM_B/C ordering for older RVC2 hardware.
        default_left = getattr(dai.CameraBoardSocket, "CAM_B", None) or getattr(
            dai.CameraBoardSocket, "LEFT", None)
        default_right = getattr(dai.CameraBoardSocket, "CAM_C", None) or getattr(
            dai.CameraBoardSocket, "RIGHT", None)
        if default_left:
            mono_sockets.insert(0, default_left)
        if default_right:
            mono_sockets.append(default_right)

    if color_socket is None or len(mono_sockets) < 2:
        raise RuntimeError(
            "Unable to determine OAK-D camera sockets; check the USB connection and board configuration.")

    return CameraSockets(color=color_socket, left=mono_sockets[0], right=mono_sockets[1])


def _build_pipeline(
    pipeline: dai.Pipeline,
    sockets: CameraSockets,
    resolution: tuple[int, int],
    fps: float,
    enable_depth: bool = True,
    fast_depth: bool = True,
    lock_af: bool = False,
    manual_focus_position: Optional[int] = None,
) -> tuple[dai.Node.Output, Optional[dai.Node.Output], Optional[object]]:
    """Create the Luxonis template-aligned capture pipeline."""

    color = pipeline.create(dai.node.Camera).build(sockets.color)
    if lock_af:
        try:
            focus_position = 130 if manual_focus_position is None else int(
                manual_focus_position
            )
            focus_position = max(0, min(255, focus_position))
            ctrl = color.initialControl
            ctrl.setAutoFocusMode(dai.CameraControl.AutoFocusMode.OFF)
            if hasattr(ctrl, "setAutoFocusEnable"):
                ctrl.setAutoFocusEnable(False)
            ctrl.setManualFocus(focus_position)
        except Exception as exc:  # noqa: BLE001
            logging.warning(
                "Unable to lock autofocus; continuing with defaults: %s", exc
            )
    sanitized_resolution = (
        resolution[0] - (resolution[0] % 16),
        resolution[1] - (resolution[1] % 2),
    )
    color_output_type = _resolve_color_output_type()
    LOGGER.info(
        "Color stream format: %s",
        "BGR888i" if color_output_type == dai.ImgFrame.Type.BGR888i else "NV12",
    )
    color_out = color.requestOutput(
        sanitized_resolution, color_output_type, fps=fps)
    control_in: Optional[object] = getattr(color, "inputControl", None)

    depth_output: Optional[dai.Node.Output] = None

    # Build stereo depth only when requested to reduce pipeline load for pure RGB streaming.
    # Downstream callers decide whether to create queues for the returned output.
    if enable_depth:
        left = pipeline.create(dai.node.Camera).build(sockets.left)
        right = pipeline.create(dai.node.Camera).build(sockets.right)
        mono_resolution = (min(1280, resolution[0]), min(800, resolution[1]))
        depth_fps = min(float(fps), _resolve_depth_max_fps())
        if depth_fps < float(fps):
            LOGGER.info(
                "Depth FPS capped for transport stability: requested %.1f, using %.1f",
                float(fps),
                depth_fps,
            )
        left_out = left.requestOutput(
            mono_resolution, dai.ImgFrame.Type.GRAY8, fps=depth_fps)
        right_out = right.requestOutput(
            mono_resolution, dai.ImgFrame.Type.GRAY8, fps=depth_fps)

        stereo = pipeline.create(dai.node.StereoDepth).build(
            left=left_out, right=right_out)
        stereo.setDepthAlign(sockets.color)
        stereo.setRectification(True)
        # Left-right check must stay enabled when aligning to COLOR (center alignment requirement).
        # Subpixel remains optional for performance control.
        stereo.setLeftRightCheck(True)
        stereo.setExtendedDisparity(False)
        stereo.setSubpixel(not fast_depth)
        stereo.setRectifyEdgeFillColor(0)
        stereo.setOutputSize(*sanitized_resolution)
        depth_output = stereo.depth

    return color_out, depth_output, control_in


def build_pipeline(
    pipeline: dai.Pipeline,
    sockets: CameraSockets,
    resolution: tuple[int, int],
    fps: float,
    enable_depth: bool = True,
    fast_depth: bool = True,
    lock_af: bool = False,
    manual_focus_position: Optional[int] = None,
) -> tuple[dai.Node.Output, Optional[dai.Node.Output], Optional[object]]:
    """Public wrapper for building the shared DepthAI pipeline."""

    return _build_pipeline(
        pipeline,
        sockets,
        resolution,
        fps,
        enable_depth=enable_depth,
        fast_depth=fast_depth,
        lock_af=lock_af,
        manual_focus_position=manual_focus_position,
    )


def _capture_color_images(
    color_queue: dai.DataOutputQueue,
    dest: Path,
    count: int,
    preview: bool,
) -> List[Path]:
    dest.mkdir(parents=True, exist_ok=True)
    window_name = "Calibration Capture"
    if preview:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    captured: List[Path] = []
    index = 1

    while index <= count:
        frame = color_queue.get().getCvFrame()
        if preview:
            cv2.imshow(window_name, frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key != ord("c"):
                continue
        else:
            input(
                f"Align the board and press Enter to capture image {index}/{count}...")

        path = dest / f"calibration_color_{index:02}.png"
        cv2.imwrite(str(path), frame)
        LOGGER.info("Saved calibration image %s", path)
        captured.append(path)
        index += 1

    if preview:
        cv2.destroyAllWindows()

    return captured


def _capture_depth_snapshot(depth_queue: dai.DataOutputQueue, dest: Path) -> Tuple[np.ndarray, Path]:
    dest.mkdir(parents=True, exist_ok=True)
    frame = depth_queue.get().getFrame()
    output_path = dest / "calibration_depth_frame.npz"
    np.savez_compressed(output_path, depth=frame.astype(np.float32))
    LOGGER.info("Saved depth frame %s", output_path)
    return frame, output_path


def capture_depth_snapshot_only(
    output_dir: Path,
    resolution: tuple[int, int] = DEFAULT_RESOLUTION,
    fps: float = DEFAULT_FPS,
    lock_af: bool = False,
    fast_depth: bool = False,
) -> Path:
    """Capture a single depth frame using a temporary DepthAI pipeline."""
    with ExitStack() as stack:
        pipeline = stack.enter_context(dai.Pipeline())
        device = pipeline.getDefaultDevice()
        if device is None:
            raise RuntimeError("No OAK-D device available for depth snapshot.")
        sockets = resolve_camera_sockets(device)
        color_out, depth_out, _control_in = _build_pipeline(
            pipeline,
            sockets,
            resolution,
            fps,
            enable_depth=True,
            fast_depth=fast_depth,
            lock_af=lock_af,
        )
        if depth_out is None:
            raise RuntimeError("Depth output unavailable from temporary pipeline.")
        # DepthAI requires every requested output to be consumed by a queue or link.
        _color_queue = color_out.createOutputQueue(maxSize=1, blocking=False)
        depth_queue = depth_out.createOutputQueue(maxSize=2, blocking=True)
        pipeline.start()
        _depth_frame, path = _capture_depth_snapshot(depth_queue, output_dir)
        pipeline.stop()
    return path


def capture_calibration_assets(config: CaptureConfig) -> Tuple[List[Path], np.ndarray]:
    with ExitStack() as stack:
        pipeline = stack.enter_context(dai.Pipeline())
        device = pipeline.getDefaultDevice()
        sockets = resolve_camera_sockets(device)
        color_out, depth_out, _control_in = _build_pipeline(
            pipeline,
            sockets,
            config.resolution,
            config.fps,
            enable_depth=True,
            fast_depth=False,
        )
        color_queue = color_out.createOutputQueue(maxSize=4, blocking=False)
        if depth_out is None:
            raise RuntimeError(
                "Depth output not available; enable depth capture for calibration.")
        depth_queue = depth_out.createOutputQueue(maxSize=2, blocking=True)

        logging.info("Connected to %s; capturing %d images.",
                     device.getMxId(), config.count)
        pipeline.start()
        images = _capture_color_images(
            color_queue, config.output_dir, config.count, config.preview)
        depth_frame, _ = _capture_depth_snapshot(
            depth_queue, config.output_dir)
        pipeline.stop()

    if not images:
        raise RuntimeError(
            "No calibration images captured; run the capture routine again.")

    return images, depth_frame
