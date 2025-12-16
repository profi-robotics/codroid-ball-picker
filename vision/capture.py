"""
Shared helpers for capturing chessboard images and depth frames from an OAK-D.
"""

from __future__ import annotations

from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import cv2
import depthai as dai
import numpy as np
import logging

LOGGER = logging.getLogger(__name__)

DEFAULT_RESOLUTION: tuple[int, int] = (1920, 1080)
DEFAULT_CAPTURE_COUNT = 15
DEFAULT_FPS = 30.0


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
        sensor_type = getattr(feature, "sensorType", None) or getattr(feature, "type", None)
        if sensor_type == dai.CameraSensorType.COLOR and color_socket is None:
            color_socket = feature.socket
        elif sensor_type == dai.CameraSensorType.MONO:
            mono_sockets.append(feature.socket)

    if color_socket is None:
        color_socket = getattr(dai.CameraBoardSocket, "CAM_A", None) or getattr(dai.CameraBoardSocket, "RGB", None)
    if len(mono_sockets) < 2:
        # Fall back to standard CAM_B/C ordering for older RVC2 hardware.
        default_left = getattr(dai.CameraBoardSocket, "CAM_B", None) or getattr(dai.CameraBoardSocket, "LEFT", None)
        default_right = getattr(dai.CameraBoardSocket, "CAM_C", None) or getattr(dai.CameraBoardSocket, "RIGHT", None)
        if default_left:
            mono_sockets.insert(0, default_left)
        if default_right:
            mono_sockets.append(default_right)

    if color_socket is None or len(mono_sockets) < 2:
        raise RuntimeError("Unable to determine OAK-D camera sockets; check the USB connection and board configuration.")

    return CameraSockets(color=color_socket, left=mono_sockets[0], right=mono_sockets[1])


def _build_pipeline(
    pipeline: dai.Pipeline,
    sockets: CameraSockets,
    resolution: tuple[int, int],
    fps: float,
) -> tuple[dai.Node.Output, dai.Node.Output]:
    """Create the Luxonis template-aligned capture pipeline."""

    color = pipeline.create(dai.node.Camera).build(sockets.color)
    sanitized_resolution = (
        resolution[0] - (resolution[0] % 16),
        resolution[1] - (resolution[1] % 2),
    )
    color_out = color.requestOutput(sanitized_resolution, dai.ImgFrame.Type.BGR888i, fps=fps)

    left = pipeline.create(dai.node.Camera).build(sockets.left)
    right = pipeline.create(dai.node.Camera).build(sockets.right)
    mono_resolution = (min(1280, resolution[0]), min(800, resolution[1]))
    left_out = left.requestOutput(mono_resolution, dai.ImgFrame.Type.GRAY8, fps=fps)
    right_out = right.requestOutput(mono_resolution, dai.ImgFrame.Type.GRAY8, fps=fps)

    stereo = pipeline.create(dai.node.StereoDepth).build(left=left_out, right=right_out)
    stereo.setDepthAlign(sockets.color)
    stereo.setRectification(True)
    stereo.setLeftRightCheck(True)
    stereo.setExtendedDisparity(False)
    stereo.setSubpixel(True)
    stereo.setRectifyEdgeFillColor(0)
    stereo.setOutputSize(*sanitized_resolution)

    return color_out, stereo.depth


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
            input(f"Align the board and press Enter to capture image {index}/{count}...")

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


def capture_calibration_assets(config: CaptureConfig) -> Tuple[List[Path], np.ndarray]:
    with ExitStack() as stack:
        pipeline = stack.enter_context(dai.Pipeline())
        device = pipeline.getDefaultDevice()
        sockets = resolve_camera_sockets(device)
        color_out, depth_out = _build_pipeline(pipeline, sockets, config.resolution, config.fps)
        color_queue = color_out.createOutputQueue(maxSize=4, blocking=False)
        depth_queue = depth_out.createOutputQueue(maxSize=2, blocking=True)

        logging.info("Connected to %s; capturing %d images.", device.getMxId(), config.count)
        pipeline.start()
        images = _capture_color_images(color_queue, config.output_dir, config.count, config.preview)
        depth_frame, _ = _capture_depth_snapshot(depth_queue, config.output_dir)
        pipeline.stop()

    if not images:
        raise RuntimeError("No calibration images captured; run the capture routine again.")

    return images, depth_frame
