"""MJPEG encoding helpers."""

from __future__ import annotations

import time
from typing import Callable, Iterable, Optional

import cv2
import numpy as np

from codroid_picker.video.frame_buffer import RawFrameBuffer

BOUNDARY = "frame"


def placeholder_frame(message: str = "Camera stream offline", size: tuple[int, int] = (640, 360)) -> np.ndarray:
    """Create a simple BGR placeholder frame."""
    width, height = size
    frame = np.full((height, width, 3), 32, dtype=np.uint8)
    cv2.putText(
        frame,
        message,
        (24, max(height // 2, 40)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (230, 230, 230),
        2,
        cv2.LINE_AA,
    )
    return frame


def resize_for_width(frame: np.ndarray, max_width: int) -> np.ndarray:
    """Downscale a frame to max width while preserving aspect ratio."""
    height, width = frame.shape[:2]
    if width <= max_width:
        return frame
    scale = max_width / float(width)
    target = (max_width, max(int(height * scale), 1))
    return cv2.resize(frame, target, interpolation=cv2.INTER_AREA)


def encode_jpeg(frame: np.ndarray, *, quality: int = 70, max_width: Optional[int] = None) -> bytes:
    """Encode a BGR frame to JPEG bytes."""
    if max_width is not None:
        frame = resize_for_width(frame, max_width)
    params = [int(cv2.IMWRITE_JPEG_QUALITY), min(max(int(quality), 30), 95)]
    ok, encoded = cv2.imencode(".jpg", frame, params)
    if not ok:
        raise RuntimeError("Failed to encode JPEG frame")
    return encoded.tobytes()


def mjpeg_frame(jpeg: bytes) -> bytes:
    """Wrap JPEG bytes in a multipart MJPEG frame."""
    header = (
        f"--{BOUNDARY}\r\n"
        "Content-Type: image/jpeg\r\n"
        f"Content-Length: {len(jpeg)}\r\n\r\n"
    ).encode("ascii")
    return header + jpeg + b"\r\n"


def mjpeg_stream(
    buffer: RawFrameBuffer,
    *,
    fps: float,
    quality: int,
    max_width: int,
    frame_transform: Callable[[np.ndarray], np.ndarray] | None = None,
) -> Iterable[bytes]:
    """Yield an endless newest-frame MJPEG stream."""
    interval_s = 1.0 / max(float(fps), 1.0)
    while True:
        snapshot = buffer.latest(copy=True)
        frame = snapshot.frame
        if frame is None:
            frame = placeholder_frame(size=(max_width, int(max_width * 9 / 16)))
        if frame_transform is not None:
            frame = frame_transform(frame)
        jpeg = encode_jpeg(frame, quality=quality, max_width=max_width)
        yield mjpeg_frame(jpeg)
        time.sleep(interval_s)
