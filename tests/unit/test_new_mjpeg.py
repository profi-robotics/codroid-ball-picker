from __future__ import annotations

import cv2
import numpy as np

from codroid_picker.video.mjpeg import encode_jpeg, mjpeg_frame, placeholder_frame


def test_encode_jpeg_returns_decodable_image() -> None:
    frame = np.zeros((120, 160, 3), dtype=np.uint8)
    jpeg = encode_jpeg(frame, quality=70, max_width=80)

    decoded = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)

    assert decoded is not None
    assert decoded.shape[1] == 80


def test_mjpeg_frame_has_boundary_and_content_type() -> None:
    jpeg = encode_jpeg(placeholder_frame(size=(160, 90)))
    chunk = mjpeg_frame(jpeg)

    assert chunk.startswith(b"--frame\r\n")
    assert b"Content-Type: image/jpeg\r\n" in chunk
    assert chunk.endswith(b"\r\n")

