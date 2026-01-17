"""Placeholder frame generation for camera displays.

Provides a standardized placeholder image when camera streams are unavailable.
"""

from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np


def placeholder_frame(
    message: str,
    size: Tuple[int, int] = (720, 1280),
    background: Tuple[int, int, int] = (11, 15, 20),
    text_color: Tuple[int, int, int] = (128, 128, 128),
) -> np.ndarray:
    """Create a placeholder frame with a centered text message.

    Args:
        message: Text message to display on the frame.
        size: Frame size as (height, width). Defaults to 720x1280.
        background: BGR color tuple for background. Defaults to dark gray.
        text_color: BGR color tuple for text. Defaults to medium gray.

    Returns:
        BGR frame with centered text message.

    Examples:
        >>> frame = placeholder_frame("Camera offline")
        >>> frame.shape
        (720, 1280, 3)
        >>> frame = placeholder_frame("Loading...", size=(480, 640))
        >>> frame.shape
        (480, 640, 3)
    """
    height, width = size
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame[:] = background

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.9
    thickness = 2

    # Calculate text size for centering
    text_size = cv2.getTextSize(message, font, font_scale, thickness)[0]
    text_x = (width - text_size[0]) // 2
    text_y = (height + text_size[1]) // 2

    cv2.putText(
        frame,
        message,
        (text_x, text_y),
        font,
        font_scale,
        text_color,
        thickness,
    )
    return frame
