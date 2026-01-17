"""Base classes and utilities for overlay system."""

from __future__ import annotations

from typing import Optional

import numpy as np


def _plate_center_px(detections: dict) -> Optional[np.ndarray]:
    """Calculate the center point of plate markers in pixels."""
    if not detections:
        return None
    centers = [det.center for det in detections.values()]
    if not centers:
        return None
    return np.mean(np.asarray(centers, dtype=np.float64), axis=0)
