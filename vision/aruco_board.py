"""
ArUco grid board specifications for camera calibration.

The board spec is stored as JSON so the printed target stays in sync with code.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np


@dataclass(frozen=True)
class ArucoBoardSpec:
    markers_x: int
    markers_y: int
    marker_length_mm: float
    marker_separation_mm: float
    dictionary: str = "DICT_4X4_50"
    first_marker_id: int = 0

    def board_size_mm(self) -> Tuple[float, float]:
        width = self.markers_x * self.marker_length_mm + (self.markers_x - 1) * self.marker_separation_mm
        height = self.markers_y * self.marker_length_mm + (self.markers_y - 1) * self.marker_separation_mm
        return float(width), float(height)

    def marker_corners(self) -> Dict[int, np.ndarray]:
        corners: Dict[int, np.ndarray] = {}
        pitch = self.marker_length_mm + self.marker_separation_mm
        for row in range(self.markers_y):
            for col in range(self.markers_x):
                marker_id = self.first_marker_id + row * self.markers_x + col
                x0 = col * pitch
                y0 = row * pitch
                corners[marker_id] = np.array(
                    [
                        [x0, y0, 0.0],
                        [x0 + self.marker_length_mm, y0, 0.0],
                        [x0 + self.marker_length_mm, y0 + self.marker_length_mm, 0.0],
                        [x0, y0 + self.marker_length_mm, 0.0],
                    ],
                    dtype=np.float64,
                )
        return corners


def load_aruco_board_spec(path: Path) -> ArucoBoardSpec:
    data = json.loads(path.read_text(encoding="utf-8"))
    return ArucoBoardSpec(
        markers_x=int(data["markers_x"]),
        markers_y=int(data["markers_y"]),
        marker_length_mm=float(data["marker_length_mm"]),
        marker_separation_mm=float(data["marker_separation_mm"]),
        dictionary=data.get("dictionary", "DICT_4X4_50"),
        first_marker_id=int(data.get("first_marker_id", 0)),
    )
