"""
ChArUco board specifications for camera calibration.

The board spec is stored as JSON so the printed target stays in sync with code.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import cv2


@dataclass(frozen=True)
class CharucoBoardSpec:
    squares_x: int
    squares_y: int
    square_length_mm: float
    marker_length_mm: float
    dictionary: str = "DICT_4X4_50"

    def board_size_mm(self) -> Tuple[float, float]:
        width = self.squares_x * self.square_length_mm
        height = self.squares_y * self.square_length_mm
        return float(width), float(height)

    def create_board(self) -> cv2.aruco_CharucoBoard:
        dictionary_id = getattr(cv2.aruco, self.dictionary)
        dictionary = cv2.aruco.getPredefinedDictionary(dictionary_id)
        return cv2.aruco.CharucoBoard(
            (self.squares_x, self.squares_y),
            self.square_length_mm,
            self.marker_length_mm,
            dictionary,
        )


def load_charuco_board_spec(path: Path) -> CharucoBoardSpec:
    data = json.loads(path.read_text(encoding="utf-8"))
    board_type = data.get("board_type", "charuco")
    if board_type != "charuco":
        raise ValueError(f"Unsupported board_type '{board_type}' in {path}")
    return CharucoBoardSpec(
        squares_x=int(data["squares_x"]),
        squares_y=int(data["squares_y"]),
        square_length_mm=float(data["square_length_mm"]),
        marker_length_mm=float(data["marker_length_mm"]),
        dictionary=data.get("dictionary", "DICT_4X4_50"),
    )

