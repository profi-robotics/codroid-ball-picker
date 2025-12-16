"""
Wrapper CLI that uses `vision.capture` to collect calibration frames from an OAK-D.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from vision.capture import (
    CaptureConfig,
    DEFAULT_CAPTURE_COUNT,
    DEFAULT_FPS,
    DEFAULT_RESOLUTION,
    capture_calibration_assets,
)


def parse_resolution(value: str) -> tuple[int, int]:
    parts = value.lower().split("x")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("Resolution must be WIDTHxHEIGHT, e.g. 640x480")
    try:
        return int(parts[0]), int(parts[1])
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Resolution values must be integers") from exc


def main() -> None:
    parser = argparse.ArgumentParser(description="Capture calibration images and a depth frame.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/calibration"),
        help="Path where captured color images and depth data are written.",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=DEFAULT_CAPTURE_COUNT,
        help="Number of chessboard images to save (default %(default)s).",
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        default=True,
        help="Show a live preview and press 'c' to capture each frame (default on).",
    )
    parser.add_argument(
        "--resolution",
        type=parse_resolution,
        default=DEFAULT_RESOLUTION,
        help="Color preview resolution WIDTHxHEIGHT (default %(default)s).",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=DEFAULT_FPS,
        help="Target frames per second for the capture pipeline.",
    )

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    config = CaptureConfig(
        output_dir=args.output_dir,
        count=args.count,
        preview=args.preview,
        resolution=args.resolution,
        fps=args.fps,
    )

    capture_calibration_assets(config)


if __name__ == "__main__":
    main()
