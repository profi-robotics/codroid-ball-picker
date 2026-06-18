"""Capture a single OAK-D depth snapshot and write it to output dir."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from vision.capture import capture_depth_snapshot_only


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--lock-af", action="store_true")
    return parser.parse_args()


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    args = parse_args()
    path = capture_depth_snapshot_only(
        output_dir=args.output_dir,
        resolution=(args.width, args.height),
        fps=args.fps,
        lock_af=args.lock_af,
        fast_depth=False,
    )
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
