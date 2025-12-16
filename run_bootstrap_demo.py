"""
Entry point for the bootstrap pick-and-place demo.

This script wires together camera capture, workspace detection, and robot motion
through the orchestrator. Replace the frame_provider stub with a real camera
read (e.g., DepthAI/OAK-D API) when running on hardware.
"""

from __future__ import annotations

import asyncio
import logging

import numpy as np

from app.orchestrator import DemoConfig, run_demo_with_frame
from vision.workspace import RigidTransform


logging.basicConfig(level=logging.INFO)


def frame_provider_stub() -> np.ndarray:
    """Placeholder frame provider returning a blank image.

    Replace with actual camera capture when deploying on the Jetson.
    """

    return np.zeros((720, 1280, 3), dtype=np.uint8)


def main() -> None:
    config = DemoConfig()
    camera_to_robot = RigidTransform(rotation=np.eye(3), translation=np.zeros(3))
    asyncio.run(run_demo_with_frame(config, frame_provider_stub, camera_to_robot))


if __name__ == "__main__":
    main()
