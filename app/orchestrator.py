"""
Bootstrap demo orchestrator.

This module wires together camera calibration, marker detection, workspace
construction, and robot motion. The orchestration is intentionally explicit and
chatty to make debugging in the lab straightforward.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Tuple

import numpy as np

from robot.websocket_client import CodroidWebSocketClient, MotionParameters
from vision import calibration, markers, workspace
from vision.workspace import RigidTransform, WorkspaceDefinition

LOGGER = logging.getLogger(__name__)


@dataclass
class DemoConfig:
    marker_ids: Tuple[int, int] = (1, 2)
    marker_size_mm: float = 40.0
    calibration_path: Path = Path("data/calibration_bundle.npz")
    websocket_uri: str = "ws://192.168.0.10:9090/codroid/ws"
    approach_height_mm: float = 30.0
    drop_offset_mm: float = 0.0


class BootstrapDemo:
    def __init__(self, config: DemoConfig) -> None:
        self.config = config
        self.calibration: Optional[calibration.CalibrationBundle] = None
        self.workspace: Optional[WorkspaceDefinition] = None
        self.robot: Optional[CodroidWebSocketClient] = None

    def load_or_create_calibration(self, image_paths=None, board_size=(7, 6), square_size_mm=25.0, depth_frame=None) -> None:
        if self.config.calibration_path.exists():
            LOGGER.info("Loading calibration from %s", self.config.calibration_path)
            self.calibration = calibration.load_calibration(self.config.calibration_path)
            return

        if image_paths is None or depth_frame is None:
            raise RuntimeError("Calibration data not found on disk; provide images and a depth frame to calibrate.")

        cam = calibration.calibrate_camera(image_paths, board_size, square_size_mm)
        plane = calibration.estimate_workspace_plane(depth_frame, cam)
        # Pixel to mm scale is estimated from the chessboard square size.
        pixel_to_mm = square_size_mm
        bundle = calibration.CalibrationBundle(camera=cam, workspace_plane=plane, pixel_to_mm=pixel_to_mm)
        calibration.save_calibration(bundle, self.config.calibration_path)
        self.calibration = bundle
        LOGGER.info("Saved calibration bundle to %s", self.config.calibration_path)

    def detect_workspace(self, frame: np.ndarray, camera_to_robot: RigidTransform) -> None:
        if self.calibration is None:
            raise RuntimeError("Calibration must be loaded before detecting workspace.")
        detected = markers.detect_markers(frame, self.config.marker_ids)
        self.workspace = workspace.compute_workspace(
            detected,
            self.calibration,
            marker_size_mm=self.config.marker_size_mm,
            camera_to_robot=camera_to_robot,
        )
        LOGGER.info(
            "Workspace detected: width=%.1fmm height=%.1fmm", self.workspace.width_mm, self.workspace.height_mm
        )

    async def _ensure_robot(self) -> CodroidWebSocketClient:
        if self.robot is None:
            self.robot = CodroidWebSocketClient(self.config.websocket_uri)
            await self.robot.connect()
        return self.robot

    async def run_motion_sequence(self) -> None:
        if self.workspace is None:
            raise RuntimeError("Workspace has not been computed.")

        robot = await self._ensure_robot()
        await robot.enable()

        params = MotionParameters(speed=50.0, acceleration=100.0, blend=0.0)
        center_xy = workspace.workspace_center(self.workspace)
        drop_xy = workspace.bottom_left(self.workspace)

        center_pose = self._pose_from_workspace(center_xy, self.config.approach_height_mm)
        drop_pose = self._pose_from_workspace(drop_xy, self.config.drop_offset_mm)

        LOGGER.info("Moving to workspace center %s", center_xy)
        await robot.movl(center_pose, params)

        LOGGER.info("Simulated grip close at center")

        LOGGER.info("Moving to bottom-left drop %s", drop_xy)
        await robot.movl(drop_pose, params)

        LOGGER.info("Simulated grip open at drop location")
        await robot.disable()

    def _pose_from_workspace(self, xy_mm: Tuple[float, float], z_offset_mm: float) -> dict:
        assert self.workspace is not None
        robot_xyz = self.workspace.workspace_to_robot(xy_mm[0], xy_mm[1], z_offset_mm)
        # Orientation left as zeroed Euler for initial bootstrap.
        return {"x": robot_xyz[0], "y": robot_xyz[1], "z": robot_xyz[2], "rx": 0.0, "ry": 0.0, "rz": 0.0}


async def run_demo_with_frame(
    config: DemoConfig,
    frame_provider: Callable[[], np.ndarray],
    camera_to_robot: RigidTransform,
) -> None:
    """
    Execute the bootstrap demo using a provided camera frame callback.

    The callback is used instead of directly acquiring frames to keep this module
    hardware-agnostic.
    """

    demo = BootstrapDemo(config)
    frame = frame_provider()
    if frame is None or frame.size == 0:
        raise RuntimeError(
            "Frame provider returned an empty frame. Verify the camera is connected and functioning (use `python -m vision.camera_check`)."
        )
    demo.load_or_create_calibration()
    demo.detect_workspace(frame, camera_to_robot)
    await demo.run_motion_sequence()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    dummy_frame = np.zeros((640, 480, 3), dtype=np.uint8)
    default_transform = RigidTransform(rotation=np.eye(3), translation=np.zeros(3))

    asyncio.run(run_demo_with_frame(DemoConfig(), lambda: dummy_frame, default_transform))
