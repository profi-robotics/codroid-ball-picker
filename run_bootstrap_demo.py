"""
Entry point for the bootstrap pick-and-place demo.

This script wires together camera capture, workspace detection, and robot motion
through the orchestrator. Replace the frame_provider stub with a real camera
read (e.g., DepthAI/OAK-D API) when running on hardware.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
from contextlib import ExitStack
from pathlib import Path
from typing import TYPE_CHECKING, Any

import cv2

import numpy as np

from app.orchestrator import BootstrapDemo, DemoConfig, run_demo_with_frame
from vision.capture import (
    CaptureConfig,
    DEFAULT_CAPTURE_COUNT,
    DEFAULT_FPS,
    DEFAULT_RESOLUTION,
    capture_calibration_assets,
    resolve_camera_sockets,
)
from vision.workspace import RigidTransform
from vision import markers, workspace, calibration
from vision.calibration import CalibrationBundle

if TYPE_CHECKING:
    import depthai as dai

logging.basicConfig(level=logging.INFO)


def frame_provider_stub() -> np.ndarray:
    """Placeholder frame provider returning a blank image.

    Replace with actual camera capture when deploying on the Jetson.
    """

    return np.zeros((720, 1280, 3), dtype=np.uint8)


def parse_resolution(value: str) -> tuple[int, int]:
    parts = value.lower().split("x")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("resolution must use the format WxH, e.g. 640x480")
    try:
        return int(parts[0]), int(parts[1])
    except ValueError as exc:
        raise argparse.ArgumentTypeError("resolution components must be integers") from exc


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Codroid bootstrap demo.")
    parser.add_argument(
        "--aruco-board-layout",
        type=Path,
        default=Path("data/markers/camera_calibration/camera_calibration_board.json"),
        help="Path to the ArUco board layout JSON used for camera calibration.",
    )
    parser.add_argument(
        "--auto-calibrate",
        action="store_true",
        help="Capture fresh ArUco board/color/depth data before computing calibration.",
    )
    parser.add_argument(
        "--capture-output-dir",
        type=Path,
        default=Path("data/calibration"),
        help="Directory where auto-calibrated frames are written.",
    )
    parser.add_argument(
        "--capture-count",
        type=int,
        default=DEFAULT_CAPTURE_COUNT,
        help="Color images collected when auto-calibrating.",
    )
    parser.add_argument(
        "--capture-preview",
        action="store_true",
        default=True,
        help="Show a preview and press 'c' to capture each auto-calibration frame.",
    )
    parser.add_argument(
        "--capture-resolution",
        type=parse_resolution,
        default=DEFAULT_RESOLUTION,
        help="Color capture resolution when auto-calibrating (default 640x480).",
    )
    parser.add_argument(
        "--capture-fps",
        type=float,
        default=DEFAULT_FPS,
        help="Target frames per second when auto-calibrating.",
    )
    parser.add_argument(
        "--use-depthai-camera",
        action="store_true",
        help="Use a live OAK-D feed as the frame provider instead of the stub.",
    )
    parser.add_argument(
        "--camera-resolution",
        type=parse_resolution,
        default=DEFAULT_RESOLUTION,
        help="Live DepthAI camera resolution WIDTHxHEIGHT (default %(default)s).",
    )
    parser.add_argument(
        "--camera-fps",
        type=float,
        default=DEFAULT_FPS,
        help="Live DepthAI camera FPS when --use-depthai-camera is enabled.",
    )
    parser.add_argument(
        "--marker-retry-attempts",
        type=int,
        default=0,
        help="Number of extra frames to sample if marker detection fails (default 0, auto-set when --use-depthai-camera).",
    )
    parser.add_argument(
        "--marker-debug-frame",
        type=Path,
        default=Path("data/marker_debug_frame.png"),
        help="Path where the last frame is written if marker detection ultimately fails.",
    )
    parser.add_argument(
        "--marker-preview",
        action="store_true",
        help="Show a live preview before marker detection and press 'm' to lock the frame.",
    )
    parser.add_argument(
        "--track-workspace",
        action="store_true",
        help="Continuously track and overlay the workspace on the live camera feed (skip robot motion).",
    )
    parser.add_argument(
        "--workspace-scale",
        type=float,
        default=1.0,
        help="Optional multiplier applied to the computed workspace dimensions.",
    )

    args = parser.parse_args()
    config = DemoConfig(
        workspace_scale=args.workspace_scale,
        aruco_board_layout_path=args.aruco_board_layout,
    )

    capture_assets = None
    if args.auto_calibrate:
        capture_config = CaptureConfig(
            output_dir=args.capture_output_dir,
            count=args.capture_count,
            preview=args.capture_preview,
            resolution=args.capture_resolution,
            fps=args.capture_fps,
        )
        capture_assets = capture_calibration_assets(capture_config)

    camera_to_robot = RigidTransform(rotation=np.eye(3), translation=np.zeros(3))
    marker_retry_attempts = args.marker_retry_attempts
    if marker_retry_attempts == 0 and args.use_depthai_camera:
        marker_retry_attempts = 20

    with ExitStack() as stack:
        frame_provider = frame_provider_stub
        if args.use_depthai_camera:
            depthai_provider = stack.enter_context(
                DepthAIFrameProvider(args.camera_resolution, args.camera_fps)
            )
            frame_provider = depthai_provider.get_frame

        if args.track_workspace:
            asyncio.run(
                track_workspace_live(
                    config,
                    frame_provider,
                    camera_to_robot,
                    calibration_capture_assets=capture_assets,
                    marker_retry_attempts=marker_retry_attempts,
                    marker_debug_frame=args.marker_debug_frame,
                    marker_preview=args.marker_preview,
                )
            )
        else:
            asyncio.run(
                run_demo_with_frame(
                    config,
                    frame_provider,
                    camera_to_robot,
                    calibration_capture_assets=capture_assets,
                    marker_retry_attempts=marker_retry_attempts,
                    marker_debug_frame=args.marker_debug_frame,
                    marker_preview=args.marker_preview,
                )
            )


class DepthAIFrameProvider:
    """Context manager that exposes depthai frames via get_frame()."""

    def __init__(self, resolution: tuple[int, int], fps: float) -> None:
        self.resolution = resolution
        self.fps = fps
        self._stack: ExitStack | None = None
        self._queue: Any = None
        self._depth_queue: Any = None
        self._pipeline: Any = None

    def __enter__(self) -> DepthAIFrameProvider:
        import depthai as dai  # Imported lazily to avoid dependency when unused.

        self._stack = ExitStack()
        pipeline = self._stack.enter_context(dai.Pipeline())
        device = pipeline.getDefaultDevice()
        sockets = resolve_camera_sockets(device)

        sanitized_resolution = (
            self.resolution[0] - (self.resolution[0] % 16),
            self.resolution[1] - (self.resolution[1] % 2),
        )
        color = pipeline.create(dai.node.Camera).build(sockets.color)
        color_out = color.requestOutput(sanitized_resolution, dai.ImgFrame.Type.BGR888i, fps=self.fps)
        self._queue = color_out.createOutputQueue(maxSize=2, blocking=True)

        left = pipeline.create(dai.node.Camera).build(sockets.left)
        right = pipeline.create(dai.node.Camera).build(sockets.right)
        mono_resolution = (min(1280, sanitized_resolution[0]), min(800, sanitized_resolution[1]))
        left_out = left.requestOutput(mono_resolution, dai.ImgFrame.Type.GRAY8, fps=self.fps)
        right_out = right.requestOutput(mono_resolution, dai.ImgFrame.Type.GRAY8, fps=self.fps)
        stereo = pipeline.create(dai.node.StereoDepth).build(left=left_out, right=right_out)
        stereo.setDepthAlign(sockets.color)
        stereo.setRectification(True)
        stereo.setLeftRightCheck(True)
        stereo.setExtendedDisparity(False)
        stereo.setSubpixel(True)
        stereo.setRectifyEdgeFillColor(0)
        stereo.setOutputSize(*sanitized_resolution)
        self._depth_queue = stereo.depth.createOutputQueue(maxSize=2, blocking=False)
        logging.info("DepthAI camera connected to %s", device.getMxId())
        pipeline.start()
        self._pipeline = pipeline
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._pipeline is not None:
            self._pipeline.stop()
        if self._stack is not None:
            self._stack.close()
        self._queue = None
        self._depth_queue = None
        self._pipeline = None

    def get_frame(self):
        if self._queue is None:
            raise RuntimeError("DepthAI frame provider is not initialized.")
        packet = self._queue.get()
        frame = packet.getCvFrame()
        if frame is None or frame.size == 0:
            raise RuntimeError("DepthAI frame provider returned an empty frame.")
        depth_frame = None
        if self._depth_queue is not None:
            depth_packet = self._depth_queue.tryGet()
            if depth_packet is not None:
                depth_frame = depth_packet.getFrame()
        return frame if depth_frame is None else (frame, depth_frame)


async def track_workspace_live(
    config: DemoConfig,
    frame_provider: callable,
    camera_to_robot: RigidTransform,
    calibration_capture_assets=None,
    marker_retry_attempts: int = 0,
    marker_debug_frame: Path | None = None,
    marker_preview: bool = False,
) -> None:
    """Continuously track markers and overlay workspace; skip robot motion."""

    demo = BootstrapDemo(config)
    image_paths = None
    depth_frame = None
    force_rebuild = False
    if calibration_capture_assets is not None:
        image_paths, depth_frame = calibration_capture_assets
        force_rebuild = True
    demo.load_or_create_calibration(image_paths=image_paths, depth_frame=depth_frame, force_rebuild=force_rebuild)

    attempts_remaining = max(1, marker_retry_attempts + 1)
    window = "Workspace Tracker"

    def _overlay(frame: np.ndarray, detections=None, ws=None, status: str | None = None, plane_offset: float | None = None) -> np.ndarray:
        vis = frame.copy()
        if detections is not None:
            for det, color in zip((detections.marker_a, detections.marker_b), ((0, 255, 0), (0, 0, 255))):
                pts = det.corners.astype(int)
                cv2.polylines(vis, [pts], isClosed=True, color=color, thickness=2)
                center = tuple(np.round(det.center).astype(int))
                cv2.circle(vis, center, 4, color, -1)
                cv2.putText(vis, f"ID {det.marker_id}", center, cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            if ws is not None:
                txt = f"Workspace: {ws.width_mm:.1f}mm x {ws.height_mm:.1f}mm"
                cv2.putText(vis, txt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        if plane_offset is not None:
            cv2.putText(
                vis,
                f"Camera-plane offset: {plane_offset:.1f} (depth units)",
                (10, 55),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (200, 200, 200),
                2,
            )
        if status:
            cv2.putText(vis, status, (10, vis.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        return vis

    baseline_plane = demo.calibration.workspace_plane if demo.calibration else None
    baseline_depth = None  # median depth baseline
    smoothed_depth = None

    missing_frames = 0
    missing_threshold = 300  # allow ~10s of occlusion at ~30 FPS before discarding the last stable workspace
    ws_smooth = None
    stable_ws = None
    stable_hits = 0
    stable_required = 60  # number of consecutive detections before updating the displayed workspace
    last_plane_offset = None

    while True:
        raw = frame_provider()
        depth_frame = None
        frame = raw
        if isinstance(raw, tuple) or isinstance(raw, list):
            if len(raw) >= 1:
                frame = raw[0]
            if len(raw) >= 2:
                depth_frame = raw[1]

        if marker_preview:
            cv2.imshow(window, _overlay(frame, status="Press 'm' to lock frame, 'q' to quit"))
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                cv2.destroyWindow(window)
                return
            if key == ord("m") or key == ord(" "):
                marker_preview = False
            else:
                continue

        detections = None
        ws = None
        last_exception = None
        plane_offset = None
        for attempt in range(1, attempts_remaining + 1):
            try:
                detections = markers.detect_markers(frame, config.marker_ids)
                temp_bundle = demo.calibration
                if depth_frame is not None and demo.calibration is not None:
                    try:
                        depth = depth_frame.astype(np.float32)
                        valid = (depth > 0) & np.isfinite(depth)
                        if np.count_nonzero(valid) > 0:
                            median_depth = float(np.median(depth[valid]))
                            if baseline_depth is None:
                                baseline_depth = median_depth
                            if smoothed_depth is None:
                                smoothed_depth = median_depth
                            else:
                                alpha = 0.02
                                smoothed_depth = (1 - alpha) * smoothed_depth + alpha * median_depth
                            plane_offset = smoothed_depth
                        scale_ratio = 1.0  # lock scale to avoid depth-induced jumps
                        temp_bundle = CalibrationBundle(
                            camera=demo.calibration.camera,
                            workspace_plane=baseline_plane if baseline_plane else demo.calibration.workspace_plane,
                            pixel_to_mm=demo.calibration.pixel_to_mm * scale_ratio,
                            plate_homography=demo.calibration.plate_homography,
                            plate_layout_path=demo.calibration.plate_layout_path,
                        )
                    except Exception:
                        temp_bundle = demo.calibration

                ws = workspace.compute_workspace(
                    detections,
                    temp_bundle,
                    marker_size_mm=config.marker_size_mm,
                    camera_to_robot=camera_to_robot,
                )
                # Smooth workspace dimensions to reduce jitter, and debounce updates until stable.
                if ws_smooth is None:
                    ws_smooth = ws
                if stable_ws is None:
                    stable_ws = ws_smooth
                stable_hits += 1
                if stable_hits >= stable_required:
                    alpha_ws = 0.1
                    ws_smooth = workspace.smooth_workspace(ws_smooth, ws, alpha_ws)
                    stable_ws = ws_smooth
                last_plane_offset = plane_offset
                last_exception = None
                missing_frames = 0
                break
            except RuntimeError as exc:
                last_exception = exc
                if "No markers detected" not in str(exc) or attempt == attempts_remaining:
                    break
                raw = frame_provider()
                depth_frame = None
                frame = raw
                if isinstance(raw, tuple) or isinstance(raw, list):
                    frame = raw[0]
                    depth_frame = raw[1] if len(raw) > 1 else None
                continue

        if last_exception is not None:
            missing_frames += 1
            if missing_frames > missing_threshold:
                stable_hits = 0
                stable_ws = None
                last_plane_offset = None
            status_text = "No markers detected; showing last workspace." if stable_ws is not None else "No markers; adjust view."
            vis = _overlay(
                frame,
                detections=None,
                ws=stable_ws,
                plane_offset=last_plane_offset,
                status=status_text,
            )
            if marker_debug_frame is not None and missing_frames >= missing_threshold and stable_ws is None:
                marker_debug_frame.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(marker_debug_frame), frame)
            cv2.imshow(window, vis)
        else:
            vis = _overlay(
                frame,
                detections=detections,
                ws=stable_ws or ws_smooth or ws,
                plane_offset=plane_offset if plane_offset is not None else last_plane_offset,
            )
            cv2.imshow(window, vis)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    cv2.destroyWindow(window)


if __name__ == "__main__":
    main()
