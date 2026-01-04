"""
Dash calibration UI for the Codroid bring-up workflow.
"""

from __future__ import annotations

import asyncio
import base64
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
from dash import Dash, Input, Output, State, callback_context, dcc, html
import dash_mantine_components as dmc

from app.orchestrator import BootstrapDemo, DemoConfig
from robot.websocket_client import CodroidWebSocketClient
from vision import aruco_board, calibration, markers, plate, workspace
from vision.camera_check import list_oak_devices
from vision.stream import DepthAIFrameStream, StreamConfig

LOGGER = logging.getLogger(__name__)

DEFAULT_BOARD_PATH = Path("data/markers/camera_calibration/camera_calibration_board.json")
DEFAULT_PLATE_PATH = Path("data/markers/robot_calibration/calibration_plate_layout.json")
DEFAULT_BUNDLE_PATH = Path("data/calibration_bundle.npz")
DEFAULT_CAPTURE_DIR = Path("data/calibration")
DEFAULT_RESOLUTION = (1280, 720)
DEFAULT_FPS = 30.0


@dataclass
class CaptureState:
    output_dir: Path = DEFAULT_CAPTURE_DIR
    frames: list[Path] = field(default_factory=list)
    depth_frame: Optional[np.ndarray] = None


@dataclass
class UISession:
    config: DemoConfig = field(default_factory=DemoConfig)
    camera_stream: Optional[DepthAIFrameStream] = None
    capture: CaptureState = field(default_factory=CaptureState)
    calibration_bundle: Optional[calibration.CalibrationBundle] = None
    workspace_definition: Optional[workspace.WorkspaceDefinition] = None
    plate_calibration: Optional[plate.PlateCalibration] = None


SESSION = UISession()


def _safe_int(value: Optional[float], fallback: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return fallback


def _safe_float(value: Optional[float], fallback: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return fallback


def _safe_path(value: Optional[str], fallback: Path) -> Path:
    if value is None:
        return fallback
    candidate = str(value).strip()
    return Path(candidate) if candidate else fallback


def _frame_index(path: Path) -> int:
    match = re.search(r"calibration_color_(\d+)\.png", path.name)
    if match:
        return int(match.group(1))
    return 0


def _list_capture_frames(output_dir: Path) -> list[Path]:
    frames = list(output_dir.glob("calibration_color_*.png"))
    frames.sort(key=_frame_index)
    return frames


def _update_capture_state(output_dir: Path) -> list[Path]:
    SESSION.capture.output_dir = output_dir
    frames = _list_capture_frames(output_dir)
    SESSION.capture.frames = frames
    return frames


def _placeholder_frame(message: str, size: Tuple[int, int] = (720, 1280)) -> np.ndarray:
    height, width = size
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    cv2.putText(
        frame,
        message,
        (40, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 255),
        2,
    )
    return frame


def _encode_frame(frame: np.ndarray) -> str:
    success, buffer = cv2.imencode(".jpg", frame)
    if not success:
        raise RuntimeError("Failed to encode camera frame.")
    encoded = base64.b64encode(buffer).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def _default_transform() -> workspace.RigidTransform:
    return workspace.RigidTransform(rotation=np.eye(3), translation=np.zeros(3))


def start_camera_stream(width: Optional[float], height: Optional[float], fps: Optional[float]) -> str:
    stop_camera_stream()
    resolution = (_safe_int(width, DEFAULT_RESOLUTION[0]), _safe_int(height, DEFAULT_RESOLUTION[1]))
    fps_value = _safe_float(fps, DEFAULT_FPS)
    stream = DepthAIFrameStream(StreamConfig(resolution=resolution, fps=fps_value))
    stream.start()
    SESSION.camera_stream = stream
    device_id = stream.device_id() or "unknown"
    return f"Camera stream running ({resolution[0]}x{resolution[1]} @ {fps_value:.1f} FPS, MXID {device_id})."


def stop_camera_stream() -> None:
    if SESSION.camera_stream is None:
        return
    SESSION.camera_stream.stop()
    SESSION.camera_stream = None


def get_latest_frame() -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    if SESSION.camera_stream is None or not SESSION.camera_stream.is_running():
        return None, None
    return SESSION.camera_stream.get_latest()


def capture_color_frame(output_dir: Path) -> Path:
    frame, _depth = get_latest_frame()
    if frame is None:
        raise RuntimeError("Camera stream is not providing frames yet.")
    output_dir.mkdir(parents=True, exist_ok=True)
    frames = _update_capture_state(output_dir)
    index = _frame_index(frames[-1]) + 1 if frames else 1
    path = output_dir / f"calibration_color_{index:02}.png"
    cv2.imwrite(str(path), frame)
    SESSION.capture.frames.append(path)
    return path


def capture_depth_snapshot(output_dir: Path) -> Path:
    _frame, depth = get_latest_frame()
    if depth is None:
        raise RuntimeError("No depth frame available from the camera stream.")
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "calibration_depth_frame.npz"
    np.savez_compressed(path, depth=depth.astype(np.float32))
    SESSION.capture.depth_frame = depth
    return path


def load_depth_snapshot(output_dir: Path) -> Optional[np.ndarray]:
    path = output_dir / "calibration_depth_frame.npz"
    if not path.exists():
        return None
    data = np.load(path, allow_pickle=True)
    depth = data.get("depth")
    if depth is None:
        return None
    return depth.astype(np.float32)


def compute_calibration_bundle(board_path: Path, bundle_path: Path, output_dir: Path) -> str:
    if not board_path.exists():
        raise FileNotFoundError(f"ChArUco board layout not found: {board_path}")
    frames = _update_capture_state(output_dir)
    if not frames:
        raise RuntimeError("No calibration frames found. Capture frames before calibrating.")
    depth_frame = SESSION.capture.depth_frame or load_depth_snapshot(output_dir)
    if depth_frame is None:
        raise RuntimeError("No depth snapshot found. Capture a depth frame before calibrating.")
    config = DemoConfig(calibration_path=bundle_path, aruco_board_layout_path=board_path)
    demo = BootstrapDemo(config)
    demo.load_or_create_calibration(image_paths=frames, depth_frame=depth_frame, force_rebuild=True)
    SESSION.calibration_bundle = demo.calibration
    return f"Calibration bundle saved to {bundle_path}."


def load_calibration_bundle(bundle_path: Path) -> str:
    if not bundle_path.exists():
        raise FileNotFoundError(f"Calibration bundle not found: {bundle_path}")
    SESSION.calibration_bundle = calibration.load_calibration(bundle_path)
    return f"Loaded calibration bundle from {bundle_path}."


def compute_workspace_from_latest(marker_ids: Tuple[int, int], marker_size_mm: float) -> workspace.WorkspaceDefinition:
    frame, _depth = get_latest_frame()
    if frame is None:
        raise RuntimeError("Camera stream is not providing frames yet.")
    if SESSION.calibration_bundle is None:
        raise RuntimeError("Calibration bundle is not loaded.")
    detections = markers.detect_markers(frame, marker_ids)
    ws = workspace.compute_workspace(
        detections,
        SESSION.calibration_bundle,
        marker_size_mm=marker_size_mm,
        camera_to_robot=_default_transform(),
    )
    SESSION.workspace_definition = ws
    return ws


def compute_plate_calibration_from_latest(layout_path: Path) -> plate.PlateCalibration:
    frame, _depth = get_latest_frame()
    if frame is None:
        raise RuntimeError("Camera stream is not providing frames yet.")
    if not layout_path.exists():
        raise FileNotFoundError(f"Calibration plate layout not found: {layout_path}")
    layout = plate.load_plate_layout(layout_path)
    result = plate.estimate_plate_calibration(frame, layout)
    SESSION.plate_calibration = result
    return result


def attach_plate_calibration(bundle_path: Path, layout_path: Path) -> str:
    if SESSION.calibration_bundle is None:
        raise RuntimeError("Calibration bundle is not loaded.")
    if SESSION.plate_calibration is None:
        raise RuntimeError("Calibration plate has not been detected yet.")
    updated = calibration.CalibrationBundle(
        camera=SESSION.calibration_bundle.camera,
        workspace_plane=SESSION.calibration_bundle.workspace_plane,
        pixel_to_mm=SESSION.calibration_bundle.pixel_to_mm,
        plate_homography=SESSION.plate_calibration.homography,
        plate_layout_path=str(layout_path),
    )
    calibration.save_calibration(updated, bundle_path)
    SESSION.calibration_bundle = updated
    return f"Updated bundle saved to {bundle_path}."


def _draw_label(frame: np.ndarray, text: str, origin: Tuple[int, int], color: Tuple[int, int, int]) -> None:
    cv2.putText(frame, text, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


def draw_charuco_overlay(frame: np.ndarray, board_path: Path) -> Tuple[np.ndarray, str, str]:
    vis = frame.copy()
    if not board_path.exists():
        status = "ChArUco layout not found."
        return vis, status, "ChArUco corners: --"
    board_spec = aruco_board.load_charuco_board_spec(board_path)
    board = board_spec.create_board()
    charuco_params = cv2.aruco.CharucoParameters()
    detector_params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.CharucoDetector(board, charuco_params, detector_params)
    gray = cv2.cvtColor(vis, cv2.COLOR_BGR2GRAY)
    charuco_corners, charuco_ids, marker_corners, marker_ids = detector.detectBoard(gray)
    marker_count = 0 if marker_ids is None else len(marker_ids)
    corner_count = 0 if charuco_ids is None else len(charuco_ids)
    if marker_ids is not None and marker_corners is not None:
        cv2.aruco.drawDetectedMarkers(vis, marker_corners, marker_ids)
    if charuco_ids is not None and charuco_corners is not None:
        cv2.aruco.drawDetectedCornersCharuco(vis, charuco_corners, charuco_ids)
    _draw_label(vis, f"Markers: {marker_count}", (10, 30), (255, 255, 255))
    _draw_label(vis, f"Corners: {corner_count}", (10, 55), (255, 255, 255))
    status = "Camera calibration overlay ready."
    metric = f"ChArUco corners: {corner_count}"
    return vis, status, metric


def draw_workspace_overlay(
    frame: np.ndarray,
    marker_ids: Tuple[int, int],
    marker_size_mm: float,
) -> Tuple[np.ndarray, str, str]:
    vis = frame.copy()
    try:
        detections = markers.detect_markers(frame, marker_ids)
        for det, color in zip((detections.marker_a, detections.marker_b), ((0, 255, 0), (0, 140, 255))):
            pts = det.corners.astype(int)
            cv2.polylines(vis, [pts], isClosed=True, color=color, thickness=2)
            center = tuple(np.round(det.center).astype(int))
            cv2.circle(vis, center, 4, color, -1)
            cv2.putText(vis, f"ID {det.marker_id}", center, cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        status = "Workspace markers detected."
    except Exception as exc:
        return vis, f"Workspace detection: {exc}", "Workspace: --"

    metric = "Workspace: --"
    if SESSION.calibration_bundle is not None:
        try:
            ws = workspace.compute_workspace(
                detections,
                SESSION.calibration_bundle,
                marker_size_mm=marker_size_mm,
                camera_to_robot=_default_transform(),
            )
            metric = f"Workspace: {ws.width_mm:.1f} x {ws.height_mm:.1f} mm"
            _draw_label(vis, metric, (10, 30), (255, 255, 255))
        except Exception as exc:
            metric = f"Workspace: {exc}"
    return vis, status, metric


def draw_plate_overlay(frame: np.ndarray, layout_path: Path) -> Tuple[np.ndarray, str, str]:
    vis = frame.copy()
    if not layout_path.exists():
        return vis, "Calibration plate layout not found.", "Plate error: --"
    plate_layout = plate.load_plate_layout(layout_path)
    try:
        calibration_result = plate.estimate_plate_calibration(frame, plate_layout)
    except Exception as exc:
        return vis, f"Plate detection: {exc}", "Plate error: --"

    for det in calibration_result.detections.values():
        pts = det.corners.astype(int)
        cv2.polylines(vis, [pts], isClosed=True, color=(255, 204, 0), thickness=2)
        center = tuple(np.round(det.center).astype(int))
        cv2.circle(vis, center, 4, (255, 204, 0), -1)
        cv2.putText(vis, f"ID {det.marker_id}", center, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 204, 0), 2)

    metric = f"Reprojection error: {calibration_result.reprojection_error_mm:.2f} mm"
    _draw_label(vis, metric, (10, 30), (255, 255, 255))
    status = f"Plate markers detected: {len(calibration_result.detections)}"
    return vis, status, metric


def build_header() -> dmc.AppShellHeader:
    return dmc.AppShellHeader(
        dmc.Group(
            [
                dmc.Stack(
                    [
                        dmc.Text("Codroid Calibration Console", className="app-title"),
                        dmc.Text("Vision + robot bring-up dashboard", size="sm", c="dimmed"),
                    ],
                    gap=2,
                ),
                dmc.Group(
                    [
                        dmc.Badge("Camera: off", id="header-camera-badge", color="gray", variant="light"),
                        dmc.Badge("Robot: idle", id="header-robot-badge", color="gray", variant="light"),
                        dmc.Button("Setup", id="open-setup", variant="outline"),
                    ],
                    gap="sm",
                ),
            ],
            justify="space-between",
            align="center",
            h=64,
            px="md",
        )
    )


def build_navbar() -> dmc.AppShellNavbar:
    return dmc.AppShellNavbar(
        dmc.Stack(
            [
                dmc.Text("Navigation", fw=600, size="sm", c="dimmed"),
                dmc.NavLink(label="Calibration", active=True),
                dmc.Divider(),
                dmc.Text("Codroid bring-up suite", size="xs", c="dimmed"),
            ],
            gap="sm",
        ),
        p="md",
    )


def build_setup_drawer() -> dmc.Drawer:
    return dmc.Drawer(
        id="setup-drawer",
        opened=False,
        title="Setup & Connections",
        padding="md",
        size="lg",
        children=dmc.Stack(
            [
                dmc.Text("Camera connection", fw=600),
                dmc.Group(
                    [
                        dmc.Button("Scan for camera", id="camera-scan", variant="light"),
                        dmc.Switch(
                            id="camera-enable",
                            label="Enable live camera stream",
                            checked=False,
                            color="teal",
                        ),
                    ],
                    gap="md",
                ),
                dmc.Group(
                    [
                        dmc.NumberInput(
                            id="camera-width",
                            label="Width",
                            value=DEFAULT_RESOLUTION[0],
                            min=320,
                            max=3840,
                        ),
                        dmc.NumberInput(
                            id="camera-height",
                            label="Height",
                            value=DEFAULT_RESOLUTION[1],
                            min=240,
                            max=2160,
                        ),
                        dmc.NumberInput(
                            id="camera-fps",
                            label="FPS",
                            value=DEFAULT_FPS,
                            min=1,
                            max=120,
                            step=1,
                        ),
                    ],
                    grow=True,
                ),
                dmc.Text(id="camera-status-text", size="sm", c="dimmed"),
                html.Div(id="camera-device-info"),
                dmc.Divider(),
                dmc.Text("Robot connection", fw=600),
                dmc.TextInput(
                    id="robot-uri",
                    label="WebSocket URI",
                    value=DemoConfig().websocket_uri,
                ),
                dmc.Button("Test robot connection", id="robot-test", variant="light"),
                dmc.Text(id="robot-status-text", size="sm", c="dimmed"),
                dmc.Group(
                    [
                        dmc.Button("Close", id="setup-close", variant="subtle"),
                    ],
                    justify="flex-end",
                ),
            ],
            gap="sm",
        ),
    )


def build_stepper_card() -> dmc.Card:
    return dmc.Card(
        [
            dmc.Group(
                [
                    dmc.Text("Calibration Steps", fw=600, size="lg"),
                    dmc.Group(
                        [
                            dmc.Button("Back", id="step-prev", variant="light", size="sm"),
                            dmc.Button("Next", id="step-next", size="sm"),
                        ],
                        gap="xs",
                    ),
                ],
                justify="space-between",
                align="center",
            ),
            dmc.Divider(),
            dmc.Stepper(
                id="calibration-stepper",
                active=0,
                orientation="vertical",
                children=[
                    dmc.StepperStep(
                        label="Camera calibration",
                        description="Intrinsics + workspace plane",
                        children=dmc.Stack(
                            [
                                dmc.Text(
                                    "Capture the ChArUco board frames and a depth snapshot.",
                                    size="sm",
                                    c="dimmed",
                                ),
                                dmc.TextInput(
                                    id="calibration-board-path",
                                    label="Board layout",
                                    value=str(DEFAULT_BOARD_PATH),
                                ),
                                dmc.TextInput(
                                    id="calibration-output-dir",
                                    label="Capture output dir",
                                    value=str(DEFAULT_CAPTURE_DIR),
                                ),
                                dmc.TextInput(
                                    id="calibration-bundle-path",
                                    label="Calibration bundle",
                                    value=str(DEFAULT_BUNDLE_PATH),
                                ),
                                dmc.Group(
                                    [
                                        dmc.Button("Capture frame", id="capture-frame", variant="outline"),
                                        dmc.Button("Capture depth", id="capture-depth", variant="outline"),
                                    ],
                                    grow=True,
                                ),
                                dmc.Group(
                                    [
                                        dmc.Button("Compute calibration", id="compute-calibration"),
                                        dmc.Button("Load bundle", id="load-calibration", variant="light"),
                                    ],
                                    grow=True,
                                ),
                                dmc.Text(id="camera-calibration-action", size="sm", c="dimmed"),
                                dmc.Text(id="camera-calibration-metrics", size="sm"),
                            ],
                            gap="xs",
                        ),
                    ),
                    dmc.StepperStep(
                        label="Workarea definition",
                        description="Detect workspace markers",
                        children=dmc.Stack(
                            [
                                dmc.Text(
                                    "Use the two corner markers to define the workarea size.",
                                    size="sm",
                                    c="dimmed",
                                ),
                                dmc.Group(
                                    [
                                        dmc.NumberInput(
                                            id="marker-id-a",
                                            label="Marker ID A",
                                            value=SESSION.config.marker_ids[0],
                                            min=0,
                                            max=49,
                                        ),
                                        dmc.NumberInput(
                                            id="marker-id-b",
                                            label="Marker ID B",
                                            value=SESSION.config.marker_ids[1],
                                            min=0,
                                            max=49,
                                        ),
                                    ],
                                    grow=True,
                                ),
                                dmc.NumberInput(
                                    id="marker-size",
                                    label="Marker size (mm)",
                                    value=SESSION.config.marker_size_mm,
                                    min=1,
                                    max=100,
                                ),
                                dmc.Button("Compute workspace", id="compute-workspace", variant="light"),
                                dmc.Text(id="workspace-action-status", size="sm", c="dimmed"),
                                dmc.Text(id="workspace-metrics", size="sm"),
                            ],
                            gap="xs",
                        ),
                    ),
                    dmc.StepperStep(
                        label="Robot calibration",
                        description="Detect plate markers",
                        children=dmc.Stack(
                            [
                                dmc.Text(
                                    "Detect the calibration plate and attach the homography.",
                                    size="sm",
                                    c="dimmed",
                                ),
                                dmc.TextInput(
                                    id="plate-layout-path",
                                    label="Plate layout",
                                    value=str(DEFAULT_PLATE_PATH),
                                ),
                                dmc.Group(
                                    [
                                        dmc.Button("Detect plate", id="compute-plate", variant="light"),
                                        dmc.Button("Attach to bundle", id="attach-plate"),
                                    ],
                                    grow=True,
                                ),
                                dmc.Text(id="plate-action-status", size="sm", c="dimmed"),
                                dmc.Text(id="plate-metrics", size="sm"),
                            ],
                            gap="xs",
                        ),
                    ),
                ],
            ),
        ],
        withBorder=True,
        radius="lg",
        className="panel-card",
    )


def build_camera_panel() -> dmc.Card:
    placeholder_src = _encode_frame(_placeholder_frame("Camera stream offline"))
    return dmc.Card(
        [
            dmc.Group(
                [
                    dmc.Text("Camera View", fw=600, size="lg"),
                    dmc.Badge("Overlay", variant="light"),
                ],
                justify="space-between",
                align="center",
            ),
            dmc.Text(id="overlay-status", size="sm", c="dimmed"),
            html.Img(id="camera-feed", src=placeholder_src, className="camera-feed"),
        ],
        withBorder=True,
        radius="lg",
        className="panel-card camera-panel",
    )


def build_calibration_tab() -> dmc.Grid:
    return dmc.Grid(
        [
            dmc.GridCol(
                dmc.Stack([build_stepper_card()], gap="md"),
                span={"base": 12, "md": 5},
            ),
            dmc.GridCol(
                dmc.Stack([build_camera_panel()], gap="md"),
                span={"base": 12, "md": 7},
            ),
        ],
        gutter="md",
    )


def build_layout() -> dmc.MantineProvider:
    theme = {
        "fontFamily": "Space Grotesk, IBM Plex Sans, sans-serif",
        "headings": {"fontFamily": "Space Grotesk, IBM Plex Sans, sans-serif"},
        "primaryColor": "teal",
    }
    return dmc.MantineProvider(
        theme=theme,
        withGlobalClasses=True,
        withCssVariables=True,
        children=html.Div(
            [
                dcc.Interval(id="camera-interval", interval=300, n_intervals=0),
                build_setup_drawer(),
                dmc.AppShell(
                    [
                        build_header(),
                        build_navbar(),
                        dmc.AppShellMain(
                            dmc.Container(
                                dmc.Tabs(
                                    [
                                        dmc.TabsList(
                                            [
                                                dmc.TabsTab("Calibration", value="calibration"),
                                            ]
                                        ),
                                        dmc.TabsPanel(build_calibration_tab(), value="calibration"),
                                    ],
                                    value="calibration",
                                    variant="pills",
                                    color="teal",
                                ),
                                size="xl",
                                className="main-panel",
                            )
                        ),
                    ],
                    header={"height": 64},
                    navbar={"width": 220, "breakpoint": "md", "collapsed": {"mobile": True}},
                    padding="md",
                ),
            ]
        ),
    )


def _format_device_list(devices: list[object]) -> html.Ul:
    if not devices:
        return html.Ul([html.Li("No OAK-D devices detected.")])
    items = []
    for idx, device in enumerate(devices, start=1):
        name = None
        mx_id_fn = getattr(device, "getMxId", None)
        device_id_fn = getattr(device, "getDeviceId", None)
        if callable(mx_id_fn):
            name = mx_id_fn()
        elif callable(device_id_fn):
            name = device_id_fn()
        else:
            name = str(device)
        state = getattr(device, "state", None)
        items.append(html.Li(f"{idx}. {name} ({state})"))
    return html.Ul(items)


async def _probe_robot(uri: str) -> dict:
    client = CodroidWebSocketClient(uri)
    await client.connect()
    try:
        response = await client.get_state()
    finally:
        await client.close()
    return response


def register_callbacks(app: Dash) -> None:
    @app.callback(
        Output("setup-drawer", "opened"),
        Input("open-setup", "n_clicks"),
        Input("setup-close", "n_clicks"),
        State("setup-drawer", "opened"),
    )
    def _toggle_setup_drawer(open_clicks: int, close_clicks: int, opened: bool) -> bool:
        trigger = callback_context.triggered_id
        if trigger == "open-setup":
            return True
        if trigger == "setup-close":
            return False
        return opened

    @app.callback(
        Output("calibration-stepper", "active"),
        Input("step-prev", "n_clicks"),
        Input("step-next", "n_clicks"),
        State("calibration-stepper", "active"),
    )
    def _update_stepper(prev_clicks: int, next_clicks: int, active: Optional[int]) -> int:
        current = active or 0
        trigger = callback_context.triggered_id
        if trigger == "step-next":
            return min(current + 1, 2)
        if trigger == "step-prev":
            return max(current - 1, 0)
        return current

    @app.callback(
        Output("camera-device-info", "children"),
        Input("camera-scan", "n_clicks"),
        prevent_initial_call=True,
    )
    def _scan_camera(_clicks: int) -> html.Ul:
        devices = list_oak_devices()
        return _format_device_list(devices)

    @app.callback(
        Output("camera-status-text", "children"),
        Output("header-camera-badge", "children"),
        Output("header-camera-badge", "color"),
        Output("camera-enable", "checked"),
        Input("camera-enable", "checked"),
        State("camera-width", "value"),
        State("camera-height", "value"),
        State("camera-fps", "value"),
    )
    def _toggle_camera(
        checked: bool,
        width: Optional[float],
        height: Optional[float],
        fps: Optional[float],
    ) -> Tuple[str, str, str, bool]:
        if checked:
            try:
                status = start_camera_stream(width, height, fps)
                return status, "Camera: live", "teal", True
            except Exception as exc:
                stop_camera_stream()
                return f"Camera error: {exc}", "Camera: error", "red", False
        stop_camera_stream()
        return "Camera stream stopped.", "Camera: off", "gray", False

    @app.callback(
        Output("robot-status-text", "children"),
        Output("header-robot-badge", "children"),
        Output("header-robot-badge", "color"),
        Input("robot-test", "n_clicks"),
        State("robot-uri", "value"),
        prevent_initial_call=True,
    )
    def _test_robot(_clicks: int, uri: str) -> Tuple[str, str, str]:
        try:
            response = asyncio.run(_probe_robot(uri))
            status = f"Robot connected. State: {response.get('state', 'ok')}"
            return status, "Robot: online", "teal"
        except Exception as exc:
            return f"Robot connection failed: {exc}", "Robot: error", "red"

    @app.callback(
        Output("camera-calibration-action", "children"),
        Input("capture-frame", "n_clicks"),
        Input("capture-depth", "n_clicks"),
        Input("compute-calibration", "n_clicks"),
        Input("load-calibration", "n_clicks"),
        State("calibration-output-dir", "value"),
        State("calibration-board-path", "value"),
        State("calibration-bundle-path", "value"),
        prevent_initial_call=True,
    )
    def _handle_camera_calibration_actions(
        _capture_frame: int,
        _capture_depth: int,
        _compute: int,
        _load: int,
        output_dir: str,
        board_path: str,
        bundle_path: str,
    ) -> str:
        trigger = callback_context.triggered_id
        output_path = _safe_path(output_dir, DEFAULT_CAPTURE_DIR)
        board_path = _safe_path(board_path, DEFAULT_BOARD_PATH)
        bundle_path = _safe_path(bundle_path, DEFAULT_BUNDLE_PATH)
        try:
            if trigger == "capture-frame":
                captured = capture_color_frame(output_path)
                return f"Captured {captured.name}."
            if trigger == "capture-depth":
                captured = capture_depth_snapshot(output_path)
                return f"Saved depth snapshot to {captured.name}."
            if trigger == "compute-calibration":
                return compute_calibration_bundle(board_path, bundle_path, output_path)
            if trigger == "load-calibration":
                return load_calibration_bundle(bundle_path)
        except Exception as exc:
            return f"Calibration action failed: {exc}"
        return "Awaiting calibration action."

    @app.callback(
        Output("workspace-action-status", "children"),
        Input("compute-workspace", "n_clicks"),
        State("marker-id-a", "value"),
        State("marker-id-b", "value"),
        State("marker-size", "value"),
        prevent_initial_call=True,
    )
    def _handle_workspace_action(
        _clicks: int,
        marker_a: Optional[float],
        marker_b: Optional[float],
        marker_size: Optional[float],
    ) -> str:
        marker_ids = (
            _safe_int(marker_a, SESSION.config.marker_ids[0]),
            _safe_int(marker_b, SESSION.config.marker_ids[1]),
        )
        marker_size_mm = _safe_float(marker_size, SESSION.config.marker_size_mm)
        try:
            ws = compute_workspace_from_latest(marker_ids, marker_size_mm)
            return f"Workspace locked: {ws.width_mm:.1f} x {ws.height_mm:.1f} mm."
        except Exception as exc:
            return f"Workspace detection failed: {exc}"

    @app.callback(
        Output("plate-action-status", "children"),
        Input("compute-plate", "n_clicks"),
        Input("attach-plate", "n_clicks"),
        State("plate-layout-path", "value"),
        State("calibration-bundle-path", "value"),
        prevent_initial_call=True,
    )
    def _handle_plate_action(
        _compute_clicks: int,
        _attach_clicks: int,
        layout_path: str,
        bundle_path: str,
    ) -> str:
        trigger = callback_context.triggered_id
        layout_path = _safe_path(layout_path, DEFAULT_PLATE_PATH)
        bundle_path = _safe_path(bundle_path, DEFAULT_BUNDLE_PATH)
        try:
            if trigger == "compute-plate":
                result = compute_plate_calibration_from_latest(layout_path)
                return f"Plate detected. Error {result.reprojection_error_mm:.2f} mm."
            if trigger == "attach-plate":
                return attach_plate_calibration(bundle_path, layout_path)
        except Exception as exc:
            return f"Plate calibration failed: {exc}"
        return "Awaiting plate action."

    @app.callback(
        Output("camera-feed", "src"),
        Output("overlay-status", "children"),
        Output("camera-calibration-metrics", "children"),
        Output("workspace-metrics", "children"),
        Output("plate-metrics", "children"),
        Input("camera-interval", "n_intervals"),
        State("calibration-stepper", "active"),
        State("calibration-board-path", "value"),
        State("marker-id-a", "value"),
        State("marker-id-b", "value"),
        State("marker-size", "value"),
        State("plate-layout-path", "value"),
    )
    def _update_camera_view(
        _ticks: int,
        active_step: Optional[int],
        board_path: str,
        marker_a: Optional[float],
        marker_b: Optional[float],
        marker_size: Optional[float],
        plate_layout: str,
    ) -> Tuple[str, str, str, str, str]:
        frame, _depth = get_latest_frame()
        if frame is None:
            placeholder = _placeholder_frame("Camera stream offline")
            return (
                _encode_frame(placeholder),
                "Camera stream is stopped.",
                "ChArUco corners: --",
                "Workspace: --",
                "Plate error: --",
            )

        step = active_step or 0
        board_path = _safe_path(board_path, DEFAULT_BOARD_PATH)
        plate_path = _safe_path(plate_layout, DEFAULT_PLATE_PATH)
        marker_ids = (
            _safe_int(marker_a, SESSION.config.marker_ids[0]),
            _safe_int(marker_b, SESSION.config.marker_ids[1]),
        )
        marker_size_mm = _safe_float(marker_size, SESSION.config.marker_size_mm)

        if step == 0:
            vis, status, metric = draw_charuco_overlay(frame, board_path)
            return _encode_frame(vis), status, metric, "Workspace: --", "Plate error: --"
        if step == 1:
            vis, status, metric = draw_workspace_overlay(frame, marker_ids, marker_size_mm)
            return _encode_frame(vis), status, "ChArUco corners: --", metric, "Plate error: --"
        vis, status, metric = draw_plate_overlay(frame, plate_path)
        return _encode_frame(vis), status, "ChArUco corners: --", "Workspace: --", metric


def create_app() -> Dash:
    app = Dash(__name__)
    app.title = "Codroid Calibration Console"
    app.layout = build_layout()
    register_callbacks(app)
    return app


app = create_app()
server = app.server


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    app.run(debug=True)
