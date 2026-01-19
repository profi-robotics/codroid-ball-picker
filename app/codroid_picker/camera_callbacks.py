"""Camera-related Dash callbacks for the calibration UI.

This module handles:
- Camera device scanning and connection
- Camera stream management and configuration
- Camera calibration workflow
- Live camera view with overlays
- Calibration progress tracking
"""

import logging
import time
from pathlib import Path
from typing import Any, Optional, Tuple

import cv2
import dash_mantine_components as dmc
from dash import callback_context, Dash, html, Input, Output, State

from app.codroid_picker.session import CALIBRATION_PROGRESS, SESSION
from app.codroid_picker.streaming import OverlayResult, OverlayTask, start_camera_stream, stop_camera_stream, MJPEG_STREAM, FRAME_PROCESSOR
from app.codroid_picker.utils import safe_float, safe_int, safe_path
from app.settings import (DEFAULT_BOARD_PATH, DEFAULT_BUNDLE_PATH, DEFAULT_CAPTURE_DIR, DEFAULT_PLATE_PATH, SETTINGS)
from vision.camera_check import list_oak_devices

# Keep underscore-prefixed aliases for backward compatibility with external imports
_safe_int = safe_int
_safe_float = safe_float
_safe_path = safe_path


def _format_fps(capture_fps: float, stream_fps: float) -> str:
    """Format FPS display text."""
    return f"Camera: {capture_fps:.1f} FPS · Stream: {stream_fps:.1f} FPS"


def _format_device_list(devices: list[object]) -> html.Ul:
    """Format device list for display."""
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


def _clear_calibration_outputs(output_path: Path) -> None:
    """Clear calibration output files."""
    if output_path.exists():
        for file in output_path.glob("*"):
            if file.is_file():
                file.unlink()


def _charuco_detection_counts(frame_paths: list[Path], board_path: Path) -> Tuple[list[int], int]:
    """Count ChArUco corners detected in each frame."""
    # Use the Charuco board spec loader and OpenCV CharucoDetector
    from vision.aruco_board import load_charuco_board_spec

    board_spec = load_charuco_board_spec(board_path)
    board = board_spec.create_board()

    # Detector params - allow inverted markers (white on black)
    detector_params = cv2.aruco.DetectorParameters()
    detector_params.detectInvertedMarker = True
    charuco_params = cv2.aruco.CharucoParameters()
    charuco_detector = cv2.aruco.CharucoDetector(board, charuco_params, detector_params)

    detection_counts: list[int] = []
    usable = 0

    for frame_path in frame_paths:
        img = cv2.imread(str(frame_path))
        if img is None:
            detection_counts.append(0)
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        try:
            charuco_corners, charuco_ids, _marker_corners, _marker_ids = charuco_detector.detectBoard(gray)
        except Exception:
            # Fall back to zero if detection fails on this image
            charuco_corners = None

        count = len(charuco_corners) if charuco_corners is not None else 0
        detection_counts.append(count)
        if count >= 16:
            usable += 1

    return detection_counts, usable


def _refresh_plate_status(frame: Any, plate_path: Path) -> None:
    """Refresh plate tracking status from frame."""
    from app.codroid_picker import state
    from vision.plate import track_plate

    try:
        result = track_plate(frame, plate_path)
        state.update_plate_status(
            count=result.marker_count,
            ok=result.all_detected,
            stable=result.is_stable,
            center_px=result.center_px,
            movement_px=result.movement_px,
            message=result.status_message,
        )
    except Exception as exc:
        state.update_plate_status(
            count=0,
            ok=False,
            stable=False,
            center_px=None,
            movement_px=None,
            message=f"Plate tracking error: {exc}",
        )


def _format_detection_metrics() -> str:
    """Format live detection metrics from SESSION.detections."""
    import app.codroid_picker as codroid_picker_module

    SESSION = codroid_picker_module.SESSION

    if not SESSION.model_loaded:
        return "Model not loaded"

    detections = SESSION.detections or []
    metrics = f"Live: {len(detections)} object(s)"

    if SESSION.last_detection_error:
        metrics = f"Error: {SESSION.last_detection_error}"

    return metrics


def register_camera_callbacks(app: Dash) -> None:
    """Register all camera-related callbacks.

    Uses lazy import from parent codroid_picker module to access helper
    functions and globals. Import happens inside callback functions to avoid
    circular dependency.
    """
    # Guard against duplicate registration
    if hasattr(app, '_camera_callbacks_registered'):
        return
    app._camera_callbacks_registered = True

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
        Output("camera-connection-state", "data"),
        Output("camera-interval", "disabled"),
        Input("camera-enable", "checked"),
        Input("auto-init", "n_intervals"),
        Input("camera-lock-af", "checked"),
        State("camera-width", "value"),
        State("camera-height", "value"),
        State("camera-fps-input", "value"),
    )
    def _toggle_camera(
        checked: bool,
        _init: int,
        lock_af: Optional[bool],
        width: Optional[float],
        height: Optional[float],
        fps: Optional[float],
    ) -> Tuple[str, str, str, bool, dict, bool]:
        trigger = callback_context.triggered_id
        desired_checked = checked
        stream_running = SESSION.camera_stream is not None and SESSION.camera_stream.is_running()

        if trigger == "camera-lock-af" and stream_running:
            try:
                assert SESSION.camera_stream is not None
                SESSION.camera_stream.set_focus_lock(bool(lock_af))
                state = "locked" if lock_af else "unlocked"
                status = f"Camera stream running; focus {state}."
                return (
                    status,
                    "Camera: live",
                    "brand",
                    True,
                    {"status": "live", "message": status},
                    False,
                )
            except Exception as exc:  # noqa: BLE001
                error = f"Camera focus control error: {exc}"
                return (
                    error,
                    "Camera: error",
                    "red",
                    True,
                    {"status": "error", "message": error},
                    False,
                )

        if trigger == "auto-init":
            desired_checked = True
        if desired_checked:
            try:
                status = start_camera_stream(width, height, fps, bool(lock_af))
                return (
                    status,
                    "Camera: live",
                    "brand",
                    True,
                    {"status": "live", "message": status},
                    False,
                )
            except RuntimeError as exc:
                # Handle specific case of no devices
                if "No available devices" in str(exc):
                    error = "No OAK-D camera detected. Ensure device is connected and powered on."
                else:
                    error = f"Camera error: {exc}"
                stop_camera_stream()
                return (
                    error,
                    "Camera: unavailable",
                    "orange",
                    False,
                    {"status": "unavailable", "message": error},
                    True,
                )
            except Exception as exc:
                stop_camera_stream()
                error = f"Camera error: {exc}"
                return (
                    error,
                    "Camera: error",
                    "yellow",
                    False,
                    {"status": "error", "message": error},
                    True,
                )
        stop_camera_stream()
        stopped = "Camera stream stopped."
        return (
            stopped,
            "Camera: off",
            "gray",
            False,
            {"status": "off", "message": stopped},
            True,
        )

    @app.callback(
        Output("camera-feed-container", "children"),
        Input("camera-connection-state", "data"),
    )
    def _update_camera_placeholder(camera_state: Optional[dict]) -> html.Div:
        """Show placeholder when camera is not connected, otherwise show video feed."""
        status = (camera_state or {}).get("status", "off")

        if status == "live":
            # Camera is connected - show the video feed
            return html.Img(
                id="camera-feed",
                src="/video_feed",
                style={
                    "width": "100%",
                    "height": "auto",
                    "maxHeight": "100%",
                    "objectFit": "contain",
                    "borderRadius": "16px",
                    "border": "1px solid rgba(240, 65, 0, 0.25)",
                    "marginTop": "8px",
                    "backgroundColor": "#0b0f14",
                },
            )
        else:
            # Camera not connected - show placeholder
            message = (camera_state or {}).get("message", "Camera not connected")
            icon = "📷" if status == "off" else "⚠️"
            color = "#666" if status == "off" else "#ff8c00"

            return html.Div(
                [
                    html.Div(
                        [
                            html.Div(icon, style={"fontSize": "64px", "marginBottom": "16px"}),
                            html.Div(
                                "Camera Not Connected" if status == "off" else "Camera Unavailable",
                                style={
                                    "fontSize": "20px",
                                    "fontWeight": 600,
                                    "color": color,
                                    "marginBottom": "8px",
                                }
                            ),
                            html.Div(
                                message,
                                style={
                                    "fontSize": "14px",
                                    "color": "#888",
                                    "textAlign": "center",
                                    "maxWidth": "400px",
                                }
                            ),
                        ],
                        style={
                            "display": "flex",
                            "flexDirection": "column",
                            "alignItems": "center",
                            "justifyContent": "center",
                            "height": "100%",
                            "minHeight": "400px",
                        }
                    )
                ],
                id="camera-feed",
                style={
                    "width": "100%",
                    "height": "auto",
                    "borderRadius": "16px",
                    "border": "1px solid rgba(240, 65, 0, 0.25)",
                    "marginTop": "8px",
                    "backgroundColor": "#0b0f14",
                    "display": "flex",
                    "alignItems": "center",
                    "justifyContent": "center",
                },
            )

    @app.callback(
        Output("camera-calibration-action", "children"),
        Output("calibration-progress-store", "data"),
        Output("calibration-stepper", "active", allow_duplicate=True),
        Input("calibrate-camera", "n_clicks"),
        Input("use-existing-calibration", "n_clicks"),
        State("settings-calib-path", "value"),
        State("calibration-stepper", "active"),
        prevent_initial_call=True,
    )
    def _handle_camera_calibration_actions(
        _calibrate: int,
        _use_existing: int,
        bundle_path: str,
        current_step: Optional[int] = 0,
    ) -> Tuple[dmc.Alert, dict, int]:
        # Import parent module to access helper functions
        import app.codroid_picker as codroid_picker_module

        trigger = callback_context.triggered_id

        output_path = DEFAULT_CAPTURE_DIR
        board_path = DEFAULT_BOARD_PATH
        bundle_path = _safe_path(bundle_path, DEFAULT_BUNDLE_PATH)

        if trigger == "use-existing-calibration":
            try:
                result = codroid_picker_module.load_calibration_bundle(
                    bundle_path)
                alert = dmc.Alert(
                    result,
                    icon="✓",
                    title="Using saved calibration",
                    color="green",
                )
                # Advance to robot calibration step
                return alert, {"progress": 0}, 1
            except Exception as exc:
                logging.error(
                    f"[CALIBRATE] Error loading saved calibration: {exc}", exc_info=True)
                alert = dmc.Alert(
                    str(exc)[:200],
                    icon="!",
                    title="Load error",
                    color="red",
                )
                return alert, {"progress": 0}, (current_step or 0)

        if trigger == "calibrate-camera":
            try:
                # Clear prior calibration artifacts on disk to avoid mixing old frames
                _clear_calibration_outputs(output_path)

                # Clear previous captures
                SESSION.capture.frames = []
                SESSION.capture.depth_frame = None

                # Capture frames for more robust calibration
                num_frames = max(1, SETTINGS.camera_calibration_frame_count)
                captured_frames = []
                CALIBRATION_PROGRESS["max_frames"] = num_frames
                CALIBRATION_PROGRESS["value"] = 1  # Start progress
                for i in range(num_frames):
                    captured = codroid_picker_module.capture_color_frame(
                        output_path)
                    captured_frames.append(captured.name)
                    # Update progress (1-30 frames = 3-98%)
                    frame_num = i + 1
                    CALIBRATION_PROGRESS["current_frame"] = frame_num
                    CALIBRATION_PROGRESS["value"] = int(
                        (frame_num / num_frames) * 95) + 3
                    # Small delay between captures to avoid motion blur
                    if (
                        i < num_frames - 1
                        and SETTINGS.camera_calibration_frame_delay_s > 0.0
                    ):
                        time.sleep(SETTINGS.camera_calibration_frame_delay_s)

                # Capture depth snapshot
                depth_path = codroid_picker_module.capture_depth_snapshot(
                    output_path)

                # Pre-flight check: ensure the saved frames actually contain ChArUco corners
                detection_counts, usable_frames = _charuco_detection_counts(
                    codroid_picker_module._list_capture_frames(
                        output_path), board_path
                )
                if usable_frames < SETTINGS.camera_calibration_min_usable_frames:
                    raise RuntimeError(
                        "No usable ChArUco detections in captured frames. "
                        f"Corner counts: {detection_counts}. "
                        "Ensure the board is visible, sharp, and well-lit."
                    )

                # Compute calibration bundle
                result = codroid_picker_module.compute_calibration_bundle(
                    board_path, bundle_path, output_path)

                alert = dmc.Alert(
                    children=[
                        dmc.Stack(
                            [
                                dmc.Text(
                                    f"✓ Captured {len(captured_frames)} RGB frames", size="sm"
                                ),
                                dmc.Text(
                                    "✓ Captured 1 depth snapshot", size="sm"),
                                dmc.Text(
                                    f"✓ Detected {usable_frames}/{num_frames} usable frames", size="sm"),
                                dmc.Text(result, size="sm", fw=600, c="teal"),
                            ],
                            gap="xs",
                        ),
                    ],
                    icon="✓",
                    title="Calibration complete",
                    color="green",
                )
                # Reset progress tracker
                CALIBRATION_PROGRESS["value"] = 0
                CALIBRATION_PROGRESS["current_frame"] = 0
                # Advance to robot calibration step
                return alert, {"progress": 0}, 1
            except Exception as exc:
                logging.error(
                    f"[CALIBRATE] Error during calibration: {exc}", exc_info=True)
                alert = dmc.Alert(
                    str(exc)[:200],
                    icon="!",
                    title="Calibration error",
                    color="red",
                )
                # Reset progress tracker on error
                CALIBRATION_PROGRESS["value"] = 0
                CALIBRATION_PROGRESS["current_frame"] = 0
                return alert, {"progress": 0}, (current_step or 0)

        return dmc.Alert("", icon="•", title="Ready", color="gray"), {"progress": 0}, (current_step or 0)

    @app.callback(
        Output("overlay-status", "children", allow_duplicate=True),
        Output("camera-fps", "children", allow_duplicate=True),
        Output("camera-calibration-metrics", "children", allow_duplicate=True),
        Output("workspace-metrics", "children", allow_duplicate=True),
        Output("plate-metrics", "children", allow_duplicate=True),
        Output("detection-metrics", "children", allow_duplicate=True),
        Output("plate-detection-state", "data", allow_duplicate=True),
        Output("calibrate-camera", "disabled", allow_duplicate=True),
        Output("markers-detected-icon", "color", allow_duplicate=True),
        Input("camera-interval", "n_intervals"),
        State("main-tabs", "value"),
        State("calibration-stepper", "active"),
        State("application-tabs", "value"),
        State("settings-marker-a", "value"),
        State("settings-marker-b", "value"),
        State("settings-marker-size", "value"),
        State("overlay-mode-select", "value"),
        prevent_initial_call=True,
    )
    def _update_camera_view(
        _ticks: int,
        active_tab: Optional[str],
        calibration_step: Optional[int],
        application_tab: Optional[str],
        marker_a: Optional[float],
        marker_b: Optional[float],
        marker_size: Optional[float],
        overlay_mode: Optional[str],
    ) -> Tuple[str, str, str, str, str, str, dict, bool, str]:
        # Import parent module to access helper functions and globals
        import app.codroid_picker as codroid_picker_module

        frame, _depth = codroid_picker_module.get_latest_frame()

        def _camera_calibration_state(
            metrics_text: str, step: int
        ) -> Tuple[bool, str]:
            if step != 0:
                return True, "gray"
            if not metrics_text or "ChArUco corners:" not in metrics_text:
                return True, "gray"
            try:
                count = int(metrics_text.split(":")[-1].strip().split()[0])
                is_disabled = count < 16
                return is_disabled, "green" if not is_disabled else "red"
            except (ValueError, IndexError):
                return True, "gray"

        plate_state = {
            "count": 0,
            "ok": False,
            "stable": False,
            "center_px": None,
            "movement_px": None,
            "timestamp": time.time(),
            "message": "Waiting for plate markers",
        }
        capture_fps = SESSION.camera_stream.get_fps() if SESSION.camera_stream else 0.0
        overlay_fps_text = _format_fps(capture_fps, MJPEG_STREAM.get_fps())

        if frame is None:
            # Don't add placeholder to MJPEG stream - let video element show its poster image
            # Only log occasionally to reduce noise (every 300 intervals = ~30 seconds at default interval)
            if _ticks % 300 == 0:
                is_running = SESSION.camera_stream and SESSION.camera_stream.is_running()
                if is_running:
                    # Only log if stream claims to be running but we're not getting frames
                    logging.warning(
                        "No frame from camera despite stream running: %s", is_running)
            return (
                "Camera stream is stopped.",
                overlay_fps_text,
                "ChArUco corners: --",
                "Workspace: --",
                "Plate error: --",
                "Detections: --",
                plate_state,
                True,
                "gray",
            )

        # Check if this is a new frame by comparing object identity
        current_frame_id = id(frame)
        is_new_frame = current_frame_id != SESSION.last_frame_id
        if is_new_frame:
            SESSION.last_frame_id = current_frame_id

        # Frame received, dispatch to background processor for heavy overlays.
        tab = active_tab or "applications"
        calibration_step = calibration_step or 0
        application_tab = application_tab or "workspace"
        board_path = DEFAULT_BOARD_PATH
        plate_path = DEFAULT_PLATE_PATH
        marker_ids = (
            _safe_int(marker_a, SESSION.marker_ids[0]),
            _safe_int(marker_b, SESSION.marker_ids[1]),
        )
        marker_size_mm = _safe_float(
            marker_size, SESSION.marker_size_mm)

        requested_overlay = overlay_mode or "auto"
        task = codroid_picker_module._select_overlay_task(
            tab,
            calibration_step,
            application_tab,
            board_path,
            marker_ids,
            marker_size_mm,
            plate_path,
        )
        if requested_overlay != "auto":
            if requested_overlay == "calibration":
                task = OverlayTask("manual-calibration",
                                   codroid_picker_module._run_charuco_task(board_path))
            elif requested_overlay == "workspace":
                task = OverlayTask(
                    "manual-workspace",
                    codroid_picker_module._run_workspace_task(
                        marker_ids, marker_size_mm),
                )
            elif requested_overlay == "plate":
                task = OverlayTask(
                    "manual-plate", codroid_picker_module._run_plate_task(plate_path))
            elif requested_overlay == "detection":
                task = OverlayTask("manual-detection",
                                   codroid_picker_module._run_detection_task())
            elif requested_overlay == "none":
                task = None
        overlay: Optional[OverlayResult] = None
        if task is not None:
            FRAME_PROCESSOR.submit(task, frame)
            overlay = FRAME_PROCESSOR.get_latest(task.key)
            if calibration_step == 1 and task.key != "calibration-plate":
                _refresh_plate_status(frame, plate_path)
        elif calibration_step == 1:
            _refresh_plate_status(frame, plate_path)

        vis = overlay.vis if overlay else frame
        status_text = overlay.status if overlay else "No overlay applied."
        metric_text = overlay.metric if overlay else "--"
        mode_label = {
            "auto": "Overlay: Auto",
            "calibration": "Overlay: Calibration",
            "workspace": "Overlay: Workspace",
            "plate": "Overlay: Plate",
            "detection": "Overlay: Detections",
            "none": "Overlay: Off",
        }.get(requested_overlay, "Overlay: Auto")
        status_text = f"{mode_label} · {status_text}"

        # Get plate status from state module
        from app.codroid_picker import state
        plate_state = state.get_plate_status_snapshot()

        if tab == "calibration":
            if calibration_step == 0:
                if is_new_frame:
                    MJPEG_STREAM.add_frame(vis)
                overlay_fps_text = _format_fps(
                    capture_fps, MJPEG_STREAM.get_fps())
                calibrate_disabled, marker_color = _camera_calibration_state(
                    metric_text, calibration_step
                )
                return (
                    status_text,
                    overlay_fps_text,
                    metric_text,
                    "Workspace: --",
                    "Plate error: --",
                    "Detections: --",
                    plate_state,
                    calibrate_disabled,
                    marker_color,
                )
            if is_new_frame:
                MJPEG_STREAM.add_frame(vis)
            overlay_fps_text = _format_fps(
                capture_fps, MJPEG_STREAM.get_fps())
            metrics_text = "ChArUco corners: --"
            calibrate_disabled, marker_color = _camera_calibration_state(
                metrics_text, calibration_step
            )
            return (
                status_text,
                overlay_fps_text,
                metrics_text,
                "Workspace: --",
                metric_text,
                "Detections: --",
                plate_state,
                calibrate_disabled,
                marker_color,
            )

        if tab == "applications":
            if is_new_frame:
                MJPEG_STREAM.add_frame(vis)
            overlay_fps_text = _format_fps(
                capture_fps, MJPEG_STREAM.get_fps())
            metrics_text = "ChArUco corners: --"
            calibrate_disabled, marker_color = _camera_calibration_state(
                metrics_text, calibration_step
            )
            return (
                status_text,
                overlay_fps_text,
                metrics_text,
                metric_text if application_tab == "workspace" else "Workspace: --",
                "Plate error: --",
                _format_detection_metrics() if application_tab == "pick-place" else "Detections: --",
                plate_state,
                calibrate_disabled,
                marker_color,
            )

        # Monitoring or any other tab: keep stream alive and surface metrics.
        if is_new_frame:
            MJPEG_STREAM.add_frame(vis)
        overlay_fps_text = _format_fps(capture_fps, MJPEG_STREAM.get_fps())
        metrics_text = "ChArUco corners: --"
        calibrate_disabled, marker_color = _camera_calibration_state(
            metrics_text, calibration_step
        )
        return (
            status_text,
            overlay_fps_text,
            metrics_text,
            "Workspace: --",
            "Plate error: --",
            "Detections: --",
            plate_state,
            calibrate_disabled,
            marker_color,
        )

    @app.callback(
        Output("calibration-progress", "value"),
        Output("calibration-progress", "style"),
        Output("calibration-progress-text", "children"),
        Output("calibration-progress-text", "style"),
        Input("calibration-progress-interval", "n_intervals"),
    )
    def _update_calibration_progress(_):
        """Update progress bar and text from global calibration progress tracker."""
        progress = CALIBRATION_PROGRESS.get("value", 0)
        current_frame = CALIBRATION_PROGRESS.get("current_frame", 0)
        max_frames = CALIBRATION_PROGRESS.get("max_frames", 0)

        # Show progress bar and text only if capturing (1-99%)
        if progress > 0 and progress < 100:
            text = f"Capturing images: {current_frame}/{max_frames}"
            return progress, {"display": "block"}, text, {"display": "block", "marginBottom": "8px"}

        return 0, {"display": "none"}, "", {"display": "none"}
