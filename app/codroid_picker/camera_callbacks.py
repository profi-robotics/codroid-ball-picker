"""Camera-related Dash callbacks for the calibration UI.

This module handles:
- Camera device scanning and connection
- Camera stream management and configuration
- Camera calibration workflow
- Live camera view with overlays
- Calibration progress tracking
"""

import logging
import threading
import time
from pathlib import Path
from typing import Any, Optional, Tuple

import cv2
import dash_mantine_components as dmc
from dash import callback_context, Dash, html, Input, Output, State, no_update
from app.codroid_picker.i18n import t

from app.codroid_picker.session import CALIBRATION_PROGRESS, SESSION
from app.codroid_picker.streaming import (
    AUTOFOCUS_SETTLE_S,
    OverlayResult,
    OverlayTask,
    FRAME_PROCESSOR,
    MJPEG_STREAM,
    RAW_STREAM,
    set_active_overlay_task,
    start_camera_stream,
    stop_camera_stream,
)
from app.codroid_picker.utils import safe_float, safe_int, safe_path
from app.settings import (DEFAULT_BOARD_PATH, DEFAULT_BUNDLE_PATH,
                          DEFAULT_CAPTURE_DIR, DEFAULT_PLATE_PATH, SETTINGS)
from vision.camera_check import list_oak_devices

# Keep underscore-prefixed aliases for backward compatibility with external imports
_safe_int = safe_int
_safe_float = safe_float
_safe_path = safe_path

CALIBRATION_ACTION_MOUNT_GUARD_S = 0.75


def _calibration_tab_action_guard_active(mounted_at: Optional[dict]) -> bool:
    """Return true while newly mounted calibration controls should ignore clicks."""
    if not mounted_at:
        return False
    try:
        age_s = time.monotonic() - float(mounted_at.get("monotonic_s", 0.0))
    except (TypeError, ValueError):
        return False
    return age_s < CALIBRATION_ACTION_MOUNT_GUARD_S

_CALIBRATION_ACTION_LOCK = threading.Lock()
_PLATE_STATUS_REFRESH_INTERVAL_S = 0.5
_plate_status_last_refresh_ts = 0.0


def _format_fps(capture_fps: float, stream_fps: float) -> str:
    """Format FPS display text."""
    return t("Camera: {capture_fps:.1f} FPS · Stream: {stream_fps:.1f} FPS", capture_fps=capture_fps, stream_fps=stream_fps)


def _format_device_list(devices: list[object]) -> html.Ul:
    """Format device list for display."""
    if not devices:
        return html.Ul([html.Li(t("No OAK-D devices detected."))])
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
    charuco_detector = cv2.aruco.CharucoDetector(
        board, charuco_params, detector_params)

    detection_counts: list[int] = []
    usable = 0

    for frame_path in frame_paths:
        img = cv2.imread(str(frame_path))
        if img is None:
            detection_counts.append(0)
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        try:
            charuco_corners, charuco_ids, _marker_corners, _marker_ids = charuco_detector.detectBoard(
                gray)
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
    from vision.plate import load_plate_layout, estimate_plate_calibration
    from app.codroid_picker.overlays.calibration import _evaluate_plate_status

    try:
        plate_layout = load_plate_layout(plate_path)
        calibration_result = estimate_plate_calibration(frame, plate_layout)
        status_data = _evaluate_plate_status(calibration_result)
        count = status_data.get("count", 0)
        if status_data.get("is_ok"):
            if status_data.get("movement_px") is not None:
                message = t("Plate markers: {count} (movement: {movement:.1f}px)", count=count, movement=status_data["movement_px"])
            else:
                message = t("Plate markers: {count}", count=count)
        else:
            message = t("Insufficient plate markers: {count}/{required}", count=count, required=status_data.get("min_required", "?"))
        state.update_plate_status(
            count=count,
            ok=status_data.get("is_ok", False),
            stable=status_data.get("stable", False),
            center_px=tuple(status_data["current_center"]) if status_data.get(
                "current_center") is not None else None,
            movement_px=status_data.get("movement_px"),
            message=message,
        )
    except Exception as exc:
        state.update_plate_status(
            count=0,
            ok=False,
            stable=False,
            center_px=None,
            movement_px=None,
            message=t("Plate tracking error: {error}", error=exc),
        )


def _format_detection_metrics() -> str:
    """Format live detection metrics from SESSION.detections."""
    import app.codroid_picker as codroid_picker_module

    SESSION = codroid_picker_module.SESSION

    if not SESSION.model_loaded:
        return t("Model not loaded")

    detections = SESSION.detections or []
    metrics = t("Live: {count} object(s)", count=len(detections))

    if SESSION.last_detection_error:
        metrics = t("Error: {error}", error=SESSION.last_detection_error)

    return metrics


def _message_params(payload: Optional[dict]) -> dict:
    """Return safe localization params from payload."""
    params = (payload or {}).get("message_params")
    return params if isinstance(params, dict) else {}


def _resolve_camera_message(camera_state: Optional[dict]) -> str:
    """Resolve camera status message in the current locale."""
    payload = camera_state or {}

    message_key = payload.get("message_key")
    if isinstance(message_key, str) and message_key:
        return t(message_key, **_message_params(payload))

    session_key = SESSION.camera_last_error_key
    if isinstance(session_key, str) and session_key:
        session_params = SESSION.camera_last_error_params
        if not isinstance(session_params, dict):
            session_params = {}
        return t(session_key, **session_params)

    raw = payload.get("message")
    if raw:
        return str(raw)
    if SESSION.camera_last_error_message:
        return SESSION.camera_last_error_message
    return t("Camera not connected.")


def _auto_program_running() -> bool:
    """Return True when Auto Pick & Place is actively running/paused."""
    try:
        from app.codroid_picker.programs import get_registry

        active_program = get_registry().active_program
        if active_program is None:
            return False
        return active_program.name == "auto_pick_place" and active_program.is_running
    except Exception:  # noqa: BLE001
        return False


def _resolve_overlay_task(
    *,
    requested_overlay: str,
    tab: str,
    calibration_step: int,
    application_tab: str,
    board_path: Path,
    marker_ids: tuple[int, int],
    marker_size_mm: float,
    plate_path: Path,
) -> tuple[Optional[OverlayTask], str]:
    """Resolve overlay task and effective mode."""
    import app.codroid_picker as codroid_picker_module

    requested = (requested_overlay or "auto").strip().lower()
    task = codroid_picker_module._select_overlay_task(
        tab,
        calibration_step,
        application_tab,
        board_path,
        marker_ids,
        marker_size_mm,
        plate_path,
    )
    effective_mode = requested

    if requested == "auto":
        if (
            tab == "applications"
            and application_tab == "pick-place"
            and _auto_program_running()
        ):
            task = OverlayTask("auto-plan", codroid_picker_module._run_auto_plan_task())
            effective_mode = "auto-plan"
        return task, effective_mode

    if requested == "calibration":
        task = OverlayTask("manual-calibration", codroid_picker_module._run_charuco_task(board_path))
    elif requested == "workspace":
        task = OverlayTask(
            "manual-workspace",
            codroid_picker_module._run_workspace_task(marker_ids, marker_size_mm),
        )
    elif requested == "plate":
        task = OverlayTask("manual-plate", codroid_picker_module._run_plate_task(plate_path))
    elif requested == "detection":
        task = OverlayTask("manual-detection", codroid_picker_module._run_detection_task())
    elif requested == "auto-plan":
        task = OverlayTask("manual-auto-plan", codroid_picker_module._run_auto_plan_task())
        effective_mode = "auto-plan"
    elif requested == "none":
        task = None
    else:
        effective_mode = "auto"

    return task, effective_mode


def _capture_depth_snapshot_for_calibration(
    codroid_picker_module: Any,
    output_path: Path,
) -> tuple[Path, bool]:
    """Capture calibration depth and return whether stream pause was required."""
    stream = SESSION.camera_stream
    if stream is None or not stream.is_running():
        raise RuntimeError(t("Camera stream stopped before depth capture."))

    config = getattr(stream, "config", None)
    enable_depth = bool(getattr(config, "enable_depth", False))
    resolution = getattr(config, "resolution", SETTINGS.camera_resolution)
    fps = float(getattr(config, "fps", SETTINGS.camera_fps))
    lock_af = bool(getattr(config, "lock_af", False))

    if enable_depth:
        try:
            depth_path = codroid_picker_module.capture_depth_snapshot(output_path)
            return depth_path, False
        except RuntimeError as exc:
            if "No depth frame available" not in str(exc):
                raise
            logging.warning(
                "[CALIBRATE] Live depth capture timed out; "
                "retrying with isolated depth pipeline."
            )

    CALIBRATION_PROGRESS["value"] = max(
        int(CALIBRATION_PROGRESS.get("value", 0) or 0),
        98,
    )
    CALIBRATION_PROGRESS["message"] = (
        "Camera will restart to capture depth. "
        "Please wait and do not refresh the browser."
    )
    stop_camera_stream(wait_for_release=True)
    depth_path = codroid_picker_module.capture_depth_snapshot_isolated(
        output_path,
        resolution,
        fps,
        lock_af=lock_af,
    )
    CALIBRATION_PROGRESS["message"] = (
        "Camera restarted. Finishing calibration..."
    )
    try:
        status = start_camera_stream(
            resolution[0],
            resolution[1],
            fps,
            lock_af=lock_af,
            enable_depth_override=enable_depth,
        )
        logging.info(
            "[CALIBRATE] Restarted camera stream after isolated depth capture: %s",
            status,
        )
        return depth_path, False
    except Exception as exc:  # noqa: BLE001
        logging.error(
            "[CALIBRATE] Unable to restart camera stream after isolated "
            "depth capture: %s",
            exc,
            exc_info=True,
        )
        return depth_path, True


def _refresh_plate_status_throttled(frame, plate_path: Path) -> None:
    """Refresh plate status at a bounded cadence from Dash callbacks."""
    global _plate_status_last_refresh_ts
    now = time.monotonic()
    if now - _plate_status_last_refresh_ts < _PLATE_STATUS_REFRESH_INTERVAL_S:
        return
    _plate_status_last_refresh_ts = now
    _refresh_plate_status(frame, plate_path)


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
    min_live_charuco_corners = 8

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
        Output("camera-enable", "checked"),
        Output("camera-connection-state", "data"),
        Output("metrics-interval", "disabled"),
        Input("camera-enable", "checked"),
        Input("hardware-init", "n_intervals"),
        Input("monitor-interval", "n_intervals"),
        Input("camera-lock-af", "checked"),
        State("camera-connection-state", "data"),
        State("camera-width", "value"),
        State("camera-height", "value"),
        State("camera-fps-input", "value"),
        prevent_initial_call=False,
    )
    def _toggle_camera(
        checked: bool,
        _init: int,
        _monitor: int,
        lock_af: Optional[bool],
        camera_state: Optional[dict],
        width: Optional[float],
        height: Optional[float],
        fps: Optional[float],
    ) -> Tuple[str, bool, dict, bool]:
        def _describe_stream() -> str:
            stream = SESSION.camera_stream
            if stream is None:
                return t("Camera stream stopped.")
            res_w, res_h = stream.config.resolution
            fps_val = stream.config.fps
            device_id = stream.device_id() or t("unknown")
            return (
                t("Camera stream running ({res_w}x{res_h} @ {fps_val:.1f} FPS, MXID {device_id}).", res_w=res_w, res_h=res_h, fps_val=fps_val, device_id=device_id)
            )

        trigger = callback_context.triggered_id
        if trigger is None:
            return (
                no_update,
                no_update,
                no_update,
                no_update,
            )

        startup_triggers = {"hardware-init"}
        if trigger in startup_triggers and not SETTINGS.camera_auto_start:
            SESSION.camera_auto_init_done = True
            stopped = t("Camera stream stopped.")
            return (
                stopped,
                False,
                {"status": "off", "message": stopped},
                True,
            )
        if trigger == "monitor-interval" and SESSION.camera_calibration_in_progress:
            return (
                no_update,
                no_update,
                no_update,
                no_update,
            )
        if trigger == "monitor-interval":
            # Never auto-retry camera startup from the periodic monitor tick.
            # Reconnect attempts should be explicit user actions (switch/button),
            # otherwise missing hardware can cause endless retry loops.
            stream_running = (
                SESSION.camera_stream is not None
                and SESSION.camera_stream.is_running()
            )
            if stream_running:
                status = _describe_stream()
                SESSION.camera_last_error_message = None
                SESSION.camera_last_error_key = None
                SESSION.camera_last_error_params = {}
                SESSION.camera_last_error_status = None
                return (
                    status,
                    True,
                    {"status": "live", "message": status},
                    False,
                )
            return (
                no_update,
                no_update,
                no_update,
                no_update,
            )
        if trigger in startup_triggers:
            # Auto-init should run once per worker process. Dash can remount
            # intervals on client reconnects; avoid restarting camera retries.
            if SESSION.camera_auto_init_done:
                stream_running = (
                    SESSION.camera_stream is not None
                    and SESSION.camera_stream.is_running()
                )
                if stream_running:
                    status = _describe_stream()
                    return (
                        status,
                        True,
                        {"status": "live", "message": status},
                        False,
                    )
                if SESSION.camera_last_error_key or SESSION.camera_last_error_message:
                    last_status = SESSION.camera_last_error_status or "unavailable"
                    message = _resolve_camera_message(camera_state)
                    return (
                        message,
                        False,
                        {"status": last_status, "message": message},
                        True,
                    )
                stopped = t("Camera stream stopped.")
                return (
                    stopped,
                    False,
                    {"status": "off", "message": stopped},
                    True,
                )
            SESSION.camera_auto_init_done = True
        stream_running = (
            SESSION.camera_stream is not None
            and SESSION.camera_stream.is_running()
        )
        checked_value = stream_running if checked is None else bool(checked)
        if (
            trigger == "camera-enable"
            and not checked_value
            and not stream_running
        ):
            # Preserve last known startup failure details when camera is already
            # down and the toggle transitions to false.
            if SESSION.camera_last_error_key or SESSION.camera_last_error_message:
                last_status = SESSION.camera_last_error_status or "unavailable"
                message = _resolve_camera_message(camera_state)
                state_payload = {
                    "status": last_status,
                    "message": message,
                }
                if SESSION.camera_last_error_key:
                    state_payload["message_key"] = SESSION.camera_last_error_key
                    state_payload["message_params"] = SESSION.camera_last_error_params
                return (
                    message,
                    False,
                    state_payload,
                    True,
                )
        desired_checked = checked_value

        if trigger == "camera-lock-af" and stream_running:
            try:
                assert SESSION.camera_stream is not None
                SESSION.camera_stream.set_focus_lock(bool(lock_af))
                state = t("locked") if lock_af else t("unlocked")
                status = t("Camera stream running; focus {state}.", state=state)
                SESSION.camera_last_error_message = None
                SESSION.camera_last_error_key = None
                SESSION.camera_last_error_params = {}
                SESSION.camera_last_error_status = None
                return (
                    status,
                    True,
                    {"status": "live", "message": status},
                    False,
                )
            except Exception as exc:  # noqa: BLE001
                error_key = "Camera focus control error: {error}"
                error_params = {"error": str(exc)}
                error = t(error_key, **error_params)
                SESSION.camera_last_error_message = error
                SESSION.camera_last_error_key = error_key
                SESSION.camera_last_error_params = error_params
                SESSION.camera_last_error_status = "error"
                return (
                    error,
                    True,
                    {
                        "status": "error",
                        "message": error,
                        "message_key": error_key,
                        "message_params": error_params,
                    },
                    False,
                )

        if trigger in startup_triggers:
            desired_checked = True
        elif trigger == "camera-enable" and desired_checked:
            # Explicit user retry path after unavailable/error.
            SESSION.camera_auto_init_done = True
        if desired_checked:
            try:
                if stream_running:
                    status = _describe_stream()
                    SESSION.camera_last_error_message = None
                    SESSION.camera_last_error_key = None
                    SESSION.camera_last_error_params = {}
                    SESSION.camera_last_error_status = None
                    return (
                        status,
                        True,
                        {"status": "live", "message": status},
                        False,
                    )
                status = start_camera_stream(
                    width,
                    height,
                    fps,
                    bool(lock_af),
                )
                SESSION.camera_last_error_message = None
                SESSION.camera_last_error_key = None
                SESSION.camera_last_error_params = {}
                SESSION.camera_last_error_status = None
                return (
                    status,
                    True,
                    {"status": "live", "message": status},
                    False,
                )
            except RuntimeError as exc:
                # Handle specific case of no devices
                if "No available devices" in str(exc):
                    error_key = (
                        "No OAK-D camera detected (retried 8 times). Connect/power the camera, then enable the camera stream to retry."
                    )
                    error_params = {}
                    error = t(error_key)
                else:
                    error_key = "Camera error: {error}"
                    error_params = {"error": str(exc)}
                    error = t(error_key, **error_params)
                SESSION.camera_last_error_message = error
                SESSION.camera_last_error_key = error_key
                SESSION.camera_last_error_params = error_params
                SESSION.camera_last_error_status = "unavailable"
                set_active_overlay_task(None)
                stop_camera_stream()
                return (
                    error,
                    False,
                    {
                        "status": "unavailable",
                        "message": error,
                        "message_key": error_key,
                        "message_params": error_params,
                    },
                    True,
                )
            except Exception as exc:
                set_active_overlay_task(None)
                stop_camera_stream()
                error_key = "Camera error: {error}"
                error_params = {"error": str(exc)}
                error = t(error_key, **error_params)
                SESSION.camera_last_error_message = error
                SESSION.camera_last_error_key = error_key
                SESSION.camera_last_error_params = error_params
                SESSION.camera_last_error_status = "error"
                return (
                    error,
                    False,
                    {
                        "status": "error",
                        "message": error,
                        "message_key": error_key,
                        "message_params": error_params,
                    },
                    True,
                )
        set_active_overlay_task(None)
        stop_camera_stream()
        SESSION.camera_last_error_message = None
        SESSION.camera_last_error_key = None
        SESSION.camera_last_error_params = {}
        SESSION.camera_last_error_status = None
        stopped = t("Camera stream stopped.")
        return (
            stopped,
            False,
            {"status": "off", "message": stopped},
            True,
        )

    @app.callback(
        Output("camera-status-text", "children", allow_duplicate=True),
        Output("camera-connection-state", "data", allow_duplicate=True),
        Output("metrics-interval", "disabled", allow_duplicate=True),
        Output("camera-lock-af", "checked"),
        Input("footer-refocus-lock", "n_clicks"),
        prevent_initial_call=True,
    )
    def _refocus_and_lock_camera(
        n_clicks: Optional[int],
    ) -> Tuple[str, dict, bool, bool]:
        """Refocus the live camera briefly, then lock at the current lens."""

        if not n_clicks:
            return (
                no_update,
                no_update,
                no_update,
                no_update,
            )

        stream = SESSION.camera_stream
        if stream is None or not stream.is_running():
            message = t("Camera stream is not running.")
            return (
                message,
                {"status": "off", "message": message},
                True,
                no_update,
            )

        try:
            position = stream.refocus_and_lock(AUTOFOCUS_SETTLE_S)
        except Exception as exc:  # noqa: BLE001
            error_key = "Camera focus control error: {error}"
            error_params = {"error": str(exc)}
            error = t(error_key, **error_params)
            SESSION.camera_last_error_message = error
            SESSION.camera_last_error_key = error_key
            SESSION.camera_last_error_params = error_params
            SESSION.camera_last_error_status = "error"
            return (
                error,
                {
                    "status": "error",
                    "message": error,
                    "message_key": error_key,
                    "message_params": error_params,
                },
                False,
                no_update,
            )

        status = t(
            "Camera refocused and locked at lens position {position}.",
            position=position,
        )
        SESSION.camera_last_error_message = None
        SESSION.camera_last_error_key = None
        SESSION.camera_last_error_params = {}
        SESSION.camera_last_error_status = None
        return (
            status,
            {"status": "live", "message": status},
            False,
            True,
        )

    @app.callback(
        Output("camera-feed-container", "children"),
        Input("camera-connection-state", "data"),
    )
    def _update_camera_placeholder(camera_state: Optional[dict]) -> html.Div:
        """Show placeholder when camera is not connected, otherwise show video feed."""
        status = (camera_state or {}).get("status", "off")
        if (
            status != "live"
            and SESSION.camera_stream is not None
            and SESSION.camera_stream.is_running()
        ):
            status = "live"
        if status == "off" and (
            SESSION.camera_last_error_key or SESSION.camera_last_error_message
        ):
            status = SESSION.camera_last_error_status or "unavailable"

        if status == "live":
            # Camera is connected - show the video feed
            return html.Div(
                [
                    html.Img(
                        id="camera-feed",
                        src="/video_feed",
                        className="camera-feed",
                    ),
                    html.Canvas(
                        id="camera-overlay-canvas",
                        className="camera-overlay-canvas",
                    ),
                ],
                id="camera-layered-feed",
                className="camera-layered-feed",
            )
        else:
            # Camera not connected - show placeholder
            message = _resolve_camera_message(camera_state)
            icon = "📷" if status == "off" else "⚠️"
            title_class = (
                "camera-placeholder-title"
                if status == "off"
                else "camera-placeholder-title is-warning"
            )

            return html.Div(
                [
                    html.Div(
                        [
                            html.Div(icon, className="camera-placeholder-icon"),
                            html.Div(
                                t("Camera Not Connected") if status == "off" else t("Camera Unavailable"),
                                className=title_class,
                            ),
                            html.Div(
                                message,
                                className="camera-placeholder-message",
                            ),
                        ],
                        className="camera-placeholder-content",
                    )
                ],
                id="camera-feed",
                className="camera-feed camera-placeholder",
            )

    @app.callback(
        Output("camera-calibration-action", "children"),
        Output("calibration-progress-store", "data"),
        Output("calibration-stepper", "active", allow_duplicate=True),
        Input("calibrate-camera", "n_clicks"),
        Input("use-existing-calibration", "n_clicks"),
        State("settings-calib-path", "value"),
        State("calibration-stepper", "active"),
        State("calibration-tab-mounted-at", "data"),
        prevent_initial_call=True,
    )
    def _handle_camera_calibration_actions(
        _calibrate: int,
        _use_existing: int,
        bundle_path: str,
        current_step: Optional[int] = 0,
        mounted_at: Optional[dict] = None,
    ) -> Tuple[dmc.Alert, dict, int]:
        # Import parent module to access helper functions
        import app.codroid_picker as codroid_picker_module

        trigger = callback_context.triggered_id
        if trigger in {"calibrate-camera", "use-existing-calibration"}:
            click_count = _calibrate if trigger == "calibrate-camera" else _use_existing
            if not click_count or _calibration_tab_action_guard_active(mounted_at):
                return no_update, no_update, no_update

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
                    title=t("Using saved calibration"),
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
                    title=t("Load error"),
                    color="red",
                )
                return alert, {"progress": 0}, (current_step or 0)

        if trigger == "calibrate-camera":
            logging.info("[CALIBRATE] Camera calibration requested")
            if not _CALIBRATION_ACTION_LOCK.acquire(blocking=False):
                alert = dmc.Alert(
                    t("Camera calibration is already running. Wait for the current capture to finish."),
                    icon="!",
                    title=t("Calibration in progress"),
                    color="yellow",
                )
                return alert, {"progress": CALIBRATION_PROGRESS.get("value", 0)}, (current_step or 0)

            SESSION.camera_calibration_in_progress = True

            try:
                stream = SESSION.camera_stream
                logging.info(
                    "[CALIBRATE] Stream state before capture: present=%s running=%s depth=%s",
                    stream is not None,
                    bool(stream and stream.is_running()),
                    bool(stream and stream.config.enable_depth),
                )

                # Clear prior calibration artifacts on disk to avoid mixing old frames
                _clear_calibration_outputs(output_path)
                logging.info("[CALIBRATE] Cleared previous calibration outputs")

                # Clear previous captures
                SESSION.capture.frames = []
                SESSION.capture.depth_frame = None

                # Capture frames for more robust calibration
                num_frames = max(1, SETTINGS.camera_calibration_frame_count)
                captured_frames = []
                CALIBRATION_PROGRESS["max_frames"] = num_frames
                CALIBRATION_PROGRESS["value"] = 1  # Start progress
                CALIBRATION_PROGRESS["message"] = ""
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

                # Capture depth from the live depth-enabled stream so calibration
                # does not tear down or restart the DepthAI pipeline.
                (
                    depth_path,
                    stream_paused_for_depth_capture,
                ) = _capture_depth_snapshot_for_calibration(
                    codroid_picker_module,
                    output_path,
                )
                logging.info("[CALIBRATE] Captured depth snapshot at %s", depth_path)

                # Pre-flight check: ensure the saved frames actually contain ChArUco corners
                detection_counts, usable_frames = _charuco_detection_counts(
                    codroid_picker_module._list_capture_frames(
                        output_path), board_path
                )
                if usable_frames < SETTINGS.camera_calibration_min_usable_frames:
                    raise RuntimeError(
                        t("No usable ChArUco detections in captured frames. Corner counts: {counts}. Ensure the board is visible, sharp, and well-lit.", counts=detection_counts)
                    )

                # Compute calibration bundle
                result = codroid_picker_module.compute_calibration_bundle(
                    board_path, bundle_path, output_path)
                logging.info("[CALIBRATE] Calibration bundle computed: %s", bundle_path)

                success_lines = [
                    dmc.Text(
                        t("✓ Captured {count} RGB frames", count=len(captured_frames)), size="sm"
                    ),
                    dmc.Text(t("✓ Captured 1 depth snapshot"), size="sm"),
                    dmc.Text(
                        t("✓ Detected {usable}/{total} usable frames", usable=usable_frames, total=num_frames), size="sm"
                    ),
                ]
                if stream_paused_for_depth_capture:
                    success_lines.append(
                        dmc.Text(
                            t("⚠ Camera stream paused after isolated depth capture. Click 'Enable camera' to resume live video."),
                            size="sm",
                            c="orange",
                        )
                    )
                success_lines.append(dmc.Text(result, size="sm", fw=600, c="teal"))

                alert = dmc.Alert(
                    children=[dmc.Stack(success_lines, gap="xs")],
                    icon="✓",
                    title=t("Calibration complete"),
                    color="green",
                )
                # Reset progress tracker
                CALIBRATION_PROGRESS["value"] = 0
                CALIBRATION_PROGRESS["current_frame"] = 0
                CALIBRATION_PROGRESS["message"] = ""
                # Advance to robot calibration step
                return alert, {"progress": 0}, 1
            except Exception as exc:
                logging.error(
                    f"[CALIBRATE] Error during calibration: {exc}", exc_info=True)
                alert = dmc.Alert(
                    str(exc)[:200],
                    icon="!",
                    title=t("Calibration error"),
                    color="red",
                )
                # Reset progress tracker on error
                CALIBRATION_PROGRESS["value"] = 0
                CALIBRATION_PROGRESS["current_frame"] = 0
                CALIBRATION_PROGRESS["message"] = ""
                return alert, {"progress": 0}, (current_step or 0)
            finally:
                SESSION.camera_calibration_in_progress = False
                _CALIBRATION_ACTION_LOCK.release()

        return dmc.Alert("", icon="•", title=t("Ready"), color="gray"), {"progress": 0}, (current_step or 0)

    @app.callback(
        Output("overlay-status", "children", allow_duplicate=True),
        Output("camera-fps", "children", allow_duplicate=True),
        Output("plate-detection-state", "data", allow_duplicate=True),
        Output("camera-ui-metrics-state", "data"),
        Input("metrics-interval", "n_intervals"),
        State("main-tabs", "value"),
        State("calibration-step-state", "data"),
        State("application-tab-state", "data"),
        State("settings-marker-a", "value"),
        State("settings-marker-b", "value"),
        State("settings-marker-size", "value"),
        State("overlay-mode-select", "value"),
        prevent_initial_call=True,
    )
    def _update_live_metrics(
        _ticks: int,
        active_tab: Optional[str],
        calibration_step_state: Optional[dict],
        application_tab_state: Optional[dict],
        marker_a: Optional[float],
        marker_b: Optional[float],
        marker_size: Optional[float],
        overlay_mode: Optional[str],
    ) -> Tuple[str, str, dict, dict]:
        frame = RAW_STREAM.get_latest_frame()
        tab = active_tab or "applications"
        calibration_step = int((calibration_step_state or {}).get("active", 0) or 0)
        application_tab = (application_tab_state or {}).get("value") or "pick-place"

        def _metrics_payload(
            *,
            camera: str = "",
            workspace: str = "",
            plate: str = "",
            detection: str = "",
            calibrate_disabled: bool = True,
            marker_color: str = "gray",
        ) -> dict:
            return {
                "camera_calibration_metrics": camera,
                "workspace_metrics": workspace,
                "plate_metrics": plate,
                "detection_metrics": detection,
                "calibrate_camera_disabled": bool(calibrate_disabled),
                "markers_detected_color": marker_color,
            }

        def _camera_calibration_state(
            metrics_text: str, step: int
        ) -> Tuple[bool, str]:
            if SESSION.camera_calibration_in_progress:
                return True, "yellow"
            if step != 0:
                return True, "gray"
            if not metrics_text or "ChArUco corners:" not in metrics_text:
                return True, "gray"
            try:
                count = int(metrics_text.split(":")[-1].strip().split()[0])
                is_disabled = count < min_live_charuco_corners
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
            "message": t("Waiting for plate markers"),
        }
        capture_fps = SESSION.camera_stream.get_fps() if SESSION.camera_stream else 0.0
        stream_fps = MJPEG_STREAM.get_fps()
        configured_fps = (
            float(SESSION.camera_stream.config.fps)
            if SESSION.camera_stream is not None
            else 0.0
        )
        fps_warning_threshold = configured_fps * 0.85 if configured_fps > 0 else 0.0
        fps_warning = (
            " ⚠️"
            if capture_fps > 0 and fps_warning_threshold > 0 and capture_fps < fps_warning_threshold
            else ""
        )
        overlay_fps_text = _format_fps(capture_fps, stream_fps) + fps_warning

        if frame is None:
            set_active_overlay_task(None)
            # Don't add placeholder to MJPEG stream - let video element show its poster image
            # Only log occasionally to reduce noise (every 30 intervals at default interval)
            if _ticks % 30 == 0:
                is_running = SESSION.camera_stream and SESSION.camera_stream.is_running()
                if is_running:
                    # Only log if stream claims to be running but we're not getting frames
                    logging.warning(
                        "No frame from camera despite stream running: %s", is_running)
            camera_metric = (
                t("ChArUco corners: --")
                if tab == "calibration" and calibration_step == 0
                else ""
            )
            workspace_metric = (
                t("Workspace: --")
                if tab == "applications" and application_tab == "workspace"
                else ""
            )
            plate_metric = (
                t("Plate error: --")
                if tab == "calibration" and calibration_step == 1
                else ""
            )
            detection_metric = (
                t("Detections: --")
                if tab == "applications" and application_tab == "pick-place"
                else ""
            )
            return (
                t("Camera stream is stopped."),
                overlay_fps_text,
                plate_state,
                _metrics_payload(
                    camera=camera_metric,
                    workspace=workspace_metric,
                    plate=plate_metric,
                    detection=detection_metric,
                    calibrate_disabled=True,
                    marker_color="gray",
                ),
            )

        # Frame received, dispatch to background processor for heavy overlays.
        board_path = DEFAULT_BOARD_PATH
        plate_path = DEFAULT_PLATE_PATH
        marker_ids = (
            _safe_int(marker_a, SESSION.marker_ids[0]),
            _safe_int(marker_b, SESSION.marker_ids[1]),
        )
        marker_size_mm = _safe_float(
            marker_size, SESSION.marker_size_mm)

        requested_overlay = (overlay_mode or "auto").strip().lower()
        task, effective_overlay = _resolve_overlay_task(
            requested_overlay=requested_overlay,
            tab=tab,
            calibration_step=calibration_step,
            application_tab=application_tab,
            board_path=board_path,
            marker_ids=marker_ids,
            marker_size_mm=marker_size_mm,
            plate_path=plate_path,
        )

        set_active_overlay_task(task)

        overlay: Optional[OverlayResult] = None
        if task is not None:
            overlay = FRAME_PROCESSOR.get_latest(task.key)
            if overlay is None:
                status_text = t("Overlay processing...")
                metric_text = "--"
            else:
                status_text = overlay.status
                metric_text = overlay.metric
            if calibration_step == 1 and task.key != "calibration-plate":
                _refresh_plate_status_throttled(frame, plate_path)
        elif calibration_step == 1:
            _refresh_plate_status_throttled(frame, plate_path)

        if task is None:
            status_text = t("No overlay applied.")
            metric_text = "--"

        mode_label = {
            "auto": t("Overlay: Auto"),
            "calibration": t("Overlay: Calibration"),
            "workspace": t("Overlay: Workspace"),
            "plate": t("Overlay: Plate"),
            "detection": t("Overlay: Detections"),
            "auto-plan": t("Overlay: Auto Plan"),
            "none": t("Overlay: Off"),
        }.get(effective_overlay, t("Overlay: Auto"))
        status_text = t("{mode} · {status}", mode=mode_label, status=status_text)

        # Get plate status from state module
        from app.codroid_picker import state
        plate_state = state.get_plate_status_snapshot()

        # Note: MJPEG streaming is fed from the camera frame hook. Overlay
        # metrics come from the background processor's browser-rendered state.

        if tab == "calibration":
            if calibration_step == 0:
                calibrate_disabled, marker_color = _camera_calibration_state(
                    metric_text, calibration_step
                )
                return (
                    status_text,
                    overlay_fps_text,
                    plate_state,
                    _metrics_payload(
                        camera=metric_text,
                        calibrate_disabled=calibrate_disabled,
                        marker_color=marker_color,
                    ),
                )
            metrics_text = ""
            calibrate_disabled, marker_color = _camera_calibration_state(
                metrics_text, calibration_step
            )
            return (
                status_text,
                overlay_fps_text,
                plate_state,
                _metrics_payload(
                    plate=metric_text,
                    calibrate_disabled=calibrate_disabled,
                    marker_color=marker_color,
                ),
            )

        if tab == "applications":
            return (
                status_text,
                overlay_fps_text,
                plate_state,
                _metrics_payload(
                    workspace=metric_text if application_tab == "workspace" else "",
                    detection=(
                        _format_detection_metrics()
                        if application_tab == "pick-place"
                        else ""
                    ),
                ),
            )

        # Monitoring or any other tab: surface metrics.
        return (
            status_text,
            overlay_fps_text,
            plate_state,
            _metrics_payload(),
        )

    @app.callback(
        Output("workspace-metrics", "children", allow_duplicate=True),
        Output("detection-metrics", "children", allow_duplicate=True),
        Input("application-ui-interval", "n_intervals"),
        State("camera-ui-metrics-state", "data"),
        prevent_initial_call=True,
    )
    def _render_application_metrics(
        _ticks: int,
        metrics_state: Optional[dict],
    ) -> Tuple[str, str]:
        metrics = metrics_state or {}
        workspace_metrics = metrics.get("workspace_metrics") or no_update
        detection_metrics = metrics.get("detection_metrics") or no_update
        return workspace_metrics, detection_metrics

    @app.callback(
        Output("camera-calibration-metrics", "children", allow_duplicate=True),
        Output("plate-metrics", "children", allow_duplicate=True),
        Output("calibrate-camera", "disabled", allow_duplicate=True),
        Output("markers-detected-icon", "color", allow_duplicate=True),
        Input("calibration-ui-interval", "n_intervals"),
        State("camera-ui-metrics-state", "data"),
        prevent_initial_call=True,
    )
    def _render_calibration_metrics(
        _ticks: int,
        metrics_state: Optional[dict],
    ) -> Tuple[str, str, bool, str]:
        metrics = metrics_state or {}
        camera_metrics = metrics.get("camera_calibration_metrics") or no_update
        plate_metrics = metrics.get("plate_metrics") or no_update
        disabled = metrics.get("calibrate_camera_disabled")
        marker_color = metrics.get("markers_detected_color") or "gray"
        return (
            camera_metrics,
            plate_metrics,
            True if disabled is None else bool(disabled),
            marker_color,
        )

    @app.callback(
        Output("calibration-progress", "value"),
        Output("calibration-progress", "style"),
        Output("calibration-progress-text", "children"),
        Output("calibration-progress-text", "style"),
        Input("calibration-ui-interval", "n_intervals"),
    )
    def _update_calibration_progress(_):
        """Update progress bar and text from global calibration progress tracker."""
        progress = CALIBRATION_PROGRESS.get("value", 0)
        current_frame = CALIBRATION_PROGRESS.get("current_frame", 0)
        max_frames = CALIBRATION_PROGRESS.get("max_frames", 0)
        message = CALIBRATION_PROGRESS.get("message", "")

        # Show progress bar and text only if capturing (1-99%)
        if progress > 0 and progress < 100:
            if message:
                text = t(str(message))
            else:
                text = t(
                    "Capturing images: {current}/{total}",
                    current=current_frame,
                    total=max_frames,
                )
            return (
                progress,
                {"display": "block"},
                text,
                {"display": "block", "marginBottom": "8px"},
            )

        return 0, {"display": "none"}, "", {"display": "none"}
