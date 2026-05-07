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
    OverlayResult,
    OverlayTask,
    FRAME_PROCESSOR,
    set_active_overlay_task,
    start_camera_stream,
    stop_camera_stream,
)
from app.codroid_picker.utils import safe_float, safe_int, safe_path
from app.codroid_picker.webrtc import should_try_webrtc
from app.settings import (DEFAULT_BOARD_PATH, DEFAULT_BUNDLE_PATH,
                          DEFAULT_CAPTURE_DIR, DEFAULT_PLATE_PATH, SETTINGS)
from vision.camera_check import list_oak_devices
from vision.stream import list_host_camera_devices

# Keep underscore-prefixed aliases for backward compatibility with external imports
_safe_int = safe_int
_safe_float = safe_float
_safe_path = safe_path

_CALIBRATION_ACTION_LOCK = threading.Lock()


def _format_fps(capture_fps: float) -> str:
    """Format FPS display text."""
    return t("Camera: {capture_fps:.1f} FPS", capture_fps=capture_fps)


def _webrtc_preview_active() -> bool:
    """Return whether the browser-side WebRTC asset owns the FPS label."""
    try:
        from app.codroid_picker.webrtc import should_try_webrtc
    except Exception:
        return False
    return should_try_webrtc()


def _format_preview_fps(capture_fps: float, warning: str) -> Any:
    """Format the preview FPS label unless WebRTC is actively rendering it."""
    if _webrtc_preview_active():
        return no_update
    return _format_fps(capture_fps) + warning


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


def _format_camera_scan_results() -> html.Ul:
    """Format OAK-D and host camera scan results."""
    items: list[Any] = []
    oak_devices = list_oak_devices()
    if oak_devices:
        for idx, device in enumerate(oak_devices, start=1):
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
            items.append(html.Li(f"OAK-D {idx}: {name} ({state})"))
    else:
        items.append(html.Li(t("No OAK-D devices detected.")))

    host_devices = list_host_camera_devices()
    if host_devices:
        for device in host_devices:
            resolution = device.get("resolution") or (0, 0)
            items.append(
                html.Li(
                    f"{device.get('name')} index {device.get('index')} "
                    f"({resolution[0]}x{resolution[1]})"
                )
            )
    else:
        items.append(html.Li(t("No host cameras detected.")))
    return html.Ul(items)


def _camera_device_label(device_id: object) -> str:
    """Return the display label for the active camera identifier."""
    return t("Host camera") if str(device_id).startswith("host:") else "MXID"


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


def _camera_toggle_no_update() -> Tuple[object, object, object, object, object, object, object]:
    """Return no-update payload for camera toggle callback outputs."""
    return (
        no_update,
        no_update,
        no_update,
        no_update,
        no_update,
        no_update,
        no_update,
    )


def _handle_toggle_camera_callback(
    checked: bool,
    lock_af: Optional[bool],
    camera_state: Optional[dict],
    width: Optional[float],
    height: Optional[float],
    fps: Optional[float],
    camera_source: Optional[str] = None,
    host_camera_index: Optional[int] = None,
) -> Tuple[str, str, str, bool, dict, bool, bool]:
    """Handle camera enable/disable/monitor actions."""
    trigger = callback_context.triggered_id
    if trigger is None:
        if checked:
            trigger = "auto-init"
        else:
            return _camera_toggle_no_update()
    stream_running = (
        SESSION.camera_stream is not None
        and SESSION.camera_stream.is_running()
    )
    if trigger == "monitor-interval" and SESSION.camera_calibration_in_progress:
        return _camera_toggle_no_update()
    if trigger in {"camera-source", "host-camera-index"} and stream_running:
        stop_camera_stream()
        stream_running = False
    if trigger == "monitor-interval":
        if not stream_running or SESSION.camera_stream is None:
            return _camera_toggle_no_update()
        stream_status = SESSION.camera_stream.status()
        if stream_status == "retrying":
            error = SESSION.camera_stream.last_error() or t("Camera unavailable")
            message = t("Camera retrying: {error}", error=error)
            return (
                message,
                t("Camera: retrying"),
                "orange",
                True,
                {"status": "retrying", "message": message, "error": error},
                True,
                False,
            )
        if stream_status == "live":
            res_w, res_h = SESSION.camera_stream.config.resolution
            fps_val = SESSION.camera_stream.config.fps
            device_id = SESSION.camera_stream.device_id() or t("unknown")
            device_label = _camera_device_label(device_id)
            message = t(
                "Camera stream running ({res_w}x{res_h} @ {fps_val:.1f} FPS, {device_label} {device_id}).",
                res_w=res_w,
                res_h=res_h,
                fps_val=fps_val,
                device_label=device_label,
                device_id=device_id,
            )
            return (
                message,
                t("Camera: live"),
                "brand",
                True,
                {"status": "live", "message": message},
                False,
                False,
            )
        return _camera_toggle_no_update()
    if trigger == "auto-init":
        # Auto-init should run once per worker process. Dash can remount
        # intervals on client reconnects; avoid restarting camera retries.
        if SESSION.camera_auto_init_done:
            return _camera_toggle_no_update()
        SESSION.camera_auto_init_done = True
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
            badge_text = (
                t("Camera: unavailable")
                if last_status == "unavailable"
                else t("Camera: error")
            )
            badge_color = "orange" if last_status == "unavailable" else "yellow"
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
                badge_text,
                badge_color,
                False,
                state_payload,
                True,
                True,
            )
    desired_checked = checked_value

    def _describe_stream() -> str:
        stream = SESSION.camera_stream
        if stream is None:
            return t("Camera stream stopped.")
        if stream.status() == "retrying":
            error = stream.last_error() or t("Camera unavailable")
            return t("Camera retrying: {error}", error=error)
        res_w, res_h = stream.config.resolution
        fps_val = stream.config.fps
        device_id = stream.device_id() or t("unknown")
        device_label = _camera_device_label(device_id)
        return (
            t(
                "Camera stream running ({res_w}x{res_h} @ {fps_val:.1f} FPS, {device_label} {device_id}).",
                res_w=res_w,
                res_h=res_h,
                fps_val=fps_val,
                device_label=device_label,
                device_id=device_id,
            )
        )

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
                t("Camera: live"),
                "brand",
                True,
                {"status": "live", "message": status},
                False,
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
                t("Camera: error"),
                "red",
                True,
                {
                    "status": "error",
                    "message": error,
                    "message_key": error_key,
                    "message_params": error_params,
                },
                False,
                False,
            )

    if trigger == "auto-init":
        desired_checked = True
    elif trigger == "camera-enable" and desired_checked:
        # Explicit user retry path after unavailable/error.
        SESSION.camera_auto_init_done = True
    if desired_checked:
        try:
            if stream_running:
                status = _describe_stream()
                stream_status = SESSION.camera_stream.status() if SESSION.camera_stream else "live"
                if stream_status == "retrying":
                    return (
                        status,
                        t("Camera: retrying"),
                        "orange",
                        True,
                        {"status": "retrying", "message": status},
                        True,
                        False,
                    )
                SESSION.camera_last_error_message = None
                SESSION.camera_last_error_key = None
                SESSION.camera_last_error_params = {}
                SESSION.camera_last_error_status = None
                return (
                    status,
                    t("Camera: live"),
                    "brand",
                    True,
                    {"status": "live", "message": status},
                    False,
                    False,
                )
            status = start_camera_stream(
                width,
                height,
                fps,
                bool(lock_af),
                enable_depth_override=False,
                source=camera_source,
                host_camera_index=safe_int(host_camera_index, 0),
            )
            stream_status = SESSION.camera_stream.status() if SESSION.camera_stream else "live"
            if stream_status == "retrying":
                SESSION.camera_last_error_message = status
                SESSION.camera_last_error_key = "Camera retrying: {error}"
                SESSION.camera_last_error_params = {
                    "error": SESSION.camera_stream.last_error() if SESSION.camera_stream else status
                }
                SESSION.camera_last_error_status = "retrying"
                return (
                    status,
                    t("Camera: retrying"),
                    "orange",
                    True,
                    {"status": "retrying", "message": status},
                    True,
                    False,
                )
            SESSION.camera_last_error_message = None
            SESSION.camera_last_error_key = None
            SESSION.camera_last_error_params = {}
            SESSION.camera_last_error_status = None
            return (
                status,
                t("Camera: live"),
                "brand",
                True,
                {"status": "live", "message": status},
                False,
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
                t("Camera: unavailable"),
                "orange",
                False,
                {
                    "status": "unavailable",
                    "message": error,
                    "message_key": error_key,
                    "message_params": error_params,
                },
                True,
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
                t("Camera: error"),
                "yellow",
                False,
                {
                    "status": "error",
                    "message": error,
                    "message_key": error_key,
                    "message_params": error_params,
                },
                True,
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
        t("Camera: off"),
        "gray",
        False,
        {"status": "off", "message": stopped},
        True,
        True,
    )


def _handle_camera_calibration_action_callback(
    bundle_path: str,
    current_step: Optional[int] = 0,
) -> Tuple[dmc.Alert, dict, int]:
    """Handle camera calibration actions."""
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

            # Capture depth snapshot. If runtime stream is RGB-only, stop the
            # stream, capture depth in an isolated helper process, and keep the
            # stream paused to avoid in-process DepthAI restarts.
            depth_path = None
            stream_paused_for_depth_capture = False
            stream = SESSION.camera_stream
            if stream is None or not stream.is_running():
                raise RuntimeError(
                    t("Camera stream stopped before depth capture."))
            if bool(stream.config.enable_depth):
                depth_path = codroid_picker_module.capture_depth_snapshot(
                    output_path)
            else:
                restore_resolution = stream.config.resolution
                restore_fps = stream.config.fps
                restore_lock_af = stream.config.lock_af
                logging.info(
                    "[CALIBRATE] Capturing depth snapshot via isolated helper process")
                stop_camera_stream(wait_for_release=True)
                stream_paused_for_depth_capture = True
                depth_path = codroid_picker_module.capture_depth_snapshot_isolated(
                    output_path,
                    restore_resolution,
                    restore_fps,
                    bool(restore_lock_af),
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
            return alert, {"progress": 0}, (current_step or 0)
        finally:
            SESSION.camera_calibration_in_progress = False
            _CALIBRATION_ACTION_LOCK.release()

    return dmc.Alert("", icon="•", title=t("Ready"), color="gray"), {"progress": 0}, (current_step or 0)


def _camera_calibration_state(
    metrics_text: str,
    step: int,
    *,
    min_live_charuco_corners: int,
) -> Tuple[bool, str]:
    """Compute calibrate button disabled state and icon color."""
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


def _update_live_metrics_callback(
    _ticks: int,
    active_tab: Optional[str],
    calibration_step: Optional[int],
    application_tab: Optional[str],
    marker_a: Optional[float],
    marker_b: Optional[float],
    marker_size: Optional[float],
    overlay_mode: Optional[str],
    *,
    min_live_charuco_corners: int,
) -> Tuple[str, str, str, str, str, str, dict, bool, str]:
    """Update overlay/metrics UI state based on latest frame and selected mode."""
    import app.codroid_picker as codroid_picker_module

    frame, _depth = codroid_picker_module.get_latest_frame()

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
    overlay_fps_text = _format_preview_fps(capture_fps, fps_warning)

    if frame is None:
        set_active_overlay_task(None)
        # Only log occasionally to reduce noise (every 300 intervals = ~30 seconds at default interval)
        if _ticks % 300 == 0:
            is_running = SESSION.camera_stream and SESSION.camera_stream.is_running()
            if is_running:
                # Only log if stream claims to be running but we're not getting frames
                logging.warning(
                    "No frame from camera despite stream running: %s", is_running)
        return (
            t("Camera stream is stopped."),
            overlay_fps_text,
            t("ChArUco corners: --"),
            t("Workspace: --"),
            t("Plate error: --"),
            t("Detections: --"),
            plate_state,
            True,
            "gray",
        )

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
            _refresh_plate_status(frame, plate_path)
    elif calibration_step == 1:
        _refresh_plate_status(frame, plate_path)

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

    if tab == "calibration":
        if calibration_step == 0:
            calibrate_disabled, marker_color = _camera_calibration_state(
                metric_text,
                calibration_step,
                min_live_charuco_corners=min_live_charuco_corners,
            )
            return (
                status_text,
                overlay_fps_text,
                metric_text,
                t("Workspace: --"),
                t("Plate error: --"),
                t("Detections: --"),
                plate_state,
                calibrate_disabled,
                marker_color,
            )
        metrics_text = t("ChArUco corners: --")
        calibrate_disabled, marker_color = _camera_calibration_state(
            metrics_text,
            calibration_step,
            min_live_charuco_corners=min_live_charuco_corners,
        )
        return (
            status_text,
            overlay_fps_text,
            metrics_text,
            t("Workspace: --"),
            metric_text,
            t("Detections: --"),
            plate_state,
            calibrate_disabled,
            marker_color,
        )

    if tab == "applications":
        metrics_text = t("ChArUco corners: --")
        calibrate_disabled, marker_color = _camera_calibration_state(
            metrics_text,
            calibration_step,
            min_live_charuco_corners=min_live_charuco_corners,
        )
        return (
            status_text,
            overlay_fps_text,
            metrics_text,
            metric_text if application_tab == "workspace" else t("Workspace: --"),
            t("Plate error: --"),
            _format_detection_metrics() if application_tab == "pick-place" else t("Detections: --"),
            plate_state,
            calibrate_disabled,
            marker_color,
        )

    # Monitoring or any other tab: surface metrics.
    metrics_text = t("ChArUco corners: --")
    calibrate_disabled, marker_color = _camera_calibration_state(
        metrics_text,
        calibration_step,
        min_live_charuco_corners=min_live_charuco_corners,
    )
    return (
        status_text,
        overlay_fps_text,
        metrics_text,
        t("Workspace: --"),
        t("Plate error: --"),
        t("Detections: --"),
        plate_state,
        calibrate_disabled,
        marker_color,
    )


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
        return _format_camera_scan_results()

    @app.callback(
        Output("camera-status-text", "children"),
        Output("header-camera-badge", "children"),
        Output("header-camera-badge", "color"),
        Output("camera-enable", "checked"),
        Output("camera-connection-state", "data"),
        Output("camera-interval", "disabled"),
        Output("metrics-interval", "disabled"),
        Input("camera-enable", "checked"),
        Input("auto-init", "n_intervals"),
        Input("monitor-interval", "n_intervals"),
        Input("camera-lock-af", "checked"),
        Input("camera-source", "value"),
        Input("host-camera-index", "value"),
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
        camera_source: Optional[str],
        host_camera_index: Optional[int],
        camera_state: Optional[dict],
        width: Optional[float],
        height: Optional[float],
        fps: Optional[float],
    ) -> Tuple[str, str, str, bool, dict, bool, bool]:
        return _handle_toggle_camera_callback(
            checked,
            lock_af,
            camera_state,
            width,
            height,
            fps,
            camera_source,
            host_camera_index,
        )

    @app.callback(
        Output("camera-feed-container", "children"),
        Input("camera-connection-state", "data"),
    )
    def _update_camera_placeholder(camera_state: Optional[dict]) -> html.Div:
        """Show placeholder when camera is not connected, otherwise show video feed."""
        status = (camera_state or {}).get("status", "off")
        if status == "off" and (
            SESSION.camera_last_error_key or SESSION.camera_last_error_message
        ):
            status = SESSION.camera_last_error_status or "unavailable"

        if should_try_webrtc():
            message = _resolve_camera_message(camera_state)
            return html.Div(
                [
                    html.Video(
                        id="camera-feed",
                        className="camera-feed camera-feed-video",
                        autoPlay=True,
                        muted=True,
                    ),
                    html.Canvas(
                        id="camera-overlay-canvas",
                        className="camera-overlay-canvas",
                    ),
                ],
                id="camera-video-wrapper",
                className="camera-video-wrapper",
                **{
                    "data-video-transport": "webrtc",
                    "data-webrtc-preferred": "true",
                    "data-camera-status": status,
                    "data-camera-message": message,
                },
            )

        if status == "live":
            return html.Div(
                t("WebRTC video transport unavailable."),
                id="camera-feed",
                className="camera-feed",
                style={
                    "width": "100%",
                    "minHeight": "360px",
                    "borderRadius": "16px",
                    "border": "1px solid rgba(240, 65, 0, 0.25)",
                    "marginTop": "8px",
                    "backgroundColor": "#0b0f14",
                    "color": "#888",
                    "display": "flex",
                    "alignItems": "center",
                    "justifyContent": "center",
                },
            )
        else:
            # Camera not connected - show placeholder
            message = _resolve_camera_message(camera_state)
            icon = "📷" if status == "off" else "⚠️"
            color = "#666" if status == "off" else "#ff8c00"

            return html.Div(
                [
                    html.Div(
                        [
                            html.Div(icon, style={
                                     "fontSize": "64px", "marginBottom": "16px"}),
                            html.Div(
                                t("Camera Not Connected") if status == "off" else t("Camera Unavailable"),
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

                # Capture depth snapshot. If runtime stream is RGB-only, stop the
                # stream, capture depth in an isolated helper process, and keep the
                # stream paused to avoid in-process DepthAI restarts.
                depth_path = None
                stream_paused_for_depth_capture = False
                stream = SESSION.camera_stream
                if stream is None or not stream.is_running():
                    raise RuntimeError(
                        t("Camera stream stopped before depth capture."))
                if bool(stream.config.enable_depth):
                    depth_path = codroid_picker_module.capture_depth_snapshot(
                        output_path)
                else:
                    restore_resolution = stream.config.resolution
                    restore_fps = stream.config.fps
                    restore_lock_af = stream.config.lock_af
                    logging.info(
                        "[CALIBRATE] Capturing depth snapshot via isolated helper process")
                    stop_camera_stream(wait_for_release=True)
                    stream_paused_for_depth_capture = True
                    depth_path = codroid_picker_module.capture_depth_snapshot_isolated(
                        output_path,
                        restore_resolution,
                        restore_fps,
                        bool(restore_lock_af),
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
                return alert, {"progress": 0}, (current_step or 0)
            finally:
                SESSION.camera_calibration_in_progress = False
                _CALIBRATION_ACTION_LOCK.release()

        return dmc.Alert("", icon="•", title=t("Ready"), color="gray"), {"progress": 0}, (current_step or 0)

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
        Input("metrics-interval", "n_intervals"),
        State("main-tabs", "value"),
        State("calibration-stepper", "active"),
        State("application-tabs", "value"),
        State("settings-marker-a", "value"),
        State("settings-marker-b", "value"),
        State("settings-marker-size", "value"),
        State("overlay-mode-select", "value"),
        prevent_initial_call=True,
    )
    def _update_live_metrics(
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
        overlay_fps_text = _format_preview_fps(capture_fps, fps_warning)

        if frame is None:
            set_active_overlay_task(None)
            # Only log occasionally to reduce noise (every 300 intervals = ~30 seconds at default interval)
            if _ticks % 300 == 0:
                is_running = SESSION.camera_stream and SESSION.camera_stream.is_running()
                if is_running:
                    # Only log if stream claims to be running but we're not getting frames
                    logging.warning(
                        "No frame from camera despite stream running: %s", is_running)
            return (
                t("Camera stream is stopped."),
                overlay_fps_text,
                t("ChArUco corners: --"),
                t("Workspace: --"),
                t("Plate error: --"),
                t("Detections: --"),
                plate_state,
                True,
                "gray",
            )

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
                _refresh_plate_status(frame, plate_path)
        elif calibration_step == 1:
            _refresh_plate_status(frame, plate_path)

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

        if tab == "calibration":
            if calibration_step == 0:
                calibrate_disabled, marker_color = _camera_calibration_state(
                    metric_text, calibration_step
                )
                return (
                    status_text,
                    overlay_fps_text,
                    metric_text,
                    t("Workspace: --"),
                    t("Plate error: --"),
                    t("Detections: --"),
                    plate_state,
                    calibrate_disabled,
                    marker_color,
                )
            metrics_text = t("ChArUco corners: --")
            calibrate_disabled, marker_color = _camera_calibration_state(
                metrics_text, calibration_step
            )
            return (
                status_text,
                overlay_fps_text,
                metrics_text,
                t("Workspace: --"),
                metric_text,
                t("Detections: --"),
                plate_state,
                calibrate_disabled,
                marker_color,
            )

        if tab == "applications":
            metrics_text = t("ChArUco corners: --")
            calibrate_disabled, marker_color = _camera_calibration_state(
                metrics_text, calibration_step
            )
            return (
                status_text,
                overlay_fps_text,
                metrics_text,
                metric_text if application_tab == "workspace" else t("Workspace: --"),
                t("Plate error: --"),
                _format_detection_metrics() if application_tab == "pick-place" else t("Detections: --"),
                plate_state,
                calibrate_disabled,
                marker_color,
            )

        # Monitoring or any other tab: surface metrics.
        metrics_text = t("ChArUco corners: --")
        calibrate_disabled, marker_color = _camera_calibration_state(
            metrics_text, calibration_step
        )
        return (
            status_text,
            overlay_fps_text,
            metrics_text,
            t("Workspace: --"),
            t("Plate error: --"),
            t("Detections: --"),
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
            text = t("Capturing images: {current}/{total}", current=current_frame, total=max_frames)
            return progress, {"display": "block"}, text, {"display": "block", "marginBottom": "8px"}

        return 0, {"display": "none"}, "", {"display": "none"}
