"""Monitoring and operation state callbacks.

This module contains Dash callbacks for:
- Auto operation config updates
- Button states during operations
- Operation lock reset
- Plate test status refresh
- Monitoring summary display
- Plate detection summary
- Verification button state
"""

from typing import Optional, Tuple

import dash_mantine_components as dmc
from dash import Dash, Input, Output, State, no_update


def register_monitoring_callbacks(app: Dash) -> None:
    """Register monitoring-related callbacks with the Dash app."""

    @app.callback(
        Output("auto-config-state", "data"),
        Input("auto-target-class", "value"),
        Input("auto-drop-ignore-radius", "value"),
        Input("plate-test-approach-height", "value"),
        Input("plate-test-pick-height", "value"),
        Input("plate-test-drop-height", "value"),
        Input("shared-tracking-mode", "value"),
        Input("shared-segment-min", "value"),
        Input("shared-segment-max", "value"),
        Input("shared-segment-fraction", "value"),
    )
    def _update_auto_config(
        target_label: Optional[str],
        drop_ignore_radius: Optional[float],
        approach_height: Optional[float],
        pick_height: Optional[float],
        drop_height: Optional[float],
        tracking_mode: Optional[str],
        segment_min: Optional[float],
        segment_max: Optional[float],
        segment_fraction: Optional[float],
    ) -> dict:
        """Update auto operation configuration from UI inputs."""
        import app.codroid_picker as codroid_picker_module
        from app.codroid_picker.session import SESSION

        config = {
            "target_label": (
                target_label or codroid_picker_module.SETTINGS.auto_target_label
            ),
            "drop_ignore_radius_mm": codroid_picker_module._safe_float(
                drop_ignore_radius, codroid_picker_module.SETTINGS.auto_drop_ignore_radius_mm
            ),
            "approach_mm": codroid_picker_module._safe_float(
                approach_height, codroid_picker_module.SETTINGS.auto_approach_mm
            ),
            "pick_mm": codroid_picker_module._safe_float(
                pick_height, codroid_picker_module.SETTINGS.auto_pick_mm
            ),
            "drop_mm": codroid_picker_module._safe_float(
                drop_height, codroid_picker_module.SETTINGS.auto_drop_mm
            ),
            "estimate_width": bool(
                SESSION.auto_config.get(
                    "estimate_width", codroid_picker_module.SETTINGS.auto_estimate_width
                )
            ),
            "tracking_mode": (tracking_mode or codroid_picker_module.SETTINGS.tracking_mode).strip().lower(),
            "segment_min_mm": codroid_picker_module._safe_float(
                segment_min, codroid_picker_module.SETTINGS.segment_min
            ),
            "segment_max_mm": codroid_picker_module._safe_float(
                segment_max, codroid_picker_module.SETTINGS.segment_max
            ),
            "segment_fraction": codroid_picker_module._safe_float(
                segment_fraction, codroid_picker_module.SETTINGS.segment_fraction
            ),
        }
        SESSION.auto_config.update(config)
        return config

    @app.callback(
        [
            Output("manual-pick-closest", "disabled"),
            Output("execute-drop", "disabled"),
            Output("manual-pick-place-target", "disabled"),
            Output("reset-operation-lock", "disabled"),
        ],
        Input("monitor-interval", "n_intervals"),
        prevent_initial_call=True,
    )
    def _update_button_states(_n: int) -> Tuple[bool, bool, bool, bool]:
        """Disable buttons during operations to prevent rapid clicking.

        Includes a 120-second watchdog to auto-reset stuck operations.
        """
        import time
        import logging
        from app.codroid_picker.session import SESSION

        # Watchdog: auto-reset if operation has been running for more than 120 seconds
        if SESSION.operation_in_progress and SESSION.operation_started_at is not None:
            elapsed = time.monotonic() - SESSION.operation_started_at
            if elapsed > 120.0:
                logging.getLogger(__name__).warning(
                    "Auto-resetting stuck operation_in_progress flag after %.1f seconds",
                    elapsed,
                )
                SESSION.operation_in_progress = False
                SESSION.operation_started_at = None

        disabled = SESSION.operation_in_progress
        # Only enable reset when operation is in progress
        reset_disabled = not SESSION.operation_in_progress
        return disabled, disabled, disabled, reset_disabled

    @app.callback(
        Output("pick-place-status", "children", allow_duplicate=True),
        Input("reset-operation-lock", "n_clicks"),
        prevent_initial_call=True,
    )
    def _reset_operation_lock(_clicks: int) -> str:
        """Emergency reset for stuck operations."""
        import app.codroid_picker as codroid_picker_module
        from app.codroid_picker.session import SESSION

        SESSION.operation_in_progress = False
        SESSION.operation_started_at = None
        codroid_picker_module.LOGGER.info(
            "Operation lock manually reset by user")
        return "Operation lock reset."

    @app.callback(
        Output("plate-test-status", "children", allow_duplicate=True),
        Input("monitor-interval", "n_intervals"),
        prevent_initial_call=True,
    )
    def _refresh_plate_test_status(_ticks: int) -> str:
        """Refresh plate test status from session state."""
        from app.codroid_picker.session import SESSION

        state = SESSION.plate_test_state or {}
        if state.get("running"):
            return state.get("message", "")
        return no_update

    @app.callback(
        Output("monitor-system-summary", "children"),
        Output("monitor-detection-summary", "children"),
        Output("monitor-pick-summary", "children"),
        Output("use-existing-calibration", "style"),
        Output("use-existing-workspace", "style"),
        Output("use-existing-calibration", "disabled"),
        Output("use-existing-robot-calibration", "disabled"),
        Input("monitor-interval", "n_intervals"),
        State("camera-connection-state", "data"),
        State("robot-connection-state", "data"),
        State("gripper-connection-state", "data"),
        State("robot-power-state", "data"),
        State("robot-speed-state", "data"),
        State("settings-calib-path", "value"),
        State("settings-robot-calib-path", "value"),
    )
    def _update_monitoring_summary(
        _monitor_ticks: int,
        camera_state: Optional[dict],
        robot_state: Optional[dict],
        gripper_state: Optional[dict],
        robot_power: Optional[dict],
        robot_speed: Optional[dict],
        bundle_path_value: Optional[str],
        robot_calib_path_value: Optional[str],
    ) -> Tuple[list, list, list, dict, dict, bool, bool]:
        """Update monitoring summary panels with system, detection, and pick info."""
        import app.codroid_picker as codroid_picker_module
        from app.codroid_picker.session import SESSION

        camera_status = (camera_state or {}).get("status", "off")
        robot_status = (robot_state or {}).get("status", "offline")
        gripper_status = (gripper_state or {}).get("status", "offline")
        calibration_state = (
            "loaded" if SESSION.calibration_bundle is not None else "missing"
        )
        workspace_state = (
            "ready" if SESSION.workspace_definition is not None else "pending"
        )
        plate_state = "ready" if SESSION.plate_calibration is not None else "pending"
        model_state = (
            SESSION.active_model_path.name if SESSION.active_model_path else "none"
        )
        detection_count = len(SESSION.detections)
        last_action = SESSION.last_action or "No recent actions."

        stream = SESSION.camera_stream
        stream_fps = stream.get_fps() if stream else 0.0
        stream_cfg_fps = (
            stream.config.fps if stream else codroid_picker_module.DEFAULT_FPS
        )
        res_w, res_h = (
            stream.config.resolution
            if stream
            else codroid_picker_module.DEFAULT_RESOLUTION
        )

        system_items = [
            dmc.Text(
                f"Camera: {camera_status} · {res_w}x{res_h} @ {stream_fps:.1f}/{stream_cfg_fps:g} FPS",
                size="sm",
            ),
            dmc.Text(
                f"Robot: {robot_status} · speed {int((robot_speed or {}).get('value', SESSION.robot_speed_rate*100))}%",
                size="sm",
            ),
            dmc.Text(
                f"Power: {('on' if (robot_power or {}).get('value') else 'off')}",
                size="sm",
            ),
            dmc.Text(f"Calibration bundle: {calibration_state}", size="sm"),
            dmc.Text(f"Workspace: {workspace_state}", size="sm"),
            dmc.Text(f"Plate calibration: {plate_state}", size="sm"),
            dmc.Text(
                f"Gripper: {gripper_status}",
                size="sm",
            ),
        ]

        detection_items = [
            dmc.Text(f"Model: {model_state}", size="sm"),
            dmc.Text(f"Detections: {detection_count}", size="sm"),
        ]
        if SESSION.last_detection_error:
            detection_items.append(
                dmc.Text(
                    f"Last error: {SESSION.last_detection_error}",
                    size="sm",
                    c="red",
                )
            )

        attempts = SESSION.pick_attempts
        successes = SESSION.pick_successes
        failures = SESSION.pick_failures
        processed = successes
        success_rate = (successes / attempts * 100.0) if attempts else None
        avg_cycle = (
            sum(SESSION.cycle_times) / len(SESSION.cycle_times)
            if SESSION.cycle_times
            else None
        )
        pick_items = [
            dmc.Text(
                f"Holding object: {'yes' if SESSION.robot_has_object else 'no'}",
                size="sm",
            ),
            dmc.Text(f"Last action: {last_action}", size="sm"),
            dmc.Text(f"Processed objects: {processed}", size="sm"),
            dmc.Text(
                f"Attempts: {attempts} · Failures: {failures}",
                size="sm",
            ),
            dmc.Text(
                f"Success rate: {success_rate:.1f}%"
                if success_rate is not None
                else "Success rate: n/a",
                size="sm",
            ),
            dmc.Text(
                f"Avg cycle: {avg_cycle:.2f}s"
                if avg_cycle is not None
                else "Avg cycle: n/a",
                size="sm",
            ),
            dmc.Text(
                f"Last cycle: {SESSION.last_cycle_s:.2f}s"
                if SESSION.last_cycle_s is not None
                else "Last cycle: n/a",
                size="sm",
            ),
        ]
        if SESSION.last_pick_mm:
            pick_items.append(
                dmc.Text(
                    f"Last pick: {SESSION.last_pick_mm[0]:.1f}, {SESSION.last_pick_mm[1]:.1f} mm",
                    size="sm",
                )
            )
        if SESSION.drop_target_mm:
            pick_items.append(
                dmc.Text(
                    f"Drop target: {SESSION.drop_target_mm[0]:.1f}, {SESSION.drop_target_mm[1]:.1f} mm",
                    size="sm",
                )
            )
        if SESSION.drop_target_orientation:
            pick_items.append(
                dmc.Text(
                    "Drop orientation: "
                    f"a={SESSION.drop_target_orientation[0]:.1f}, "
                    f"b={SESSION.drop_target_orientation[1]:.1f}, "
                    f"c={SESSION.drop_target_orientation[2]:.1f}",
                    size="sm",
                )
            )
        if SESSION.drop_target_robot:
            pick_items.append(
                dmc.Text(
                    "Drop robot pose: "
                    f"x={SESSION.drop_target_robot[0]:.1f}, "
                    f"y={SESSION.drop_target_robot[1]:.1f}, "
                    f"z={SESSION.drop_target_robot[2]:.1f}",
                    size="sm",
                )
            )
        bundle_path = codroid_picker_module._safe_path(
            bundle_path_value, codroid_picker_module.DEFAULT_BUNDLE_PATH
        )
        bundle_style = {}
        workspace_style = (
            {}
            if codroid_picker_module.DEFAULT_WORKSPACE_PATH.exists()
            else {"display": "none"}
        )
        bundle_disabled = not bundle_path.exists()
        robot_path = codroid_picker_module._safe_path(
            robot_calib_path_value, codroid_picker_module.DEFAULT_ROBOT_CALIBRATION_PATH
        )
        robot_disabled = not robot_path.exists()

        return (
            system_items,
            detection_items,
            pick_items,
            bundle_style,
            workspace_style,
            bundle_disabled,
            robot_disabled,
        )

    @app.callback(
        Output("plate-detection-summary", "children"),
        Output("plate-detection-summary", "color"),
        Output("start-robot-calibration", "disabled"),
        Output("start-robot-calibration", "variant"),
        Input("plate-detection-state", "data"),
        Input("robot-connection-state", "data"),
        State("robot-calibration-state", "data"),
    )
    def _update_plate_summary(
        plate_state: Optional[dict],
        robot_conn_state: Optional[dict],
        calib_state: Optional[dict],
    ) -> Tuple[str, str, bool, str]:
        """Update plate detection summary and calibration button state."""
        import app.codroid_picker as codroid_picker_module
        from app.codroid_picker import state

        plate_data = plate_state or {}
        count = plate_data.get("count", 0)
        ok = bool(plate_data.get("ok"))
        stable = bool(plate_data.get("stable", ok))
        calib_status = (calib_state or {}).get("status")
        in_progress = calib_status in {
            "running",
            "awaiting_verification",
            "verifying",
        }
        if state.get_calibration_guidance_snapshot().get("status") in {
            "running",
            "awaiting_verification",
            "verifying",
        }:
            in_progress = True
        robot_online = (robot_conn_state or {}).get("status") == "online"
        posture_ready, posture_message = codroid_picker_module._robot_posture_ready()

        # Determine message based on the actual blocking issue
        if in_progress:
            if calib_status == "awaiting_verification":
                message = "Awaiting verification move."
                color = "yellow"
            elif calib_status == "verifying":
                message = "Verification in progress..."
                color = "blue"
            else:
                message = "Calibration in progress..."
                color = "blue"
        elif not robot_online:
            message = (
                "Robot not connected. Click 'Test robot connection' to establish connection."
            )
            color = "red"
        elif not ok:
            message = plate_data.get(
                "message") or f"Detected {count} marker(s)."
            color = "yellow"
        elif not stable:
            message = plate_data.get(
                "message"
            ) or "Calibration plate moved; hold it steady."
            color = "yellow"
        elif not posture_ready:
            message = posture_message
            color = "yellow"
        else:
            message = "Plate markers stable. Ready to calibrate."
            color = "green"

        disabled = (
            not ok or not stable or not posture_ready or in_progress or not robot_online
        )
        variant = "filled" if not disabled else "light"
        return message, color, disabled, variant

    @app.callback(
        Output("run-verification-inline", "disabled"),
        Input("plate-detection-state", "data"),
        Input("robot-calibration-state", "data"),
        Input("robot-connection-state", "data"),
    )
    def _sync_verification_inline_button(
        plate_state: Optional[dict],
        calibration_state: Optional[dict],
        robot_state: Optional[dict],
    ) -> bool:
        """Sync inline verification button disabled state."""
        from app.codroid_picker.session import SESSION

        plate_ok = bool((plate_state or {}).get("ok"))
        plate_stable = bool((plate_state or {}).get("stable", plate_ok))
        guidance = calibration_state or {}
        status = guidance.get("status")
        in_progress = status in {"running",
                                 "awaiting_verification", "verifying"}
        robot_online = (robot_state or {}).get("status") == "online"
        has_calibration_points = len(SESSION.robot_calibration_points) >= 3
        return not (
            plate_ok
            and plate_stable
            and robot_online
            and has_calibration_points
            and not in_progress
        )
