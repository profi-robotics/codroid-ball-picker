"""Settings save/reset callbacks.

This module contains Dash callbacks for:
- Settings save and reset operations
"""

from typing import Optional, Tuple

from dash import Dash, Input, Output, State, callback_context
from app.codroid_picker.i18n import t


def register_settings_callbacks(app: Dash) -> None:
    """Register settings callbacks with the Dash app."""

    @app.callback(
        Output("settings-status", "children"),
        Output("settings-robot-host", "value"),
        Output("settings-robot-coordinate-id", "value"),
        Output("settings-gripper-ip", "value"),
        Output("settings-gripper-type", "value"),
        Output("settings-marker-a", "value"),
        Output("settings-marker-b", "value"),
        Output("settings-marker-size", "value"),
        Output("settings-calib-path", "value"),
        Output("settings-workspace-path", "value"),
        Output("settings-robot-calib-path", "value"),
        Output("settings-auto-target-label", "value"),
        Output("settings-auto-drop-ignore-radius", "value"),
        Output("settings-auto-intrusion-match", "value"),
        Output("settings-auto-detection-stale", "value"),
        Output("settings-auto-idle-sleep", "value"),
        Output("settings-auto-estimate-width", "checked"),
        Output("settings-auto-approach-height", "value"),
        Output("settings-auto-pick-height", "value"),
        Output("settings-auto-drop-height", "value"),
        Output("settings-camera-calib-frames", "value"),
        Output("settings-camera-calib-delay", "value"),
        Output("settings-camera-calib-min-usable", "value"),
        Output("settings-pick-follow-timeout", "value"),
        Output("settings-pick-follow-interval", "value"),
        Output("settings-pick-follow-segment", "value"),
        Output("settings-pick-follow-settle", "value"),
        Output("settings-pick-follow-tolerance", "value"),
        Output("settings-rescue-jog-hold", "value"),
        Output("settings-coord5-max-z", "value"),
        Input("settings-save", "n_clicks"),
        Input("settings-reset", "n_clicks"),
        State("settings-robot-host", "value"),
        State("settings-robot-coordinate-id", "value"),
        State("settings-gripper-ip", "value"),
        State("settings-gripper-type", "value"),
        State("settings-marker-a", "value"),
        State("settings-marker-b", "value"),
        State("settings-marker-size", "value"),
        State("settings-calib-path", "value"),
        State("settings-workspace-path", "value"),
        State("settings-robot-calib-path", "value"),
        State("settings-auto-target-label", "value"),
        State("settings-auto-drop-ignore-radius", "value"),
        State("settings-auto-intrusion-match", "value"),
        State("settings-auto-detection-stale", "value"),
        State("settings-auto-idle-sleep", "value"),
        State("settings-auto-estimate-width", "checked"),
        State("settings-auto-approach-height", "value"),
        State("settings-auto-pick-height", "value"),
        State("settings-auto-drop-height", "value"),
        State("settings-camera-calib-frames", "value"),
        State("settings-camera-calib-delay", "value"),
        State("settings-camera-calib-min-usable", "value"),
        State("settings-pick-follow-timeout", "value"),
        State("settings-pick-follow-interval", "value"),
        State("settings-pick-follow-segment", "value"),
        State("settings-pick-follow-settle", "value"),
        State("settings-pick-follow-tolerance", "value"),
        State("settings-rescue-jog-hold", "value"),
        State("settings-coord5-max-z", "value"),
        prevent_initial_call=True,
    )
    def _handle_settings(
        _save: int,
        _reset: int,
        robot_host: Optional[str],
        robot_coordinate_id: Optional[float],
        gripper_ip: Optional[str],
        gripper_type: Optional[str],
        marker_a: Optional[float],
        marker_b: Optional[float],
        marker_size: Optional[float],
        calib_path: Optional[str],
        workspace_path: Optional[str],
        robot_calib_path: Optional[str],
        auto_target_label: Optional[str],
        auto_drop_ignore_radius: Optional[float],
        auto_intrusion_match: Optional[float],
        auto_detection_stale: Optional[float],
        auto_idle_sleep: Optional[float],
        auto_estimate_width: Optional[bool],
        auto_approach_height: Optional[float],
        auto_pick_height: Optional[float],
        auto_drop_height: Optional[float],
        camera_calib_frames: Optional[float],
        camera_calib_delay: Optional[float],
        camera_calib_min_usable: Optional[float],
        pick_follow_timeout: Optional[float],
        pick_follow_interval: Optional[float],
        pick_follow_segment: Optional[float],
        pick_follow_settle: Optional[float],
        pick_follow_tolerance: Optional[float],
        rescue_jog_hold: Optional[float],
        coord5_max_z: Optional[float],
    ) -> Tuple[
        str,
        Optional[str],
        int,
        Optional[str],
        Optional[str],
        Optional[int],
        Optional[int],
        Optional[float],
        str,
        str,
        str,
        str,
        float,
        float,
        float,
        float,
        bool,
        float,
        float,
        float,
        int,
        float,
        int,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
    ]:
        """Handle settings save and reset operations."""
        import app.codroid_picker as codroid_picker_module
        import app.settings as app_settings
        from app.codroid_picker.session import SESSION, _refresh_session_auto_config_defaults
        from onrobot.gripper_profiles import DEFAULT_GRIPPER_TYPE

        trigger = callback_context.triggered_id

        default_marker_a = 1
        default_marker_b = 2
        default_marker_size = 40.0
        default_calib = str(codroid_picker_module.DEFAULT_BUNDLE_PATH)
        default_workspace = str(codroid_picker_module.DEFAULT_WORKSPACE_PATH)
        default_robot_calib = str(
            codroid_picker_module.DEFAULT_ROBOT_CALIBRATION_PATH)
        default_gripper_ip = app_settings.DEFAULT_GRIPPER_IP
        default_gripper_type = DEFAULT_GRIPPER_TYPE
        default_auto_target_label = app_settings._BASE_AUTO_TARGET_LABEL
        default_auto_drop_ignore_radius = app_settings._BASE_AUTO_DROP_IGNORE_RADIUS_MM
        default_auto_intrusion_match = app_settings._BASE_AUTO_INTRUSION_MATCH_MM
        default_auto_detection_stale = app_settings._BASE_AUTO_DETECTION_STALE_S
        default_auto_idle_sleep = app_settings._BASE_AUTO_IDLE_SLEEP_S
        default_auto_estimate_width = app_settings._BASE_AUTO_ESTIMATE_WIDTH
        default_auto_approach_height = app_settings._BASE_AUTO_APPROACH_MM
        default_auto_pick_height = app_settings._BASE_AUTO_PICK_MM
        default_auto_drop_height = app_settings._BASE_AUTO_DROP_MM

        default_camera_calib_frames = app_settings._BASE_CAMERA_CALIBRATION_FRAMES
        default_camera_calib_delay = app_settings._BASE_CAMERA_CALIBRATION_DELAY_S
        default_camera_calib_min_usable = app_settings._BASE_CAMERA_CALIBRATION_MIN_USABLE_FRAMES
        default_pick_follow_timeout = app_settings._BASE_PICK_FOLLOW_TIMEOUT_S
        default_pick_follow_interval = app_settings._BASE_PICK_FOLLOW_UPDATE_S
        default_pick_follow_segment = app_settings._BASE_PICK_FOLLOW_SEGMENT_S
        default_pick_follow_settle = app_settings._BASE_PICK_FOLLOW_SETTLE_S
        default_pick_follow_tolerance = app_settings._BASE_PICK_FOLLOW_TOLERANCE_MM
        default_rescue_jog_hold = app_settings._BASE_RESCUE_JOG_HOLD_S
        default_coord5_max_z = app_settings._BASE_COORD5_MAX_Z_MM
        default_robot_coordinate_id = app_settings._BASE_ROBOT_COORDINATE_ID

        status_text = t("Settings saved.")
        if trigger == "settings-reset":
            status_text = t("Settings reset to defaults (env + built-ins).")
            host_val = None
            gripper_ip_val = None
            gripper_type_val = default_gripper_type
            marker_a_val = default_marker_a
            marker_b_val = default_marker_b
            marker_size_val = default_marker_size
            calib_path_val = default_calib
            workspace_path_val = default_workspace
            robot_calib_path_val = default_robot_calib
            target_label_val = default_auto_target_label
            drop_ignore_radius_val = default_auto_drop_ignore_radius
            intrusion_match_val = default_auto_intrusion_match
            detection_stale_val = default_auto_detection_stale
            idle_sleep_val = default_auto_idle_sleep
            estimate_width_val = default_auto_estimate_width
            approach_height_val = default_auto_approach_height
            pick_height_val = default_auto_pick_height
            drop_height_val = default_auto_drop_height
            robot_coordinate_id_val = default_robot_coordinate_id
            pick_follow_timeout_val = default_pick_follow_timeout
            pick_follow_interval_val = default_pick_follow_interval
            pick_follow_segment_val = default_pick_follow_segment
            pick_follow_settle_val = default_pick_follow_settle
            pick_follow_tolerance_val = default_pick_follow_tolerance
            rescue_jog_hold_val = default_rescue_jog_hold
            coord5_max_z_val = default_coord5_max_z
        else:
            host_val = (robot_host or "").strip() or None
            gripper_ip_val = (gripper_ip or "").strip() or None
            gripper_type_val = (
                gripper_type or "").strip() or default_gripper_type
            if gripper_type_val not in codroid_picker_module.GRIPPER_PROFILE_KEYS:
                gripper_type_val = default_gripper_type
            marker_a_val = int(
                marker_a) if marker_a is not None else default_marker_a
            marker_b_val = int(
                marker_b) if marker_b is not None else default_marker_b
            marker_size_val = (
                float(marker_size) if marker_size is not None else default_marker_size
            )
            calib_path_val = (calib_path or "").strip() or default_calib
            workspace_path_val = (
                workspace_path or "").strip() or default_workspace
            robot_calib_path_val = (
                robot_calib_path or "").strip() or default_robot_calib
            target_label_val = (auto_target_label or "").strip(
            ) or default_auto_target_label
            drop_ignore_radius_val = codroid_picker_module._safe_float(
                auto_drop_ignore_radius, default_auto_drop_ignore_radius
            )
            intrusion_match_val = codroid_picker_module._safe_float(
                auto_intrusion_match, default_auto_intrusion_match
            )
            detection_stale_val = codroid_picker_module._safe_float(
                auto_detection_stale, default_auto_detection_stale
            )
            idle_sleep_val = codroid_picker_module._safe_float(
                auto_idle_sleep, default_auto_idle_sleep
            )
            estimate_width_val = (
                bool(auto_estimate_width)
                if auto_estimate_width is not None
                else default_auto_estimate_width
            )
            approach_height_val = codroid_picker_module._safe_float(
                auto_approach_height, default_auto_approach_height
            )
            pick_height_val = codroid_picker_module._safe_float(
                auto_pick_height, default_auto_pick_height
            )
            drop_height_val = codroid_picker_module._safe_float(
                auto_drop_height, default_auto_drop_height
            )
            robot_coordinate_id_val = codroid_picker_module._safe_int(
                robot_coordinate_id, default_robot_coordinate_id
            )
            pick_follow_timeout_val = codroid_picker_module._safe_float(
                pick_follow_timeout, default_pick_follow_timeout
            )
            pick_follow_interval_val = codroid_picker_module._safe_float(
                pick_follow_interval, default_pick_follow_interval
            )
            pick_follow_segment_val = codroid_picker_module._safe_float(
                pick_follow_segment, default_pick_follow_segment
            )
            pick_follow_settle_val = codroid_picker_module._safe_float(
                pick_follow_settle, default_pick_follow_settle
            )
            pick_follow_tolerance_val = codroid_picker_module._safe_float(
                pick_follow_tolerance, default_pick_follow_tolerance
            )
            rescue_jog_hold_val = codroid_picker_module._safe_float(
                rescue_jog_hold, default_rescue_jog_hold
            )
            coord5_max_z_val = codroid_picker_module._safe_float(
                coord5_max_z, default_coord5_max_z
            )

        codroid_picker_module._commit_settings(
            robot_host=host_val,
            gripper_ip=gripper_ip_val,
            gripper_type=gripper_type_val,
            marker_ids=(marker_a_val, marker_b_val),
            marker_size_mm=marker_size_val,
            calibration_bundle_path=calib_path_val,
            workspace_path=workspace_path_val,
            robot_calibration_path=robot_calib_path_val,
            auto_target_label=target_label_val,
            auto_drop_ignore_radius_mm=drop_ignore_radius_val,
            auto_intrusion_match_mm=intrusion_match_val,
            auto_detection_stale_s=detection_stale_val,
            auto_idle_sleep_s=idle_sleep_val,
            auto_estimate_width=estimate_width_val,
            auto_approach_mm=approach_height_val,
            auto_pick_mm=pick_height_val,
            auto_drop_mm=drop_height_val,
            pick_follow_timeout_s=pick_follow_timeout_val,
            pick_follow_update_interval_s=pick_follow_interval_val,
            pick_follow_segment_hold_s=pick_follow_segment_val,
            pick_follow_settle_s=pick_follow_settle_val,
            pick_follow_tolerance_mm=pick_follow_tolerance_val,
            rescue_jog_hold_s=rescue_jog_hold_val,
            coord5_max_z_mm=coord5_max_z_val,
            robot_coordinate_id=robot_coordinate_id_val,
        )

        # Refresh global settings
        app_settings.reload_defaults()
        codroid_picker_module.CODROID_DEFAULTS = app_settings.CODROID_DEFAULTS
        codroid_picker_module.DEFAULT_ROBOT_URI = app_settings.DEFAULT_ROBOT_URI
        codroid_picker_module.GRIPPER_DEFAULTS = app_settings.GRIPPER_DEFAULTS
        codroid_picker_module.DEFAULT_GRIPPER_IP = app_settings.DEFAULT_GRIPPER_IP
        codroid_picker_module.refresh_gripper_constants()
        _refresh_session_auto_config_defaults()
        codroid_picker_module.set_asset_paths(
            calibration_bundle_path=calib_path_val,
            workspace_path=workspace_path_val,
            robot_calibration_path=robot_calib_path_val,
        )
        codroid_picker_module.DEFAULT_BUNDLE_PATH = app_settings.DEFAULT_BUNDLE_PATH
        codroid_picker_module.DEFAULT_WORKSPACE_PATH = app_settings.DEFAULT_WORKSPACE_PATH
        codroid_picker_module.DEFAULT_ROBOT_CALIBRATION_PATH = (
            app_settings.DEFAULT_ROBOT_CALIBRATION_PATH
        )
        SESSION.marker_ids = (marker_a_val, marker_b_val)
        SESSION.marker_size_mm = marker_size_val
        codroid_picker_module.ROBOT_COORDINATE_ID = codroid_picker_module.SETTINGS.robot_coordinate_id

        host_display = host_val or codroid_picker_module.CODROID_DEFAULTS.get(
            "host")
        gripper_ip_display = gripper_ip_val or app_settings.DEFAULT_GRIPPER_IP

        return (
            status_text,
            host_display,
            robot_coordinate_id_val,
            gripper_ip_display,
            gripper_type_val,
            marker_a_val,
            marker_b_val,
            marker_size_val,
            calib_path_val,
            workspace_path_val,
            robot_calib_path_val,
            target_label_val,
            drop_ignore_radius_val,
            intrusion_match_val,
            detection_stale_val,
            idle_sleep_val,
            estimate_width_val,
            approach_height_val,
            pick_height_val,
            drop_height_val,
            camera_calib_frames,
            camera_calib_delay,
            camera_calib_min_usable,
            pick_follow_timeout_val,
            pick_follow_interval_val,
            pick_follow_segment_val,
            pick_follow_settle_val,
            pick_follow_tolerance_val,
            rescue_jog_hold_val,
            coord5_max_z_val,
        )
