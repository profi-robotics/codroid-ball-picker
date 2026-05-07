# Codroid Picker package

import importlib
import sys
from types import ModuleType


_LAZY_EXPORTS = {
    # Camera calibration
    "get_latest_frame": ("app.codroid_picker.camera_calibration", "get_latest_frame"),
    "load_calibration_bundle": ("app.codroid_picker.camera_calibration", "load_calibration_bundle"),
    "load_workspace_definition": ("app.codroid_picker.camera_calibration", "load_workspace_definition"),
    "compute_workspace_from_latest": (
        "app.codroid_picker.camera_calibration",
        "compute_workspace_from_latest",
    ),
    "_clear_workspace_definition": (
        "app.codroid_picker.camera_calibration",
        "_clear_workspace_definition",
    ),
    "capture_color_frame": ("app.codroid_picker.camera_calibration", "capture_color_frame"),
    "capture_depth_snapshot": ("app.codroid_picker.camera_calibration", "capture_depth_snapshot"),
    "capture_depth_snapshot_isolated": (
        "app.codroid_picker.camera_calibration",
        "capture_depth_snapshot_isolated",
    ),
    "compute_calibration_bundle": (
        "app.codroid_picker.camera_calibration",
        "compute_calibration_bundle",
    ),
    "compute_plate_calibration_from_latest": (
        "app.codroid_picker.camera_calibration",
        "compute_plate_calibration_from_latest",
    ),
    # Persistence + plate state
    "_save_drop_target": ("app.codroid_picker.persistence", "_save_drop_target"),
    "_plate_is_stable": ("app.codroid_picker.persistence", "_plate_is_stable"),
    "_set_plate_test_state": ("app.codroid_picker.persistence", "_set_plate_test_state"),
    "_update_robot_calibration_meta": (
        "app.codroid_picker.persistence",
        "_update_robot_calibration_meta",
    ),
    "_resolve_plate_test_targets": (
        "app.codroid_picker.plate_targets",
        "_resolve_plate_test_targets",
    ),
    "_resolve_plate_test_coord5_targets": (
        "app.codroid_picker.plate_targets",
        "_resolve_plate_test_coord5_targets",
    ),
    # Robot plate calibration
    "_apply_robot_calibration_results": (
        "app.codroid_picker.robot_plate_calibration",
        "_apply_robot_calibration_results",
    ),
    "_ensure_plate_stable": (
        "app.codroid_picker.robot_plate_calibration",
        "_ensure_plate_stable",
    ),
    # Robot procedures
    "_get_robot_session": ("app.codroid_picker.procedures", "_get_robot_session"),
    "_current_coordinate_id": ("app.codroid_picker.procedures", "_current_coordinate_id"),
    "_robot_to_workspace_mm": ("app.codroid_picker.procedures", "_robot_to_workspace_mm"),
    "_robot_posture_ready": ("app.codroid_picker.procedures", "_robot_posture_ready"),
    "_ensure_coordinate5_active": ("app.codroid_picker.procedures", "_ensure_coordinate5_active"),
    "_set_robot_guidance": ("app.codroid_picker.procedures", "_set_robot_guidance"),
    "_clear_robot_accuracy_guidance": (
        "app.codroid_picker.procedures",
        "_clear_robot_accuracy_guidance",
    ),
    "_load_robot_calibration": ("app.codroid_picker.procedures", "_load_robot_calibration"),
    "_acknowledge_robot_error": (
        "app.codroid_picker.procedures",
        "_acknowledge_robot_error",
    ),
    "_ensure_posture_axes": ("app.codroid_picker.procedures", "_ensure_posture_axes"),
    # Robot error detection
    "_detect_emergency_stop_status": (
        "app.codroid_picker.error_detection",
        "_detect_emergency_stop_status",
    ),
    "_detect_overspeed_status": (
        "app.codroid_picker.error_detection",
        "_detect_overspeed_status",
    ),
    "_detect_joint_protection_status": (
        "app.codroid_picker.error_detection",
        "_detect_joint_protection_status",
    ),
    "_detect_wrong_tool_status": (
        "app.codroid_picker.error_detection",
        "_detect_wrong_tool_status",
    ),
    "_detect_drag_not_allowed_status": (
        "app.codroid_picker.error_detection",
        "_detect_drag_not_allowed_status",
    ),
    "_detect_rescue_mode_status": (
        "app.codroid_picker.error_detection",
        "_detect_rescue_mode_status",
    ),
    "_activate_robot_error": (
        "app.codroid_picker.error_detection",
        "_activate_robot_error",
    ),
    "_error_snapshot": ("app.codroid_picker.error_detection", "_error_snapshot"),
    "_update_robot_error_state": (
        "app.codroid_picker.error_detection",
        "_update_robot_error_state",
    ),
    # Robot utilities + transforms
    "_probe_robot": ("app.codroid_picker.robot_utils", "_probe_robot"),
    "_with_robot_api": ("app.codroid_picker.robot_utils", "_with_robot_api"),
    "_get_robot_api": ("app.codroid_picker.robot_utils", "_get_robot_api"),
    "_ensure_robot_coord5_for_uri": (
        "app.codroid_picker.robot_utils",
        "_ensure_robot_coord5_for_uri",
    ),
    "_robot_control_mode": ("app.codroid_picker.robot_utils", "_robot_control_mode"),
    "_robot_control_released": (
        "app.codroid_picker.robot_utils",
        "_robot_control_released",
    ),
    "_log_move_target": ("app.codroid_picker.robot_utils", "_log_move_target"),
    "_coord5_to_workspace_mm": ("app.codroid_picker.transforms", "_coord5_to_workspace_mm"),
    "_pose_from_coord5_mm": ("app.codroid_picker.transforms", "_pose_from_coord5_mm"),
    # Operation state
    "_begin_operation": ("app.codroid_picker.state", "_begin_operation"),
    "_finish_operation": ("app.codroid_picker.state", "_finish_operation"),
    "_operation_snapshot": ("app.codroid_picker.state", "_operation_snapshot"),
    "_pause_operation": ("app.codroid_picker.state", "_pause_operation"),
    "_update_operation_message": (
        "app.codroid_picker.state",
        "_update_operation_message",
    ),
    "_request_stop_operation": ("app.codroid_picker.state", "_request_stop_operation"),
    "_reset_operation_state": ("app.codroid_picker.state", "_reset_operation_state"),
    "_resume_operation": ("app.codroid_picker.state", "resume_operation"),
    # Calibration menu helpers
    "_calibration_data_complete": (
        "app.codroid_picker.calibration_menu",
        "_calibration_data_complete",
    ),
    "_calibration_metadata_entries": (
        "app.codroid_picker.calibration_menu",
        "_calibration_metadata_entries",
    ),
    "_reset_calibration_assets": (
        "app.codroid_picker.calibration_menu",
        "_reset_calibration_assets",
    ),
    # Gripper helpers
    "refresh_gripper_constants": (
        "app.codroid_picker.gripper_utils",
        "refresh_gripper_constants",
    ),
    "_record_gripper_sync": (
        "app.codroid_picker.gripper_utils",
        "_record_gripper_sync",
    ),
    "_recent_gripper_sync": (
        "app.codroid_picker.gripper_utils",
        "_recent_gripper_sync",
    ),
    "_restore_calibration_gripper_width": (
        "app.codroid_picker.gripper_utils",
        "_restore_calibration_gripper_width",
    ),
    "_resolve_gripper_limits": (
        "app.codroid_picker.gripper_utils",
        "_resolve_gripper_limits",
    ),
    "_command_gripper": ("app.codroid_picker.gripper_utils", "_command_gripper"),
    "_prepare_gripper_for_calibration": (
        "app.codroid_picker.gripper_utils",
        "_prepare_gripper_for_calibration",
    ),
    # Models + GPU helpers
    "_model_options": ("app.codroid_picker.models", "_model_options"),
    "model_label_options": ("app.codroid_picker.models", "model_label_options"),
    "run_model_benchmark": ("app.codroid_picker.models", "run_model_benchmark"),
    "format_model_benchmark": (
        "app.codroid_picker.models",
        "format_model_benchmark",
    ),
    "activate_model": ("app.codroid_picker.models", "activate_model"),
    "save_uploaded_model": ("app.codroid_picker.models", "save_uploaded_model"),
    "_list_capture_frames": ("app.codroid_picker.models", "_list_capture_frames"),
    "format_gpu_status": ("app.codroid_picker.gpu_utils", "format_gpu_status"),
    # Auto operation + overlays
    "_format_detection_list": ("app.codroid_picker.auto_operation", "_format_detection_list"),
    "_run_detection": ("app.codroid_picker.auto_operation", "_run_detection"),
    "_detection_coord5_xy": ("app.codroid_picker.auto_operation", "_detection_coord5_mm"),
    "_select_overlay_task": (
        "app.codroid_picker.overlay_tasks",
        "_select_overlay_task",
    ),
    "_run_charuco_task": ("app.codroid_picker.overlay_tasks", "_run_charuco_task"),
    "_run_workspace_task": ("app.codroid_picker.overlay_tasks", "_run_workspace_task"),
    "_run_plate_task": ("app.codroid_picker.overlay_tasks", "_run_plate_task"),
    "_run_detection_task": ("app.codroid_picker.overlay_tasks", "_run_detection_task"),
    "_run_drop_marker_task": ("app.codroid_picker.overlay_tasks", "_run_drop_marker_task"),
    "_run_auto_plan_task": ("app.codroid_picker.overlay_tasks", "_run_auto_plan_task"),
    # Settings helpers
    "_save_settings": ("app.settings", "_save_settings"),
    "_commit_settings": ("app.settings", "_commit_settings"),
    "set_asset_paths": ("app.settings", "set_asset_paths"),
    "reload_defaults": ("app.settings", "reload_defaults"),
    # Safe type helpers
    "_safe_int": ("app.codroid_picker.utils.safe_types", "safe_int"),
    "_safe_float": ("app.codroid_picker.utils.safe_types", "safe_float"),
    "_safe_path": ("app.codroid_picker.utils.safe_types", "safe_path"),
    # Dash and Mantine
    "callback_context": ("dash", "callback_context"),
    "no_update": ("dash", "no_update"),
    "dmc": ("dash_mantine_components", None),
    # API classes and exceptions
    "CodroidAPI": ("codroid_api", "CodroidAPI"),
    "RobotSession": ("codroid_api", "RobotSession"),
    "DEFAULT_FLANGE_BUTTON_PORTS": ("codroid_api", "DEFAULT_FLANGE_BUTTON_PORTS"),
    "OperationAbort": ("app.codroid_picker.exceptions", "OperationAbort"),
    # Session exports
    "SESSION": ("app.codroid_picker.session", "SESSION"),
    "GripperDevice": ("app.codroid_picker.session", "GripperDevice"),
    "TwoFG7Gripper": ("app.codroid_picker.session", "TwoFG7Gripper"),
    # Settings/constants (fallbacks if codroid_picker_main is unavailable)
    "SETTINGS": ("app.settings", "SETTINGS"),
    "CODROID_DEFAULTS": ("app.settings", "CODROID_DEFAULTS"),
    "DEFAULT_BUNDLE_PATH": ("app.settings", "DEFAULT_BUNDLE_PATH"),
    "DEFAULT_WORKSPACE_PATH": ("app.settings", "DEFAULT_WORKSPACE_PATH"),
    "DEFAULT_ROBOT_CALIBRATION_PATH": ("app.settings", "DEFAULT_ROBOT_CALIBRATION_PATH"),
    "DEFAULT_PLATE_PATH": ("app.settings", "DEFAULT_PLATE_PATH"),
    "DEFAULT_RESOLUTION": ("app.settings", "DEFAULT_RESOLUTION"),
    "DEFAULT_FPS": ("app.settings", "DEFAULT_FPS"),
    "DEFAULT_MODEL_DIR": ("app.settings", "DEFAULT_MODEL_DIR"),
    "DEFAULT_ROBOT_URI": ("app.settings", "DEFAULT_ROBOT_URI"),
    "GRIPPER_DEFAULTS": ("app.settings", "GRIPPER_DEFAULTS"),
    "DEFAULT_GRIPPER_IP": ("app.settings", "DEFAULT_GRIPPER_IP"),
}

_EXTRA_EXPORT_MODULES = (
    "app.codroid_picker.session",
    "app.codroid_picker.camera_calibration",
    "app.codroid_picker.persistence",
    "app.codroid_picker.robot_plate_calibration",
    "app.codroid_picker.procedures",
    "app.codroid_picker.error_detection",
    "app.codroid_picker.robot_utils",
    "app.codroid_picker.transforms",
    "app.codroid_picker.state",
    "app.codroid_picker.calibration_menu",
    "app.codroid_picker.gripper_utils",
    "app.codroid_picker.models",
    "app.codroid_picker.gpu_utils",
    "app.codroid_picker.auto_operation",
    "app.codroid_picker.overlay_tasks",
    "app.codroid_picker.exceptions",
    "app.codroid_picker.utils.safe_types",
    "app.settings",
    "codroid_api",
    "dash",
    "dash_mantine_components",
)


def _load_module(module_name: str) -> ModuleType:
    module = sys.modules.get(module_name)
    if module is None:
        module = importlib.import_module(module_name)
    return module


def __getattr__(name: str):
    """Lazy attribute access for codroid_picker and exported helpers."""
    if "codroid_picker_main" in sys.modules:
        main_module = sys.modules["codroid_picker_main"]
        if hasattr(main_module, name):
            return getattr(main_module, name)

    export = _LAZY_EXPORTS.get(name)
    if export is not None:
        module_name, attr_name = export
        module = _load_module(module_name)
        if attr_name is None:
            return module
        if hasattr(module, attr_name):
            return getattr(module, attr_name)

    for module_name in _EXTRA_EXPORT_MODULES:
        module = sys.modules.get(module_name)
        if module is not None and hasattr(module, name):
            return getattr(module, name)

    raise AttributeError(f"module 'app.codroid_picker' has no attribute '{name}'")
