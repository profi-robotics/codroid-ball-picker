from __future__ import annotations

import numpy as np
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Deque, Optional, Tuple

from onrobot.device import Device as GripperDevice
from onrobot.twofg import TWOFG as TwoFG7Gripper
from vision import calibration, plate, workspace
from vision.stream import DepthAIFrameStream

import app.settings as app_settings
from app.settings import SETTINGS
from codroid_api.robot_session import RobotSession


@dataclass
class CaptureState:
    output_dir: Path = app_settings.DEFAULT_CAPTURE_DIR
    frames: list[Path] = field(default_factory=list)
    depth_frame: Optional[np.ndarray] = None


@dataclass
class UISession:
    marker_ids: Tuple[int, int] = field(default_factory=lambda: SETTINGS.marker_ids)
    marker_size_mm: float = SETTINGS.marker_size_mm
    camera_stream: Optional[DepthAIFrameStream] = None
    gripper_device: Optional[GripperDevice] = None
    gripper: Optional[TwoFG7Gripper] = None
    capture: CaptureState = field(default_factory=CaptureState)
    calibration_bundle: Optional[calibration.CalibrationBundle] = None
    workspace_definition: Optional[workspace.WorkspaceDefinition] = None
    plate_calibration: Optional[plate.PlateCalibration] = None
    active_model_path: Optional[Path] = None
    yolo_model: Optional[object] = None
    model_loaded: bool = False
    detections: list[dict] = field(default_factory=list)
    last_detection_error: Optional[str] = None
    auto_task: Optional[object] = None
    auto_running: bool = False
    auto_message: str = ""
    auto_stop_requested: bool = False
    auto_waiting_at_drop: bool = False
    auto_config: dict = field(
        default_factory=lambda: {
            "target_label": SETTINGS.auto_target_label,
            "drop_ignore_radius_mm": SETTINGS.auto_drop_ignore_radius_mm,
            "approach_mm": SETTINGS.auto_approach_mm,
            "pick_mm": SETTINGS.auto_pick_mm,
            "drop_mm": SETTINGS.auto_drop_mm,
            "estimate_width": SETTINGS.auto_estimate_width,
            "tracking_mode": SETTINGS.tracking_mode,
            "segment_min_mm": SETTINGS.segment_min,
            "segment_max_mm": SETTINGS.segment_max,
            "segment_fraction": SETTINGS.segment_fraction,
            "cluster_neighbor_mm": SETTINGS.auto_cluster_neighbor_mm,
            "rotation_clearance_mm": SETTINGS.auto_rotation_clearance_mm,
            "rotation_step_deg": SETTINGS.auto_rotation_step_deg,
            "rotation_max_delta_deg": SETTINGS.auto_rotation_max_delta_deg,
            "collision_body_radius_mm": SETTINGS.auto_collision_body_radius_mm,
            "collision_finger_reach_mm": SETTINGS.auto_collision_finger_reach_mm,
            "separation_push_mm": SETTINGS.auto_separation_push_mm,
            "separation_entry_mm": SETTINGS.auto_separation_entry_mm,
            "separation_max_attempts": SETTINGS.auto_separation_max_attempts,
            "rescan_delay_s": SETTINGS.auto_rescan_delay_s,
        }
    )
    auto_detection_updated_at: float = 0.0
    safety_config: dict = field(
        default_factory=lambda: {
            "enabled": SETTINGS.camera_safety_enabled,
            "zone": SETTINGS.camera_safety_zone,
            "labels": list(SETTINGS.camera_safety_labels),
            "check_interval_ms": SETTINGS.camera_safety_check_interval_ms,
        }
    )
    drop_marker_id: int = app_settings.DEFAULT_DROP_MARKER_ID
    drop_target_mm: Optional[Tuple[float, float]] = None
    drop_target_orientation: Optional[Tuple[float, float, float]] = None
    drop_target_robot: Optional[Tuple[float, float, float]] = None
    drop_target_coord_id: Optional[int] = None
    plate_reference_homography: Optional[np.ndarray] = None
    robot_session: Optional[RobotSession] = None
    robot_connection_attempted: bool = False
    robot_last_connection_error: Optional[str] = None
    robot_probe_inflight: bool = False
    robot_last_probe_started_at: float = 0.0
    last_pick_mm: Optional[Tuple[float, float]] = None
    last_action: Optional[str] = None
    last_plate_test_capture: Optional[str] = None
    plate_test_state: dict = field(default_factory=dict)
    last_frame_id: int = 0
    continuous_detection: bool = False
    detection_confidence: float = 0.4
    detection_iou: float = 0.45
    detection_max: int = 10
    detection_classes: Optional[str] = None
    gripper_sync: dict = field(default_factory=dict)
    gripper_orientation: Optional[str] = None
    gripper_min_width: Optional[float] = None
    gripper_max_width: Optional[float] = None
    gripper_busy: Optional[bool] = None
    gripper_grip_detected: Optional[bool] = None
    gripper_force: Optional[float] = None
    gripper_status_code: Optional[int] = None
    robot_speed_rate: float = 0.25
    robot_speed_last_set: float = 0.0
    robot_status_last_poll: float = 0.0
    robot_power_poll_error_logged_at: float = 0.0
    robot_power_toggle_time: float = 0.0
    camera_calibration_in_progress: bool = False
    camera_auto_init_done: bool = False
    camera_last_error_message: Optional[str] = None
    camera_last_error_key: Optional[str] = None
    camera_last_error_params: dict[str, Any] = field(default_factory=dict)
    camera_last_error_status: Optional[str] = None
    robot_calibration_points: list[dict] = field(default_factory=list)
    robot_calibration_min_z: Optional[float] = None
    robot_calibration_baseline_px: Optional[np.ndarray] = None
    robot_calibration_gripper_width: Optional[float] = None
    robot_calibration_accuracy: Optional[dict[str, float]] = None
    plate_to_robot_transform: Optional[workspace.RigidTransform] = None
    plate_origin_mm: Optional[np.ndarray] = None
    coord5_transform: Optional[workspace.RigidTransform] = None
    robot_has_object: bool = False
    operation_in_progress: bool = False
    operation_started_at: Optional[float] = None
    pick_attempts: int = 0
    pick_successes: int = 0
    pick_failures: int = 0
    last_cycle_s: Optional[float] = None
    cycle_times: Deque[float] = field(default_factory=lambda: deque(maxlen=50))
    calibration_menu_open: bool = True


CALIBRATION_PROGRESS = {"value": 0, "current_frame": 0, "max_frames": 0}

SESSION = UISession()


def _refresh_session_auto_config_defaults() -> None:
    """Overwrite the auto-config store with the persisted defaults."""
    SESSION.auto_config.update(
        {
            "target_label": SETTINGS.auto_target_label,
            "drop_ignore_radius_mm": SETTINGS.auto_drop_ignore_radius_mm,
            "approach_mm": SETTINGS.auto_approach_mm,
            "pick_mm": SETTINGS.auto_pick_mm,
            "drop_mm": SETTINGS.auto_drop_mm,
            "estimate_width": bool(
                SESSION.auto_config.get("estimate_width", SETTINGS.auto_estimate_width)
            ),
            "tracking_mode": SESSION.auto_config.get(
                "tracking_mode", SETTINGS.tracking_mode
            ),
            "segment_min_mm": SESSION.auto_config.get(
                "segment_min_mm", SETTINGS.segment_min
            ),
            "segment_max_mm": SESSION.auto_config.get(
                "segment_max_mm", SETTINGS.segment_max
            ),
            "segment_fraction": SESSION.auto_config.get(
                "segment_fraction", SETTINGS.segment_fraction
            ),
            "cluster_neighbor_mm": SESSION.auto_config.get(
                "cluster_neighbor_mm", SETTINGS.auto_cluster_neighbor_mm
            ),
            "rotation_clearance_mm": SESSION.auto_config.get(
                "rotation_clearance_mm", SETTINGS.auto_rotation_clearance_mm
            ),
            "rotation_step_deg": SESSION.auto_config.get(
                "rotation_step_deg", SETTINGS.auto_rotation_step_deg
            ),
            "rotation_max_delta_deg": SESSION.auto_config.get(
                "rotation_max_delta_deg", SETTINGS.auto_rotation_max_delta_deg
            ),
            "collision_body_radius_mm": SESSION.auto_config.get(
                "collision_body_radius_mm", SETTINGS.auto_collision_body_radius_mm
            ),
            "collision_finger_reach_mm": SESSION.auto_config.get(
                "collision_finger_reach_mm", SETTINGS.auto_collision_finger_reach_mm
            ),
            "separation_push_mm": SESSION.auto_config.get(
                "separation_push_mm", SETTINGS.auto_separation_push_mm
            ),
            "separation_entry_mm": SESSION.auto_config.get(
                "separation_entry_mm", SETTINGS.auto_separation_entry_mm
            ),
            "separation_max_attempts": SESSION.auto_config.get(
                "separation_max_attempts", SETTINGS.auto_separation_max_attempts
            ),
            "rescan_delay_s": SESSION.auto_config.get(
                "rescan_delay_s", SETTINGS.auto_rescan_delay_s
            ),
        }
    )


def _refresh_session_safety_config_defaults() -> None:
    """Overwrite safety-config with persisted defaults."""
    SESSION.safety_config.update(
        {
            "enabled": SETTINGS.camera_safety_enabled,
            "zone": SETTINGS.camera_safety_zone,
            "labels": list(SETTINGS.camera_safety_labels),
            "check_interval_ms": SETTINGS.camera_safety_check_interval_ms,
        }
    )


_refresh_session_auto_config_defaults()
_refresh_session_safety_config_defaults()
