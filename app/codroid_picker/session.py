from __future__ import annotations

import numpy as np
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Deque, Optional, Protocol, Tuple

from onrobot.device import Device as GripperDevice
from onrobot.twofg import TWOFG as TwoFG7Gripper
from vision import calibration, plate, workspace

import app.settings as app_settings
from app.settings import SETTINGS
from codroid_api.robot_session import RobotSession


@dataclass
class CaptureState:
    output_dir: Path = app_settings.DEFAULT_CAPTURE_DIR
    frames: list[Path] = field(default_factory=list)
    depth_frame: Optional[np.ndarray] = None


def _coerce_float(value: Any, default: float) -> float:
    """Return value as float, falling back to default when conversion fails."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _coerce_int(value: Any, default: int) -> int:
    """Return value as int, falling back to default when conversion fails."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _default_auto_label_profile() -> dict[str, Any]:
    """Build default per-label pick settings from global auto/gripper defaults."""
    gripper_profile = SETTINGS.gripper_profile
    vacuum_threshold = (
        float(getattr(gripper_profile, "vacuum_default", 40))
        * float(getattr(gripper_profile, "vacuum_hold_threshold_ratio", 0.8))
    )
    return {
        "approach_mm": float(SETTINGS.auto_approach_mm),
        "pick_mm": float(SETTINGS.auto_pick_mm),
        "gripper_close_mm": float(gripper_profile.close_width_default),
        "gripper_force": float(
            min(gripper_profile.force_default * 2, gripper_profile.force_max)
        ),
        "gripper_speed": 80.0,
        "soft_gentle": bool(getattr(gripper_profile, "gentle_default", False)),
        "vacuum_setpoint": float(getattr(gripper_profile, "vacuum_default", 40)),
        "vacuum_threshold": vacuum_threshold,
        "grip_timeout_s": float(SETTINGS.plate_test_grip_timeout_s),
    }


def normalize_auto_label_profile(raw: Optional[dict] = None) -> dict[str, Any]:
    """Normalize a persisted or UI-provided label profile."""
    defaults = _default_auto_label_profile()
    source = raw if isinstance(raw, dict) else {}
    return {
        "approach_mm": _coerce_float(source.get("approach_mm"), defaults["approach_mm"]),
        "pick_mm": _coerce_float(source.get("pick_mm"), defaults["pick_mm"]),
        "gripper_close_mm": _coerce_float(
            source.get("gripper_close_mm"), defaults["gripper_close_mm"]
        ),
        "gripper_force": _coerce_float(
            source.get("gripper_force"), defaults["gripper_force"]
        ),
        "gripper_speed": _coerce_float(
            source.get("gripper_speed"), defaults["gripper_speed"]
        ),
        "soft_gentle": bool(source.get("soft_gentle", defaults["soft_gentle"])),
        "vacuum_setpoint": _coerce_float(
            source.get("vacuum_setpoint"), defaults["vacuum_setpoint"]
        ),
        "vacuum_threshold": _coerce_float(
            source.get("vacuum_threshold"), defaults["vacuum_threshold"]
        ),
        "grip_timeout_s": _coerce_float(
            source.get("grip_timeout_s"), defaults["grip_timeout_s"]
        ),
    }


def normalize_auto_label_profiles(
    labels: list[str],
    profiles: Optional[dict[str, dict]] = None,
) -> dict[str, dict[str, Any]]:
    """Return normalized profiles for selected labels."""
    source = profiles if isinstance(profiles, dict) else {}
    normalized: dict[str, dict[str, Any]] = {}
    for label in labels:
        clean_label = str(label).strip()
        if clean_label:
            normalized[clean_label] = normalize_auto_label_profile(
                source.get(clean_label)
            )
    return normalized


def _default_auto_config() -> dict:
    target_labels = [
        str(label).strip()
        for label in SETTINGS.auto_target_labels
        if str(label).strip()
    ]
    if not target_labels and SETTINGS.auto_target_label:
        target_labels = [SETTINGS.auto_target_label]
    return {
        "target_label": SETTINGS.auto_target_label,
        "target_labels": target_labels,
        "label_profiles": normalize_auto_label_profiles(
            target_labels, SETTINGS.auto_label_profiles
        ),
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


class CameraFrameStream(Protocol):
    config: object

    def start(self) -> None: ...
    def stop(self) -> None: ...
    def is_running(self) -> bool: ...
    def status(self) -> str: ...
    def last_error(self) -> Optional[str]: ...
    def device_id(self) -> Optional[str]: ...
    def get_latest(self) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]: ...
    def get_fps(self) -> float: ...
    def set_focus_lock(self, lock: bool, lens_position: Optional[int] = None) -> None: ...


@dataclass
class CameraState:
    stream: Optional[CameraFrameStream] = None
    calibration_bundle: Optional[calibration.CalibrationBundle] = None
    workspace_definition: Optional[workspace.WorkspaceDefinition] = None
    calibration_in_progress: bool = False
    auto_init_done: bool = False
    last_error_message: Optional[str] = None
    last_error_key: Optional[str] = None
    last_error_params: dict[str, Any] = field(default_factory=dict)
    last_error_status: Optional[str] = None


@dataclass
class RobotState:
    gripper_device: Optional[GripperDevice] = None
    gripper: Optional[Any] = None
    gripper_adapter: Optional[Any] = None
    gripper_key: Optional[str] = None
    gripper_display_name: Optional[str] = None
    gripper_product_code: Optional[int] = None
    gripper_dimensions: Optional[Any] = None
    robot_session: Optional[RobotSession] = None
    connection_attempted: bool = False
    last_connection_error: Optional[str] = None
    probe_inflight: bool = False
    last_probe_started_at: float = 0.0
    control_mode: str = "acquired"
    control_released_at: float = 0.0
    speed_rate: float = 0.25
    speed_last_set: float = 0.0
    status_last_poll: float = 0.0
    power_poll_error_logged_at: float = 0.0
    power_toggle_time: float = 0.0


@dataclass
class AutoState:
    task: Optional[object] = None
    running: bool = False
    message: str = ""
    stop_requested: bool = False
    waiting_at_drop: bool = False
    plan_snapshot: dict = field(default_factory=dict)
    plan_stage: str = "idle"
    plan_updated_at: float = 0.0
    config: dict = field(default_factory=_default_auto_config)
    detection_updated_at: float = 0.0


@dataclass
class MonitoringState:
    detection_history: Deque[dict] = field(default_factory=lambda: deque(maxlen=90))
    pick_attempts: int = 0
    pick_successes: int = 0
    pick_failures: int = 0
    last_cycle_s: Optional[float] = None
    cycle_times: Deque[float] = field(default_factory=lambda: deque(maxlen=50))


@dataclass
class UISession:
    marker_ids: Tuple[int, int] = field(default_factory=lambda: SETTINGS.marker_ids)
    marker_size_mm: float = SETTINGS.marker_size_mm
    camera: CameraState = field(default_factory=CameraState)
    robot: RobotState = field(default_factory=RobotState)
    auto: AutoState = field(default_factory=AutoState)
    monitoring: MonitoringState = field(default_factory=MonitoringState)
    capture: CaptureState = field(default_factory=CaptureState)
    plate_calibration: Optional[plate.PlateCalibration] = None
    active_model_path: Optional[Path] = None
    yolo_model: Optional[object] = None
    model_loaded: bool = False
    model_labels: list[str] = field(default_factory=list)
    model_benchmark: dict = field(default_factory=dict)
    detections: list[dict] = field(default_factory=list)
    last_detection_error: Optional[str] = None
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
    detection_frame_skip: int = SETTINGS.detection_frame_skip
    detection_frame_index: int = 0
    gripper_sync: dict = field(default_factory=dict)
    gripper_orientation: Optional[str] = None
    gripper_min_width: Optional[float] = None
    gripper_max_width: Optional[float] = None
    gripper_busy: Optional[bool] = None
    gripper_grip_detected: Optional[bool] = None
    gripper_force: Optional[float] = None
    gripper_status_code: Optional[int] = None
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
    calibration_menu_open: bool = True

    @property
    def camera_stream(self) -> Optional[CameraFrameStream]:
        return self.camera.stream

    @camera_stream.setter
    def camera_stream(self, value: Optional[CameraFrameStream]) -> None:
        self.camera.stream = value

    @property
    def calibration_bundle(self) -> Optional[calibration.CalibrationBundle]:
        return self.camera.calibration_bundle

    @calibration_bundle.setter
    def calibration_bundle(self, value: Optional[calibration.CalibrationBundle]) -> None:
        self.camera.calibration_bundle = value

    @property
    def workspace_definition(self) -> Optional[workspace.WorkspaceDefinition]:
        return self.camera.workspace_definition

    @workspace_definition.setter
    def workspace_definition(self, value: Optional[workspace.WorkspaceDefinition]) -> None:
        self.camera.workspace_definition = value

    @property
    def gripper_device(self) -> Optional[GripperDevice]:
        return self.robot.gripper_device

    @gripper_device.setter
    def gripper_device(self, value: Optional[GripperDevice]) -> None:
        self.robot.gripper_device = value

    @property
    def gripper(self) -> Optional[Any]:
        return self.robot.gripper

    @gripper.setter
    def gripper(self, value: Optional[Any]) -> None:
        self.robot.gripper = value

    @property
    def gripper_adapter(self) -> Optional[Any]:
        return self.robot.gripper_adapter

    @gripper_adapter.setter
    def gripper_adapter(self, value: Optional[Any]) -> None:
        self.robot.gripper_adapter = value

    @property
    def robot_session(self) -> Optional[RobotSession]:
        return self.robot.robot_session

    @robot_session.setter
    def robot_session(self, value: Optional[RobotSession]) -> None:
        self.robot.robot_session = value

    @property
    def robot_connection_attempted(self) -> bool:
        return self.robot.connection_attempted

    @robot_connection_attempted.setter
    def robot_connection_attempted(self, value: bool) -> None:
        self.robot.connection_attempted = value

    @property
    def robot_last_connection_error(self) -> Optional[str]:
        return self.robot.last_connection_error

    @robot_last_connection_error.setter
    def robot_last_connection_error(self, value: Optional[str]) -> None:
        self.robot.last_connection_error = value

    @property
    def robot_probe_inflight(self) -> bool:
        return self.robot.probe_inflight

    @robot_probe_inflight.setter
    def robot_probe_inflight(self, value: bool) -> None:
        self.robot.probe_inflight = value

    @property
    def robot_last_probe_started_at(self) -> float:
        return self.robot.last_probe_started_at

    @robot_last_probe_started_at.setter
    def robot_last_probe_started_at(self, value: float) -> None:
        self.robot.last_probe_started_at = value

    @property
    def robot_control_mode(self) -> str:
        return self.robot.control_mode

    @robot_control_mode.setter
    def robot_control_mode(self, value: str) -> None:
        self.robot.control_mode = value

    @property
    def robot_control_released_at(self) -> float:
        return self.robot.control_released_at

    @robot_control_released_at.setter
    def robot_control_released_at(self, value: float) -> None:
        self.robot.control_released_at = value

    @property
    def robot_speed_rate(self) -> float:
        return self.robot.speed_rate

    @robot_speed_rate.setter
    def robot_speed_rate(self, value: float) -> None:
        self.robot.speed_rate = value

    @property
    def robot_speed_last_set(self) -> float:
        return self.robot.speed_last_set

    @robot_speed_last_set.setter
    def robot_speed_last_set(self, value: float) -> None:
        self.robot.speed_last_set = value

    @property
    def robot_status_last_poll(self) -> float:
        return self.robot.status_last_poll

    @robot_status_last_poll.setter
    def robot_status_last_poll(self, value: float) -> None:
        self.robot.status_last_poll = value

    @property
    def robot_power_poll_error_logged_at(self) -> float:
        return self.robot.power_poll_error_logged_at

    @robot_power_poll_error_logged_at.setter
    def robot_power_poll_error_logged_at(self, value: float) -> None:
        self.robot.power_poll_error_logged_at = value

    @property
    def robot_power_toggle_time(self) -> float:
        return self.robot.power_toggle_time

    @robot_power_toggle_time.setter
    def robot_power_toggle_time(self, value: float) -> None:
        self.robot.power_toggle_time = value

    @property
    def auto_task(self) -> Optional[object]:
        return self.auto.task

    @auto_task.setter
    def auto_task(self, value: Optional[object]) -> None:
        self.auto.task = value

    @property
    def auto_running(self) -> bool:
        return self.auto.running

    @auto_running.setter
    def auto_running(self, value: bool) -> None:
        self.auto.running = value

    @property
    def auto_message(self) -> str:
        return self.auto.message

    @auto_message.setter
    def auto_message(self, value: str) -> None:
        self.auto.message = value

    @property
    def auto_stop_requested(self) -> bool:
        return self.auto.stop_requested

    @auto_stop_requested.setter
    def auto_stop_requested(self, value: bool) -> None:
        self.auto.stop_requested = value

    @property
    def auto_waiting_at_drop(self) -> bool:
        return self.auto.waiting_at_drop

    @auto_waiting_at_drop.setter
    def auto_waiting_at_drop(self, value: bool) -> None:
        self.auto.waiting_at_drop = value

    @property
    def auto_plan_snapshot(self) -> dict:
        return self.auto.plan_snapshot

    @auto_plan_snapshot.setter
    def auto_plan_snapshot(self, value: dict) -> None:
        self.auto.plan_snapshot = value

    @property
    def auto_plan_stage(self) -> str:
        return self.auto.plan_stage

    @auto_plan_stage.setter
    def auto_plan_stage(self, value: str) -> None:
        self.auto.plan_stage = value

    @property
    def auto_plan_updated_at(self) -> float:
        return self.auto.plan_updated_at

    @auto_plan_updated_at.setter
    def auto_plan_updated_at(self, value: float) -> None:
        self.auto.plan_updated_at = value

    @property
    def auto_config(self) -> dict:
        return self.auto.config

    @auto_config.setter
    def auto_config(self, value: dict) -> None:
        self.auto.config = value

    @property
    def auto_detection_updated_at(self) -> float:
        return self.auto.detection_updated_at

    @auto_detection_updated_at.setter
    def auto_detection_updated_at(self, value: float) -> None:
        self.auto.detection_updated_at = value

    @property
    def monitor_detection_history(self) -> Deque[dict]:
        return self.monitoring.detection_history

    @monitor_detection_history.setter
    def monitor_detection_history(self, value: Deque[dict]) -> None:
        self.monitoring.detection_history = value

    @property
    def pick_attempts(self) -> int:
        return self.monitoring.pick_attempts

    @pick_attempts.setter
    def pick_attempts(self, value: int) -> None:
        self.monitoring.pick_attempts = value

    @property
    def pick_successes(self) -> int:
        return self.monitoring.pick_successes

    @pick_successes.setter
    def pick_successes(self, value: int) -> None:
        self.monitoring.pick_successes = value

    @property
    def pick_failures(self) -> int:
        return self.monitoring.pick_failures

    @pick_failures.setter
    def pick_failures(self, value: int) -> None:
        self.monitoring.pick_failures = value

    @property
    def last_cycle_s(self) -> Optional[float]:
        return self.monitoring.last_cycle_s

    @last_cycle_s.setter
    def last_cycle_s(self, value: Optional[float]) -> None:
        self.monitoring.last_cycle_s = value

    @property
    def cycle_times(self) -> Deque[float]:
        return self.monitoring.cycle_times

    @cycle_times.setter
    def cycle_times(self, value: Deque[float]) -> None:
        self.monitoring.cycle_times = value

    @property
    def camera_calibration_in_progress(self) -> bool:
        return self.camera.calibration_in_progress

    @camera_calibration_in_progress.setter
    def camera_calibration_in_progress(self, value: bool) -> None:
        self.camera.calibration_in_progress = value

    @property
    def camera_auto_init_done(self) -> bool:
        return self.camera.auto_init_done

    @camera_auto_init_done.setter
    def camera_auto_init_done(self, value: bool) -> None:
        self.camera.auto_init_done = value

    @property
    def camera_last_error_message(self) -> Optional[str]:
        return self.camera.last_error_message

    @camera_last_error_message.setter
    def camera_last_error_message(self, value: Optional[str]) -> None:
        self.camera.last_error_message = value

    @property
    def camera_last_error_key(self) -> Optional[str]:
        return self.camera.last_error_key

    @camera_last_error_key.setter
    def camera_last_error_key(self, value: Optional[str]) -> None:
        self.camera.last_error_key = value

    @property
    def camera_last_error_params(self) -> dict[str, Any]:
        return self.camera.last_error_params

    @camera_last_error_params.setter
    def camera_last_error_params(self, value: dict[str, Any]) -> None:
        self.camera.last_error_params = value

    @property
    def camera_last_error_status(self) -> Optional[str]:
        return self.camera.last_error_status

    @camera_last_error_status.setter
    def camera_last_error_status(self, value: Optional[str]) -> None:
        self.camera.last_error_status = value


CALIBRATION_PROGRESS = {"value": 0, "current_frame": 0, "max_frames": 0}

SESSION = UISession()


def _refresh_session_auto_config_defaults() -> None:
    """Overwrite the auto-config store with the persisted defaults."""
    target_labels = [
        str(label).strip()
        for label in SETTINGS.auto_target_labels
        if str(label).strip()
    ]
    if not target_labels and SETTINGS.auto_target_label:
        target_labels = [SETTINGS.auto_target_label]
    SESSION.auto_config.update(
        {
            "target_label": SETTINGS.auto_target_label,
            "target_labels": target_labels,
            "label_profiles": normalize_auto_label_profiles(
                target_labels, SETTINGS.auto_label_profiles
            ),
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
    SESSION.detection_frame_skip = _coerce_int(
        SETTINGS.detection_frame_skip, SETTINGS.detection_frame_skip
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
