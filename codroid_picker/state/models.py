"""Typed state domains for the FastAPI runtime."""

from __future__ import annotations

import time
from typing import Any, Optional

from pydantic import BaseModel, Field


class CameraState(BaseModel):
    source: str = "oak"
    running: bool = False
    status: str = "off"
    device_id: Optional[str] = None
    capture_fps: float = 0.0
    frame_age_s: Optional[float] = None
    resolution: tuple[int, int] = (1920, 1080)
    focus_locked: bool = False
    depth_enabled: bool = False
    last_error: Optional[str] = None


class VideoState(BaseModel):
    transport: str = "mjpeg"
    overlay_mode: str = "auto"
    stream_fps: float = 0.0
    encoded_fps: float = 0.0
    jpeg_quality: int = 70
    max_width: int = 960
    clients: int = 0
    frames_encoded: int = 0
    frames_dropped: int = 0
    fallback: str = "snapshot"
    last_error: Optional[str] = None


class DetectionState(BaseModel):
    worker_status: str = "stopped"
    ml_available: Optional[bool] = None
    active_model: Optional[str] = None
    labels: list[str] = Field(default_factory=list)
    backend_device: Optional[str] = None
    confidence: float = 0.4
    iou: float = 0.45
    max_detections: int = 10
    classes: Optional[str] = None
    frame_skip: int = 3
    detections: list[dict[str, Any]] = Field(default_factory=list)
    latency_ms: Optional[float] = None
    benchmark: dict[str, Any] = Field(default_factory=lambda: {"status": "not_run"})
    updated_at: float = 0.0
    last_error: Optional[str] = None


class RobotState(BaseModel):
    status: str = "offline"
    control_mode: str = "acquired"
    power: Optional[bool] = None
    speed_rate: float = 0.25
    coordinate_id: int = 5
    coordinate_frame: dict[str, Any] = Field(default_factory=dict)
    uri: Optional[str] = None
    posture: Optional[dict[str, float]] = None
    di_state: dict[str, int] = Field(default_factory=dict)
    stale: bool = True
    last_seen_at: Optional[float] = None
    poll_age_s: Optional[float] = None
    active_error: Optional[str] = None
    error_message: Optional[str] = None
    rescue_mode: bool = False
    message: str = "Robot offline."


class GripperState(BaseModel):
    status: str = "offline"
    ip: Optional[str] = None
    gripper_type: str = "auto"
    display_name: str = "Gripper"
    capabilities: dict[str, Any] = Field(default_factory=dict)
    width_mm: Optional[float] = None
    force: Optional[float] = None
    speed: Optional[float] = None
    vacuum: Optional[dict[str, float]] = None
    orientation: Optional[str] = None
    holding: Optional[bool] = None
    busy: Optional[bool] = None
    message: str = "Gripper offline."


class OperationState(BaseModel):
    status: str = "idle"
    operation_id: Optional[str] = None
    label: str = ""
    message: str = "Program idle."
    source: str = ""
    started_at: Optional[float] = None
    updated_at: float = Field(default_factory=time.time)
    pause_reason: Optional[str] = None
    pause_owner: Optional[str] = None
    stop_requested: bool = False


class CalibrationState(BaseModel):
    camera_status: str = "idle"
    camera_progress: float = 0.0
    workspace_status: str = "missing"
    robot_status: str = "idle"
    step: int = 0
    robot_points: list[dict[str, Any]] = Field(default_factory=list)
    robot_expected_points: int = 3
    robot_verification_ready: bool = False
    robot_controller_applied: bool = False
    robot_controller_frame: dict[str, Any] = Field(default_factory=dict)
    plate: dict[str, Any] = Field(default_factory=lambda: {"ok": False, "stable": False, "count": 0})
    accuracy: Optional[dict[str, float]] = None
    artifacts: dict[str, Optional[str]] = Field(default_factory=dict)
    message: str = "Calibration idle."


class AutoPickState(BaseModel):
    running: bool = False
    target_labels: list[str] = Field(default_factory=lambda: ["orange", "sports ball"])
    label_profiles: dict[str, dict[str, Any]] = Field(default_factory=dict)
    drop_target: Optional[dict[str, Any]] = None
    plan_snapshot: dict[str, Any] = Field(default_factory=dict)
    selected_candidate: Optional[dict[str, Any]] = None
    stop_requested: bool = False
    prerequisites: dict[str, bool] = Field(default_factory=dict)
    message: str = "Auto idle."


class SafetyState(BaseModel):
    enabled: bool = True
    zone: str = "workspace"
    labels: list[str] = Field(default_factory=lambda: ["person", "people"])
    blocked: bool = False
    matches: list[dict[str, Any]] = Field(default_factory=list)
    action: str = "none"
    owner: Optional[str] = None


class MonitoringState(BaseModel):
    pick_attempts: int = 0
    pick_successes: int = 0
    pick_failures: int = 0
    last_cycle_s: Optional[float] = None
    cycle_times: list[float] = Field(default_factory=list)
    detection_history: list[dict[str, Any]] = Field(default_factory=list)
    system: dict[str, Any] = Field(default_factory=dict)
    gpu: dict[str, Any] = Field(default_factory=dict)
    plate_test: dict[str, Any] = Field(default_factory=dict)


class SettingsState(BaseModel):
    version: int = 1
    saved_at: Optional[float] = None
    reconnect_required: list[str] = Field(default_factory=list)


class AppState(BaseModel):
    camera: CameraState = Field(default_factory=CameraState)
    video: VideoState = Field(default_factory=VideoState)
    detection: DetectionState = Field(default_factory=DetectionState)
    robot: RobotState = Field(default_factory=RobotState)
    gripper: GripperState = Field(default_factory=GripperState)
    operation: OperationState = Field(default_factory=OperationState)
    calibration: CalibrationState = Field(default_factory=CalibrationState)
    auto_pick: AutoPickState = Field(default_factory=AutoPickState)
    safety: SafetyState = Field(default_factory=SafetyState)
    monitoring: MonitoringState = Field(default_factory=MonitoringState)
    settings: SettingsState = Field(default_factory=SettingsState)
