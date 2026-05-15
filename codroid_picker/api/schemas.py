"""Pydantic schemas for the FastAPI API."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, Field


class ActionResponse(BaseModel):
    ok: bool = True
    message: str
    state: dict[str, Any] = Field(default_factory=dict)
    operation_id: Optional[str] = None


class CameraStartRequest(BaseModel):
    source: Optional[str] = None
    width: Optional[int] = None
    height: Optional[int] = None
    fps: Optional[float] = None
    host_camera_index: Optional[int] = None
    host_camera_device: Optional[str] = None
    enable_depth: Optional[bool] = None
    lock_af: bool = False


class CameraFocusRequest(BaseModel):
    lock: bool
    lens_position: Optional[int] = None


class OverlayRequest(BaseModel):
    mode: str


class RobotConnectRequest(BaseModel):
    uri: Optional[str] = None
    host: Optional[str] = None
    robot_port: Optional[int] = None
    user_port: Optional[int] = None

    def resolved_uri(self, default_uri: str) -> str:
        if self.uri:
            return self.uri
        if self.host and self.robot_port:
            return f"ws://{self.host}:{self.robot_port}/"
        return default_uri


class RobotPowerRequest(BaseModel):
    on: bool


class RobotSpeedRequest(BaseModel):
    rate: float


class RobotJogRequest(BaseModel):
    axis: str
    direction: int
    step_mm: Optional[float] = None
    step_deg: Optional[float] = None
    mode: str = "standard"
    hold_s: float = 0.25


class Pose(BaseModel):
    x: float
    y: float
    z: float
    a: float = 0.0
    b: float = 0.0
    c: float = 0.0


class RobotMoveRequest(BaseModel):
    pose: Pose
    coordinate_id: int = 5
    timeout_s: float = 15.0


class RobotErrorAckRequest(BaseModel):
    error_type: str


class GripperConnectRequest(BaseModel):
    ip: Optional[str] = None
    type: str = "auto"
    t_index: int = 0


class GripperOpenRequest(BaseModel):
    width_mm: Optional[float] = None
    speed: Optional[int] = None
    force: Optional[int] = None


class GripperCloseRequest(BaseModel):
    width_mm: Optional[float] = None
    force: Optional[int] = None
    speed: Optional[int] = None
    gentle: Optional[bool] = None
    vacuum_setpoint: Optional[int] = None
    vacuum_threshold: Optional[float] = None


class GripperOrientationRequest(BaseModel):
    orientation: str


class DetectionParamsRequest(BaseModel):
    confidence: Optional[float] = None
    iou: Optional[float] = None
    max_detections: Optional[int] = None
    classes: Optional[str] = None
    frame_skip: Optional[int] = None


class DetectionRunRequest(BaseModel):
    force: bool = True


class ModelActivateRequest(BaseModel):
    path: Optional[Path] = None


class AutoConfigRequest(BaseModel):
    target_labels: Optional[list[str]] = None
    label_profiles: Optional[dict[str, dict[str, Any]]] = None
    plan_snapshot: Optional[dict[str, Any]] = None


class OperationPauseRequest(BaseModel):
    reason: str = "operator"


class PickDropRequest(BaseModel):
    drop_height_mm: float = 10.0


class PlateTestRunRequest(BaseModel):
    approach_mm: float = -30.0
    pick_mm: float = 0.0
    drop_mm: float = 0.0
    grip_timeout_s: float = 2.5
    use_base_coordinates: Optional[bool] = None
    dry_run: bool = False


class CalibrationResetRequest(BaseModel):
    scope: str = "all"


class PathRequest(BaseModel):
    path: Optional[Path] = None


class CameraCalibrationCaptureRequest(BaseModel):
    frame_count: Optional[int] = None
    delay_s: Optional[float] = None
    min_usable_frames: Optional[int] = None
    output_path: Optional[Path] = None


class WorkspaceComputeRequest(BaseModel):
    marker_ids: tuple[int, int] = (1, 2)
    marker_size_mm: float = 40.0


class RobotCalibrationPointRequest(BaseModel):
    index: Optional[int] = None


class SettingsUpdateRequest(BaseModel):
    updates: dict[str, Any]


class LocaleRequest(BaseModel):
    language: str


def action(message: str, *, state: dict[str, Any] | None = None, operation_id: str | None = None) -> ActionResponse:
    return ActionResponse(message=message, state=state or {}, operation_id=operation_id)
