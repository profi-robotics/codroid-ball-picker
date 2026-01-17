"""State management for the calibration UI.

This module defines Pydantic models for tracking operation state, errors,
and calibration progress. It provides thread-safe helper functions to manage
the shared state across different UI components and robot operations.
"""

import threading
import time
from typing import Optional, Dict, Any

from pydantic import BaseModel, Field


class OperationState(BaseModel):
    """Tracks the current robot operation state."""
    status: str = "idle"  # idle, running, paused, stopped, error
    source: str = ""  # Source callback/function that started the operation
    label: str = ""  # Human-readable operation name
    message: str = "Program idle."
    run_id: int = 0  # Incremented for each new operation
    started_at: Optional[float] = None
    updated_at: float = Field(default_factory=time.time)
    pause_reason: Optional[str] = None


class ErrorState(BaseModel):
    """Tracks robot error conditions (emergency stop, overspeed, etc)."""
    active: bool = False
    # emergency, overspeed, joint_protection, wrong_tool, drag
    error_type: Optional[str] = None
    message: str = ""
    detected_at: Optional[float] = None
    acknowledged: bool = False
    physical_pressed: bool = False  # True if physical button was pressed


class CalibrationStepperState(BaseModel):
    """Tracks robot calibration stepper wizard progress."""
    status: str = "idle"  # idle, active, complete
    step: int = 0
    message: str = "Robot calibration idle. Click 'Begin calibrating robot' to start."
    verification_ready: bool = False
    modal_suppressed: bool = False
    accuracy: Optional[Dict[str, float]] = None
    accuracy_ready: bool = False
    accuracy_passed: bool = False
    timestamp: float = Field(default_factory=time.time)


class PlateTrackingState(BaseModel):
    """Tracks calibration plate detection status."""
    count: int = 0
    ok: bool = False
    stable: bool = False
    center_px: Optional[tuple] = None
    movement_px: Optional[float] = None
    timestamp: float = Field(default_factory=time.time)
    message: str = "Waiting for plate markers"


# Global state instances with thread-safe locks
_operation_lock = threading.Lock()
_error_lock = threading.Lock()

operation_state = OperationState()
error_state = ErrorState()
calibration_stepper_state = CalibrationStepperState()
plate_tracking_state = PlateTrackingState()


# Operation state helpers
def is_operation_running() -> bool:
    """Check if any robot operation is currently running."""
    with _operation_lock:
        return operation_state.status in {"running", "paused"}


def is_operation_idle() -> bool:
    """Check if no robot operation is running."""
    with _operation_lock:
        return operation_state.status == "idle"


def is_operation_paused() -> bool:
    """Check if the current operation is paused."""
    with _operation_lock:
        return operation_state.status == "paused"


def get_operation_status() -> str:
    """Get the current operation status."""
    with _operation_lock:
        return operation_state.status


def get_operation_run_id() -> int:
    """Get the current operation run ID."""
    with _operation_lock:
        return operation_state.run_id


def get_operation_snapshot() -> dict:
    """Get a thread-safe snapshot of the operation state."""
    with _operation_lock:
        return operation_state.model_dump()


def update_operation_state(**updates: Any) -> dict:
    """Update operation state fields and return snapshot."""
    with _operation_lock:
        for key, value in updates.items():
            if hasattr(operation_state, key):
                setattr(operation_state, key, value)
        operation_state.updated_at = time.time()
        return operation_state.model_dump()


def begin_operation(source: str, label: str, message: Optional[str] = None) -> int:
    """Start a new operation. Returns the run_id or raises if already running."""
    with _operation_lock:
        if operation_state.status in {"running", "paused"}:
            raise RuntimeError("Another program is already running.")

        operation_state.run_id += 1
        operation_state.status = "running"
        operation_state.source = source
        operation_state.label = label
        operation_state.message = message or f"{label} running."
        operation_state.started_at = time.time()
        operation_state.updated_at = time.time()
        operation_state.pause_reason = None

        return operation_state.run_id


def finish_operation(run_id: int, *, status: str, message: str) -> None:
    """Finish an operation if it matches the expected run_id."""
    with _operation_lock:
        if operation_state.run_id != run_id:
            return  # Operation was superseded

        operation_state.status = status
        operation_state.message = message
        operation_state.updated_at = time.time()
        operation_state.pause_reason = None

        if status == "idle":
            operation_state.source = ""
            operation_state.label = ""
            operation_state.started_at = None


def pause_operation(reason: str, message: str) -> None:
    """Pause the current operation."""
    with _operation_lock:
        if operation_state.status == "running":
            operation_state.status = "paused"
            operation_state.pause_reason = reason
            operation_state.message = message
            operation_state.updated_at = time.time()


def resume_operation(message: str) -> None:
    """Resume a paused operation."""
    with _operation_lock:
        if operation_state.status == "paused":
            operation_state.status = "running"
            operation_state.pause_reason = None
            operation_state.message = message
            operation_state.updated_at = time.time()


def stop_operation(message: str) -> None:
    """Stop the current operation."""
    with _operation_lock:
        if operation_state.status in {"running", "paused"}:
            operation_state.status = "stopped"
            operation_state.message = message
            operation_state.updated_at = time.time()


def update_operation_message(message: str, *, label: Optional[str] = None) -> None:
    """Update the operation message and optionally the label."""
    with _operation_lock:
        if operation_state.status in {"running", "paused"}:
            operation_state.message = message
            if label is not None:
                operation_state.label = label
            operation_state.updated_at = time.time()


# Error state helpers
def set_error_state(
    error_type: str,
    message: str,
    *,
    physical_pressed: bool = False,
    acknowledge_existing: bool = False
) -> None:
    """Set an error condition."""
    with _error_lock:
        if acknowledge_existing and error_state.active:
            error_state.acknowledged = True
            error_state.physical_pressed = physical_pressed
        else:
            error_state.active = True
            error_state.error_type = error_type
            error_state.message = message
            error_state.detected_at = time.time()
            error_state.acknowledged = False
            error_state.physical_pressed = physical_pressed


def clear_error_state() -> None:
    """Clear the current error state."""
    with _error_lock:
        error_state.active = False
        error_state.error_type = None
        error_state.message = ""
        error_state.detected_at = None
        error_state.acknowledged = False
        error_state.physical_pressed = False


def get_error_snapshot() -> dict:
    """Get a thread-safe snapshot of the error state."""
    with _error_lock:
        return error_state.model_dump()


def is_error_active() -> bool:
    """Check if there's an active error."""
    with _error_lock:
        return error_state.active


def is_error_acknowledged() -> bool:
    """Check if the current error has been acknowledged."""
    with _error_lock:
        return error_state.acknowledged


# Calibration stepper helpers
def update_calibration_guidance(
    message: str,
    *,
    step: Optional[int] = None,
    status: Optional[str] = None,
    verification_ready: Optional[bool] = None,
    modal_suppressed: Optional[bool] = None
) -> None:
    """Update robot calibration guidance state."""
    calibration_stepper_state.message = message

    if step is not None:
        calibration_stepper_state.step = step
    if status is not None:
        calibration_stepper_state.status = status
    if verification_ready is not None:
        calibration_stepper_state.verification_ready = verification_ready
    if modal_suppressed is not None:
        calibration_stepper_state.modal_suppressed = modal_suppressed

    calibration_stepper_state.timestamp = time.time()


def clear_calibration_accuracy() -> None:
    """Clear calibration accuracy data."""
    calibration_stepper_state.accuracy = None
    calibration_stepper_state.accuracy_ready = False
    calibration_stepper_state.accuracy_passed = False


def set_calibration_accuracy(accuracy: Dict[str, float]) -> None:
    """Set calibration accuracy measurements."""
    calibration_stepper_state.accuracy = accuracy
    calibration_stepper_state.accuracy_ready = True
    calibration_stepper_state.accuracy_passed = all(
        v <= threshold for v, threshold in zip(
            accuracy.values(),
            [2.0, 2.0, 2.0]  # mm thresholds for x, y, z
        )
    )


def get_calibration_guidance_snapshot() -> dict:
    """Get calibration guidance state snapshot."""
    return calibration_stepper_state.model_dump()


def build_robot_calibration_state_payload() -> dict:
    """Build robot calibration state payload for UI callbacks."""
    guidance = get_calibration_guidance_snapshot()
    return {
        "status": guidance.get("status", "idle"),
        "message": guidance.get("message", ""),
        "step": guidance.get("step", 0),
        "verification_ready": guidance.get("verification_ready", False),
        "accuracy": guidance.get("accuracy"),
        "accuracy_ready": guidance.get("accuracy_ready", False),
        "accuracy_passed": guidance.get("accuracy_passed", False),
    }


# Plate tracking helpers
def update_plate_status(
    ok: bool,
    stable: bool,
    message: str,
    *,
    count: Optional[int] = None,
    center_px: Optional[tuple] = None,
    movement_px: Optional[float] = None
) -> None:
    """Update plate tracking state."""
    plate_tracking_state.ok = ok
    plate_tracking_state.stable = stable
    plate_tracking_state.message = message
    plate_tracking_state.timestamp = time.time()

    if count is not None:
        plate_tracking_state.count = count
    if center_px is not None:
        plate_tracking_state.center_px = center_px
    if movement_px is not None:
        plate_tracking_state.movement_px = movement_px


def get_plate_status_snapshot() -> dict:
    """Get plate tracking state snapshot."""
    return plate_tracking_state.model_dump()


# Backwards compatibility helpers for existing code patterns
def _operation_snapshot() -> dict:
    """Legacy alias for get_operation_snapshot()."""
    return get_operation_snapshot()


def _update_operation_state(**updates: Any) -> dict:
    """Legacy alias for update_operation_state()."""
    return update_operation_state(**updates)


def _begin_operation(source: str, label: str, message: Optional[str] = None) -> int:
    """Legacy alias for begin_operation()."""
    return begin_operation(source, label, message)


def _finish_operation(run_id: int, *, status: str, message: str) -> None:
    """Legacy alias for finish_operation()."""
    return finish_operation(run_id, status=status, message=message)


def _pause_operation(reason: str, message: str) -> None:
    """Legacy alias for pause_operation()."""
    return pause_operation(reason, message)


def _update_operation_message(message: str, *, label: Optional[str] = None) -> None:
    """Legacy alias for update_operation_message()."""
    return update_operation_message(message, label=label)
