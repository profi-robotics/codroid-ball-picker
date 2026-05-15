"""Minimal program result types used by services."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from codroid_picker.state.store import StateStore


class ProgramStatus(StrEnum):
    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPING = "stopping"
    STOPPED = "stopped"
    COMPLETED = "completed"
    ERROR = "error"


@dataclass(frozen=True)
class ProgramResult:
    status: ProgramStatus
    message: str
    data: dict[str, Any] = field(default_factory=dict)


class ProgramController:
    """Small operation-state adapter used by service-layer programs."""

    def __init__(self, state: StateStore, *, source: str, label: str) -> None:
        self.state = state
        self.source = source
        self.label = label
        self.operation_id: str | None = None

    def start(self, message: str) -> str:
        self.operation_id = self.state.begin_operation(source=self.source, label=self.label, message=message)
        return self.operation_id

    def pause(self, reason: str = "operator") -> None:
        self.state.pause_operation(reason, f"{self.label} paused.", owner="operator")

    def resume(self) -> None:
        self.state.resume_operation(f"{self.label} resumed.", owner="operator")

    def stop(self) -> None:
        self.state.request_stop(f"{self.label} stop requested.")

    def finish(self, status: ProgramStatus, message: str) -> ProgramResult:
        final_status = "idle" if status in {ProgramStatus.COMPLETED, ProgramStatus.STOPPED} else status.value
        self.state.finish_operation(self.operation_id, status=final_status, message=message)
        return ProgramResult(status=status, message=message)

    def should_stop(self) -> bool:
        return bool(self.state.domain("operation").get("stop_requested"))

    def status(self) -> ProgramStatus:
        raw = str(self.state.domain("operation").get("status") or "idle")
        try:
            return ProgramStatus(raw)
        except ValueError:
            return ProgramStatus.IDLE
