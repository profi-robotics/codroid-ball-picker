"""Detection-driven camera safety pause service."""

from __future__ import annotations

import asyncio
from typing import Any

from codroid_picker.config import AppSettings
from codroid_picker.core.safety_guard import desired_operation_action, evaluate_safety_state, normalize_safety_labels
from codroid_picker.services.calibration_service import CalibrationService
from codroid_picker.state.store import StateStore


class SafetyService:
    def __init__(self, settings: AppSettings, state: StateStore, calibration: CalibrationService) -> None:
        self.settings = settings
        self.state = state
        self.calibration = calibration
        self._task: asyncio.Task[None] | None = None

    def start(self) -> None:
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._loop())

    async def shutdown(self) -> None:
        if self._task is None:
            return
        self._task.cancel()
        try:
            await self._task
        except asyncio.CancelledError:
            pass
        self._task = None

    def evaluate_once(self) -> dict[str, Any]:
        safety = self.state.domain("safety")
        detection = self.state.domain("detection")
        operation = self.state.domain("operation")
        labels = normalize_safety_labels(safety.get("labels") or self.settings.camera_safety_labels)
        raw_detections = detection.get("detections") or []
        detections = [dict(item) for item in raw_detections if isinstance(item, dict)]
        blocked, matches = evaluate_safety_state(
            detections,
            enabled=bool(safety.get("enabled", self.settings.camera_safety_enabled)),
            labels=labels,
            zone=str(safety.get("zone") or self.settings.camera_safety_zone),
            workspace_def=self.calibration.workspace_definition,
        )
        action = desired_operation_action(
            status=str(operation.get("status") or "idle"),
            pause_reason=operation.get("pause_reason") if isinstance(operation.get("pause_reason"), str) else None,
            blocked=blocked,
        )
        if action == "pause":
            self.state.pause_operation("person_detected", "Camera safety pause: configured object detected.", owner="safety")
        elif action == "resume":
            self.state.resume_operation("Camera safety clear; operation resumed.", owner="safety")
        owner = "safety" if blocked or action in {"pause", "resume"} else None
        return self.state.update_domain("safety", blocked=blocked, matches=matches, action=action, labels=labels, owner=owner)

    async def _loop(self) -> None:
        interval_s = max(float(self.settings.camera_safety_check_interval_ms) / 1000.0, 0.1)
        while True:
            self.evaluate_once()
            await asyncio.sleep(interval_s)
