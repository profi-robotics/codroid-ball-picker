"""Auto pick operation orchestration shell."""

from __future__ import annotations

import asyncio
import contextlib
import time
from typing import Any, Optional

from codroid_picker.config import AppSettings
from codroid_picker.errors import OperationConflictError
from codroid_picker.services.detection_service import DetectionService
from codroid_picker.services.pick_place_service import PickPlaceService
from codroid_picker.state.store import StateStore


class AutoPickService:
    def __init__(self, settings: AppSettings, state: StateStore, detection: DetectionService, pick_place: PickPlaceService) -> None:
        self.settings = settings
        self.state = state
        self.detection = detection
        self.pick_place = pick_place
        self._task: Optional[asyncio.Task[None]] = None
        self._operation_id: Optional[str] = None

    def config(self) -> dict[str, Any]:
        return self.state.domain("auto_pick")

    def update_config(self, updates: dict[str, Any]) -> dict[str, Any]:
        return self.state.update_domain("auto_pick", **updates)

    async def start(self) -> str:
        if self._task is not None and not self._task.done():
            raise OperationConflictError("Auto pick is already running.")
        self._operation_id = self.state.begin_operation(
            source="auto",
            label="Auto pick & place",
            message="Auto pick loop started.",
        )
        self.state.update_domain("auto_pick", running=True, stop_requested=False, message="Auto pick loop running.")
        self._task = asyncio.create_task(self._loop(self._operation_id))
        return self._operation_id

    async def stop(self) -> str:
        self.state.request_stop("Auto stop requested.")
        self.state.update_domain("auto_pick", stop_requested=True, message="Stopping auto pick loop.")
        if self._task is not None:
            try:
                await asyncio.wait_for(asyncio.shield(self._task), timeout=2.0)
            except asyncio.TimeoutError:
                self._task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await self._task
        if self._operation_id is not None:
            self.state.finish_operation(self._operation_id, status="idle", message="Auto pick stopped.")
        self.state.update_domain("auto_pick", running=False, stop_requested=False, message="Auto idle.")
        return "Auto pick stopped."

    async def shutdown(self) -> None:
        if self._task is not None and not self._task.done():
            await self.stop()

    async def _loop(self, operation_id: str) -> None:
        try:
            while True:
                operation = self.state.domain("operation")
                if operation.get("stop_requested"):
                    break
                if operation.get("status") == "paused":
                    await asyncio.sleep(max(float(self.settings.auto_idle_sleep_s), 0.1))
                    continue
                if self.state.domain("safety").get("blocked"):
                    self.state.update_domain("auto_pick", message="Safety guard is blocking auto pick.")
                    await asyncio.sleep(max(float(self.settings.auto_idle_sleep_s), 0.1))
                    continue
                current = self.state.domain("detection").get("detections", [])
                if isinstance(current, list) and current:
                    detections = current
                else:
                    detections = self.detection.run_once(force=False)
                try:
                    candidate, plan, candidates = self.pick_place.plan_next_target(
                        [dict(item) for item in detections if isinstance(item, dict)]
                    )
                except RuntimeError as exc:
                    self.state.update_domain(
                        "auto_pick",
                        plan_snapshot={"available": False, "stage": "detect", "message": str(exc)},
                        message=str(exc),
                    )
                    await asyncio.sleep(max(float(self.settings.auto_idle_sleep_s), 0.1))
                    continue
                self.state.update_domain(
                    "auto_pick",
                    message=f"Executing auto pick with {len(candidates)} ranked candidates.",
                )
                started = time.monotonic()
                attempts_before = int(self.state.domain("monitoring").get("pick_attempts", 0))
                try:
                    result = await self.pick_place.pick_candidate(candidate, plan, operation_id=operation_id)
                except Exception as exc:  # noqa: BLE001
                    attempts_after = int(self.state.domain("monitoring").get("pick_attempts", 0))
                    if attempts_after == attempts_before:
                        self.pick_place.record_cycle(success=False, started=started)
                    self.state.update_domain(
                        "auto_pick",
                        plan_snapshot={
                            **dict(self.state.domain("auto_pick").get("plan_snapshot") or {}),
                            "stage": "error",
                            "message": str(exc),
                        },
                        message=f"Auto pick failed; retrying: {exc}",
                    )
                    await asyncio.sleep(max(float(self.settings.auto_idle_sleep_s), 0.1))
                    continue
                self.state.update_domain("auto_pick", message=result)
                await asyncio.sleep(max(float(self.settings.auto_idle_sleep_s), 0.1))
            self.state.finish_operation(operation_id, status="idle", message="Auto pick stopped.")
            self.state.update_domain("auto_pick", running=False, stop_requested=False, message="Auto idle.")
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001
            self.state.finish_operation(operation_id, status="error", message=str(exc))
            self.state.update_domain("auto_pick", running=False, message=str(exc))
