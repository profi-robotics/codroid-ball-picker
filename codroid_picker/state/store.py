"""Thread-safe state store with delta publishing."""

from __future__ import annotations

import threading
import time
import uuid
from typing import Any, Callable, Optional, cast

from codroid_picker.errors import OperationConflictError
from codroid_picker.state.events import StateEvent, StatePublisher
from codroid_picker.state.models import AppState

STATE_DOMAINS = (
    "camera",
    "video",
    "detection",
    "robot",
    "gripper",
    "operation",
    "calibration",
    "auto_pick",
    "safety",
    "monitoring",
    "settings",
)


class StateStore:
    """Owns all mutable state for the process."""

    def __init__(self, publisher: Optional[StatePublisher] = None) -> None:
        self._lock = threading.RLock()
        self._state = AppState()
        self._seq = 0
        self._ts = time.time()
        self._publisher = publisher

    @property
    def seq(self) -> int:
        with self._lock:
            return self._seq

    def set_publisher(self, publisher: StatePublisher) -> None:
        self._publisher = publisher

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return self._snapshot_locked()

    def domain(self, name: str) -> dict[str, Any]:
        with self._lock:
            value = getattr(self._state, name)
            return cast(dict[str, Any], value.model_dump(mode="json"))

    def update_domain(self, name: str, **updates: Any) -> dict[str, Any]:
        if name not in STATE_DOMAINS:
            raise KeyError(name)
        with self._lock:
            model = getattr(self._state, name)
            setattr(self._state, name, model.model_copy(update=updates))
            return self._publish_locked([name])

    def mutate(self, name: str, mutator: Callable[[Any], Any]) -> dict[str, Any]:
        if name not in STATE_DOMAINS:
            raise KeyError(name)
        with self._lock:
            model = getattr(self._state, name)
            replacement = mutator(model)
            if replacement is not None:
                setattr(self._state, name, replacement)
            return self._publish_locked([name])

    def begin_operation(self, *, source: str, label: str, message: str) -> str:
        with self._lock:
            current = self._state.operation
            if current.status in {"running", "paused"}:
                raise OperationConflictError("Another operation is already running.")
            operation_id = uuid.uuid4().hex
            self._state.operation = current.model_copy(
                update={
                    "status": "running",
                    "operation_id": operation_id,
                    "source": source,
                    "label": label,
                    "message": message,
                    "started_at": time.time(),
                    "updated_at": time.time(),
                    "pause_reason": None,
                    "pause_owner": None,
                    "stop_requested": False,
                }
            )
            self._publish_locked(["operation"])
            return operation_id

    def finish_operation(
        self,
        operation_id: Optional[str],
        *,
        status: str,
        message: str,
    ) -> dict[str, Any]:
        with self._lock:
            current = self._state.operation
            if operation_id is not None and current.operation_id != operation_id:
                return self._snapshot_locked()
            reset = status == "idle"
            self._state.operation = current.model_copy(
                update={
                    "status": status,
                    "message": message,
                    "updated_at": time.time(),
                    "pause_reason": None,
                    "pause_owner": None,
                    "stop_requested": False,
                    "operation_id": None if reset else current.operation_id,
                    "source": "" if reset else current.source,
                    "label": "" if reset else current.label,
                    "started_at": None if reset else current.started_at,
                }
            )
            return self._publish_locked(["operation"])

    def request_stop(self, message: str = "Stop requested.") -> dict[str, Any]:
        with self._lock:
            self._state.operation = self._state.operation.model_copy(
                update={
                    "status": "stopping" if self._state.operation.status in {"running", "paused"} else self._state.operation.status,
                    "stop_requested": True,
                    "message": message,
                    "updated_at": time.time(),
                }
            )
            return self._publish_locked(["operation"])

    def pause_operation(self, reason: str, message: str = "Operation paused.", *, owner: str = "operator") -> dict[str, Any]:
        with self._lock:
            current = self._state.operation
            if current.status != "running":
                return self._snapshot_locked()
            self._state.operation = current.model_copy(
                update={
                    "status": "paused",
                    "pause_reason": reason,
                    "pause_owner": owner,
                    "message": message,
                    "updated_at": time.time(),
                }
            )
            return self._publish_locked(["operation"])

    def resume_operation(self, message: str = "Operation resumed.", *, owner: Optional[str] = None) -> dict[str, Any]:
        with self._lock:
            current = self._state.operation
            if current.status != "paused":
                return self._snapshot_locked()
            if owner is not None and current.pause_owner not in {None, owner}:
                return self._snapshot_locked()
            self._state.operation = current.model_copy(
                update={
                    "status": "running",
                    "pause_reason": None,
                    "pause_owner": None,
                    "message": message,
                    "updated_at": time.time(),
                }
            )
            return self._publish_locked(["operation"])

    def reset_operation(self, message: str = "Program idle.") -> dict[str, Any]:
        with self._lock:
            self._state.operation = self._state.operation.model_copy(
                update={
                    "status": "idle",
                    "operation_id": None,
                    "label": "",
                    "message": message,
                    "source": "",
                    "started_at": None,
                    "updated_at": time.time(),
                    "pause_reason": None,
                    "pause_owner": None,
                    "stop_requested": False,
                }
            )
            return self._publish_locked(["operation"])

    def _snapshot_locked(self) -> dict[str, Any]:
        return {
            "seq": self._seq,
            "ts": self._ts,
            **{
                domain: getattr(self._state, domain).model_dump(mode="json")
                for domain in STATE_DOMAINS
            },
        }

    def _publish_locked(self, changed: list[str]) -> dict[str, Any]:
        self._seq += 1
        self._ts = time.time()
        snapshot = self._snapshot_locked()
        payload = {
            "type": "state.delta",
            "seq": self._seq,
            "ts": self._ts,
            "changed": changed,
            **{domain: snapshot[domain] for domain in changed},
        }
        if self._publisher is not None:
            self._publisher.publish(StateEvent(self._seq, "state.delta", payload))
        return snapshot
