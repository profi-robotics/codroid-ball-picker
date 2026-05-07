"""Cached UI snapshots and Server-Sent Events helpers.

The snapshot service keeps fast UI state reads separate from hardware command
callbacks. It intentionally publishes cached/session state only; robot motion
and safety callbacks remain responsible for side-effectful commands.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Callable, Deque, Iterable, Optional

LOGGER = logging.getLogger(__name__)

SNAPSHOT_FIELDS = (
    "camera",
    "robot",
    "gripper",
    "operation",
    "safety",
    "model",
    "detections",
    "monitoring",
)


def _json_safe(value: Any) -> Any:
    """Return a JSON-serializable representation of ``value``."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "model_dump"):
        return _json_safe(value.model_dump())
    if hasattr(value, "__dict__"):
        return _json_safe(vars(value))
    return str(value)


def diff_snapshot_fields(previous: Optional[dict], current: dict) -> list[str]:
    """Return top-level snapshot fields that changed, ignoring seq/timestamp."""
    if previous is None:
        return list(SNAPSHOT_FIELDS)
    changed: list[str] = []
    for field in SNAPSHOT_FIELDS:
        if previous.get(field) != current.get(field):
            changed.append(field)
    return changed


def snapshot_delta(snapshot: dict, changed: list[str]) -> dict:
    """Return the SSE payload containing only fields that changed."""
    return {
        "seq": snapshot["seq"],
        "ts": snapshot["ts"],
        "changed": list(changed),
        **{field: snapshot[field] for field in changed if field in snapshot},
    }


def format_sse_event(
    event: str,
    payload: dict,
    *,
    event_id: Optional[int] = None,
) -> str:
    """Format one Server-Sent Event frame."""
    lines: list[str] = []
    if event_id is not None:
        lines.append(f"id: {int(event_id)}")
    lines.append(f"event: {event}")
    data = json.dumps(payload, separators=(",", ":"), ensure_ascii=False)
    for line in data.splitlines() or ["{}"]:
        lines.append(f"data: {line}")
    lines.append("")
    lines.append("")
    return "\n".join(lines)


class SingleFlightWorker:
    """Small guard that prevents overlapping worker executions."""

    def __init__(self) -> None:
        self._lock = threading.Lock()

    def run(self, fn: Callable[[], Any]) -> bool:
        """Run ``fn`` only when no previous execution is still active."""
        if not self._lock.acquire(blocking=False):
            return False
        try:
            fn()
            return True
        finally:
            self._lock.release()


@dataclass
class SnapshotEvent:
    """A published UI snapshot event."""

    seq: int
    event: str
    payload: dict


class UiSnapshotService:
    """Publish cached UI state snapshots for SSE clients."""

    def __init__(
        self,
        collector: Optional[Callable[[], dict]] = None,
        *,
        poll_interval_s: float = 0.3,
        max_events: int = 128,
    ) -> None:
        self._collector = collector or collect_ui_snapshot
        self._poll_interval_s = max(float(poll_interval_s), 0.1)
        self._events: Deque[SnapshotEvent] = deque(maxlen=max_events)
        self._condition = threading.Condition()
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._seq = 0
        self._latest: Optional[dict] = None
        self._publisher = SingleFlightWorker()

    def start(self) -> None:
        """Start the background snapshot publisher."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._run,
            name="UiSnapshotService",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        """Stop the background publisher."""
        self._running = False
        with self._condition:
            self._condition.notify_all()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        self._thread = None

    def latest(self) -> Optional[dict]:
        """Return the latest published snapshot."""
        with self._condition:
            return None if self._latest is None else dict(self._latest)

    def force_publish(self) -> Optional[dict]:
        """Publish a snapshot immediately if no publisher is active."""
        result: dict[str, Any] = {"snapshot": None}

        def _publish() -> None:
            result["snapshot"] = self._publish_if_changed(force=True)

        self._publisher.run(_publish)
        return result["snapshot"]

    def events_since(self, last_seq: int) -> list[SnapshotEvent]:
        """Return buffered events newer than ``last_seq``."""
        with self._condition:
            return [event for event in self._events if event.seq > last_seq]

    def iter_sse(self, last_seq: int = 0, *, heartbeat_s: float = 10.0) -> Iterable[str]:
        """Yield SSE frames for a connected client."""
        self.start()
        latest = self.latest()
        if latest is None:
            latest = self.force_publish()
        if latest is not None and int(latest.get("seq", 0)) > last_seq:
            yield format_sse_event("snapshot", latest, event_id=int(latest["seq"]))
            last_seq = int(latest["seq"])

        next_heartbeat = time.monotonic() + heartbeat_s
        while True:
            with self._condition:
                timeout = max(next_heartbeat - time.monotonic(), 0.0)
                self._condition.wait(timeout=timeout)
                events = [event for event in self._events if event.seq > last_seq]

            if events:
                for event in events:
                    yield format_sse_event(
                        event.event,
                        event.payload,
                        event_id=event.seq,
                    )
                    last_seq = event.seq
                next_heartbeat = time.monotonic() + heartbeat_s
                continue

            if time.monotonic() >= next_heartbeat:
                yield format_sse_event(
                    "heartbeat",
                    {"ts": time.time(), "seq": last_seq},
                    event_id=last_seq if last_seq > 0 else None,
                )
                next_heartbeat = time.monotonic() + heartbeat_s

    def _run(self) -> None:
        while self._running:
            self._publisher.run(self._publish_if_changed)
            time.sleep(self._poll_interval_s)

    def _publish_if_changed(self, *, force: bool = False) -> Optional[dict]:
        try:
            snapshot = self._collector()
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("UI snapshot collection failed: %s", exc, exc_info=True)
            return None

        changed = diff_snapshot_fields(self._latest, snapshot)
        if not force and not changed:
            return self._latest

        self._seq += 1
        now = time.time()
        snapshot = {
            "seq": self._seq,
            "ts": now,
            **{field: _json_safe(snapshot.get(field, {})) for field in SNAPSHOT_FIELDS},
            "changed": changed,
        }
        if force or self._latest is None:
            payload = snapshot
        else:
            payload = snapshot_delta(snapshot, changed)
        event = SnapshotEvent(self._seq, "snapshot", payload)
        with self._condition:
            self._latest = snapshot
            self._events.append(event)
            self._condition.notify_all()
        return snapshot


def _camera_snapshot(session: Any) -> dict:
    stream = getattr(session, "camera_stream", None)
    if stream is None:
        last_status = getattr(session, "camera_last_error_status", None)
        message = getattr(session, "camera_last_error_message", None)
        if last_status in {"unavailable", "error", "retrying"}:
            status = last_status
            badge = "Camera: unavailable" if status == "unavailable" else "Camera: error"
            color = "orange" if status == "unavailable" else "yellow"
        else:
            status = "off"
            badge = "Camera: off"
            color = "gray"
            message = "Camera stream stopped."
        return {
            "status": status,
            "message": message or "Camera stream stopped.",
            "badge": badge,
            "badge_color": color,
            "checked": False,
            "degraded": ["camera_retrying"] if status == "retrying" else [],
        }

    stream_status = stream.status() if hasattr(stream, "status") else "running"
    if stream_status == "retrying":
        error = stream.last_error() if hasattr(stream, "last_error") else "camera unavailable"
        message = f"Camera retrying: {error}"
        return {
            "status": "retrying",
            "message": message,
            "badge": "Camera: retrying",
            "badge_color": "orange",
            "checked": True,
            "degraded": ["camera_retrying"],
        }

    if stream_status == "live":
        cfg = getattr(stream, "config", None)
        resolution = getattr(cfg, "resolution", (0, 0))
        fps = float(getattr(cfg, "fps", 0.0) or 0.0)
        device_id = stream.device_id() if hasattr(stream, "device_id") else None
        device_label = "Host camera" if str(device_id).startswith("host:") else "MXID"
        message = (
            f"Camera stream running ({resolution[0]}x{resolution[1]} @ "
            f"{fps:.1f} FPS, {device_label} {device_id or 'unknown'})."
        )
        return {
            "status": "live",
            "message": message,
            "badge": "Camera: live",
            "badge_color": "brand",
            "checked": True,
            "capture_fps": stream.get_fps() if hasattr(stream, "get_fps") else 0.0,
            "degraded": [],
        }

    return {
        "status": stream_status,
        "message": f"Camera stream {stream_status}.",
        "badge": f"Camera: {stream_status}",
        "badge_color": "gray",
        "checked": True,
        "degraded": [],
    }


def _robot_snapshot(session: Any) -> dict:
    robot_session = getattr(session, "robot_session", None)
    control_mode = getattr(session, "robot_control_mode", "acquired")
    released = control_mode == "released"
    if robot_session is None:
        status = "offline"
        message = "Robot: --" if not released else "Robot control released."
        return {
            "status": status,
            "message": message,
            "badge": "Robot: released" if released else "Robot: offline",
            "badge_color": "yellow" if released else "gray",
            "control": {
                "mode": control_mode,
                "released_at": getattr(session, "robot_control_released_at", 0.0),
                "message": message if released else "",
            },
            "power": {"value": None, "timestamp": 0.0},
            "position": "X=--.--- Y=--.--- Z=--.---  A=--.--- B=--.--- C=--.---",
            "degraded": ["hardware_poll_stale"],
        }

    snapshot = robot_session.snapshot()
    connected = bool(getattr(snapshot, "robot_connected", False))
    status = "online" if connected and not released else "offline"
    if released:
        message = "Robot control released. Acquire control to reconnect."
        badge = "Robot: released"
        color = "yellow"
    elif connected:
        target = robot_session.resolve_robot_target(getattr(snapshot, "robot_uri", ""))
        message = f"Robot connected to {target['host']}:{target['port']}."
        badge = "Robot: online"
        color = "brand"
    else:
        message = getattr(session, "robot_last_connection_error", None) or "Robot: --"
        badge = "Robot: offline"
        color = "gray"

    pos_text = "X=--.--- Y=--.--- Z=--.---  A=--.--- B=--.--- C=--.---"
    if getattr(snapshot, "posture_seen", False):
        pos = snapshot.posture
        pos_text = (
            f"X={pos.x:.3f} Y={pos.y:.3f} Z={pos.z:.3f}  "
            f"A={pos.a:.3f} B={pos.b:.3f} C={pos.c:.3f}"
        )
        age_fn = getattr(snapshot, "posture_age_seconds", None)
        age_s = age_fn() if callable(age_fn) else None
        if isinstance(age_s, (int, float)) and age_s > 2.0:
            pos_text = f"{pos_text} - stale {age_s:.1f}s"

    power_on = robot_session.robot_power_on()
    degraded = []
    status_age_fn = getattr(snapshot, "robot_status_age_seconds", None)
    status_age = status_age_fn() if callable(status_age_fn) else None
    if status_age is None or status_age > 5.0:
        degraded.append("hardware_poll_stale")

    return {
        "status": status,
        "message": message,
        "badge": badge,
        "badge_color": color,
        "control": {
            "mode": control_mode,
            "released_at": getattr(session, "robot_control_released_at", 0.0),
            "message": message if released else "",
        },
        "power": {
            "value": power_on,
            "timestamp": float(getattr(session, "robot_status_last_poll", 0.0) or 0.0),
        },
        "position": pos_text,
        "di_state": dict(getattr(snapshot, "di_state", {}) or {}),
        "degraded": degraded,
    }


def _gripper_snapshot(session: Any) -> dict:
    adapter = getattr(session, "gripper_adapter", None)
    gripper = getattr(session, "gripper", None)
    if adapter is None or gripper is None:
        return {
            "status": "offline",
            "message": "Gripper: off",
            "badge": "Gripper: off",
            "badge_color": "gray",
            "degraded": ["hardware_poll_stale"],
        }
    display = getattr(session.robot, "gripper_display_name", None) or "Gripper"
    state = adapter.to_state() if hasattr(adapter, "to_state") else {}
    return {
        "status": "online",
        "message": f"{display} connected.",
        "badge": "Gripper: online",
        "badge_color": "brand",
        "state": state,
        "width": getattr(session, "gripper_min_width", None),
        "busy": getattr(session, "gripper_busy", None),
        "gripped": getattr(session, "gripper_grip_detected", None),
        "force": getattr(session, "gripper_force", None),
        "degraded": [],
    }


def collect_ui_snapshot() -> dict:
    """Collect one cached UI snapshot from session state."""
    from app.codroid_picker.app_instance import registry
    from app.codroid_picker.auto_plan_state import DEFAULT_PLAN_STALE_S, get_auto_plan_snapshot
    from app.codroid_picker.state import _operation_snapshot

    session = registry().session
    operation = _operation_snapshot()
    detections = list(getattr(session, "detections", []) or [])
    safety = dict(getattr(session, "safety_config", {}) or {})
    safety_state = {
        "blocked": False,
        "matches": 0,
        "action": "none",
        "active": operation.get("status") in {"running", "paused"},
        "degraded": [],
    }

    return {
        "camera": _camera_snapshot(session),
        "robot": _robot_snapshot(session),
        "gripper": _gripper_snapshot(session),
        "operation": operation,
        "safety": {
            **safety_state,
            "config": safety,
        },
        "model": {
            "loaded": bool(getattr(session, "model_loaded", False)),
            "active": (
                getattr(session, "active_model_path", None).name
                if getattr(session, "active_model_path", None)
                else None
            ),
            "labels": list(getattr(session, "model_labels", []) or []),
            "benchmark": dict(getattr(session, "model_benchmark", {}) or {}),
        },
        "detections": {
            "count": len(detections),
            "items": detections,
            "last_error": getattr(session, "last_detection_error", None),
            "updated_at": getattr(session, "auto_detection_updated_at", 0.0),
        },
        "monitoring": {
            "pick_attempts": int(getattr(session, "pick_attempts", 0) or 0),
            "pick_successes": int(getattr(session, "pick_successes", 0) or 0),
            "pick_failures": int(getattr(session, "pick_failures", 0) or 0),
            "last_cycle_s": getattr(session, "last_cycle_s", None),
            "cycle_times": list(getattr(session, "cycle_times", []) or []),
            "auto_plan": get_auto_plan_snapshot(stale_after_s=DEFAULT_PLAN_STALE_S),
        },
    }


_SERVICE: Optional[UiSnapshotService] = None
_SERVICE_LOCK = threading.Lock()


def get_ui_snapshot_service() -> UiSnapshotService:
    """Return the process-wide UI snapshot service."""
    global _SERVICE
    with _SERVICE_LOCK:
        if _SERVICE is None:
            _SERVICE = UiSnapshotService()
        return _SERVICE
