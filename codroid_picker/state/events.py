"""State event publishing and Server-Sent Event formatting."""

from __future__ import annotations

import json
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, Iterable, Optional


@dataclass(frozen=True)
class StateEvent:
    seq: int
    event: str
    payload: dict[str, Any]


def format_sse_event(event: str, payload: dict[str, Any], *, event_id: Optional[int] = None) -> str:
    """Return one SSE frame."""
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


class StatePublisher:
    """Ring-buffered publisher for status SSE clients."""

    def __init__(self, *, max_events: int = 256) -> None:
        self._events: Deque[StateEvent] = deque(maxlen=max_events)
        self._condition = threading.Condition()

    def publish(self, event: StateEvent) -> None:
        with self._condition:
            self._events.append(event)
            self._condition.notify_all()

    def events_since(self, last_seq: int) -> list[StateEvent]:
        with self._condition:
            return [event for event in self._events if event.seq > last_seq]

    def iter_sse(
        self,
        snapshot: dict[str, Any],
        *,
        last_seq: int = 0,
        heartbeat_s: float = 10.0,
    ) -> Iterable[str]:
        """Yield SSE frames from the current snapshot and subsequent events."""
        current_seq = int(snapshot.get("seq", 0))
        if current_seq > last_seq:
            yield format_sse_event(
                "state.snapshot",
                {"type": "state.snapshot", **snapshot},
                event_id=current_seq,
            )
            last_seq = current_seq

        next_heartbeat = time.monotonic() + heartbeat_s
        while True:
            with self._condition:
                timeout = max(next_heartbeat - time.monotonic(), 0.0)
                self._condition.wait(timeout=timeout)
                events = [event for event in self._events if event.seq > last_seq]

            if events:
                for event in events:
                    yield format_sse_event(event.event, event.payload, event_id=event.seq)
                    last_seq = event.seq
                next_heartbeat = time.monotonic() + heartbeat_s
                continue

            if time.monotonic() >= next_heartbeat:
                yield format_sse_event(
                    "heartbeat",
                    {"type": "heartbeat", "seq": last_seq, "ts": time.time()},
                    event_id=last_seq if last_seq > 0 else None,
                )
                next_heartbeat = time.monotonic() + heartbeat_s
