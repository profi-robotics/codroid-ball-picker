"""Tests for UI snapshot diffing and SSE helpers."""

from __future__ import annotations

import json
import threading
import time
import unittest

from app.codroid_picker.ui_snapshots import (
    SingleFlightWorker,
    UiSnapshotService,
    diff_snapshot_fields,
    format_sse_event,
    snapshot_delta,
)


def _snapshot(camera_status: str = "off") -> dict:
    return {
        "camera": {"status": camera_status},
        "robot": {"status": "offline"},
        "gripper": {"status": "offline"},
        "operation": {"status": "idle"},
        "safety": {"blocked": False},
        "model": {"loaded": False},
        "detections": {"count": 0},
        "monitoring": {"pick_attempts": 0},
    }


class UiSnapshotTests(unittest.TestCase):
    def test_diff_snapshot_fields_ignores_sequence_metadata(self) -> None:
        previous = {"seq": 1, "ts": 1.0, **_snapshot("live")}
        current = {"seq": 2, "ts": 2.0, **_snapshot("live")}

        self.assertEqual(diff_snapshot_fields(previous, current), [])

    def test_diff_snapshot_fields_reports_changed_top_level_fields(self) -> None:
        previous = _snapshot("off")
        current = _snapshot("retrying")

        self.assertEqual(diff_snapshot_fields(previous, current), ["camera"])

    def test_format_sse_event_emits_id_event_and_json_data(self) -> None:
        frame = format_sse_event("snapshot", {"seq": 3, "camera": {"status": "live"}}, event_id=3)

        self.assertIn("id: 3\n", frame)
        self.assertIn("event: snapshot\n", frame)
        data_line = next(line for line in frame.splitlines() if line.startswith("data: "))
        self.assertEqual(json.loads(data_line.removeprefix("data: ")), {"seq": 3, "camera": {"status": "live"}})

    def test_snapshot_delta_contains_only_changed_fields(self) -> None:
        snapshot = {"seq": 7, "ts": 1.5, **_snapshot("live"), "changed": ["camera"]}

        self.assertEqual(
            snapshot_delta(snapshot, ["camera"]),
            {"seq": 7, "ts": 1.5, "changed": ["camera"], "camera": {"status": "live"}},
        )

    def test_service_increments_sequence_when_snapshot_changes(self) -> None:
        states = [_snapshot("off"), _snapshot("live")]

        def _collector() -> dict:
            return states.pop(0) if states else _snapshot("live")

        service = UiSnapshotService(_collector, poll_interval_s=1.0)
        first = service.force_publish()
        second = service.force_publish()

        self.assertIsNotNone(first)
        self.assertIsNotNone(second)
        assert first is not None
        assert second is not None
        self.assertEqual(first["seq"], 1)
        self.assertEqual(second["seq"], 2)
        self.assertEqual(second["changed"], ["camera"])

    def test_service_buffers_delta_events_after_initial_publish(self) -> None:
        states = [_snapshot("off"), _snapshot("live")]

        def _collector() -> dict:
            return states.pop(0) if states else _snapshot("live")

        service = UiSnapshotService(_collector, poll_interval_s=1.0)
        service.force_publish()
        second = service._publish_if_changed()
        assert second is not None

        events = service.events_since(1)

        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].payload["changed"], ["camera"])
        self.assertEqual(events[0].payload["camera"], {"status": "live"})
        self.assertNotIn("robot", events[0].payload)

    def test_single_flight_worker_rejects_overlapping_runs(self) -> None:
        worker = SingleFlightWorker()
        entered = threading.Event()
        release = threading.Event()
        results: list[bool] = []

        def _slow() -> None:
            entered.set()
            release.wait(timeout=1.0)

        thread = threading.Thread(target=lambda: results.append(worker.run(_slow)))
        thread.start()
        self.assertTrue(entered.wait(timeout=1.0))

        self.assertFalse(worker.run(lambda: None))
        release.set()
        thread.join(timeout=1.0)
        self.assertEqual(results, [True])

    def test_iter_sse_yields_heartbeat_when_no_snapshot_changes(self) -> None:
        service = UiSnapshotService(lambda: _snapshot("off"), poll_interval_s=1.0)
        service.force_publish()
        events = service.iter_sse(last_seq=1, heartbeat_s=0.01)

        frame = next(events)

        self.assertIn("event: heartbeat", frame)
        self.assertIn("data: ", frame)


if __name__ == "__main__":
    unittest.main()
