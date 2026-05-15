from __future__ import annotations

import json

from codroid_picker.state import StatePublisher, StateStore, format_sse_event


def test_state_store_publishes_delta_events() -> None:
    publisher = StatePublisher()
    store = StateStore(publisher)

    snapshot = store.update_domain("camera", status="live", running=True)
    events = publisher.events_since(0)

    assert snapshot["seq"] == 1
    assert len(events) == 1
    assert events[0].event == "state.delta"
    assert events[0].payload["changed"] == ["camera"]
    assert events[0].payload["camera"]["status"] == "live"


def test_format_sse_event_round_trips_json_data() -> None:
    frame = format_sse_event("state.delta", {"seq": 3, "ok": True}, event_id=3)

    assert "id: 3\n" in frame
    assert "event: state.delta\n" in frame
    data_line = next(line for line in frame.splitlines() if line.startswith("data: "))
    assert json.loads(data_line.removeprefix("data: ")) == {"seq": 3, "ok": True}


def test_state_store_sequence_increments_for_representative_domains() -> None:
    publisher = StatePublisher()
    store = StateStore(publisher)

    store.update_domain("robot", status="online")
    store.update_domain("gripper", status="online")
    store.update_domain("detection", detections=[{"label": "orange"}])
    store.update_domain("safety", blocked=True)
    snapshot = store.update_domain("monitoring", plate_test={"status": "dry_run"})

    assert snapshot["seq"] == 5
    assert [event.payload["changed"][0] for event in publisher.events_since(0)] == [
        "robot",
        "gripper",
        "detection",
        "safety",
        "monitoring",
    ]


def test_state_publisher_yields_initial_snapshot_and_delta() -> None:
    publisher = StatePublisher()
    store = StateStore(publisher)
    snapshot = store.update_domain("camera", status="live")
    frames = publisher.iter_sse(snapshot, heartbeat_s=0.0)

    first = next(frames)
    store.update_domain("robot", status="online")
    second = next(frames)

    assert "event: state.snapshot" in first
    first_data = json.loads(next(line for line in first.splitlines() if line.startswith("data: ")).removeprefix("data: "))
    assert first_data["type"] == "state.snapshot"
    assert first_data["camera"]["status"] == "live"
    assert "event: state.delta" in second
    second_data = json.loads(next(line for line in second.splitlines() if line.startswith("data: ")).removeprefix("data: "))
    assert second_data["changed"] == ["robot"]
    assert second_data["robot"]["status"] == "online"
