from __future__ import annotations

import numpy as np
import pytest

from codroid_picker.config import AppSettings
from codroid_picker.core.models import benchmark_rating, extract_model_labels, list_model_files
from codroid_picker.core.robot_errors import detect_robot_error
from codroid_picker.core.safety_guard import desired_operation_action, evaluate_safety_state, normalize_safety_labels
from codroid_picker.core.target_planner import plan_pick_strategy, rank_targets
from codroid_picker.hardware.robot_codroid import CLEAR_ERROR_METHODS
from codroid_picker.services.detection_service import DetectionService
from codroid_picker.state.store import StateStore
from codroid_picker.video.frame_buffer import RawFrameBuffer
from codroid_picker.video.overlays import render_overlay


def test_robot_error_detection_finds_legacy_status_patterns() -> None:
    assert detect_robot_error({"emergencyStop": True})[0] == "emergency"
    assert detect_robot_error({"alarms": ["joint collision warning"]})[0] == "joint_protection"
    assert detect_robot_error({"state": 3}) == (None, None, True)


def test_safety_guard_normalizes_and_requests_owned_pause_resume() -> None:
    labels = normalize_safety_labels(["Person", "person", " people "])
    blocked, matches = evaluate_safety_state(
        [{"label": "person", "workspace_mm": [10, 10]}, {"label": "orange", "workspace_mm": [20, 20]}],
        enabled=True,
        labels=labels,
        zone="full_frame",
        workspace_def=None,
    )

    assert labels == ["person", "people"]
    assert blocked is True
    assert len(matches) == 1
    assert desired_operation_action(status="running", pause_reason=None, blocked=True) == "pause"
    assert desired_operation_action(status="paused", pause_reason="person_detected", blocked=False) == "resume"


def test_overlay_modes_change_frame_pixels() -> None:
    frame = np.zeros((180, 240, 3), dtype=np.uint8)
    state = {
        "detection": {"detections": [{"label": "orange", "confidence": 0.9, "bbox": [20, 30, 70, 90]}]},
        "calibration": {"workspace_status": "loaded", "plate": {"ok": True, "count": 4}},
        "auto_pick": {"plan_snapshot": {"available": True, "mode": "rotate_pick", "candidate_count": 1}},
    }

    rendered = render_overlay(frame, mode="auto_plan", state=state)

    assert int(rendered.sum()) > int(frame.sum())


def test_overlay_draws_safety_and_drop_target_context() -> None:
    frame = np.zeros((180, 240, 3), dtype=np.uint8)
    state = {
        "auto_pick": {
            "drop_target": {"x": 40.0, "y": 50.0},
            "plan_snapshot": {"available": True, "selected": {"coord5_xy": [90.0, 75.0]}, "candidate_count": 1},
        },
        "robot": {"status": "online", "coordinate_id": 5, "posture": {"x": 1, "y": 2, "z": 3}, "stale": False},
        "safety": {"blocked": True, "matches": [{"label": "person"}]},
    }

    rendered = render_overlay(frame, mode="auto", state=state)

    assert int(rendered.sum()) > int(frame.sum())
    assert rendered[5, 5].sum() > 0


@pytest.mark.parametrize("mode", ["auto", "calibration", "workspace", "plate", "detection", "auto_plan"])
def test_each_active_overlay_mode_changes_pixels(mode: str) -> None:
    frame = np.zeros((180, 240, 3), dtype=np.uint8)
    state = {
        "detection": {"detections": [{"label": "orange", "confidence": 0.9, "bbox": [20, 30, 70, 90]}]},
        "calibration": {"workspace_status": "ready", "plate": {"ok": True, "count": 4}},
        "auto_pick": {
            "drop_target": {"x": 40, "y": 60},
            "plan_snapshot": {"available": True, "mode": "rotate_pick", "candidate_count": 1, "selected": {"coord5_xy": [90, 75]}},
        },
        "robot": {"status": "online", "coordinate_id": 5, "posture": {"x": 1, "y": 2, "z": 3}, "stale": False},
        "safety": {"blocked": False, "matches": []},
    }

    rendered = render_overlay(frame, mode=mode, state=state)

    assert int(np.abs(rendered.astype(int) - frame.astype(int)).sum()) > 0


def test_none_overlay_leaves_frame_unchanged() -> None:
    frame = np.full((64, 96, 3), 42, dtype=np.uint8)

    rendered = render_overlay(frame, mode="none", state={})

    assert np.array_equal(rendered, frame)


def test_detection_class_filter_parser_accepts_comma_separated_ids() -> None:
    assert DetectionService._parse_classes("0, 32") == [0, 32]
    assert DetectionService._parse_classes("") is None
    assert DetectionService._parse_classes("person") is None


def test_operation_pause_owner_prevents_wrong_resume_owner() -> None:
    state = StateStore()
    state.begin_operation(source="auto", label="Auto", message="Running.")
    state.pause_operation("person_detected", "Safety pause.", owner="safety")

    state.resume_operation("Operator tried resume.", owner="operator")
    assert state.domain("operation")["status"] == "paused"
    assert state.domain("operation")["pause_owner"] == "safety"

    state.resume_operation("Safety clear.", owner="safety")
    assert state.domain("operation")["status"] == "running"
    assert state.domain("operation")["pause_owner"] is None


def test_native_target_planner_prefers_clearer_target_and_falls_back_to_separation() -> None:
    detections = [
        {"id": "crowded", "coord5_mm": [0.0, 0.0]},
        {"id": "blocker", "coord5_mm": [8.0, 0.0]},
        {"id": "clear", "coord5_mm": [80.0, 0.0]},
    ]
    ranked = rank_targets(
        detections,
        robot_xy=(0.0, 0.0),
        coord5_fn=lambda item: tuple(item["coord5_mm"]),
        neighbor_radius_mm=20.0,
    )

    assert ranked[0].detection["id"] == "clear"

    plan = plan_pick_strategy(
        ranked[1],
        ranked,
        current_c_deg=0.0,
        rotation_step_deg=45.0,
        rotation_max_delta_deg=0.0,
        body_radius_mm=20.0,
        finger_reach_mm=20.0,
        clearance_mm=5.0,
        separation_entry_mm=10.0,
        separation_push_mm=15.0,
    )
    assert plan.mode == "separate_then_rescan"
    assert plan.separation is not None


def test_model_helpers_and_robot_recovery_mapping_are_native() -> None:
    class Model:
        names = {1: "orange", 0: "person", 2: ""}

    assert extract_model_labels(Model()) == ["person", "orange"]
    assert benchmark_rating(80.0) == "usable"
    assert benchmark_rating(250.0) == "warning"
    assert benchmark_rating(500.0) == "poor"
    assert CLEAR_ERROR_METHODS["wrong_tool"][0] == "clear_tool_error"
    assert "clear_robot_error" in CLEAR_ERROR_METHODS["drag_not_allowed"]


def test_model_file_discovery_filters_supported_suffixes(tmp_path) -> None:
    for name in ("a.pt", "b.onnx", "c.engine", "notes.txt", "nested.pt"):
        path = tmp_path / name
        if name == "nested.pt":
            path.mkdir()
        else:
            path.write_text("x")

    assert [path.name for path in list_model_files(tmp_path)] == ["a.pt", "b.onnx", "c.engine"]


def test_label_extraction_supports_callable_names() -> None:
    class Model:
        def names(self) -> list[str]:
            return ["person", "orange"]

    assert extract_model_labels(Model()) == ["person", "orange"]


def test_codroid_api_recovery_methods_exist_when_dependency_is_available() -> None:
    codroid_client = pytest.importorskip("codroid_api.client")
    api = codroid_client.CodroidAPI
    for methods in CLEAR_ERROR_METHODS.values():
        assert any(hasattr(api, name) for name in methods)
    for name in ("hold_jog", "stop_command", "set_speed_multiplier"):
        assert hasattr(api, name)


def test_detection_activation_reports_missing_ml_extras(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    model_path = tmp_path / "model.pt"
    model_path.write_bytes(b"fake")
    monkeypatch.setattr("codroid_picker.services.detection_service.importlib.util.find_spec", lambda name: None)
    settings = AppSettings(detection_worker=False, model_dir=tmp_path)
    state = StateStore()
    service = DetectionService(settings, state, RawFrameBuffer())

    result = service.activate(model_path)

    assert result["active_model"] is None
    assert result["ml_available"] is False
    assert "ML extras are not installed" in result["last_error"]


def test_detection_history_is_trimmed_to_recent_entries(tmp_path) -> None:
    service = DetectionService(AppSettings(detection_worker=False, model_dir=tmp_path), StateStore(), RawFrameBuffer())

    for index in range(95):
        service._publish_detections([{"label": "orange", "confidence": index / 100}], latency_ms=float(index))

    history = service.state.domain("monitoring")["detection_history"]
    assert len(history) == 90
    assert history[0]["latency_ms"] == 5.0
    assert history[-1]["latency_ms"] == 94.0


def test_detection_frame_skip_returns_cached_detections(tmp_path) -> None:
    state = StateStore()
    service = DetectionService(AppSettings(detection_worker=False, detection_frame_skip=3, model_dir=tmp_path), state, RawFrameBuffer())
    state.update_domain("detection", detections=[{"label": "cached"}], frame_skip=3)

    detections = service.run_once(force=False)

    assert detections == [{"label": "cached"}]
