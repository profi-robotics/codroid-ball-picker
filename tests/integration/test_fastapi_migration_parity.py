from __future__ import annotations

import re
from pathlib import Path

import cv2
import numpy as np
from fastapi.testclient import TestClient

from codroid_picker.app import create_app


def _fake_env(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("CODROID_USE_FAKE_HARDWARE", "true")
    monkeypatch.setenv("CODROID_SETTINGS_PATH", str(tmp_path / "app_settings.json"))
    monkeypatch.setenv("CODROID_ROBOT_CALIBRATION_PATH", str(tmp_path / "robot_calibration.json"))
    monkeypatch.setenv("CODROID_DROP_TARGET_PATH", str(tmp_path / "drop_target.json"))
    monkeypatch.setenv("CODROID_PLATE_TEST_CAPTURE_DIR", str(tmp_path / "plate_runs"))
    monkeypatch.setenv("CODROID_VERIFICATION_SETTLE_S", "0")


def _move_robot(client: TestClient, x: float, y: float, z: float) -> None:
    response = client.post(
        "/robot/move",
        json={"pose": {"x": x, "y": y, "z": z, "a": 0, "b": 0, "c": 0}, "coordinate_id": 0, "timeout_s": 1},
    )
    assert response.status_code == 200


def _seed_robot_calibration(client: TestClient) -> None:
    assert client.post("/robot/connect", json={}).status_code == 200
    assert client.post("/calibration/robot/start").status_code == 200
    _move_robot(client, 70.0, 70.0, 12.0)
    assert client.post("/calibration/robot/point", json={"index": 1}).status_code == 200
    _move_robot(client, 230.0, 70.0, 12.0)
    assert client.post("/calibration/robot/point", json={"index": 2}).status_code == 200
    _move_robot(client, 230.0, 210.0, 12.0)
    assert client.post("/calibration/robot/point", json={"index": 3}).status_code == 200
    assert client.post("/calibration/robot/finish").status_code == 200


def test_every_declared_ui_command_has_a_route() -> None:
    with TestClient(create_app()) as client:
        html = client.get("/").text
        paths = sorted(set(re.findall(r'data-command="([^"]+)"', html)))
        routes = {route.path for route in client.app.routes}

    assert paths
    assert not [path for path in paths if path not in routes]


def test_every_declared_ui_payload_source_is_supported() -> None:
    with TestClient(create_app()) as client:
        html = client.get("/").text
        js = client.get("/static/js/app.js").text

    sources = sorted(set(re.findall(r'data-payload-source="([^"]+)"', html)))
    handled = set(re.findall(r'if \(source === "([^"]+)"\)', js))

    assert sources
    assert not [source for source in sources if source not in handled]


def test_every_declared_overlay_mode_is_accepted() -> None:
    with TestClient(create_app()) as client:
        html = client.get("/").text
        modes = sorted(set(re.findall(r'data-overlay="([^"]+)"', html)))
        for mode in modes:
            response = client.put("/camera/overlay", json={"mode": mode})
            assert response.status_code == 200, mode
            assert response.json()["state"]["video"]["overlay_mode"] == mode.replace("-", "_")

    assert modes == ["auto", "auto_plan", "calibration", "detection", "none", "plate", "workspace"]


def test_declared_ui_requirements_are_implemented_in_frontend_logic() -> None:
    with TestClient(create_app()) as client:
        html = client.get("/").text
        js = client.get("/static/js/sse.js").text

    requirements = sorted(set(re.findall(r'data-requires="([^"]+)"', html)))
    for requirement in requirements:
        assert f'data-requires="{requirement}"' in js or f"data-requires='{requirement}'" in js

    assert requirements == ["auto-ready", "robot-verification"]


def test_robot_error_dialogs_have_command_buttons() -> None:
    with TestClient(create_app()) as client:
        html = client.get("/").text

    dialogs = sorted(set(re.findall(r'data-error-dialog="([^"]+)"', html)))
    for dialog in dialogs:
        pattern = rf'data-error-dialog="{re.escape(dialog)}"[\s\S]*?data-command="/robot/errors/ack"'
        assert re.search(pattern, html), dialog

    assert dialogs == ["drag", "emergency", "joint_protection", "overspeed", "wrong_tool"]


def test_snapshot_overlay_modes_change_jpeg_pixels() -> None:
    with TestClient(create_app()) as client:
        ctx = client.app.state.context
        ctx.state.update_domain(
            "detection",
            detections=[{"label": "orange", "confidence": 0.9, "bbox": [20, 30, 120, 140]}],
        )
        plain = client.get("/video/snapshot.jpg?overlay=none").content
        overlaid = client.get("/video/snapshot.jpg?overlay=detection").content

    plain_frame = cv2.imdecode(np.frombuffer(plain, dtype=np.uint8), cv2.IMREAD_COLOR)
    overlaid_frame = cv2.imdecode(np.frombuffer(overlaid, dtype=np.uint8), cv2.IMREAD_COLOR)

    assert plain_frame is not None
    assert overlaid_frame is not None
    assert plain_frame.shape == overlaid_frame.shape
    assert int(np.abs(overlaid_frame.astype(int) - plain_frame.astype(int)).sum()) > 0


def test_each_snapshot_overlay_mode_changes_jpeg_pixels() -> None:
    with TestClient(create_app()) as client:
        ctx = client.app.state.context
        ctx.state.update_domain(
            "detection",
            detections=[{"label": "orange", "confidence": 0.9, "bbox": [20, 30, 120, 140]}],
        )
        ctx.state.update_domain("calibration", workspace_status="ready", plate={"ok": True, "stable": True, "count": 4})
        ctx.state.update_domain(
            "auto_pick",
            drop_target={"x": 40, "y": 60},
            plan_snapshot={"available": True, "mode": "rotate_pick", "candidate_count": 1, "selected": {"coord5_xy": [80, 90]}},
        )
        plain = client.get("/video/snapshot.jpg?overlay=none").content
        plain_frame = cv2.imdecode(np.frombuffer(plain, dtype=np.uint8), cv2.IMREAD_COLOR)
        assert plain_frame is not None
        for mode in ["auto", "calibration", "workspace", "plate", "detection", "auto_plan"]:
            content = client.get(f"/video/snapshot.jpg?overlay={mode}").content
            frame = cv2.imdecode(np.frombuffer(content, dtype=np.uint8), cv2.IMREAD_COLOR)
            assert frame is not None
            assert int(np.abs(frame.astype(int) - plain_frame.astype(int)).sum()) > 0, mode


def test_api_contract_shapes(monkeypatch, tmp_path: Path) -> None:
    _fake_env(monkeypatch, tmp_path)
    monkeypatch.setenv("CODROID_MODEL_DIR", str(tmp_path / "models"))
    (tmp_path / "models").mkdir()
    (tmp_path / "models" / "ball.pt").write_bytes(b"fake")

    with TestClient(create_app()) as client:
        state = client.get("/state")
        health = client.get("/health")
        video_stats = client.get("/video/stats")
        models = client.get("/models")
        benchmark = client.post("/models/benchmark")
        auto_config = client.get("/auto/config")
        detection_params = client.get("/detection/params")
        response = client.post("/operation/reset")

    assert state.status_code == 200
    assert {"camera", "robot", "gripper", "operation", "detection", "monitoring"} <= set(state.json())
    assert health.status_code == 200
    assert {"ok", "version", "state_seq", "camera", "robot", "worker"} <= set(health.json())
    assert video_stats.status_code == 200
    assert {"transport", "clients", "frames_encoded", "encoded_fps"} <= set(video_stats.json())
    assert models.status_code == 200
    assert models.json()["models"][0]["name"] == "ball.pt"
    assert benchmark.status_code == 200
    assert benchmark.json()["status"] == "unavailable"
    assert auto_config.status_code == 200
    assert {"running", "target_labels", "plan_snapshot", "drop_target"} <= set(auto_config.json())
    assert detection_params.status_code == 200
    assert {"confidence", "iou", "max_detections", "frame_skip", "detections"} <= set(detection_params.json())
    assert response.status_code == 200
    assert response.json()["ok"] is True
    assert response.json()["message"] == "Operation reset."


def test_benchmark_reports_unavailable_without_active_model() -> None:
    with TestClient(create_app()) as client:
        response = client.post("/models/benchmark")
        state = client.get("/state").json()

    assert response.status_code == 200
    assert response.json()["status"] == "unavailable"
    assert state["detection"]["benchmark"]["status"] == "unavailable"


def test_rescue_jog_uses_fake_robot_without_error(monkeypatch, tmp_path: Path) -> None:
    _fake_env(monkeypatch, tmp_path)
    with TestClient(create_app()) as client:
        assert client.post("/robot/connect", json={}).status_code == 200
        response = client.post("/robot/jog", json={"axis": "j1", "direction": 1, "mode": "rescue", "hold_s": 0.01})

    assert response.status_code == 200
    assert response.json()["message"] == "Jogged j1."


def test_plate_test_route_runs_with_fake_hardware(monkeypatch, tmp_path: Path) -> None:
    _fake_env(monkeypatch, tmp_path)
    with TestClient(create_app()) as client:
        _seed_robot_calibration(client)
        _move_robot(client, 100.0, 60.0, 20.0)
        assert client.post("/drop-target/teach").status_code == 200

        response = client.post(
            "/plate-test/run",
            json={"approach_mm": 5, "pick_mm": 0, "drop_mm": 0, "grip_timeout_s": 0},
        )
        state = client.get("/state").json()

    assert response.status_code == 200
    assert response.json()["state"]["plate_test"]["status"] == "completed"
    assert state["monitoring"]["plate_test"]["successes"] >= 1
    assert list((tmp_path / "plate_runs").glob("plate_test_*.json"))


def test_plate_test_dry_run_does_not_require_drop_target(monkeypatch, tmp_path: Path) -> None:
    _fake_env(monkeypatch, tmp_path)
    with TestClient(create_app()) as client:
        _seed_robot_calibration(client)

        response = client.post(
            "/plate-test/run",
            json={"approach_mm": 5, "pick_mm": 0, "drop_mm": 0, "grip_timeout_s": 0, "dry_run": True},
        )

    assert response.status_code == 200
    summary = response.json()["state"]["plate_test"]
    assert summary["status"] == "dry_run"
    assert summary["dry_run"] is True
    assert summary["results"][0]["approach_pose"]


def test_robot_polling_updates_state_with_fake_hardware(monkeypatch, tmp_path: Path) -> None:
    _fake_env(monkeypatch, tmp_path)
    with TestClient(create_app()) as client:
        assert client.post("/robot/connect", json={}).status_code == 200
        state = client.get("/state").json()

    assert state["robot"]["posture"] is not None
    assert state["robot"]["last_seen_at"] is not None


def test_operator_resume_does_not_clear_safety_owned_pause(monkeypatch, tmp_path: Path) -> None:
    _fake_env(monkeypatch, tmp_path)
    with TestClient(create_app()) as client:
        ctx = client.app.state.context
        ctx.state.begin_operation(source="auto", label="Auto", message="Running.")
        ctx.state.pause_operation("person_detected", "Safety pause.", owner="safety")

        response = client.post("/operation/resume")
        state = client.get("/state").json()

    assert response.status_code == 200
    assert state["operation"]["status"] == "paused"
    assert state["operation"]["pause_owner"] == "safety"
    assert state["operation"]["pause_reason"] == "person_detected"
