from __future__ import annotations

import time
from pathlib import Path

from fastapi.testclient import TestClient

from codroid_picker.app import create_app


def _use_temp_robot_calibration(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("CODROID_USE_FAKE_HARDWARE", "true")
    monkeypatch.setenv("CODROID_SETTINGS_PATH", str(tmp_path / "app_settings.json"))
    monkeypatch.setenv("CODROID_WORKSPACE_PATH", str(tmp_path / "workspace_definition.npz"))
    monkeypatch.setenv("CODROID_CALIBRATION_BUNDLE_PATH", str(tmp_path / "calibration_bundle.npz"))
    monkeypatch.setenv("CODROID_DROP_TARGET_PATH", str(tmp_path / "drop_target.json"))
    monkeypatch.setenv("CODROID_ROBOT_CALIBRATION_PATH", str(tmp_path / "robot_calibration.json"))
    monkeypatch.setenv("CODROID_VERIFICATION_SETTLE_S", "0")
    monkeypatch.setenv("CODROID_VERIFICATION_RAISE_MM", "10")


def _move_robot(client: TestClient, x: float, y: float, z: float) -> None:
    response = client.post(
        "/robot/move",
        json={"pose": {"x": x, "y": y, "z": z, "a": 0, "b": 0, "c": 0}, "coordinate_id": 0, "timeout_s": 1},
    )
    assert response.status_code == 200


def test_health_and_state_endpoints() -> None:
    with TestClient(create_app()) as client:
        health = client.get("/health")
        state = client.get("/state")

    assert health.status_code == 200
    assert health.json()["ok"] is True
    assert state.status_code == 200
    assert "camera" in state.json()


def test_video_snapshot_returns_jpeg() -> None:
    with TestClient(create_app()) as client:
        response = client.get("/video/snapshot.jpg")

    assert response.status_code == 200
    assert response.headers["content-type"] == "image/jpeg"
    assert response.content.startswith(b"\xff\xd8")


def test_operator_page_contains_restored_workflow_regions() -> None:
    with TestClient(create_app()) as client:
        response = client.get("/")

    assert response.status_code == 200
    html = response.text
    assert "Setup & Connections" in html
    assert "Tracking Controls" in html
    assert "Model idle." in html
    assert "Plate Pick & Place Test" in html
    assert "Calibration Steps" in html
    assert "System Summary" in html
    assert "Robot Health" in html
    assert "Emergency stop triggered" in html
    assert "Rescue joint control" in html


def test_locale_switch_persists_and_serves_uk_catalog(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("CODROID_SETTINGS_PATH", str(tmp_path / "app_settings.json"))
    with TestClient(create_app()) as client:
        switch = client.post("/locale", json={"language": "uk"})
        page = client.get("/")

    assert switch.status_code == 200
    assert switch.json()["state"]["language"] == "uk"
    assert 'lang="uk"' in page.text
    assert "Налаштування та підключення" in page.text
    assert (tmp_path / "app_settings.json").exists()


def test_fake_camera_start_updates_state() -> None:
    with TestClient(create_app()) as client:
        response = client.post(
            "/camera/start",
            json={"source": "fake", "width": 320, "height": 180, "fps": 5},
        )
        state = client.get("/state").json()

    assert response.status_code == 200
    assert response.json()["ok"] is True
    assert state["camera"]["status"] == "live"
    assert state["camera"]["running"] is True


def test_overlay_selection_publishes_video_state() -> None:
    with TestClient(create_app()) as client:
        response = client.put("/camera/overlay", json={"mode": "auto_plan"})
        state = client.get("/state").json()

    assert response.status_code == 200
    assert state["video"]["overlay_mode"] == "auto_plan"


def test_robot_speed_requires_connection_without_500() -> None:
    with TestClient(create_app()) as client:
        response = client.post("/robot/speed", json={"rate": 0.5})

    assert response.status_code == 409
    assert response.json()["message"] == "Robot is not connected."


def test_robot_speed_updates_connected_fake_robot(monkeypatch, tmp_path: Path) -> None:
    _use_temp_robot_calibration(monkeypatch, tmp_path)
    with TestClient(create_app()) as client:
        assert client.post("/robot/connect", json={}).status_code == 200
        response = client.post("/robot/speed", json={"rate": 0.5})
        state = client.get("/state").json()

    assert response.status_code == 200
    assert state["robot"]["speed_rate"] == 0.5


def test_manual_pick_closest_executes_with_seeded_detection() -> None:
    with TestClient(create_app()) as client:
        ctx = client.app.state.context
        ctx.state.update_domain(
            "detection",
            detections=[{"label": "orange", "confidence": 0.9, "coord5_mm": [25.0, -10.0]}],
            updated_at=time.time(),
        )

        response = client.post("/pick/closest")
        state = client.get("/state").json()

    assert response.status_code == 200
    assert response.json()["ok"] is True
    assert "Pick completed" in response.json()["message"]
    assert state["operation"]["status"] == "idle"
    assert state["monitoring"]["pick_successes"] >= 1


def test_pick_and_drop_executes_with_seeded_drop_target() -> None:
    with TestClient(create_app()) as client:
        ctx = client.app.state.context
        ctx.calibration.drop_target = {
            "x": 100.0,
            "y": 50.0,
            "robot_x": 100.0,
            "robot_y": 50.0,
            "robot_z": -40.0,
            "a": 0.0,
            "b": 0.0,
            "c": 0.0,
            "coord_id": 5,
        }
        ctx.state.update_domain("auto_pick", drop_target=ctx.calibration.drop_target)
        ctx.state.update_domain(
            "detection",
            detections=[{"label": "orange", "confidence": 0.9, "coord5_mm": [25.0, -10.0]}],
            updated_at=time.time(),
        )

        response = client.post("/pick/pick-and-drop")
        state = client.get("/state").json()

    assert response.status_code == 200
    assert response.json()["ok"] is True
    assert response.json()["message"] == "Drop completed."
    assert state["operation"]["status"] == "idle"
    assert state["gripper"]["holding"] is False


def test_auto_pick_loop_executes_seeded_detection_then_stops() -> None:
    with TestClient(create_app()) as client:
        ctx = client.app.state.context
        ctx.calibration.drop_target = {
            "x": 100.0,
            "y": 50.0,
            "robot_x": 100.0,
            "robot_y": 50.0,
            "robot_z": -40.0,
            "a": 0.0,
            "b": 0.0,
            "c": 0.0,
            "coord_id": 5,
        }
        ctx.state.update_domain("auto_pick", drop_target=ctx.calibration.drop_target)
        ctx.state.update_domain(
            "detection",
            detections=[{"label": "orange", "confidence": 0.9, "coord5_mm": [25.0, -10.0]}],
            updated_at=time.time(),
        )

        start = client.post("/auto/start")
        time.sleep(0.2)
        stop = client.post("/auto/stop")
        state = client.get("/state").json()

    assert start.status_code == 200
    assert stop.status_code == 200
    assert state["auto_pick"]["running"] is False
    assert state["auto_pick"]["plan_snapshot"]["available"] is True
    assert state["monitoring"]["pick_successes"] >= 1


def test_robot_calibration_teach_capture_and_finish(monkeypatch, tmp_path: Path) -> None:
    _use_temp_robot_calibration(monkeypatch, tmp_path)
    with TestClient(create_app()) as client:
        assert client.post("/robot/connect", json={}).status_code == 200

        start = client.post("/calibration/robot/start")
        _move_robot(client, 70.0, 70.0, 12.0)
        point_1 = client.post("/calibration/robot/point", json={"index": 1})
        _move_robot(client, 230.0, 70.0, 12.0)
        point_2 = client.post("/calibration/robot/point", json={"index": 2})
        _move_robot(client, 230.0, 210.0, 12.0)
        point_3 = client.post("/calibration/robot/point", json={"index": 3})
        finish = client.post("/calibration/robot/finish")
        state = client.get("/state").json()

    assert start.status_code == 200
    assert point_1.status_code == 200
    assert point_2.status_code == 200
    assert point_3.status_code == 200
    assert finish.status_code == 200
    assert state["calibration"]["robot_status"] == "accuracy_ready"
    assert state["calibration"]["robot_verification_ready"] is True
    assert state["calibration"]["robot_controller_applied"] is True
    assert state["calibration"]["robot_controller_frame"]["id"] == 5.0
    assert len(state["calibration"]["robot_points"]) == 3
    assert state["calibration"]["accuracy"]["max_xy_mm"] < 1e-6
    assert state["robot"]["coordinate_id"] == 5
    assert (tmp_path / "robot_calibration.json").exists()


def test_robot_calibration_verify_moves_through_saved_points(monkeypatch, tmp_path: Path) -> None:
    _use_temp_robot_calibration(monkeypatch, tmp_path)
    with TestClient(create_app()) as client:
        assert client.post("/robot/connect", json={}).status_code == 200
        assert client.post("/calibration/robot/start").status_code == 200
        _move_robot(client, 70.0, 70.0, 12.0)
        assert client.post("/calibration/robot/point", json={"index": 1}).status_code == 200
        _move_robot(client, 230.0, 70.0, 12.0)
        assert client.post("/calibration/robot/point", json={"index": 2}).status_code == 200
        _move_robot(client, 230.0, 210.0, 12.0)
        assert client.post("/calibration/robot/point", json={"index": 3}).status_code == 200
        assert client.post("/calibration/robot/finish").status_code == 200

        verify = client.post("/calibration/robot/verify")
        state = client.get("/state").json()

    assert verify.status_code == 200
    assert verify.json()["message"] == "Robot calibration verification complete."
    assert state["calibration"]["robot_status"] == "complete"
    assert state["operation"]["status"] == "idle"
    assert state["robot"]["coordinate_id"] == 5
    assert state["robot"]["posture"]["x"] == 0.0
    assert state["robot"]["posture"]["y"] == 0.0
