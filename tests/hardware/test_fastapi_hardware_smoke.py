from __future__ import annotations

import os
from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from codroid_picker.app import create_app


pytestmark = pytest.mark.hardware


def _enabled(name: str) -> bool:
    return os.getenv(name) == "1"


def _client() -> Iterator[TestClient]:
    with TestClient(create_app()) as client:
        yield client


def _require_smoke() -> None:
    if not _enabled("CODROID_HARDWARE_SMOKE"):
        pytest.skip("Set CODROID_HARDWARE_SMOKE=1 on the robot cell to run hardware smoke tests.")


def _require(name: str) -> None:
    _require_smoke()
    if not _enabled(name):
        pytest.skip(f"Set {name}=1 when that hardware is connected and ready.")


def test_fastapi_hardware_health_route() -> None:
    _require_smoke()
    with TestClient(create_app()) as client:
        assert client.get("/health").status_code == 200


def test_camera_start_snapshot_overlay_and_stop() -> None:
    _require("CODROID_HARDWARE_CAMERA")
    for client in _client():
        assert client.get("/camera/devices").status_code == 200
        start = client.post("/camera/start", json={})
        assert start.status_code == 200, start.text
        snapshot = client.get("/video/snapshot.jpg?overlay=auto")
        assert snapshot.status_code == 200
        assert snapshot.content.startswith(b"\xff\xd8")
        assert client.put("/camera/overlay", json={"mode": "detection"}).status_code == 200
        assert client.post("/camera/stop").status_code == 200


def test_robot_connect_acquire_speed_stop_and_release() -> None:
    _require("CODROID_HARDWARE_ROBOT")
    for client in _client():
        assert client.post("/robot/connect", json={}).status_code == 200
        assert client.post("/robot/acquire").status_code == 200
        assert client.post("/robot/speed", json={"rate": 0.25}).status_code == 200
        assert client.post("/robot/stop").status_code == 200
        state = client.get("/state").json()
        assert state["robot"]["poll_age_s"] is not None
        assert client.post("/robot/release").status_code == 200


@pytest.mark.parametrize("axis", ["x", "y", "z", "a", "b", "c"])
def test_robot_tcp_jog_axes(axis: str) -> None:
    _require("CODROID_HARDWARE_ROBOT_JOG")
    for client in _client():
        assert client.post("/robot/connect", json={}).status_code == 200
        assert client.post("/robot/acquire").status_code == 200
        response = client.post("/robot/jog", json={"axis": axis, "direction": 1, "mode": "standard", "hold_s": 0.05})
        assert response.status_code == 200, response.text
        assert client.post("/robot/stop").status_code == 200
        assert client.post("/robot/release").status_code == 200


@pytest.mark.parametrize("axis", ["j1", "j2", "j3", "j4", "j5", "j6"])
def test_robot_rescue_jog_axes(axis: str) -> None:
    _require("CODROID_HARDWARE_ROBOT_RESCUE")
    for client in _client():
        assert client.post("/robot/connect", json={}).status_code == 200
        assert client.post("/robot/acquire").status_code == 200
        response = client.post("/robot/jog", json={"axis": axis, "direction": 1, "mode": "rescue", "hold_s": 0.05})
        assert response.status_code == 200, response.text
        assert client.post("/robot/stop").status_code == 200
        assert client.post("/robot/release").status_code == 200


@pytest.mark.parametrize("error_type", ["emergency", "overspeed", "joint_protection", "wrong_tool", "drag_not_allowed"])
def test_robot_recovery_acknowledgement(error_type: str) -> None:
    _require("CODROID_HARDWARE_ROBOT_RECOVERY")
    for client in _client():
        assert client.post("/robot/connect", json={}).status_code == 200
        assert client.post("/robot/acquire").status_code == 200
        response = client.post("/robot/errors/ack", json={"error_type": error_type})
        assert response.status_code == 200, response.text
        assert client.post("/robot/release").status_code == 200


def test_gripper_connect_open_close_orientation_and_disconnect() -> None:
    _require("CODROID_HARDWARE_GRIPPER")
    for client in _client():
        assert client.post("/gripper/connect", json={}).status_code == 200
        assert client.post("/gripper/open", json={"width_mm": 60, "speed": 40}).status_code == 200
        assert client.post("/gripper/close", json={"width_mm": 20, "force": 20, "speed": 30}).status_code == 200
        orientation = client.post("/gripper/orientation", json={"orientation": "inward"})
        assert orientation.status_code == 200
        assert client.post("/gripper/disconnect").status_code == 200


def test_plate_test_dry_run_with_loaded_calibration() -> None:
    _require("CODROID_HARDWARE_PLATE_DRY_RUN")
    for client in _client():
        assert client.post("/calibration/robot/load", json={}).status_code == 200
        response = client.post(
            "/plate-test/run",
            json={"approach_mm": 5, "pick_mm": 0, "drop_mm": 0, "grip_timeout_s": 0, "dry_run": True},
        )
        assert response.status_code == 200, response.text
        summary = response.json()["state"]["plate_test"]
        assert summary["dry_run"] is True
        assert summary["results"]
