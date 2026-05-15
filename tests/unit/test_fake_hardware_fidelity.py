from __future__ import annotations

import asyncio
import time

from codroid_picker.hardware.fakes import FakeCameraStream, FakeGripperClient, FakeRobotClient, FakeStreamConfig


def test_fake_robot_can_publish_di_errors_and_stale_posture() -> None:
    robot = FakeRobotClient()
    robot.connected = True
    robot.di_state = {"41": 1}
    robot.raw_status = {"overspeed": True}
    robot.pose_available = False

    snapshot = robot.snapshot()

    assert snapshot["pose"] is None
    assert snapshot["di_state"] == {"41": 1}
    assert snapshot["raw_status"] == {"overspeed": True}


def test_fake_robot_records_clear_error_calls() -> None:
    robot = FakeRobotClient()
    robot.raw_status = {"drag_not_allowed": True}

    asyncio.run(robot.clear_error("drag_not_allowed"))

    assert robot.clear_error_calls == ["drag_not_allowed"]
    assert robot.raw_status == {}


def test_fake_gripper_can_simulate_failed_grip_and_busy_state() -> None:
    gripper = FakeGripperClient()
    gripper.connect(ip="127.0.0.1", gripper_type="auto")
    gripper.busy = True
    gripper.fail_next_close = True

    gripper.close(width_mm=12)
    telemetry = gripper.telemetry()

    assert telemetry["holding"] is False
    assert telemetry["busy"] is True
    assert telemetry["width_mm"] == 12


def test_fake_camera_can_drop_frames_and_restart() -> None:
    stream = FakeCameraStream(FakeStreamConfig(resolution=(80, 45), fps=60.0, frame_drop_every=2))
    stream.start()
    time.sleep(0.08)
    stream.stop()
    first_generated = stream.generated_frames
    first_dropped = stream.dropped_frames

    stream.start()
    time.sleep(0.04)
    stream.stop()

    assert first_generated > 0
    assert first_dropped > 0
    assert stream.generated_frames > first_generated
    assert stream.restart_count == 1
