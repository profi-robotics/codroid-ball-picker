from __future__ import annotations

from codroid_picker.hardware.robot_codroid import CodroidRobotClient


class _Pose:
    x = 1.0
    y = 2.0
    z = 3.0
    a = 4.0
    b = 5.0
    c = 6.0


class _SessionWithEmptySnapshots:
    def position_snapshot(self) -> _Pose:
        return _Pose()

    def robot_status_snapshot(self) -> None:
        return None

    def robot_power_on(self) -> None:
        return None

    def posture_seen(self) -> bool:
        return True

    def is_connected(self, uri: str) -> bool:
        return True

    def flange_button_states(self) -> None:
        return None

    def coordinate_frame_snapshot(self) -> None:
        return None


def test_codroid_snapshot_tolerates_none_payloads() -> None:
    client = CodroidRobotClient.__new__(CodroidRobotClient)
    client._session = _SessionWithEmptySnapshots()
    client._uri = "ws://robot.test/"

    snapshot = client.snapshot()

    assert snapshot["connected"] is True
    assert snapshot["pose"] == {
        "x": 1.0,
        "y": 2.0,
        "z": 3.0,
        "a": 4.0,
        "b": 5.0,
        "c": 6.0,
    }
    assert snapshot["di_state"] == {}
    assert snapshot["coordinate_id"] is None
    assert snapshot["coordinate_frame"] == {}
    assert snapshot["raw_status"] == {}
