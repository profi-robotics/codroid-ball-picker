"""Tests for camera calibration depth-capture helpers."""

from __future__ import annotations

import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from app.codroid_picker.camera_callbacks import (
    _capture_depth_snapshot_for_calibration,
)
from app.codroid_picker.session import CALIBRATION_PROGRESS, SESSION


class CameraCalibrationFlowTests(unittest.TestCase):
    def setUp(self) -> None:
        self._snapshot = {
            "camera_stream": SESSION.camera_stream,
            "camera_last_error_message": SESSION.camera_last_error_message,
            "camera_last_error_key": SESSION.camera_last_error_key,
            "camera_last_error_params": dict(SESSION.camera_last_error_params),
            "camera_last_error_status": SESSION.camera_last_error_status,
            "calibration_progress": dict(CALIBRATION_PROGRESS),
        }

    def tearDown(self) -> None:
        SESSION.camera_stream = self._snapshot["camera_stream"]
        SESSION.camera_last_error_message = self._snapshot["camera_last_error_message"]
        SESSION.camera_last_error_key = self._snapshot["camera_last_error_key"]
        SESSION.camera_last_error_params = self._snapshot["camera_last_error_params"]
        SESSION.camera_last_error_status = self._snapshot["camera_last_error_status"]
        CALIBRATION_PROGRESS.clear()
        CALIBRATION_PROGRESS.update(self._snapshot["calibration_progress"])

    def test_depth_disabled_stream_uses_isolated_capture(self) -> None:
        isolated_calls: list[tuple[Path, tuple[int, int], float, bool]] = []

        def isolated_capture(output, resolution, fps, lock_af=False):
            isolated_calls.append((output, resolution, fps, lock_af))
            return Path("isolated_depth.npz")

        stream = SimpleNamespace(
            config=SimpleNamespace(
                enable_depth=False,
                resolution=(1920, 1080),
                fps=30.0,
                lock_af=True,
            ),
            is_running=lambda: True,
        )
        SESSION.camera_stream = stream

        fake_module = SimpleNamespace(
            capture_depth_snapshot=lambda _output: Path("should_not_happen.npz"),
            capture_depth_snapshot_isolated=isolated_capture,
        )

        with (
            patch(
                "app.codroid_picker.camera_callbacks.stop_camera_stream"
            ) as stop_mock,
            patch(
                "app.codroid_picker.camera_callbacks.start_camera_stream",
                return_value="Camera stream running.",
            ) as start_mock,
        ):
            depth_path, paused = _capture_depth_snapshot_for_calibration(
                fake_module,
                Path("data/calibration"),
            )

        self.assertEqual(depth_path, Path("isolated_depth.npz"))
        self.assertFalse(paused)
        self.assertEqual(
            CALIBRATION_PROGRESS["message"],
            "Camera restarted. Finishing calibration...",
        )
        stop_mock.assert_called_once_with(wait_for_release=True)
        start_mock.assert_called_once_with(
            1920,
            1080,
            30.0,
            lock_af=True,
            enable_depth_override=False,
        )
        self.assertEqual(
            isolated_calls,
            [(Path("data/calibration"), (1920, 1080), 30.0, True)],
        )

    def test_depth_enabled_uses_in_stream_capture(self) -> None:
        stream = SimpleNamespace(
            config=SimpleNamespace(
                enable_depth=True,
                resolution=(1280, 720),
                fps=25.0,
                lock_af=False,
            ),
            is_running=lambda: True,
        )
        SESSION.camera_stream = stream

        in_stream_calls: list[Path] = []
        fake_module = SimpleNamespace(
            capture_depth_snapshot=lambda output: (
                in_stream_calls.append(output) or Path("direct_depth.npz")
            ),
            capture_depth_snapshot_isolated=lambda *_args, **_kwargs: Path(
                "should_not_happen.npz"
            ),
        )

        with patch(
            "app.codroid_picker.camera_callbacks.stop_camera_stream"
        ) as stop_mock, patch(
            "app.codroid_picker.camera_callbacks.start_camera_stream"
        ) as start_mock:
            depth_path, paused = _capture_depth_snapshot_for_calibration(
                fake_module,
                Path("data/calibration"),
            )

        self.assertEqual(depth_path, Path("direct_depth.npz"))
        self.assertFalse(paused)
        stop_mock.assert_not_called()
        start_mock.assert_not_called()
        self.assertEqual(in_stream_calls, [Path("data/calibration")])

    def test_depth_enabled_timeout_falls_back_to_isolated_capture(self) -> None:
        isolated_calls: list[tuple[Path, tuple[int, int], float, bool]] = []

        def fail_live_capture(_output):
            raise RuntimeError(
                "No depth frame available from the camera stream."
            )

        def isolated_capture(output, resolution, fps, lock_af=False):
            isolated_calls.append((output, resolution, fps, lock_af))
            return Path("isolated_depth.npz")

        stream = SimpleNamespace(
            config=SimpleNamespace(
                enable_depth=True,
                resolution=(1280, 720),
                fps=15.0,
                lock_af=False,
            ),
            is_running=lambda: True,
        )
        SESSION.camera_stream = stream

        fake_module = SimpleNamespace(
            capture_depth_snapshot=fail_live_capture,
            capture_depth_snapshot_isolated=isolated_capture,
        )

        with (
            patch(
                "app.codroid_picker.camera_callbacks.stop_camera_stream"
            ) as stop_mock,
            patch(
                "app.codroid_picker.camera_callbacks.start_camera_stream",
                return_value="Camera stream running.",
            ) as start_mock,
        ):
            depth_path, paused = _capture_depth_snapshot_for_calibration(
                fake_module,
                Path("data/calibration"),
            )

        self.assertEqual(depth_path, Path("isolated_depth.npz"))
        self.assertFalse(paused)
        self.assertEqual(
            CALIBRATION_PROGRESS["message"],
            "Camera restarted. Finishing calibration...",
        )
        stop_mock.assert_called_once_with(wait_for_release=True)
        start_mock.assert_called_once_with(
            1280,
            720,
            15.0,
            lock_af=False,
            enable_depth_override=True,
        )
        self.assertEqual(
            isolated_calls,
            [(Path("data/calibration"), (1280, 720), 15.0, False)],
        )

    def test_isolated_capture_reports_paused_when_restart_fails(self) -> None:
        def isolated_capture(output, resolution, fps, lock_af=False):
            return Path("isolated_depth.npz")

        stream = SimpleNamespace(
            config=SimpleNamespace(
                enable_depth=False,
                resolution=(1920, 1080),
                fps=30.0,
                lock_af=False,
            ),
            is_running=lambda: True,
        )
        SESSION.camera_stream = stream

        fake_module = SimpleNamespace(
            capture_depth_snapshot=lambda _output: Path("should_not_happen.npz"),
            capture_depth_snapshot_isolated=isolated_capture,
        )

        with (
            patch("app.codroid_picker.camera_callbacks.stop_camera_stream"),
            patch(
                "app.codroid_picker.camera_callbacks.start_camera_stream",
                side_effect=RuntimeError("camera busy"),
            ),
        ):
            depth_path, paused = _capture_depth_snapshot_for_calibration(
                fake_module,
                Path("data/calibration"),
            )

        self.assertEqual(depth_path, Path("isolated_depth.npz"))
        self.assertTrue(paused)


if __name__ == "__main__":
    unittest.main()
