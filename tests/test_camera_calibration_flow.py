"""Tests for camera calibration depth-capture helpers."""

from __future__ import annotations

import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from app.codroid_picker.camera_callbacks import (
    _capture_depth_snapshot_for_calibration,
)
from app.codroid_picker.session import SESSION


class CameraCalibrationFlowTests(unittest.TestCase):
    def setUp(self) -> None:
        self._snapshot = {
            "camera_stream": SESSION.camera_stream,
            "camera_last_error_message": SESSION.camera_last_error_message,
            "camera_last_error_key": SESSION.camera_last_error_key,
            "camera_last_error_params": dict(SESSION.camera_last_error_params),
            "camera_last_error_status": SESSION.camera_last_error_status,
        }

    def tearDown(self) -> None:
        SESSION.camera_stream = self._snapshot["camera_stream"]
        SESSION.camera_last_error_message = self._snapshot["camera_last_error_message"]
        SESSION.camera_last_error_key = self._snapshot["camera_last_error_key"]
        SESSION.camera_last_error_params = self._snapshot["camera_last_error_params"]
        SESSION.camera_last_error_status = self._snapshot["camera_last_error_status"]

    def test_depth_capture_uses_live_stream_without_pausing(self) -> None:
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

        in_stream_calls: list[Path] = []
        fake_module = SimpleNamespace(
            capture_depth_snapshot=lambda output: (
                in_stream_calls.append(output) or Path("direct_depth.npz")
            ),
            capture_depth_snapshot_isolated=lambda *_args, **_kwargs: Path(
                "should_not_happen.npz"
            ),
        )

        with patch("app.codroid_picker.camera_callbacks.stop_camera_stream") as stop_mock:
            depth_path, paused = _capture_depth_snapshot_for_calibration(
                fake_module,
                Path("data/calibration"),
            )

        self.assertEqual(depth_path, Path("direct_depth.npz"))
        self.assertFalse(paused)
        stop_mock.assert_not_called()
        self.assertEqual(in_stream_calls, [Path("data/calibration")])

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

        with patch("app.codroid_picker.camera_callbacks.stop_camera_stream") as stop_mock:
            depth_path, paused = _capture_depth_snapshot_for_calibration(
                fake_module,
                Path("data/calibration"),
            )

        self.assertEqual(depth_path, Path("direct_depth.npz"))
        self.assertFalse(paused)
        stop_mock.assert_not_called()
        self.assertEqual(in_stream_calls, [Path("data/calibration")])

if __name__ == "__main__":
    unittest.main()
