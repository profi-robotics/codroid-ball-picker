"""Tests for camera retry state and frame processing throttles."""

from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np

from app.codroid_picker import streaming
from app.codroid_picker.session import SESSION


class _RetryingStream:
    def __init__(self, config, frame_hook=None) -> None:
        self.config = config
        self._running = False
        self._frame_hook = frame_hook
        self.stopped = False

    def start(self) -> None:
        self._running = True

    def stop(self) -> None:
        self._running = False
        self.stopped = True

    def is_running(self) -> bool:
        return self._running

    def status(self) -> str:
        return "retrying"

    def last_error(self) -> str:
        return "X_LINK_DEVICE_ALREADY_IN_USE"

    def device_id(self) -> str:
        return "unknown"


class _LiveStream(_RetryingStream):
    def __init__(self, config, frame_hook=None, *, camera_index: int = 0) -> None:
        super().__init__(config, frame_hook)
        self.camera_index = camera_index

    def status(self) -> str:
        return "live"

    def device_id(self) -> str:
        return f"host:{self.camera_index}"


class StreamingPerformanceTests(unittest.TestCase):
    def tearDown(self) -> None:
        streaming.stop_camera_stream()

    def test_start_camera_stream_surfaces_retrying_state(self) -> None:
        previous_stream = SESSION.camera_stream
        SESSION.camera_stream = None
        try:
            with patch.object(streaming, "DepthAIFrameStream", _RetryingStream):
                message = streaming.start_camera_stream(640, 480, 30)

            self.assertIn("Camera retrying", message)
            self.assertIsNotNone(SESSION.camera_stream)
            self.assertEqual(SESSION.camera_stream.status(), "retrying")
        finally:
            streaming.stop_camera_stream()
            SESSION.camera_stream = previous_stream

    def test_stop_camera_stream_disables_retrying_stream(self) -> None:
        stream = _RetryingStream(streaming.StreamConfig(), None)
        stream.start()
        previous_stream = SESSION.camera_stream
        SESSION.camera_stream = stream
        try:
            streaming.stop_camera_stream()
            self.assertTrue(stream.stopped)
            self.assertIsNone(SESSION.camera_stream)
        finally:
            SESSION.camera_stream = previous_stream

    def test_start_camera_stream_can_use_host_camera_backend(self) -> None:
        previous_stream = SESSION.camera_stream
        SESSION.camera_stream = None
        try:
            with patch.object(streaming, "HostCameraFrameStream", _LiveStream):
                message = streaming.start_camera_stream(
                    640,
                    480,
                    30,
                    source="host",
                    host_camera_index=1,
                    enable_depth_override=True,
                )

            self.assertIn("Host camera host:1", message)
            self.assertIsNotNone(SESSION.camera_stream)
            assert SESSION.camera_stream is not None
            self.assertEqual(SESSION.camera_stream.status(), "live")
            self.assertFalse(SESSION.camera_stream.config.enable_depth)
        finally:
            streaming.stop_camera_stream()
            SESSION.camera_stream = previous_stream

    def test_frame_processor_throttles_overlay_submissions(self) -> None:
        processor = streaming.FrameProcessor(max_queue=4)
        calls: list[int] = []

        def _runner(frame: np.ndarray) -> streaming.OverlayResult:
            calls.append(1)
            return streaming.OverlayResult(frame, "ok", "metric")

        task = streaming.OverlayTask("test", _runner)
        frame = np.zeros((4, 4, 3), dtype=np.uint8)

        with patch.object(streaming, "_OVERLAY_MAX_FPS", 1.0):
            processor.submit(task, frame)
            processor.submit(task, frame)

        self.assertLessEqual(len(processor._queue), 1)
        processor.stop()


if __name__ == "__main__":
    unittest.main()
