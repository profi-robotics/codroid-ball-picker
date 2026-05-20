"""Tests for streaming frame admission."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import threading
import time
import unittest
from unittest.mock import patch

import cv2
import numpy as np
from dash import Dash, html

from app.codroid_picker import overlay_tasks
from app.codroid_picker import streaming
from app.codroid_picker.session import SESSION
from app.codroid_picker.streaming import (
    DepthAIFrameStream,
    FrameProcessor,
    MotionJPEGStream,
    OverlayResult,
    OverlayTask,
    StreamConfig,
)
from app.codroid_picker.video_routes import register_video_routes
from vision.aruco_board import load_charuco_board_spec


class MotionJPEGStreamTests(unittest.TestCase):
    def test_add_frame_rate_limit_uses_admission_time(self) -> None:
        stream = MotionJPEGStream(min_push_interval_s=0.1)
        frame = np.zeros((8, 8, 3), dtype=np.uint8)

        with patch(
            "app.codroid_picker.streaming.time.monotonic",
            side_effect=[10.0, 10.05, 10.11],
        ):
            stream.add_frame(frame)
            stream.add_frame(frame)
            stream.add_frame(frame)

        self.assertEqual(stream.frame_count, 2)
        self.assertEqual(stream.get_frame_with_index()[1], 2)


class FocusLockTests(unittest.TestCase):
    def test_refocus_and_lock_uses_latest_lens_position(self) -> None:
        stream = DepthAIFrameStream(StreamConfig())
        stream._running = True
        stream._last_lens_pos = 177
        calls: list[tuple[bool, int | None]] = []

        def fake_set_focus(lock: bool, lens_position: int | None = None) -> None:
            calls.append((lock, lens_position))

        with patch.object(stream, "set_focus_lock", side_effect=fake_set_focus):
            position = stream.refocus_and_lock(settle_s=0.0)

        self.assertEqual(position, 177)
        self.assertEqual(calls, [(False, None), (True, 177)])

    def test_refocus_and_lock_falls_back_to_previous_manual_position(self) -> None:
        stream = DepthAIFrameStream(StreamConfig(manual_focus_position=142))
        stream._running = True
        calls: list[tuple[bool, int | None]] = []

        def fake_set_focus(lock: bool, lens_position: int | None = None) -> None:
            calls.append((lock, lens_position))

        with patch.object(stream, "set_focus_lock", side_effect=fake_set_focus):
            position = stream.refocus_and_lock(settle_s=0.0)

        self.assertEqual(position, 142)
        self.assertEqual(calls, [(False, None), (True, 142)])


class FakeFrameProcessor:
    def __init__(self, latest: OverlayResult | None = None) -> None:
        self.latest = latest
        self.submitted: list[tuple[streaming.OverlayTaskSnapshot, np.ndarray]] = []
        self.stored: list[tuple[str, int, np.ndarray, OverlayResult]] = []

    def submit(self, task: streaming.OverlayTaskSnapshot, frame: np.ndarray) -> bool:
        self.submitted.append((task, frame.copy()))
        return True

    def get_latest(self, _key: str) -> OverlayResult | None:
        return self.latest

    def store_latest(
        self,
        key: str,
        generation: int,
        frame: np.ndarray,
        result: OverlayResult,
    ) -> None:
        self.stored.append((key, generation, frame.copy(), result))
        self.latest = result

    def clear(self) -> None:
        self.submitted.clear()
        self.stored.clear()
        self.latest = None


class FakeMjpegStream:
    def __init__(self) -> None:
        self.frames: list[np.ndarray] = []

    def add_frame(self, frame: np.ndarray) -> None:
        self.frames.append(frame.copy())


class ProcessCameraFrameTests(unittest.TestCase):
    def tearDown(self) -> None:
        streaming.set_active_overlay_task(None)

    def test_overlay_task_generation_invalidates_stale_results(self) -> None:
        first_task = OverlayTask(
            "test-overlay",
            lambda image: OverlayResult(image, "ok", "metric"),
        )
        second_task = OverlayTask(
            "other-overlay",
            lambda image: OverlayResult(image, "ok", "metric"),
        )

        streaming.set_active_overlay_task(first_task)
        first_snapshot = streaming.get_active_overlay_task()
        assert first_snapshot is not None

        streaming.set_active_overlay_task(second_task)

        self.assertFalse(
            streaming.is_active_overlay_task(
                first_snapshot.task.key,
                first_snapshot.generation,
            )
        )

    def test_stale_worker_result_is_not_stored_after_overlay_turns_off(self) -> None:
        frame = np.full((4, 4, 3), 7, dtype=np.uint8)
        started = threading.Event()
        release = threading.Event()
        processor = FrameProcessor()

        def runner(image: np.ndarray) -> OverlayResult:
            started.set()
            release.wait(timeout=1.0)
            return OverlayResult(image, "ok", "metric")

        task = OverlayTask("slow-overlay", runner)

        try:
            streaming.set_active_overlay_task(task)
            snapshot = streaming.get_active_overlay_task()
            assert snapshot is not None
            processor.submit(snapshot, frame)
            self.assertTrue(started.wait(timeout=1.0))

            streaming.set_active_overlay_task(None)
            release.set()
            time.sleep(0.05)

            self.assertIsNone(processor.get_latest(task.key))
        finally:
            release.set()
            processor.stop()

    def test_store_latest_exposes_browser_overlay_state(self) -> None:
        frame = np.full((4, 5, 3), 7, dtype=np.uint8)
        processor = FrameProcessor()
        task = OverlayTask(
            "test-overlay",
            lambda image: OverlayResult(image, "ok", "metric"),
        )

        try:
            streaming.set_active_overlay_task(task)
            snapshot = streaming.get_active_overlay_task()
            assert snapshot is not None
            processor.store_latest(
                task.key,
                snapshot.generation,
                frame,
                OverlayResult(
                    frame,
                    "ok",
                    "metric",
                    primitives=[{"type": "box", "x1": 1, "y1": 2, "x2": 3, "y2": 4}],
                    has_targets=True,
                ),
            )

            state = processor.get_active_state_payload()

            self.assertEqual(state["key"], task.key)
            self.assertEqual(state["frame_width"], 5)
            self.assertEqual(state["frame_height"], 4)
            self.assertEqual(state["status"], "ok")
            self.assertEqual(state["metric"], "metric")
            self.assertTrue(state["has_targets"])
            self.assertEqual(len(state["primitives"]), 1)
        finally:
            processor.stop()

    def test_processing_payload_keeps_latest_frame_size(self) -> None:
        frame = np.full((4, 5, 3), 7, dtype=np.uint8)
        started = threading.Event()
        release = threading.Event()
        processor = FrameProcessor()

        def runner(image: np.ndarray) -> OverlayResult:
            started.set()
            release.wait(timeout=1.0)
            return OverlayResult(image, "ok", "metric")

        task = OverlayTask("slow-overlay", runner)

        try:
            streaming.set_active_overlay_task(task)
            snapshot = streaming.get_active_overlay_task()
            assert snapshot is not None

            self.assertTrue(processor.submit(snapshot, frame))
            self.assertTrue(started.wait(timeout=1.0))

            state = processor.get_active_state_payload()

            self.assertEqual(state["status"], "Overlay processing...")
            self.assertEqual(state["frame_width"], 5)
            self.assertEqual(state["frame_height"], 4)
        finally:
            release.set()
            processor.stop()

    def test_empty_overlay_state_replaces_server_state(self) -> None:
        frame = np.full((4, 4, 3), 7, dtype=np.uint8)
        processor = FrameProcessor()
        task = OverlayTask(
            "test-overlay",
            lambda image: OverlayResult(image, "ok", "metric"),
        )

        try:
            streaming.set_active_overlay_task(task)
            snapshot = streaming.get_active_overlay_task()
            assert snapshot is not None
            processor.store_latest(
                task.key,
                snapshot.generation,
                frame,
                OverlayResult(
                    frame,
                    "Objects detected: 1",
                    "Detections: 1",
                    primitives=[{"type": "box", "x1": 1, "y1": 1, "x2": 3, "y2": 3}],
                    has_targets=True,
                ),
            )
            processor.store_latest(
                task.key,
                snapshot.generation,
                frame,
                OverlayResult(frame, "No objects detected.", "Detections: 0"),
            )

            state = processor.get_active_state_payload()

            self.assertEqual(state["status"], "No objects detected.")
            self.assertFalse(state["has_targets"])
            self.assertEqual(state["primitives"], [])
        finally:
            processor.stop()

    def test_overlay_state_clears_on_mode_change(self) -> None:
        frame = np.full((4, 4, 3), 7, dtype=np.uint8)
        processor = FrameProcessor()
        first_task = OverlayTask(
            "test-overlay",
            lambda image: OverlayResult(image, "ok", "metric"),
        )
        second_task = OverlayTask(
            "other-overlay",
            lambda image: OverlayResult(image, "ok", "metric"),
        )

        try:
            with patch.object(streaming, "FRAME_PROCESSOR", processor):
                streaming.set_active_overlay_task(first_task)
                snapshot = streaming.get_active_overlay_task()
                assert snapshot is not None
                processor.store_latest(
                    first_task.key,
                    snapshot.generation,
                    frame,
                    OverlayResult(
                        frame,
                        "ok",
                        "metric",
                        primitives=[{"type": "point", "x": 1, "y": 1}],
                        has_targets=True,
                    ),
                )
                self.assertTrue(processor.get_active_state_payload()["has_targets"])
                streaming.set_active_overlay_task(second_task)

            state = processor.get_active_state_payload()
            self.assertEqual(state["key"], second_task.key)
            self.assertEqual(state["primitives"], [])
            self.assertFalse(state["has_targets"])
        finally:
            processor.stop()

    def test_get_active_state_payload_marks_stale_overlay(self) -> None:
        frame = np.full((4, 4, 3), 7, dtype=np.uint8)
        processor = FrameProcessor()
        task = OverlayTask(
            "test-overlay",
            lambda image: OverlayResult(image, "ok", "metric"),
        )

        try:
            streaming.set_active_overlay_task(task)
            snapshot = streaming.get_active_overlay_task()
            assert snapshot is not None
            processor.store_latest(
                task.key,
                snapshot.generation,
                frame,
                OverlayResult(
                    frame,
                    "ok",
                    "metric",
                    primitives=[{"type": "point", "x": 1, "y": 1}],
                    has_targets=True,
                ),
            )

            with patch("app.codroid_picker.streaming.time.time", return_value=time.time() + 10.0):
                state = processor.get_active_state_payload()

            self.assertTrue(state["stale"])
        finally:
            processor.stop()

    def test_frame_processor_rejects_submit_while_worker_busy(self) -> None:
        frame = np.full((4, 4, 3), 7, dtype=np.uint8)
        started = threading.Event()
        release = threading.Event()
        processor = FrameProcessor()
        task = OverlayTask(
            "slow-overlay",
            lambda image: (
                started.set(),
                release.wait(timeout=1.0),
                OverlayResult(image, "ok", "metric"),
            )[2],
        )

        try:
            streaming.set_active_overlay_task(task)
            snapshot = streaming.get_active_overlay_task()
            assert snapshot is not None

            self.assertTrue(processor.submit(snapshot, frame))
            self.assertTrue(started.wait(timeout=1.0))
            self.assertFalse(processor.submit(snapshot, frame))
        finally:
            release.set()
            processor.stop()

    def test_process_camera_frame_publishes_raw_without_overlay(self) -> None:
        frame = np.full((4, 4, 3), 7, dtype=np.uint8)
        mjpeg_stream = FakeMjpegStream()

        with patch.object(streaming, "MJPEG_STREAM", mjpeg_stream):
            streaming.set_active_overlay_task(None)
            streaming.process_camera_frame(frame)

        self.assertEqual(len(mjpeg_stream.frames), 1)
        np.testing.assert_array_equal(mjpeg_stream.frames[0], frame)

    def test_process_camera_frame_publishes_raw_and_submits_overlay(self) -> None:
        frame = np.full((4, 4, 3), 7, dtype=np.uint8)
        task = OverlayTask(
            "test-overlay",
            lambda image: OverlayResult(image, "ok", "metric"),
        )
        processor = FakeFrameProcessor()
        mjpeg_stream = FakeMjpegStream()

        with (
            patch.object(streaming, "FRAME_PROCESSOR", processor),
            patch.object(streaming, "MJPEG_STREAM", mjpeg_stream),
        ):
            streaming.set_active_overlay_task(task)
            streaming.process_camera_frame(frame)

        self.assertEqual(len(processor.submitted), 1)
        self.assertEqual(processor.submitted[0][0].task, task)
        np.testing.assert_array_equal(processor.submitted[0][1], frame)
        self.assertEqual(processor.stored, [])
        self.assertEqual(len(mjpeg_stream.frames), 1)
        np.testing.assert_array_equal(mjpeg_stream.frames[0], frame)


class OverlayRouteTests(unittest.TestCase):
    def tearDown(self) -> None:
        streaming.set_active_overlay_task(None)

    def test_overlay_state_route_returns_stable_empty_payload(self) -> None:
        app = Dash(__name__)
        app.layout = html.Div()
        register_video_routes(app)

        response = app.server.test_client().get("/overlay_state.json")

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertEqual(payload["primitives"], [])
        self.assertFalse(payload["has_targets"])
        self.assertIn("status", payload)


class OverlayPrimitiveTests(unittest.TestCase):
    @staticmethod
    def _marker_image(marker_id: int, size_px: int = 80) -> np.ndarray:
        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        marker = np.zeros((size_px, size_px), dtype=np.uint8)
        cv2.aruco.generateImageMarker(dictionary, marker_id, size_px, marker, 1)
        return cv2.cvtColor(marker, cv2.COLOR_GRAY2BGR)

    def test_charuco_runner_emits_semantic_primitives(self) -> None:
        board_path = Path(
            "data/markers/camera_calibration/camera_calibration_board.json"
        )
        board = load_charuco_board_spec(board_path).create_board()
        image = board.generateImage((500, 700), marginSize=40)
        frame = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        result = overlay_tasks._run_charuco_task(board_path)(frame)

        primitive_types = {item["type"] for item in result.primitives}
        self.assertTrue(result.has_targets)
        self.assertIn("circle", primitive_types)
        self.assertIn("polygon", primitive_types)
        self.assertTrue(
            all(
                "fill" not in item
                for item in result.primitives
                if item["type"] == "polygon"
            )
        )

    def test_workspace_runner_emits_semantic_primitives(self) -> None:
        frame = np.full((500, 700, 3), 255, dtype=np.uint8)
        for marker_id, center in ((1, (120, 100)), (2, (540, 380))):
            marker = self._marker_image(marker_id)
            x = center[0] - marker.shape[1] // 2
            y = center[1] - marker.shape[0] // 2
            frame[y:y + marker.shape[0], x:x + marker.shape[1]] = marker

        old_bundle = SESSION.calibration_bundle
        old_workspace = SESSION.workspace_definition
        SESSION.calibration_bundle = SimpleNamespace(pixel_to_mm=20.0 / 80.0)
        SESSION.workspace_definition = None
        try:
            result = overlay_tasks._run_workspace_task((1, 2), 20.0)(frame)
        finally:
            SESSION.calibration_bundle = old_bundle
            SESSION.workspace_definition = old_workspace

        primitive_types = {item["type"] for item in result.primitives}
        self.assertTrue(result.has_targets)
        self.assertIn("line", primitive_types)
        self.assertIn("polygon", primitive_types)
        self.assertTrue(
            all(
                "fill" not in item
                for item in result.primitives
                if item["type"] == "polygon"
            )
        )

    def test_plate_runner_emits_semantic_primitives(self) -> None:
        layout_path = Path(
            "data/markers/robot_calibration/calibration_plate_layout.json"
        )
        scale = 4
        frame = np.full((280 * scale + 80, 300 * scale + 80, 3), 255, dtype=np.uint8)
        marker_centers = (
            (20, (20, 20)),
            (21, (150, 20)),
            (22, (280, 20)),
            (23, (20, 140)),
            (24, (280, 140)),
            (25, (20, 260)),
            (26, (150, 260)),
            (27, (280, 260)),
        )
        for marker_id, center_mm in marker_centers:
            marker = self._marker_image(marker_id, 20 * scale)
            x = int(center_mm[0] * scale + 40 - marker.shape[1] / 2)
            y = int(center_mm[1] * scale + 40 - marker.shape[0] / 2)
            frame[y:y + marker.shape[0], x:x + marker.shape[1]] = marker

        old_workspace = SESSION.workspace_definition
        SESSION.workspace_definition = None
        try:
            result = overlay_tasks._run_plate_task(layout_path)(frame)
        finally:
            SESSION.workspace_definition = old_workspace

        primitive_types = {item["type"] for item in result.primitives}
        self.assertTrue(result.has_targets)
        self.assertIn("circle", primitive_types)
        self.assertIn("polygon", primitive_types)
        self.assertTrue(
            all(
                "fill" not in item
                for item in result.primitives
                if item["type"] == "polygon"
            )
        )

    def test_detection_runner_emits_box_primitives(self) -> None:
        frame = np.zeros((32, 32, 3), dtype=np.uint8)
        old_detections = SESSION.detections
        SESSION.detections = [
            {
                "bbox": [2, 3, 12, 14],
                "label": "ball",
                "confidence": 0.91,
            }
        ]

        try:
            with patch(
                "app.codroid_picker.overlay_tasks.draw_detection_overlay",
                return_value=(frame.copy(), "Objects detected: 1", "Detections: 1"),
            ):
                result = overlay_tasks._run_detection_task()(frame)
        finally:
            SESSION.detections = old_detections

        self.assertTrue(result.has_targets)
        self.assertEqual(result.primitives[0]["type"], "box")
        self.assertEqual(result.primitives[0]["label"], "ball: 0.91")


if __name__ == "__main__":
    unittest.main()
