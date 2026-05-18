"""Unit tests for camera safety guard and detection mode wiring."""

from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np

from app.codroid_picker.auto_operation import _run_detection
from app.codroid_picker.safety_guard import (
    desired_operation_action,
    evaluate_heuristic_intrusion,
    evaluate_safety_state,
    normalize_safety_labels,
    normalize_safety_mode,
    preprocess_gray,
)
from app.codroid_picker.session import SESSION


class _DummyBoxes:
    def __init__(self) -> None:
        self.xyxy = np.array([[10.0, 20.0, 30.0, 40.0]], dtype=np.float32)
        self.conf = np.array([0.95], dtype=np.float32)
        self.cls = np.array([0], dtype=np.float32)


class _DummyResult:
    def __init__(self) -> None:
        self.boxes = _DummyBoxes()


class _DummyModel:
    def __init__(self) -> None:
        self.names = {0: "person"}
        self.last_shape = None
        self.last_kwargs = {}

    def predict(self, frame, **kwargs):
        self.last_shape = frame.shape
        self.last_kwargs = kwargs
        return [_DummyResult()]


class CameraSafetyTests(unittest.TestCase):
    """Validate safety label matching and zone decisions."""

    def test_normalize_labels_deduplicates_and_trims(self) -> None:
        normalized = normalize_safety_labels(
            [" person ", "PERSON", "", "people", "people", None]  # type: ignore[list-item]
        )
        self.assertEqual(normalized, ["person", "people"])

    def test_normalize_mode_uses_legacy_enabled_fallback(self) -> None:
        self.assertEqual(normalize_safety_mode("off", True), "off")
        self.assertEqual(normalize_safety_mode(None, False), "off")
        self.assertEqual(normalize_safety_mode(None, True), "model")

    def test_evaluate_safety_state_full_frame(self) -> None:
        blocked, matches = evaluate_safety_state(
            [
                {"label": "person"},
                {"label": "orange"},
            ],
            enabled=True,
            labels=["person", "people"],
            zone="full_frame",
            workspace_def=None,
        )
        self.assertTrue(blocked)
        self.assertEqual(len(matches), 1)

    def test_evaluate_safety_state_workspace_zone_filters_outside(self) -> None:
        workspace_def = SimpleNamespace(width_mm=100.0, height_mm=80.0)
        blocked, matches = evaluate_safety_state(
            [
                {"label": "person", "workspace_mm": (120.0, 10.0)},
                {"label": "person", "workspace_mm": (40.0, 20.0)},
            ],
            enabled=True,
            labels=["person"],
            zone="workspace",
            workspace_def=workspace_def,
        )
        self.assertTrue(blocked)
        self.assertEqual(len(matches), 1)

    def test_desired_operation_action_pause_resume_gating(self) -> None:
        self.assertEqual(
            desired_operation_action(
                status="running",
                pause_reason=None,
                blocked=True,
            ),
            "pause",
        )
        self.assertEqual(
            desired_operation_action(
                status="paused",
                pause_reason="person_detected",
                blocked=False,
            ),
            "resume",
        )
        self.assertEqual(
            desired_operation_action(
                status="paused",
                pause_reason="operator",
                blocked=False,
            ),
            "none",
        )

    def test_heuristic_intrusion_detects_large_change(self) -> None:
        baseline_frame = np.zeros((100, 100, 3), dtype=np.uint8)
        baseline = preprocess_gray(baseline_frame)
        frame = baseline_frame.copy()
        frame[20:80, 20:80] = 255
        blocked, metrics = evaluate_heuristic_intrusion(
            frame,
            baseline,
            roi_mask=None,
            ignore_mask=None,
            threshold=0.01,
            min_area_ratio=0.01,
        )
        self.assertTrue(blocked)
        self.assertGreater(metrics["changed_ratio"], 0.01)

    def test_run_detection_skips_workspace_crop_when_disabled(self) -> None:
        prev_model = SESSION.yolo_model
        prev_workspace = SESSION.workspace_definition
        prev_calib = SESSION.calibration_bundle

        model = _DummyModel()
        SESSION.yolo_model = model
        SESSION.workspace_definition = object()
        SESSION.calibration_bundle = object()

        from app.codroid_picker import auto_operation

        original_workspace_corners = auto_operation._workspace_corners_px

        def _fail_if_called(_workspace_def):
            raise AssertionError("workspace crop helper should not be called")

        auto_operation._workspace_corners_px = _fail_if_called
        try:
            detections = _run_detection(
                np.zeros((100, 200, 3), dtype=np.uint8),
                confidence=0.4,
                iou=0.45,
                max_detections=10,
                class_filter=None,
                crop_to_workspace=False,
            )
            self.assertEqual(model.last_shape, (100, 200, 3))
            self.assertEqual(len(detections), 1)
            self.assertEqual(detections[0]["label"], "person")
        finally:
            auto_operation._workspace_corners_px = original_workspace_corners
            SESSION.yolo_model = prev_model
            SESSION.workspace_definition = prev_workspace
            SESSION.calibration_bundle = prev_calib

    def test_run_detection_clamps_confidence_and_iou(self) -> None:
        prev_model = SESSION.yolo_model

        model = _DummyModel()
        SESSION.yolo_model = model

        try:
            detections = _run_detection(
                np.zeros((100, 200, 3), dtype=np.uint8),
                confidence=4.0,
                iou=-2.0,
                max_detections=0,
                class_filter=None,
                crop_to_workspace=False,
            )
            self.assertEqual(len(detections), 1)
            self.assertEqual(model.last_kwargs["conf"], 1.0)
            self.assertEqual(model.last_kwargs["iou"], 0.0)
            self.assertEqual(model.last_kwargs["max_det"], 1)
        finally:
            SESSION.yolo_model = prev_model


if __name__ == "__main__":
    unittest.main()
