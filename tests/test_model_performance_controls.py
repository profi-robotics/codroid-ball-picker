"""Tests for model labels, benchmark ratings, and YOLO frame skipping."""

from __future__ import annotations

import types
import unittest

from app.codroid_picker.auto_operation import _should_run_yolo_inference
from app.codroid_picker.models import _benchmark_rating, extract_model_labels
from app.codroid_picker.session import SESSION


class ModelPerformanceControlTests(unittest.TestCase):
    def setUp(self) -> None:
        self._snapshot = {
            "detections": list(SESSION.detections),
            "detection_frame_skip": SESSION.detection_frame_skip,
            "detection_frame_index": SESSION.detection_frame_index,
        }

    def tearDown(self) -> None:
        SESSION.detections = self._snapshot["detections"]
        SESSION.detection_frame_skip = self._snapshot["detection_frame_skip"]
        SESSION.detection_frame_index = self._snapshot["detection_frame_index"]

    def test_extract_model_labels_from_dict_names(self) -> None:
        model = types.SimpleNamespace(names={2: "blue", 0: "orange", 1: "cube"})

        labels = extract_model_labels(model)

        self.assertEqual(labels, ["orange", "cube", "blue"])

    def test_extract_model_labels_from_list_names(self) -> None:
        model = types.SimpleNamespace(names=["orange", "sports ball"])

        labels = extract_model_labels(model)

        self.assertEqual(labels, ["orange", "sports ball"])

    def test_benchmark_rating_thresholds(self) -> None:
        self.assertEqual(_benchmark_rating(120.0), "usable")
        self.assertEqual(_benchmark_rating(250.0), "warning")
        self.assertEqual(_benchmark_rating(500.0), "poor")

    def test_frame_skip_gates_cached_detection_refreshes(self) -> None:
        SESSION.detections = [{"label": "orange"}]
        SESSION.detection_frame_skip = 3
        SESSION.detection_frame_index = 0

        decisions = [_should_run_yolo_inference() for _ in range(4)]

        self.assertEqual(decisions, [True, False, False, True])


if __name__ == "__main__":
    unittest.main()
