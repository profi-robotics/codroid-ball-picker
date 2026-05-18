"""Tests for live detection overlay state updates."""

from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np

from app.codroid_picker.overlays.detection import _refresh_detections
from app.codroid_picker.session import SESSION


class DetectionOverlayTests(unittest.TestCase):
    def setUp(self) -> None:
        self._snapshot = {
            "model_loaded": SESSION.model_loaded,
            "yolo_model": SESSION.yolo_model,
            "detections": list(SESSION.detections),
            "last_detection_error": SESSION.last_detection_error,
            "auto_detection_updated_at": SESSION.auto_detection_updated_at,
        }

    def tearDown(self) -> None:
        SESSION.model_loaded = self._snapshot["model_loaded"]
        SESSION.yolo_model = self._snapshot["yolo_model"]
        SESSION.detections = self._snapshot["detections"]
        SESSION.last_detection_error = self._snapshot["last_detection_error"]
        SESSION.auto_detection_updated_at = self._snapshot[
            "auto_detection_updated_at"
        ]

    def test_refresh_clears_stale_detections_when_model_returns_none(self) -> None:
        SESSION.model_loaded = True
        SESSION.yolo_model = object()
        SESSION.detections = [{"label": "orange", "confidence": 0.55}]
        SESSION.auto_detection_updated_at = 0.0

        with patch(
            "app.codroid_picker.auto_operation._run_detection_serialized",
            return_value=[],
        ):
            _refresh_detections(np.zeros((10, 10, 3), dtype=np.uint8))

        self.assertEqual(SESSION.detections, [])
        self.assertIsNone(SESSION.last_detection_error)


if __name__ == "__main__":
    unittest.main()
