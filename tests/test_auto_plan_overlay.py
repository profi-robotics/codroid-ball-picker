"""Tests for auto-plan overlay rendering."""

from __future__ import annotations

import unittest

import numpy as np

from app.codroid_picker.auto_plan_state import clear_auto_plan_snapshot, publish_auto_plan_snapshot
from app.codroid_picker.overlays.auto_plan import draw_auto_plan_overlay
from app.codroid_picker.session import SESSION


class AutoPlanOverlayTests(unittest.TestCase):
    def setUp(self) -> None:
        self._snapshot = dict(SESSION.auto_plan_snapshot)
        clear_auto_plan_snapshot()

    def tearDown(self) -> None:
        SESSION.auto_plan_snapshot = self._snapshot

    def test_overlay_handles_missing_snapshot(self) -> None:
        frame = np.zeros((120, 160, 3), dtype=np.uint8)
        vis, status, metric = draw_auto_plan_overlay(frame)
        self.assertEqual(vis.shape, frame.shape)
        self.assertIn("unavailable", status.lower())
        self.assertIn("Auto plan", metric)

    def test_overlay_draws_with_basic_snapshot(self) -> None:
        publish_auto_plan_snapshot(
            stage="candidate_selected",
            message="Candidate selected.",
            ranked_candidates=[
                {
                    "label": "orange",
                    "confidence": 0.9,
                    "center": (70.0, 60.0),
                    "bbox": (50.0, 40.0, 90.0, 80.0),
                    "coord5_mm": (10.0, 20.0),
                }
            ],
            selected_candidate={
                "label": "orange",
                "confidence": 0.9,
                "center": (70.0, 60.0),
                "bbox": (50.0, 40.0, 90.0, 80.0),
                "coord5_mm": (10.0, 20.0),
            },
            gripper_close_mm=12.0,
        )
        frame = np.zeros((120, 160, 3), dtype=np.uint8)
        vis, status, metric = draw_auto_plan_overlay(frame)
        self.assertEqual(vis.shape, frame.shape)
        self.assertIn("candidate_selected", status)
        self.assertIn("Auto plan", metric)


if __name__ == "__main__":
    unittest.main()
