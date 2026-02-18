"""Tests for shared auto-plan snapshot helpers."""

from __future__ import annotations

import time
import unittest

from app.codroid_picker.auto_plan_state import (
    clear_auto_plan_snapshot,
    get_auto_plan_snapshot,
    publish_auto_plan_snapshot,
)
from app.codroid_picker.session import SESSION
from app.codroid_picker.target_planner import TargetCandidate


class AutoPlanStateTests(unittest.TestCase):
    def setUp(self) -> None:
        self._snapshot = {
            "auto_plan_snapshot": dict(SESSION.auto_plan_snapshot),
            "auto_plan_stage": SESSION.auto_plan_stage,
            "auto_plan_updated_at": SESSION.auto_plan_updated_at,
        }
        clear_auto_plan_snapshot()

    def tearDown(self) -> None:
        SESSION.auto_plan_snapshot = self._snapshot["auto_plan_snapshot"]
        SESSION.auto_plan_stage = self._snapshot["auto_plan_stage"]
        SESSION.auto_plan_updated_at = self._snapshot["auto_plan_updated_at"]

    def test_publish_normalizes_candidates(self) -> None:
        candidate = TargetCandidate(
            detection={
                "label": "orange",
                "confidence": 0.91,
                "center": (120.0, 140.0),
                "bbox": (90.0, 110.0, 150.0, 170.0),
                "workspace_mm": (50.0, 75.0),
            },
            coord5_xy=(12.5, 24.5),
            distance_to_robot_mm=111.0,
            nearest_neighbor_mm=28.0,
            neighbor_count=1,
        )

        snapshot = publish_auto_plan_snapshot(
            stage="ranked",
            message="Targets ranked.",
            ranked_candidates=[candidate],
            selected_candidate=candidate,
            approach_mm=-25.0,
            pick_mm=0.0,
            drop_mm=8.0,
            orientation=(0.0, 0.0, 20.0),
        )

        self.assertEqual(snapshot["stage"], "ranked")
        self.assertEqual(SESSION.auto_plan_stage, "ranked")
        self.assertEqual(len(snapshot["ranked_candidates"]), 1)
        self.assertEqual(snapshot["ranked_candidates"][0]["label"], "orange")
        self.assertEqual(snapshot["selected_candidate"]["coord5_xy"], [12.5, 24.5])

    def test_get_snapshot_marks_stale(self) -> None:
        publish_auto_plan_snapshot(stage="ranked", message="x")
        SESSION.auto_plan_snapshot["updated_at"] = time.time() - 5.0

        snapshot = get_auto_plan_snapshot(stale_after_s=2.5)

        self.assertTrue(snapshot["available"])
        self.assertTrue(snapshot["stale"])


if __name__ == "__main__":
    unittest.main()
