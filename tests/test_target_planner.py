"""Unit tests for collision-aware target planning."""

from __future__ import annotations

import unittest

from app.codroid_picker.target_planner import plan_pick_strategy, rank_targets


class TargetPlannerTests(unittest.TestCase):
    """Validate ranking and strategy selection behavior."""

    @staticmethod
    def _coord5(det: dict):
        return det.get("coord5_mm")

    def test_rank_prefers_clearer_candidate_when_distance_is_similar(self) -> None:
        detections = [
            {"id": "clustered", "coord5_mm": (10.0, 10.0)},
            {"id": "neighbor", "coord5_mm": (18.0, 10.0)},
            {"id": "clear", "coord5_mm": (40.0, 10.0)},
        ]
        ranked = rank_targets(
            detections,
            robot_xy=(0.0, 0.0),
            coord5_fn=self._coord5,
            neighbor_radius_mm=20.0,
        )
        self.assertEqual(ranked[0].detection["id"], "clear")

    def test_plan_rotate_pick_for_clear_target(self) -> None:
        detections = [
            {"id": "target", "coord5_mm": (40.0, 0.0)},
            {"id": "n1", "coord5_mm": (80.0, 0.0)},
        ]
        ranked = rank_targets(
            detections,
            robot_xy=(0.0, 0.0),
            coord5_fn=self._coord5,
            neighbor_radius_mm=30.0,
        )
        plan = plan_pick_strategy(
            ranked[0],
            ranked,
            current_c_deg=0.0,
            rotation_step_deg=15.0,
            rotation_max_delta_deg=90.0,
            body_radius_mm=10.0,
            finger_reach_mm=15.0,
            clearance_mm=6.0,
            separation_entry_mm=10.0,
            separation_push_mm=25.0,
        )
        self.assertEqual(plan.mode, "rotate_pick")
        self.assertLess(abs(plan.final_c), 1e-6)

    def test_plan_rotate_pick_with_nonzero_safe_angle(self) -> None:
        detections = [
            {"id": "target", "coord5_mm": (0.0, 0.0)},
            {"id": "blocker", "coord5_mm": (28.0, 0.0)},
            {"id": "other", "coord5_mm": (0.0, 45.0)},
        ]
        ranked = rank_targets(
            detections,
            robot_xy=(-50.0, 0.0),
            coord5_fn=self._coord5,
            neighbor_radius_mm=60.0,
        )
        target = next(item for item in ranked if item.detection["id"] == "target")
        plan = plan_pick_strategy(
            target,
            ranked,
            current_c_deg=0.0,
            rotation_step_deg=15.0,
            rotation_max_delta_deg=90.0,
            body_radius_mm=12.0,
            finger_reach_mm=18.0,
            clearance_mm=8.0,
            separation_entry_mm=10.0,
            separation_push_mm=25.0,
        )
        self.assertEqual(plan.mode, "rotate_pick")
        self.assertNotEqual(abs(plan.final_c), 0.0)

    def test_plan_falls_back_to_separation_when_no_angle_is_safe(self) -> None:
        detections = [
            {"id": "target", "coord5_mm": (0.0, 0.0)},
            {"id": "n1", "coord5_mm": (14.0, 0.0)},
            {"id": "n2", "coord5_mm": (-14.0, 0.0)},
            {"id": "n3", "coord5_mm": (0.0, 14.0)},
            {"id": "n4", "coord5_mm": (0.0, -14.0)},
        ]
        ranked = rank_targets(
            detections,
            robot_xy=(0.0, -100.0),
            coord5_fn=self._coord5,
            neighbor_radius_mm=25.0,
        )
        target = next(item for item in ranked if item.detection["id"] == "target")
        plan = plan_pick_strategy(
            target,
            ranked,
            current_c_deg=0.0,
            rotation_step_deg=15.0,
            rotation_max_delta_deg=90.0,
            body_radius_mm=12.0,
            finger_reach_mm=20.0,
            clearance_mm=10.0,
            separation_entry_mm=8.0,
            separation_push_mm=25.0,
        )
        self.assertEqual(plan.mode, "separate_then_rescan")
        self.assertIsNotNone(plan.separation)


if __name__ == "__main__":
    unittest.main()
