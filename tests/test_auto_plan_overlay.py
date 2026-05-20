"""Tests for auto-plan overlay rendering."""

from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np

from app.codroid_picker.auto_plan_state import clear_auto_plan_snapshot, publish_auto_plan_snapshot
from app.codroid_picker import overlay_tasks
from app.codroid_picker.overlays.auto_plan import (
    _robot_posture_pixel,
    build_auto_plan_primitives,
    draw_auto_plan_overlay,
)
from app.codroid_picker.session import SESSION
from app.settings import SETTINGS


class AutoPlanOverlayTests(unittest.TestCase):
    def setUp(self) -> None:
        self._snapshot = dict(SESSION.auto_plan_snapshot)
        self._workspace_definition = SESSION.workspace_definition
        clear_auto_plan_snapshot()

    def tearDown(self) -> None:
        SESSION.auto_plan_snapshot = self._snapshot
        SESSION.workspace_definition = self._workspace_definition

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

    def test_overlay_prefers_live_robot_origin_for_approach_line(self) -> None:
        publish_auto_plan_snapshot(
            stage="executing_pick",
            selected_candidate={
                "label": "orange",
                "confidence": 0.9,
                "center": (80.0, 60.0),
                "bbox": (60.0, 40.0, 100.0, 80.0),
            },
        )
        frame = np.zeros((120, 160, 3), dtype=np.uint8)
        with patch(
            "app.codroid_picker.overlays.auto_plan._robot_posture_pixel",
            return_value=(10, 10),
        ):
            vis, _status, _metric = draw_auto_plan_overlay(frame)
        self.assertGreater(int(vis[10, 10].sum()), 0)

    def test_auto_plan_task_emits_structured_primitives(self) -> None:
        publish_auto_plan_snapshot(
            stage="executing_pick",
            message="Executing pick motion.",
            ranked_candidates=[
                {
                    "label": "orange",
                    "confidence": 0.9,
                    "center": (80.0, 60.0),
                    "bbox": (60.0, 40.0, 100.0, 80.0),
                }
            ],
            selected_candidate={
                "label": "orange",
                "confidence": 0.9,
                "center": (80.0, 60.0),
                "bbox": (60.0, 40.0, 100.0, 80.0),
            },
        )
        frame = np.zeros((120, 160, 3), dtype=np.uint8)

        with patch(
            "app.codroid_picker.overlays.auto_plan._robot_posture_pixel",
            return_value=(10, 10),
        ):
            result = overlay_tasks._run_auto_plan_task()(frame)

        self.assertTrue(result.has_targets)
        primitive_types = {item["type"] for item in result.primitives}
        self.assertIn("text", primitive_types)
        self.assertIn("arrow", primitive_types)
        self.assertIn("circle", primitive_types)
        self.assertNotEqual(primitive_types, {"polygon"})

    def test_primitives_use_live_tcp_as_approach_start(self) -> None:
        publish_auto_plan_snapshot(
            stage="executing_pick",
            message="Executing pick motion.",
            selected_candidate={
                "label": "orange",
                "confidence": 0.9,
                "center": (80.0, 60.0),
                "bbox": (60.0, 40.0, 100.0, 80.0),
            },
        )

        with patch(
            "app.codroid_picker.overlays.auto_plan._robot_posture_pixel",
            return_value=(10, 10),
        ):
            primitives = build_auto_plan_primitives((120, 160, 3))

        approach = next(
            item for item in primitives
            if item["type"] == "arrow" and item.get("label") == "Approach"
        )
        self.assertEqual(approach["from"], {"x": 10, "y": 10})
        self.assertEqual(approach["to"], {"x": 80, "y": 60})
        self.assertEqual(approach["label_at"], "start")
        text_values = [item.get("text") for item in primitives if item["type"] == "text"]
        self.assertIn("Auto plan: executing_pick", text_values)
        self.assertIn("Executing pick motion.", text_values)

    def test_primitives_omit_approach_when_robot_posture_missing(self) -> None:
        publish_auto_plan_snapshot(
            stage="candidate_selected",
            selected_candidate={
                "label": "orange",
                "confidence": 0.9,
                "center": (80.0, 60.0),
                "bbox": (60.0, 40.0, 100.0, 80.0),
            },
        )

        with patch(
            "app.codroid_picker.overlays.auto_plan._robot_posture_pixel",
            return_value=None,
        ):
            primitives = build_auto_plan_primitives((120, 160, 3))

        self.assertFalse(
            any(item["type"] == "arrow" and item.get("label") == "Approach" for item in primitives)
        )
        self.assertTrue(any(item["type"] == "cross" for item in primitives))

    def test_coord5_robot_posture_projection_uses_xy_footprint(self) -> None:
        class FakePosition:
            x = 1.0
            y = 2.0
            z = 3.0

        class FakeRobotSession:
            def posture_seen(self) -> bool:
                return True

            def position_snapshot(self) -> FakePosition:
                return FakePosition()

        SESSION.workspace_definition = object()

        with (
            patch(
                "app.codroid_picker.overlays.auto_plan._get_robot_session",
                return_value=FakeRobotSession(),
            ),
            patch(
                "app.codroid_picker.overlays.auto_plan._current_coordinate_id",
                return_value=SETTINGS.robot_coordinate_id,
            ),
            patch(
                "app.codroid_picker.overlays.auto_plan._coord5_to_workspace_mm",
                return_value=(7.0, 8.0),
            ) as coord5_to_workspace,
            patch(
                "app.codroid_picker.overlays.auto_plan._workspace_mm_to_pixel",
                return_value=(9, 10),
            ),
        ):
            px = _robot_posture_pixel()

        self.assertEqual(px, (9, 10))
        self.assertEqual(coord5_to_workspace.call_args.args[1], (1.0, 2.0))

    def test_robot_posture_projection_skips_unknown_coordinate_frame(self) -> None:
        class FakePosition:
            x = 1.0
            y = 2.0
            z = 3.0

        class FakeRobotSession:
            def posture_seen(self) -> bool:
                return True

            def position_snapshot(self) -> FakePosition:
                return FakePosition()

        SESSION.workspace_definition = object()

        with (
            patch(
                "app.codroid_picker.overlays.auto_plan._get_robot_session",
                return_value=FakeRobotSession(),
            ),
            patch(
                "app.codroid_picker.overlays.auto_plan._current_coordinate_id",
                return_value=None,
            ),
        ):
            px = _robot_posture_pixel()

        self.assertIsNone(px)

    def test_primitives_omit_stale_route_arrows(self) -> None:
        publish_auto_plan_snapshot(
            stage="executing_pick",
            selected_candidate={
                "label": "orange",
                "confidence": 0.9,
                "center": (80.0, 60.0),
                "bbox": (60.0, 40.0, 100.0, 80.0),
            },
        )

        with (
            patch(
                "app.codroid_picker.overlays.auto_plan.get_auto_plan_snapshot",
                return_value={
                    "available": True,
                    "stale": True,
                    "stage": "executing_pick",
                    "message": "Executing pick motion.",
                    "ranked_candidates": [],
                    "selected_candidate": {
                        "label": "orange",
                        "confidence": 0.9,
                        "center_px": [80.0, 60.0],
                    },
                    "pick_plan": {},
                },
            ),
            patch(
                "app.codroid_picker.overlays.auto_plan._robot_posture_pixel",
                return_value=(10, 10),
            ),
        ):
            primitives = build_auto_plan_primitives((120, 160, 3))

        self.assertFalse(any(item["type"] == "arrow" for item in primitives))
        self.assertTrue(any(item["type"] == "cross" for item in primitives))

    def test_primitives_omit_route_arrows_with_off_frame_endpoint(self) -> None:
        publish_auto_plan_snapshot(
            stage="executing_pick",
            selected_candidate={
                "label": "orange",
                "confidence": 0.9,
                "center": (80.0, 60.0),
                "bbox": (60.0, 40.0, 100.0, 80.0),
            },
        )

        with patch(
            "app.codroid_picker.overlays.auto_plan._robot_posture_pixel",
            return_value=(-20, 10),
        ):
            primitives = build_auto_plan_primitives((120, 160, 3))

        self.assertFalse(
            any(item["type"] == "arrow" and item.get("label") == "Approach" for item in primitives)
        )
        self.assertTrue(any(item["type"] == "cross" for item in primitives))


if __name__ == "__main__":
    unittest.main()
