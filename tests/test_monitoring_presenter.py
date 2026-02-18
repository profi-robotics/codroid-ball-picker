"""Tests for monitoring dashboard presenter helpers."""

from __future__ import annotations

from collections import deque
import time
import types
import unittest

from app.codroid_picker.ui_callbacks.monitoring import (
    _append_detection_history,
    _build_auto_plan_items,
    _build_cycle_trend_figure,
    _build_detection_trend_figure,
)


class MonitoringPresenterTests(unittest.TestCase):
    def test_detection_history_trims_window(self) -> None:
        now_ts = time.time()
        session = types.SimpleNamespace(
            monitor_detection_history=deque(
                [
                    {"ts": now_ts - 500.0, "count": 1},
                    {"ts": now_ts - 2.0, "count": 2},
                ],
                maxlen=90,
            )
        )

        history = _append_detection_history(session, 3, now_ts=now_ts, window_s=180.0)

        self.assertEqual(len(history), 2)
        self.assertEqual(history[-1]["count"], 3)

    def test_trend_figures_handle_empty_and_non_empty(self) -> None:
        empty_det = _build_detection_trend_figure([])
        empty_cycle = _build_cycle_trend_figure([])
        self.assertEqual(len(empty_det.data), 0)
        self.assertEqual(len(empty_cycle.data), 0)

        filled_det = _build_detection_trend_figure([{"ts": 1.0, "count": 2}, {"ts": 2.0, "count": 5}])
        filled_cycle = _build_cycle_trend_figure([1.2, 1.0, 1.5])
        self.assertEqual(len(filled_det.data), 1)
        self.assertEqual(len(filled_cycle.data), 1)

    def test_auto_plan_items_empty_and_populated(self) -> None:
        empty_items = _build_auto_plan_items({"available": False})
        self.assertGreaterEqual(len(empty_items), 1)

        populated_items = _build_auto_plan_items(
            {
                "available": True,
                "stale": False,
                "age_s": 0.2,
                "stage": "ranked",
                "message": "Targets ranked.",
                "selected_candidate": {"label": "orange", "coord5_xy": [10.0, 20.0]},
                "pick_plan": {"mode": "rotate_pick", "final_c": 15.0},
                "gripper_close_mm": 14.0,
                "drop_target_mm": [30.0, 40.0],
                "ranked_candidates": [
                    {
                        "rank": 1,
                        "label": "orange",
                        "neighbor_count": 0,
                        "nearest_neighbor_mm": 35.0,
                    }
                ],
            }
        )
        self.assertGreaterEqual(len(populated_items), 4)


if __name__ == "__main__":
    unittest.main()
