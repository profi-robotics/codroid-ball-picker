"""Tests for overlay resolution behavior in camera callbacks."""

from __future__ import annotations

from pathlib import Path
import unittest
from unittest.mock import patch

from app.codroid_picker.camera_callbacks import _resolve_overlay_task


class OverlaySelectionTests(unittest.TestCase):
    def test_manual_auto_plan_mode(self) -> None:
        task, effective = _resolve_overlay_task(
            requested_overlay="auto-plan",
            tab="applications",
            calibration_step=0,
            application_tab="pick-place",
            board_path=Path("data/markers/camera_calibration/camera_calibration_board.json"),
            marker_ids=(1, 2),
            marker_size_mm=40.0,
            plate_path=Path("data/markers/robot_calibration/calibration_plate_layout.json"),
        )

        self.assertIsNotNone(task)
        self.assertEqual(task.key, "manual-auto-plan")
        self.assertEqual(effective, "auto-plan")

    def test_auto_switches_to_auto_plan_when_running(self) -> None:
        with patch("app.codroid_picker.camera_callbacks._auto_program_running", return_value=True):
            task, effective = _resolve_overlay_task(
                requested_overlay="auto",
                tab="applications",
                calibration_step=0,
                application_tab="pick-place",
                board_path=Path("data/markers/camera_calibration/camera_calibration_board.json"),
                marker_ids=(1, 2),
                marker_size_mm=40.0,
                plate_path=Path("data/markers/robot_calibration/calibration_plate_layout.json"),
            )

        self.assertIsNotNone(task)
        self.assertEqual(task.key, "auto-plan")
        self.assertEqual(effective, "auto-plan")

    def test_auto_pick_place_remains_none_when_not_running(self) -> None:
        with patch("app.codroid_picker.camera_callbacks._auto_program_running", return_value=False):
            task, effective = _resolve_overlay_task(
                requested_overlay="auto",
                tab="applications",
                calibration_step=0,
                application_tab="pick-place",
                board_path=Path("data/markers/camera_calibration/camera_calibration_board.json"),
                marker_ids=(1, 2),
                marker_size_mm=40.0,
                plate_path=Path("data/markers/robot_calibration/calibration_plate_layout.json"),
            )

        self.assertIsNone(task)
        self.assertEqual(effective, "auto")


if __name__ == "__main__":
    unittest.main()
