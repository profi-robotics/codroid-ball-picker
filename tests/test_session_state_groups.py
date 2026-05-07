from __future__ import annotations

import unittest

from app.codroid_picker.session import UISession


class SessionStateGroupingTests(unittest.TestCase):
    def test_camera_aliases_write_into_camera_state(self) -> None:
        session = UISession()
        session.camera_last_error_message = "boom"
        self.assertEqual(session.camera.last_error_message, "boom")
        session.camera.auto_init_done = True
        self.assertTrue(session.camera_auto_init_done)

    def test_robot_aliases_write_into_robot_state(self) -> None:
        session = UISession()
        session.robot_control_mode = "released"
        session.robot_speed_rate = 0.5
        self.assertEqual(session.robot.control_mode, "released")
        self.assertAlmostEqual(session.robot.speed_rate, 0.5)

    def test_auto_aliases_write_into_auto_state(self) -> None:
        session = UISession()
        session.auto_plan_stage = "pick"
        session.auto_config = {"target_label": "orange"}
        self.assertEqual(session.auto.plan_stage, "pick")
        self.assertEqual(session.auto.config["target_label"], "orange")


if __name__ == "__main__":
    unittest.main()
