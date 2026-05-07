"""Tests auto program publishes planning snapshots during decision flow."""

from __future__ import annotations

import types
import unittest
from unittest.mock import AsyncMock, patch

from app.codroid_picker.programs.auto_pick_place import AutoPickPlaceProgram
from app.codroid_picker.programs.base import OperationAbort, ProgramStatus
from app.codroid_picker.session import SESSION
from app.codroid_picker.target_planner import PickPlan


class _FakeRobotSession:
    def posture_seen(self) -> bool:
        return True

    def position_snapshot(self):
        return types.SimpleNamespace(x=0.0, y=0.0, z=100.0, a=0.0, b=0.0, c=0.0)


class _FakeRegistry:
    def __init__(self) -> None:
        self.settings = types.SimpleNamespace(
            auto_target_labels=["orange"],
            auto_drop_ignore_radius_mm=60.0,
            auto_approach_mm=-30.0,
            auto_pick_mm=0.0,
            auto_drop_mm=0.0,
            auto_estimate_width=False,
            tracking_mode="move",
            auto_cluster_neighbor_mm=35.0,
            auto_rotation_clearance_mm=8.0,
            auto_rotation_step_deg=15.0,
            auto_rotation_max_delta_deg=90.0,
            auto_collision_body_radius_mm=12.0,
            auto_collision_finger_reach_mm=20.0,
            auto_separation_push_mm=28.0,
            auto_separation_entry_mm=10.0,
            auto_separation_max_attempts=2,
            auto_rescan_delay_s=0.0,
            auto_idle_sleep_s=0.0,
            auto_hold_s=0.0,
            plate_test_grip_timeout_s=1.0,
            gripper_profile=types.SimpleNamespace(
                force_default=20,
                force_max=40,
                close_width_default=10.0,
                open_width_default=100.0,
            ),
        )
        self.robot_coordinate_id = 5
        self.gripper = None

    def get_or_create_robot_session(self):
        return _FakeRobotSession()


class _FakeRobotAPI:
    async def power_on(self) -> None:
        return None


class AutoPlanProgramUpdateTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self._snapshot = {
            "workspace_definition": SESSION.workspace_definition,
            "drop_target_mm": SESSION.drop_target_mm,
            "drop_target_robot": SESSION.drop_target_robot,
            "model_loaded": SESSION.model_loaded,
            "camera_stream": SESSION.camera_stream,
            "calibration_bundle": SESSION.calibration_bundle,
            "auto_config": SESSION.auto_config,
            "auto_waiting_at_drop": SESSION.auto_waiting_at_drop,
        }
        SESSION.workspace_definition = object()
        SESSION.drop_target_mm = (10.0, 10.0)
        SESSION.drop_target_robot = (100.0, 120.0, -50.0)
        SESSION.model_loaded = True
        SESSION.calibration_bundle = object()
        SESSION.auto_config = {}
        SESSION.auto_waiting_at_drop = False

    def tearDown(self) -> None:
        for key, value in self._snapshot.items():
            setattr(SESSION, key, value)

    async def test_execute_publishes_rank_and_selection(self) -> None:
        program = AutoPickPlaceProgram()
        fake_registry = _FakeRegistry()
        candidate = types.SimpleNamespace(
            coord5_xy=(12.0, 22.0),
            detection={"label": "orange", "workspace_mm": (12.0, 22.0)},
        )

        with (
            patch("app.codroid_picker.programs.auto_pick_place.registry", return_value=fake_registry),
            patch("app.codroid_picker.programs.auto_pick_place.ensure_operation_active"),
            patch(
                "app.codroid_picker.programs.auto_pick_place.ensure_coordinate_active",
                new=AsyncMock(return_value=None),
            ),
            patch(
                "app.codroid_picker.programs.auto_pick_place.plan_pick_strategy",
                return_value=PickPlan(mode="rotate_pick", final_c=0.0, separation=None),
            ),
            patch(
                "app.codroid_picker.auto_operation._refresh_follow_detections",
                side_effect=[[{"label": "orange", "workspace_mm": (12.0, 22.0)}], OperationAbort("stop")],
            ),
            patch("app.codroid_picker.auto_operation._detection_coord5_mm", return_value=(12.0, 22.0)),
            patch.object(program, "_filter_targets", return_value=[{"label": "orange", "workspace_mm": (12.0, 22.0)}]),
            patch.object(program, "_rank_target_candidates", return_value=[candidate]),
            patch.object(program, "_execute_pick", new=AsyncMock(return_value=None)),
            patch.object(program, "_execute_drop", new=AsyncMock(return_value=None)),
            patch("app.codroid_picker.programs.auto_pick_place.publish_auto_plan_snapshot") as publish_mock,
            patch("app.codroid_picker.programs.auto_pick_place.update_auto_plan_snapshot") as update_mock,
            patch("app.codroid_picker.programs.auto_pick_place._record_pick_cycle"),
        ):
            result = await program.execute(_FakeRobotAPI())

        self.assertEqual(result.status, ProgramStatus.COMPLETED)
        published_stages = [call.kwargs.get("stage") for call in publish_mock.call_args_list]
        self.assertIn("ranked", published_stages)
        self.assertIn("candidate_selected", published_stages)
        updated_stages = [call.kwargs.get("stage") for call in update_mock.call_args_list]
        self.assertIn("cycle_complete", updated_stages)

    async def test_execute_pick_uses_label_specific_grip_parameters(self) -> None:
        program = AutoPickPlaceProgram()
        fake_registry = _FakeRegistry()

        with (
            patch("app.codroid_picker.programs.auto_pick_place.registry", return_value=fake_registry),
            patch(
                "app.codroid_picker.programs.auto_pick_place.wait_for_operation_resume",
                new=AsyncMock(return_value=None),
            ),
            patch.object(program, "_move_linear_with_hold", new=AsyncMock(return_value=None)) as move_mock,
            patch(
                "app.codroid_picker.programs.auto_pick_place.command_gripper",
                return_value="closed",
            ) as gripper_mock,
            patch(
                "app.codroid_picker.programs.auto_pick_place.await_grip_status",
                new=AsyncMock(return_value=True),
            ) as grip_status_mock,
            patch("app.codroid_picker.programs.auto_pick_place.publish_auto_plan_snapshot"),
        ):
            await program._execute_pick(
                _FakeRobotAPI(),
                target_coord5=(12.0, 22.0),
                approach_mm=-44.0,
                pick_mm=4.0,
                estimate_width=False,
                target={"label": "cube"},
                target_labels=["cube"],
                tracking_mode="move",
                orientation=(0.0, 0.0, 0.0),
                final_c=15.0,
                gripper_close_mm=17.0,
                gripper_force=33.0,
                gripper_speed=12.0,
                grip_timeout_s=1.7,
            )

        move_targets = [call.args[1] for call in move_mock.call_args_list]
        self.assertTrue(any(target["z"] == -44.0 for target in move_targets))
        self.assertTrue(any(target["z"] == 4.0 for target in move_targets))
        gripper_mock.assert_any_call(
            17.0,
            "closed",
            force=33,
            speed=12,
        )
        grip_status_mock.assert_awaited_with(1.7)


if __name__ == "__main__":
    unittest.main()
