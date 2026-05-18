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
            auto_socket_port=19093,
            auto_socket_read_timeout_ms=120000,
            auto_transit_clearance_mm=10.0,
            auto_tolerance_mm=2.0,
            auto_z_tolerance_mm=3.0,
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
    def __init__(self) -> None:
        self.auto_socket_contexts: list[_FakeAutoSocketContext] = []
        self.calls: list[str] = []
        self.config = types.SimpleNamespace(host="192.168.101.100")

    def _local_source_ip_for(self, host: str) -> str:
        self.calls.append(f"_local_source_ip_for:{host}")
        return "192.168.101.110"

    async def power_on(self) -> None:
        self.calls.append("power_on")
        return None

    async def set_auto_mode(self) -> None:
        self.calls.append("set_auto_mode")

    async def set_manual_mode(self) -> None:
        self.calls.append("set_manual_mode")

    async def set_current_coordinate_id(self, coordinate_id: int) -> None:
        self.calls.append(f"set_current_coordinate_id:{coordinate_id}")

    async def stop_auto_socket_project(self) -> None:
        self.calls.append("stop_auto_socket_project")

    def auto_socket_mode(self, config):
        self.calls.append("auto_socket_mode")
        context = _FakeAutoSocketContext(config)
        self.auto_socket_contexts.append(context)
        return context


class _FakeAutoSocketContext:
    def __init__(self, config) -> None:
        self.config = config
        self.entered = False
        self.exited = False
        self.exit_exc_type = None

    async def __aenter__(self):
        self.entered = True
        return self

    async def __aexit__(self, exc_type, exc, tb):
        self.exited = True
        self.exit_exc_type = exc_type
        return False


class _FailingAutoSocketRobotAPI(_FakeRobotAPI):
    def auto_socket_mode(self, config):
        self.calls.append("auto_socket_mode")
        raise RuntimeError("socket unavailable")


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
            "robot_has_object": SESSION.robot_has_object,
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
            robot_api = _FakeRobotAPI()
            result = await program.execute(robot_api)

        self.assertEqual(result.status, ProgramStatus.COMPLETED)
        self.assertEqual(len(robot_api.auto_socket_contexts), 1)
        socket_context = robot_api.auto_socket_contexts[0]
        self.assertTrue(socket_context.entered)
        self.assertTrue(socket_context.exited)
        self.assertLess(
            robot_api.calls.index("set_auto_mode"),
            robot_api.calls.index("auto_socket_mode"),
        )
        self.assertIn("set_current_coordinate_id:5", robot_api.calls)
        self.assertEqual(robot_api.calls[-1], "set_manual_mode")
        self.assertEqual(socket_context.config.socket_port, 19093)
        self.assertEqual(socket_context.config.project_socket_host, "192.168.101.110")
        self.assertEqual(socket_context.config.accept_timeout_s, 25.0)
        self.assertEqual(socket_context.config.read_timeout_ms, 120000)
        self.assertEqual(socket_context.config.move_speed, "V100")
        self.assertEqual(socket_context.config.move_acc, "ACC100")
        self.assertTrue(socket_context.config.install_project)
        self.assertTrue(socket_context.config.start_project)
        self.assertTrue(socket_context.config.stop_project_on_exit)
        published_stages = [call.kwargs.get("stage") for call in publish_mock.call_args_list]
        self.assertIn("ranked", published_stages)
        self.assertIn("candidate_selected", published_stages)
        updated_stages = [call.kwargs.get("stage") for call in update_mock.call_args_list]
        self.assertIn("cycle_complete", updated_stages)

    async def test_execute_fails_fast_when_auto_socket_cannot_start(self) -> None:
        program = AutoPickPlaceProgram()
        fake_registry = _FakeRegistry()

        robot_api = _FailingAutoSocketRobotAPI()
        with (
            patch("app.codroid_picker.programs.auto_pick_place.registry", return_value=fake_registry),
            patch(
                "app.codroid_picker.programs.auto_pick_place.ensure_coordinate_active",
                new=AsyncMock(return_value=None),
            ),
            patch.object(program, "_execute_pick", new=AsyncMock()) as pick_mock,
            patch.object(program, "_execute_drop", new=AsyncMock()) as drop_mock,
            patch("app.codroid_picker.programs.auto_pick_place.update_auto_plan_snapshot") as update_mock,
        ):
            result = await program.execute(robot_api)

        self.assertEqual(result.status, ProgramStatus.ERROR)
        self.assertIn("Auto socket motion setup failed", result.message)
        pick_mock.assert_not_called()
        drop_mock.assert_not_called()
        self.assertEqual(update_mock.call_args.kwargs.get("stage"), "error")
        self.assertEqual(
            [
                "power_on",
                "set_auto_mode",
                "set_current_coordinate_id:5",
                "_local_source_ip_for:192.168.101.100",
                "auto_socket_mode",
                "stop_auto_socket_project",
                "set_manual_mode",
            ],
            robot_api.calls,
        )

    async def test_move_to_drop_without_object_waits_at_approach_only(self) -> None:
        program = AutoPickPlaceProgram()
        fake_registry = _FakeRegistry()
        SESSION.robot_has_object = False

        with (
            patch("app.codroid_picker.programs.auto_pick_place.registry", return_value=fake_registry),
            patch(
                "app.codroid_picker.programs.auto_pick_place.wait_for_operation_resume",
                new=AsyncMock(return_value=None),
            ),
            patch.object(program, "_move_linear_with_hold", new=AsyncMock()) as move_mock,
            patch("app.codroid_picker.programs.auto_pick_place.update_auto_plan_snapshot") as update_mock,
        ):
            await program._move_to_drop(
                _FakeRobotAPI(),
                approach_mm=-30.0,
                drop_mm=0.0,
                open_gripper=False,
                orientation=(0.0, 0.0, 0.0),
            )

        labels = [call.kwargs["label"] for call in move_mock.call_args_list]
        self.assertEqual(
            labels,
            ["move_to_drop_lift", "move_to_drop_transit", "move_to_drop_approach"],
        )
        self.assertNotIn("move_to_drop_pose", labels)
        self.assertIn(
            "No object gripped",
            update_mock.call_args.kwargs.get("message"),
        )

    async def test_move_to_drop_with_object_descends_and_opens_gripper(self) -> None:
        program = AutoPickPlaceProgram()
        fake_registry = _FakeRegistry()
        SESSION.robot_has_object = True

        with (
            patch("app.codroid_picker.programs.auto_pick_place.registry", return_value=fake_registry),
            patch(
                "app.codroid_picker.programs.auto_pick_place.wait_for_operation_resume",
                new=AsyncMock(return_value=None),
            ),
            patch.object(program, "_move_linear_with_hold", new=AsyncMock()) as move_mock,
            patch("app.codroid_picker.programs.auto_pick_place.command_gripper") as gripper_mock,
        ):
            await program._move_to_drop(
                _FakeRobotAPI(),
                approach_mm=-30.0,
                drop_mm=0.0,
                open_gripper=True,
                orientation=(0.0, 0.0, 0.0),
            )

        labels = [call.kwargs["label"] for call in move_mock.call_args_list]
        self.assertEqual(
            labels,
            [
                "move_to_drop_lift",
                "move_to_drop_transit",
                "move_to_drop_pose",
                "move_to_drop_retreat",
            ],
        )
        gripper_mock.assert_called_once_with(100.0, "opened")
        self.assertFalse(SESSION.robot_has_object)

    async def test_move_linear_fails_fast_when_auto_socket_disconnected(self) -> None:
        program = AutoPickPlaceProgram()
        fake_registry = _FakeRegistry()
        robot_api = types.SimpleNamespace(
            _auto_socket_backend=types.SimpleNamespace(is_connected=lambda: False),
        )

        with patch(
            "app.codroid_picker.programs.auto_pick_place.registry",
            return_value=fake_registry,
        ):
            with self.assertRaisesRegex(RuntimeError, "Auto socket disconnected"):
                await program._move_linear_with_hold(
                    robot_api,
                    {
                        "x": 1.0,
                        "y": 2.0,
                        "z": 3.0,
                        "a": 0.0,
                        "b": 0.0,
                        "c": 0.0,
                    },
                    5,
                    label="socket_test",
                    hold_s=0.0,
                )


if __name__ == "__main__":
    unittest.main()
