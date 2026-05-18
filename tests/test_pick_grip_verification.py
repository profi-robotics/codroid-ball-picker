"""Tests for grip-verification behavior in auto and manual pick flows."""

from __future__ import annotations

import types
import unittest
from unittest.mock import AsyncMock, patch

from app.codroid_picker.programs.auto_pick_place import (
    AutoPickPlaceProgram,
    EmptyGripError,
)
from app.codroid_picker.programs.base import OperationAbort, ProgramStatus
from app.codroid_picker.programs.manual_pick_place import ManualPickPlaceProgram
from app.codroid_picker.session import SESSION


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
            auto_transit_clearance_mm=10.0,
            auto_socket_port=19093,
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

    async def __aenter__(self):
        self.entered = True
        return self

    async def __aexit__(self, exc_type, exc, tb):
        self.exited = True
        return False


class PickGripVerificationTests(unittest.IsolatedAsyncioTestCase):
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
        SESSION.robot_has_object = False

    def tearDown(self) -> None:
        for key, value in self._snapshot.items():
            setattr(SESSION, key, value)

    async def test_auto_empty_grip_skips_drop_and_drop_fallback(self) -> None:
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
                return_value=types.SimpleNamespace(mode="rotate_pick", final_c=0.0, separation=None),
            ),
            patch(
                "app.codroid_picker.auto_operation._refresh_follow_detections",
                side_effect=[[{"label": "orange", "workspace_mm": (12.0, 22.0)}], OperationAbort("stop")],
            ),
            patch("app.codroid_picker.auto_operation._detection_coord5_mm", return_value=(12.0, 22.0)),
            patch.object(program, "_filter_targets", return_value=[{"label": "orange", "workspace_mm": (12.0, 22.0)}]),
            patch.object(program, "_rank_target_candidates", return_value=[candidate]),
            patch.object(program, "_execute_pick", new=AsyncMock(side_effect=EmptyGripError("empty"))),
            patch.object(program, "_execute_drop", new=AsyncMock()) as drop_mock,
            patch.object(program, "_move_to_drop", new=AsyncMock()) as move_to_drop_mock,
        ):
            result = await program.execute(_FakeRobotAPI())

        self.assertEqual(result.status, ProgramStatus.COMPLETED)
        self.assertEqual(result.data["cycles_completed"], 0)
        self.assertEqual(result.data["cycles_failed"], 1)
        drop_mock.assert_not_called()
        move_to_drop_mock.assert_not_called()

    async def test_auto_success_executes_drop_and_counts_completed(self) -> None:
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
                return_value=types.SimpleNamespace(mode="rotate_pick", final_c=0.0, separation=None),
            ),
            patch(
                "app.codroid_picker.auto_operation._refresh_follow_detections",
                side_effect=[[{"label": "orange", "workspace_mm": (12.0, 22.0)}], OperationAbort("stop")],
            ),
            patch("app.codroid_picker.auto_operation._detection_coord5_mm", return_value=(12.0, 22.0)),
            patch.object(program, "_filter_targets", return_value=[{"label": "orange", "workspace_mm": (12.0, 22.0)}]),
            patch.object(program, "_rank_target_candidates", return_value=[candidate]),
            patch.object(program, "_execute_pick", new=AsyncMock(return_value=None)),
            patch.object(program, "_execute_drop", new=AsyncMock()) as drop_mock,
        ):
            result = await program.execute(_FakeRobotAPI())

        self.assertEqual(result.status, ProgramStatus.COMPLETED)
        self.assertEqual(result.data["cycles_completed"], 1)
        self.assertEqual(result.data["cycles_failed"], 0)
        drop_mock.assert_called_once()

    async def test_auto_pick_closes_to_resolved_narrowest_width(self) -> None:
        program = AutoPickPlaceProgram()
        fake_registry = _FakeRegistry()

        with (
            patch("app.codroid_picker.programs.auto_pick_place.registry", return_value=fake_registry),
            patch(
                "app.codroid_picker.programs.auto_pick_place.wait_for_operation_resume",
                new=AsyncMock(return_value=None),
            ),
            patch.object(program, "_move_linear_with_hold", new=AsyncMock(return_value=None)),
            patch("app.codroid_picker.programs.auto_pick_place.log_move_target"),
            patch("app.codroid_picker.programs.auto_pick_place.publish_auto_plan_snapshot"),
            patch(
                "app.codroid_picker.programs.auto_pick_place.resolve_gripper_close_width",
                return_value=0.0,
            ) as close_width_mock,
            patch(
                "app.codroid_picker.programs.auto_pick_place.command_gripper",
                return_value="ok",
            ) as command_mock,
            patch(
                "app.codroid_picker.programs.auto_pick_place.await_grip_status",
                new=AsyncMock(return_value=True),
            ),
        ):
            await program._execute_pick(
                _FakeRobotAPI(),
                target_coord5=(12.0, 22.0),
                approach_mm=-30.0,
                drop_mm=0.0,
                pick_mm=0.0,
                estimate_width=True,
                target={
                    "label": "orange",
                    "workspace_mm": (12.0, 22.0),
                    "bbox": (0.0, 0.0, 100.0, 80.0),
                },
                target_labels=["orange"],
                tracking_mode="move",
                orientation=(0.0, 0.0, 0.0),
                final_c=0.0,
            )

        close_width_mock.assert_called_once_with()
        command_mock.assert_any_call(0.0, "closed", force=40)

    async def test_manual_pick_drop_returns_error_on_empty_grip(self) -> None:
        fake_registry = _FakeRegistry()
        program = ManualPickPlaceProgram(
            action="pick_drop",
            pick_approach_pose_fn=lambda: {"x": 10.0, "y": 20.0, "z": -30.0, "a": 0.0, "b": 0.0, "c": 0.0},
            pick_pose_fn=lambda: {"x": 10.0, "y": 20.0, "z": 0.0, "a": 0.0, "b": 0.0, "c": 0.0},
            tracking_mode="move",
        )
        program.move_with_hold = AsyncMock(return_value=None)  # type: ignore[method-assign]
        robot_api = _FakeRobotAPI()
        SESSION.robot_has_object = True

        with (
            patch("app.codroid_picker.programs.manual_pick_place.registry", return_value=fake_registry),
            patch(
                "app.codroid_picker.programs.manual_pick_place.ensure_coordinate_active",
                new=AsyncMock(return_value=None),
            ),
            patch(
                "app.codroid_picker.programs.manual_pick_place.wait_for_operation_resume",
                new=AsyncMock(return_value=None),
            ),
            patch(
                "app.codroid_picker.programs.manual_pick_place.await_grip_status",
                new=AsyncMock(return_value=False),
            ),
            patch(
                "app.codroid_picker.programs.manual_pick_place.resolve_gripper_close_width",
                return_value=10.0,
            ),
            patch(
                "app.codroid_picker.programs.manual_pick_place.command_gripper",
                return_value="ok",
            ) as command_mock,
        ):
            result = await program._execute_pick(robot_api)

        self.assertEqual(result.status, ProgramStatus.ERROR)
        self.assertFalse(SESSION.robot_has_object)
        self.assertEqual(command_mock.call_count, 2)

    async def test_manual_pick_sets_robot_has_object_on_grip_success(self) -> None:
        fake_registry = _FakeRegistry()
        program = ManualPickPlaceProgram(
            action="pick",
            pick_approach_pose_fn=lambda: {"x": 10.0, "y": 20.0, "z": -30.0, "a": 0.0, "b": 0.0, "c": 0.0},
            pick_pose_fn=lambda: {"x": 10.0, "y": 20.0, "z": 0.0, "a": 0.0, "b": 0.0, "c": 0.0},
            tracking_mode="move",
        )
        program.move_with_hold = AsyncMock(return_value=None)  # type: ignore[method-assign]
        robot_api = _FakeRobotAPI()
        SESSION.robot_has_object = False

        with (
            patch("app.codroid_picker.programs.manual_pick_place.registry", return_value=fake_registry),
            patch(
                "app.codroid_picker.programs.manual_pick_place.ensure_coordinate_active",
                new=AsyncMock(return_value=None),
            ),
            patch(
                "app.codroid_picker.programs.manual_pick_place.wait_for_operation_resume",
                new=AsyncMock(return_value=None),
            ),
            patch(
                "app.codroid_picker.programs.manual_pick_place.await_grip_status",
                new=AsyncMock(return_value=True),
            ),
            patch(
                "app.codroid_picker.programs.manual_pick_place.resolve_gripper_close_width",
                return_value=10.0,
            ),
            patch(
                "app.codroid_picker.programs.manual_pick_place.command_gripper",
                return_value="ok",
            ),
        ):
            result = await program._execute_pick(robot_api)

        self.assertEqual(result.status, ProgramStatus.COMPLETED)
        self.assertTrue(SESSION.robot_has_object)
        self.assertEqual(robot_api.auto_socket_contexts, [])


if __name__ == "__main__":
    unittest.main()
