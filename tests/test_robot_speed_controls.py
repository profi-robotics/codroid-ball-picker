from types import SimpleNamespace

from app.codroid_picker.robot_callbacks.controls import _active_auto_control_api


def test_active_auto_control_api_returns_running_auto_robot_api() -> None:
    robot_api = object()
    active_program = SimpleNamespace(
        name="auto_pick_place",
        is_running=True,
        robot_api=robot_api,
    )

    assert _active_auto_control_api(active_program) is robot_api


def test_active_auto_control_api_ignores_non_auto_program() -> None:
    active_program = SimpleNamespace(
        name="plate_test",
        is_running=True,
        robot_api=object(),
    )

    assert _active_auto_control_api(active_program) is None


def test_active_auto_control_api_ignores_stopped_auto_program() -> None:
    active_program = SimpleNamespace(
        name="auto_pick_place",
        is_running=False,
        robot_api=object(),
    )

    assert _active_auto_control_api(active_program) is None
