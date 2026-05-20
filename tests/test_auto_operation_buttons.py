from app.codroid_picker.robot_callbacks.auto_operation import (
    _auto_button_disabled_state,
)


def test_auto_start_click_optimistically_enables_stop() -> None:
    assert _auto_button_disabled_state(
        all_ready=True,
        motion_test_ready=True,
        running=False,
        registry_running=False,
        auto_control_running=False,
        controller_auto_mode=False,
        trigger_id="auto-start",
    ) == (True, True, False)


def test_auto_buttons_follow_running_registry_state() -> None:
    assert _auto_button_disabled_state(
        all_ready=True,
        motion_test_ready=True,
        running=False,
        registry_running=True,
        auto_control_running=False,
        controller_auto_mode=False,
        trigger_id=None,
    ) == (True, True, False)


def test_auto_buttons_enable_stop_for_controller_auto_mode() -> None:
    assert _auto_button_disabled_state(
        all_ready=True,
        motion_test_ready=True,
        running=False,
        registry_running=False,
        auto_control_running=False,
        controller_auto_mode=True,
        trigger_id=None,
    ) == (False, False, False)


def test_stale_running_store_does_not_disable_start() -> None:
    assert _auto_button_disabled_state(
        all_ready=True,
        motion_test_ready=True,
        running=True,
        registry_running=False,
        auto_control_running=False,
        controller_auto_mode=False,
        trigger_id=None,
    ) == (False, False, False)


def test_auto_stop_click_disables_stop_while_stop_is_requested() -> None:
    assert _auto_button_disabled_state(
        all_ready=True,
        motion_test_ready=True,
        running=True,
        registry_running=True,
        auto_control_running=True,
        controller_auto_mode=True,
        trigger_id="auto-stop",
    ) == (True, True, True)


def test_auto_buttons_enable_stop_for_last_auto_program_state() -> None:
    assert _auto_button_disabled_state(
        all_ready=True,
        motion_test_ready=True,
        running=False,
        registry_running=False,
        auto_control_running=False,
        controller_auto_mode=False,
        trigger_id=None,
        last_auto_program=True,
    ) == (False, False, False)
