from app.codroid_picker.robot_callbacks.auto_operation import (
    _auto_button_disabled_state,
    _operation_state_flags,
)


def test_auto_start_click_optimistically_disables_start_buttons() -> None:
    assert _auto_button_disabled_state(
        all_ready=True,
        motion_test_ready=True,
        registry_running=False,
        trigger_id="auto-start",
    ) == (True, True)


def test_auto_buttons_follow_running_registry_state() -> None:
    assert _auto_button_disabled_state(
        all_ready=True,
        motion_test_ready=True,
        registry_running=True,
        trigger_id=None,
    ) == (True, True)


def test_auto_buttons_enable_start_when_ready_and_idle() -> None:
    assert _auto_button_disabled_state(
        all_ready=True,
        motion_test_ready=True,
        registry_running=False,
        trigger_id=None,
    ) == (False, False)


def test_auto_buttons_disable_start_when_prerequisites_missing() -> None:
    assert _auto_button_disabled_state(
        all_ready=False,
        motion_test_ready=True,
        registry_running=False,
        trigger_id=None,
    ) == (True, False)


def test_auto_buttons_disable_motion_test_when_robot_not_connected() -> None:
    assert _auto_button_disabled_state(
        all_ready=True,
        motion_test_ready=False,
        registry_running=False,
        trigger_id=None,
    ) == (False, True)


def test_operation_state_flags_detect_active_auto_program() -> None:
    assert _operation_state_flags(
        {"status": "running", "source": "auto_pick_place"}
    ) == (True, True)


def test_operation_state_flags_keep_non_auto_program_separate() -> None:
    assert _operation_state_flags(
        {"status": "running", "source": "verification"}
    ) == (True, False)
