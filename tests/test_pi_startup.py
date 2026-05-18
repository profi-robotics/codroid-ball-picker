"""Tests for Pi-friendly startup behavior."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from app.codroid_picker.layout import build_layout
from app.codroid_picker.models import auto_load_configured_model
from app.codroid_picker.persistence import _restore_persisted_state
from app.codroid_picker.robot_callbacks.calibration import (
    should_open_robot_calibration_modal,
)
from app.codroid_picker.state import (
    build_robot_calibration_state_payload,
    get_calibration_guidance_snapshot,
    update_calibration_guidance,
)
from app.codroid_picker.ui_callbacks.lazy_layout import (
    render_main_tab,
    sidebar_layout_state,
)


def _walk_components(component):
    yield component
    children = getattr(component, "children", None)
    if children is None:
        return
    if isinstance(children, (list, tuple)):
        for child in children:
            yield from _walk_components(child)
    else:
        yield from _walk_components(children)


def _component_ids(component) -> set[str]:
    return {
        component_id
        for item in _walk_components(component)
        if isinstance((component_id := getattr(item, "id", None)), str)
    }


def test_robot_calibration_modal_stays_closed_when_suppressed() -> None:
    state = {
        "status": "awaiting_verification",
        "modal_suppressed": True,
    }

    assert not should_open_robot_calibration_modal(state, "robot-status-interval")
    assert not should_open_robot_calibration_modal(
        {"status": "complete", "modal_suppressed": True},
        "robot-status-interval",
    )
    assert not should_open_robot_calibration_modal(
        {"status": "error", "modal_suppressed": True},
        "robot-status-interval",
    )


def test_starting_new_robot_calibration_run_clears_modal_suppression() -> None:
    original = get_calibration_guidance_snapshot()
    try:
        update_calibration_guidance(
            "Closed during current run.",
            status="awaiting_verification",
            step=5,
            verification_ready=True,
            modal_suppressed=True,
        )
        assert build_robot_calibration_state_payload()["modal_suppressed"] is True

        update_calibration_guidance(
            "Robot calibration started.",
            status="running",
            step=1,
            verification_ready=False,
            modal_suppressed=False,
        )
        payload = build_robot_calibration_state_payload()

        assert payload["modal_suppressed"] is False
        assert should_open_robot_calibration_modal(payload, "robot-calibration-state")
    finally:
        update_calibration_guidance(
            original.get("message", ""),
            status=original.get("status"),
            step=original.get("step"),
            verification_ready=original.get("verification_ready"),
            modal_suppressed=original.get("modal_suppressed"),
        )


def test_persisted_model_restore_remembers_path_without_loading_yolo(
    tmp_path,
    monkeypatch,
) -> None:
    model_path = tmp_path / "remembered.pt"
    model_path.write_bytes(b"model")
    session = SimpleNamespace(
        calibration_bundle=None,
        workspace_definition=None,
        active_model_path=None,
    )
    settings = SimpleNamespace(
        active_model=str(model_path),
        last_model=None,
        model_auto_load=False,
        drop_target=None,
    )
    fake_registry = SimpleNamespace(
        session=session,
        settings=settings,
        bundle_path=tmp_path / "missing_bundle.npz",
        workspace_path=tmp_path / "missing_workspace.npz",
        robot_calibration_path=tmp_path / "missing_robot.json",
        drop_target_path=tmp_path / "missing_drop.json",
        model_dir=tmp_path,
    )

    monkeypatch.setattr(
        "app.codroid_picker.persistence.registry",
        lambda: fake_registry,
    )

    _restore_persisted_state()

    assert session.active_model_path == Path(model_path)


def test_persisted_model_restore_defers_enabled_autoload_to_hardware_init(
    tmp_path,
    monkeypatch,
) -> None:
    model_path = tmp_path / "remembered.pt"
    model_path.write_bytes(b"model")
    session = SimpleNamespace(
        calibration_bundle=None,
        workspace_definition=None,
        active_model_path=None,
    )
    settings = SimpleNamespace(
        active_model=str(model_path),
        last_model=None,
        model_auto_load=True,
        drop_target=None,
    )
    fake_registry = SimpleNamespace(
        session=session,
        settings=settings,
        bundle_path=tmp_path / "missing_bundle.npz",
        workspace_path=tmp_path / "missing_workspace.npz",
        robot_calibration_path=tmp_path / "missing_robot.json",
        drop_target_path=tmp_path / "missing_drop.json",
        model_dir=tmp_path,
    )

    monkeypatch.setattr(
        "app.codroid_picker.persistence.registry",
        lambda: fake_registry,
    )

    _restore_persisted_state()

    assert session.active_model_path == Path(model_path)


def test_model_autoload_helper_activates_remembered_path(tmp_path) -> None:
    model_path = tmp_path / "remembered.pt"
    model_path.write_bytes(b"model")
    settings = SimpleNamespace(
        active_model=None,
        last_model=str(model_path),
        model_auto_load=True,
    )
    session = SimpleNamespace(
        active_model_path=None,
        model_loaded=False,
        last_detection_error=None,
    )
    calls = []

    def activate(path: Path) -> str:
        calls.append(path)
        session.active_model_path = path
        session.model_loaded = True
        return f"Model loaded: {path.name}."

    payload = auto_load_configured_model(
        settings=settings,
        session=session,
        activate_fn=activate,
    )

    assert calls == [model_path]
    assert payload["status"] == "loaded"
    assert payload["model"] == str(model_path)


def test_initial_layout_defers_tab_content_and_hardware_init() -> None:
    ids = _component_ids(build_layout())

    assert "monitor-detection-trend" not in ids
    assert "monitor-cycle-trend" not in ids
    assert "run-detection" not in ids
    assert "calibration-stepper" not in ids
    assert "main-tab-content" in ids
    assert "sidebar-toggle" in ids
    assert "sidebar-visibility" in ids
    assert "model-load-state" in ids
    assert "monitoring-graph-interval" in ids
    assert "application-ui-interval" in ids
    assert "calibration-ui-interval" in ids
    assert "camera-interval" not in ids
    assert "auto-init" not in ids
    assert {"ui-ready", "hardware-init"}.issubset(ids)


def test_lazy_layout_mounts_application_tab_and_enables_application_interval() -> None:
    content, app_disabled, calibration_disabled, monitoring_disabled = render_main_tab(
        "applications",
        {"value": "workspace"},
        {"active": 1},
    )
    ids = _component_ids(content)

    assert "application-tabs" in ids
    assert "compute-workspace" in ids
    assert "run-detection" in ids
    assert app_disabled is False
    assert calibration_disabled is True
    assert monitoring_disabled is True


def test_lazy_layout_mounts_calibration_tab_and_enables_calibration_interval() -> None:
    content, app_disabled, calibration_disabled, monitoring_disabled = render_main_tab(
        "calibration",
        {"value": "pick-place"},
        {"active": 1},
    )
    ids = _component_ids(content)

    assert "calibration-stepper" in ids
    assert "calibrate-camera" in ids
    assert "start-robot-calibration" in ids
    assert app_disabled is True
    assert calibration_disabled is False
    assert monitoring_disabled is True


def test_lazy_layout_mounts_monitoring_tab_and_enables_monitoring_interval() -> None:
    content, app_disabled, calibration_disabled, monitoring_disabled = render_main_tab(
        "monitoring",
        {"value": "pick-place"},
        {"active": 0},
    )
    ids = _component_ids(content)

    assert "monitor-detection-trend" in ids
    assert "monitor-cycle-trend" in ids
    assert app_disabled is True
    assert calibration_disabled is True
    assert monitoring_disabled is False


def test_sidebar_layout_state_collapses_controls_and_expands_camera() -> None:
    sidebar_style, camera_span, label = sidebar_layout_state(True)

    assert sidebar_style == {"display": "none"}
    assert camera_span == {"base": 12, "md": 12, "lg": 12}
    assert label == "Show controls"

    sidebar_style, camera_span, label = sidebar_layout_state(False)

    assert sidebar_style == {}
    assert camera_span == {"base": 12, "md": 8, "lg": 9}
    assert label == "Hide controls"
