from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def _disable_hardware_autoconnect_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CODROID_ROBOT_AUTOCONNECT", "false")
    monkeypatch.setenv("CODROID_GRIPPER_AUTOCONNECT", "false")


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Default unmarked tests to unit for consistent marker policy."""
    marker_names = {"unit", "integration", "hardware", "legacy"}
    legacy_files: dict[Path, bool] = {}
    for item in items:
        path = Path(str(item.fspath))
        if path not in legacy_files:
            text = path.read_text(encoding="utf-8", errors="ignore") if path.exists() else ""
            legacy_files[path] = "app.codroid_picker" in text or "aiortc" in text or "dash" in text.lower()
        if legacy_files[path]:
            item.add_marker(pytest.mark.legacy)
        assigned = {marker.name for marker in item.iter_markers()}
        if assigned.isdisjoint(marker_names):
            item.add_marker(pytest.mark.unit)
