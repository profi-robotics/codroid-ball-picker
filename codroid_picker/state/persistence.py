"""State restoration helpers for file-backed runtime metadata."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from codroid_picker.core.persistence import read_json
from codroid_picker.state.store import StateStore


def restore_drop_target(state: StateStore, path: Path) -> None:
    payload: Any = read_json(path, default=None)
    if isinstance(payload, dict):
        state.update_domain("auto_pick", plan_snapshot={"drop_target": payload})

