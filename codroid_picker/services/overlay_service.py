"""Overlay mode state facade."""

from __future__ import annotations

from codroid_picker.state.store import StateStore


class OverlayService:
    def __init__(self, state: StateStore) -> None:
        self.state = state
        self.mode = "auto"

    def set_mode(self, mode: str) -> str:
        normalized = mode.replace("-", "_")
        if normalized not in {"auto", "calibration", "workspace", "plate", "detection", "auto_plan", "none"}:
            normalized = "auto"
        self.mode = normalized
        self.state.update_domain("video", overlay_mode=normalized, last_error=None)
        return normalized
