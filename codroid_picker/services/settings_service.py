"""Runtime settings service."""

from __future__ import annotations

import time
from typing import Any

from codroid_picker.config import AppSettings, SettingsRepository
from codroid_picker.state.store import StateStore


class SettingsService:
    def __init__(self, settings: AppSettings, repository: SettingsRepository, state: StateStore) -> None:
        self.current = settings
        self.repository = repository
        self.state = state
        self._publish()

    def as_dict(self) -> dict[str, Any]:
        return self.current.model_dump(mode="json")

    def update(self, updates: dict[str, Any]) -> AppSettings:
        previous = self.current
        payload = previous.model_dump()
        payload.update(updates)
        self.current = AppSettings(**payload)
        self.repository.save(self.current)
        reconnect_required = []
        if any(key in updates for key in ("robot_host", "robot_port", "user_port")):
            reconnect_required.append("robot")
        if any(key in updates for key in ("gripper_ip", "gripper_type")):
            reconnect_required.append("gripper")
        self.state.update_domain(
            "settings",
            version=self.state.domain("settings").get("version", 0) + 1,
            saved_at=time.time(),
            reconnect_required=reconnect_required,
        )
        return self.current

    def reset(self) -> AppSettings:
        self.current = AppSettings()
        self.repository.save(self.current)
        self._publish()
        return self.current

    def _publish(self) -> None:
        self.state.update_domain(
            "settings",
            version=1,
            saved_at=None,
            reconnect_required=[],
        )
        self.state.update_domain(
            "camera",
            source=self.current.camera_source,
            resolution=self.current.camera_resolution,
            depth_enabled=self.current.camera_enable_depth,
        )
        self.state.update_domain(
            "detection",
            frame_skip=self.current.detection_frame_skip,
        )
        self.state.update_domain(
            "robot",
            coordinate_id=self.current.robot_coordinate_id,
            uri=self.current.default_robot_uri,
        )
        self.state.update_domain(
            "safety",
            enabled=self.current.camera_safety_enabled,
            zone=self.current.camera_safety_zone,
            labels=list(self.current.camera_safety_labels),
        )
