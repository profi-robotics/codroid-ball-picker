"""Camera safety guard helpers for the FastAPI runtime."""

from __future__ import annotations

from typing import Iterable, Optional

from vision import workspace


def normalize_safety_labels(labels: Optional[Iterable[str]]) -> list[str]:
    """Normalize user-provided labels into unique lower-case tokens."""
    if not labels:
        return []
    normalized: list[str] = []
    seen: set[str] = set()
    for raw in labels:
        token = str(raw or "").strip().lower()
        if not token or token in seen:
            continue
        seen.add(token)
        normalized.append(token)
    return normalized


def is_human_detection(detection: dict[str, object], labels: list[str]) -> bool:
    if not labels:
        return False
    label = str(detection.get("label", "")).strip().lower()
    return label in labels


def is_detection_in_zone(
    detection: dict[str, object],
    *,
    zone: str,
    workspace_def: Optional[workspace.WorkspaceDefinition],
) -> bool:
    mode = str(zone or "workspace").strip().lower()
    if mode == "full_frame":
        return True
    if workspace_def is None:
        return False
    workspace_xy = detection.get("workspace_mm") or detection.get("coord5_mm")
    if not isinstance(workspace_xy, (list, tuple)) or len(workspace_xy) < 2:
        return False
    try:
        x_val = float(workspace_xy[0])
        y_val = float(workspace_xy[1])
    except (TypeError, ValueError):
        return False
    return 0.0 <= x_val <= workspace_def.width_mm and 0.0 <= y_val <= workspace_def.height_mm


def evaluate_safety_state(
    detections: list[dict[str, object]],
    *,
    enabled: bool,
    labels: list[str],
    zone: str,
    workspace_def: Optional[workspace.WorkspaceDefinition],
) -> tuple[bool, list[dict[str, object]]]:
    if not enabled:
        return False, []
    matches = [
        detection
        for detection in detections
        if is_human_detection(detection, labels)
        and is_detection_in_zone(detection, zone=zone, workspace_def=workspace_def)
    ]
    return bool(matches), matches


def desired_operation_action(*, status: str, pause_reason: Optional[str], blocked: bool) -> str:
    if status == "running" and blocked:
        return "pause"
    if status == "paused" and pause_reason == "person_detected" and not blocked:
        return "resume"
    return "none"
