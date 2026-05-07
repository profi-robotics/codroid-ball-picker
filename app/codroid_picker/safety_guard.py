"""Camera safety guard helpers.

Pure helper functions for evaluating whether person detections should
pause robot motion.
"""

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


def is_human_detection(detection: dict, labels: list[str]) -> bool:
    """Return True when detection label matches configured safety labels."""
    if not labels:
        return False
    label = str(detection.get("label", "")).strip().lower()
    return label in labels


def _workspace_contains(
    workspace_def: workspace.WorkspaceDefinition,
    workspace_xy: tuple[float, float],
) -> bool:
    """Check if workspace-space coordinate is within workspace extents."""
    return (
        0.0 <= workspace_xy[0] <= workspace_def.width_mm
        and 0.0 <= workspace_xy[1] <= workspace_def.height_mm
    )


def is_detection_in_zone(
    detection: dict,
    *,
    zone: str,
    workspace_def: Optional[workspace.WorkspaceDefinition],
) -> bool:
    """Check whether detection is in configured safety zone."""
    mode = str(zone or "workspace").strip().lower()
    if mode == "full_frame":
        return True
    if workspace_def is None:
        return False
    workspace_xy = detection.get("workspace_mm")
    if workspace_xy is None or len(workspace_xy) < 2:
        return False
    try:
        xy = (float(workspace_xy[0]), float(workspace_xy[1]))
    except (TypeError, ValueError):
        return False
    return _workspace_contains(workspace_def, xy)


def evaluate_safety_state(
    detections: list[dict],
    *,
    enabled: bool,
    labels: list[str],
    zone: str,
    workspace_def: Optional[workspace.WorkspaceDefinition],
) -> tuple[bool, list[dict]]:
    """Return (blocked, matched_detections)."""
    if not enabled:
        return False, []

    matched: list[dict] = []
    for detection in detections:
        if not is_human_detection(detection, labels):
            continue
        if not is_detection_in_zone(
            detection,
            zone=zone,
            workspace_def=workspace_def,
        ):
            continue
        matched.append(detection)

    return bool(matched), matched


def desired_operation_action(
    *,
    status: str,
    pause_reason: Optional[str],
    blocked: bool,
) -> str:
    """Resolve desired action for operation state transitions.

    Returns one of: "pause", "resume", "none".
    """
    if status == "running" and blocked:
        return "pause"
    if status == "paused" and pause_reason == "person_detected" and not blocked:
        return "resume"
    return "none"
