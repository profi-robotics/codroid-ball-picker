"""Shared auto-planning snapshot helpers for monitoring and overlays."""

from __future__ import annotations

import time
from typing import Any, Iterable, Optional, Sequence

from app.codroid_picker.session import SESSION
from app.codroid_picker.target_planner import PickPlan, TargetCandidate

DEFAULT_PLAN_STALE_S = 2.5
TOP_RANKED_LIMIT = 3


def _coerce_float(value: Any) -> Optional[float]:
    """Best-effort float conversion."""
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _coerce_pair(value: Any) -> Optional[list[float]]:
    """Convert a pair-like value to [x, y]."""
    if value is None:
        return None
    if not isinstance(value, Sequence) or len(value) < 2:
        return None
    x = _coerce_float(value[0])
    y = _coerce_float(value[1])
    if x is None or y is None:
        return None
    return [x, y]


def _coerce_triplet(value: Any) -> Optional[list[float]]:
    """Convert a 3-item value to [x, y, z]."""
    if value is None:
        return None
    if not isinstance(value, Sequence) or len(value) < 3:
        return None
    x = _coerce_float(value[0])
    y = _coerce_float(value[1])
    z = _coerce_float(value[2])
    if x is None or y is None or z is None:
        return None
    return [x, y, z]


def _normalize_pick_plan(plan: Optional[PickPlan]) -> Optional[dict]:
    """Convert PickPlan into plain JSON-safe payload."""
    if plan is None:
        return None
    separation = None
    if plan.separation is not None:
        separation = {
            "blocker_xy": _coerce_pair(plan.separation.blocker_xy),
            "entry_xy": _coerce_pair(plan.separation.entry_xy),
            "push_end_xy": _coerce_pair(plan.separation.push_end_xy),
        }
    return {
        "mode": str(plan.mode),
        "final_c": float(plan.final_c),
        "separation": separation,
    }


def _normalize_candidate(candidate: Any, *, rank: Optional[int] = None) -> Optional[dict]:
    """Normalize TargetCandidate to payload used by UI/overlay layers."""
    if candidate is None:
        return None

    if isinstance(candidate, TargetCandidate):
        detection = candidate.detection or {}
        payload = {
            "coord5_xy": _coerce_pair(candidate.coord5_xy),
            "distance_to_robot_mm": float(candidate.distance_to_robot_mm),
            "nearest_neighbor_mm": float(candidate.nearest_neighbor_mm),
            "neighbor_count": int(candidate.neighbor_count),
        }
    elif isinstance(candidate, dict):
        detection = candidate.get("detection", candidate)
        payload = {
            "coord5_xy": _coerce_pair(candidate.get("coord5_xy")),
            "distance_to_robot_mm": _coerce_float(candidate.get("distance_to_robot_mm")),
            "nearest_neighbor_mm": _coerce_float(candidate.get("nearest_neighbor_mm")),
            "neighbor_count": int(candidate.get("neighbor_count", 0) or 0),
        }
    else:
        return None

    center = None
    bbox = None
    workspace_mm = None
    label = ""
    confidence = None
    if isinstance(detection, dict):
        center = _coerce_pair(detection.get("center"))
        bbox = detection.get("bbox")
        workspace_mm = _coerce_pair(detection.get("workspace_mm"))
        label = str(detection.get("label", "")).strip()
        confidence = _coerce_float(detection.get("confidence"))

    payload.update(
        {
            "rank": int(rank) if rank is not None else None,
            "label": label,
            "confidence": confidence,
            "center_px": center,
            "bbox": list(bbox) if isinstance(bbox, (tuple, list)) and len(bbox) >= 4 else None,
            "workspace_mm": workspace_mm,
        }
    )
    return payload


def publish_auto_plan_snapshot(
    *,
    stage: str,
    message: str = "",
    ranked_candidates: Optional[Iterable[Any]] = None,
    selected_candidate: Optional[Any] = None,
    pick_plan: Optional[PickPlan] = None,
    approach_mm: Optional[float] = None,
    pick_mm: Optional[float] = None,
    drop_mm: Optional[float] = None,
    orientation: Optional[Sequence[float]] = None,
    gripper_close_mm: Optional[float] = None,
    gripper_open_mm: Optional[float] = None,
    meta: Optional[dict] = None,
) -> dict:
    """Publish a fresh auto-planning snapshot."""
    ranked_payload: list[dict] = []
    if ranked_candidates is not None:
        for idx, candidate in enumerate(ranked_candidates, start=1):
            if idx > TOP_RANKED_LIMIT:
                break
            normalized = _normalize_candidate(candidate, rank=idx)
            if normalized is not None:
                ranked_payload.append(normalized)

    selected_payload = _normalize_candidate(selected_candidate, rank=1)
    now_ts = time.time()
    snapshot = {
        "stage": str(stage),
        "message": str(message),
        "updated_at": now_ts,
        "ranked_candidates": ranked_payload,
        "selected_candidate": selected_payload,
        "pick_plan": _normalize_pick_plan(pick_plan),
        "approach_mm": _coerce_float(approach_mm),
        "pick_mm": _coerce_float(pick_mm),
        "drop_mm": _coerce_float(drop_mm),
        "orientation": _coerce_triplet(orientation),
        "gripper_close_mm": _coerce_float(gripper_close_mm),
        "gripper_open_mm": _coerce_float(gripper_open_mm),
        "drop_target_mm": _coerce_pair(SESSION.drop_target_mm),
        "drop_target_robot": _coerce_triplet(SESSION.drop_target_robot),
        "meta": dict(meta or {}),
    }
    SESSION.auto_plan_snapshot = snapshot
    SESSION.auto_plan_stage = snapshot["stage"]
    SESSION.auto_plan_updated_at = now_ts
    return snapshot


def update_auto_plan_snapshot(**updates: Any) -> dict:
    """Partially update the existing auto-plan snapshot."""
    current = dict(SESSION.auto_plan_snapshot or {})

    if "ranked_candidates" in updates:
        ranked = updates.pop("ranked_candidates")
        ranked_payload: list[dict] = []
        if ranked is not None:
            for idx, candidate in enumerate(ranked, start=1):
                if idx > TOP_RANKED_LIMIT:
                    break
                normalized = _normalize_candidate(candidate, rank=idx)
                if normalized is not None:
                    ranked_payload.append(normalized)
        current["ranked_candidates"] = ranked_payload

    if "selected_candidate" in updates:
        current["selected_candidate"] = _normalize_candidate(
            updates.pop("selected_candidate"), rank=1
        )

    if "pick_plan" in updates:
        current["pick_plan"] = _normalize_pick_plan(updates.pop("pick_plan"))

    if "meta" in updates:
        existing_meta = dict(current.get("meta") or {})
        existing_meta.update(dict(updates.pop("meta") or {}))
        current["meta"] = existing_meta

    for key in (
        "stage",
        "message",
        "approach_mm",
        "pick_mm",
        "drop_mm",
        "gripper_close_mm",
        "gripper_open_mm",
    ):
        if key in updates:
            current[key] = updates.pop(key)

    if "orientation" in updates:
        current["orientation"] = _coerce_triplet(updates.pop("orientation"))

    if "drop_target_mm" in updates:
        current["drop_target_mm"] = _coerce_pair(updates.pop("drop_target_mm"))

    if "drop_target_robot" in updates:
        current["drop_target_robot"] = _coerce_triplet(updates.pop("drop_target_robot"))

    # Carry live drop target defaults when caller did not provide explicit overrides.
    current.setdefault("drop_target_mm", _coerce_pair(SESSION.drop_target_mm))
    current.setdefault("drop_target_robot", _coerce_triplet(SESSION.drop_target_robot))

    now_ts = time.time()
    current["updated_at"] = now_ts

    SESSION.auto_plan_snapshot = current
    SESSION.auto_plan_stage = str(current.get("stage", "idle"))
    SESSION.auto_plan_updated_at = now_ts
    return current


def clear_auto_plan_snapshot(message: str = "") -> None:
    """Clear the current auto-plan snapshot."""
    SESSION.auto_plan_snapshot = {}
    SESSION.auto_plan_stage = "idle"
    SESSION.auto_plan_updated_at = 0.0
    if message:
        SESSION.auto_message = message


def get_auto_plan_snapshot(*, stale_after_s: float = DEFAULT_PLAN_STALE_S) -> dict:
    """Return the latest plan snapshot enriched with availability/stale fields."""
    snapshot = dict(SESSION.auto_plan_snapshot or {})
    if not snapshot:
        return {
            "available": False,
            "stale": True,
            "age_s": None,
            "stage": "idle",
            "message": "",
        }

    updated_at = _coerce_float(snapshot.get("updated_at"))
    if updated_at is None:
        updated_at = 0.0
    age_s = max(0.0, time.time() - updated_at)
    snapshot["available"] = True
    snapshot["age_s"] = age_s
    snapshot["stale"] = age_s > float(stale_after_s)
    snapshot.setdefault("stage", "idle")
    snapshot.setdefault("message", "")
    return snapshot
