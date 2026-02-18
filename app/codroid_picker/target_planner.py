"""Collision-aware pick target ranking and planning helpers."""

from __future__ import annotations

from dataclasses import dataclass
from math import atan2, cos, hypot, radians
from typing import Callable, Optional, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class TargetCandidate:
    """Prepared target candidate with scoring metrics."""

    detection: dict
    coord5_xy: Tuple[float, float]
    distance_to_robot_mm: float
    nearest_neighbor_mm: float
    neighbor_count: int


@dataclass(frozen=True)
class SeparationPlan:
    """Side push trajectory to separate clustered objects."""

    blocker_xy: Tuple[float, float]
    entry_xy: Tuple[float, float]
    push_end_xy: Tuple[float, float]


@dataclass(frozen=True)
class PickPlan:
    """Pick strategy for a selected candidate."""

    mode: str  # "rotate_pick" | "separate_then_rescan"
    final_c: float
    separation: Optional[SeparationPlan] = None


def _wrap_degrees(angle: float) -> float:
    """Normalize angle to [-180, 180)."""
    wrapped = (float(angle) + 180.0) % 360.0 - 180.0
    if wrapped == -180.0:
        return 180.0
    return wrapped


def _candidate_angles(current_c: float, max_delta_deg: float, step_deg: float) -> list[float]:
    """Generate C-axis candidates with small deltas first."""
    max_delta = max(0.0, float(max_delta_deg))
    step = max(1.0, float(step_deg))
    angles = [_wrap_degrees(current_c)]
    delta = step
    while delta <= max_delta + 1e-9:
        angles.append(_wrap_degrees(current_c + delta))
        angles.append(_wrap_degrees(current_c - delta))
        delta += step
    return angles


def rank_targets(
    detections: Sequence[dict],
    *,
    robot_xy: Tuple[float, float],
    coord5_fn: Callable[[dict], Optional[Tuple[float, float]]],
    neighbor_radius_mm: float,
) -> list[TargetCandidate]:
    """Rank detections by local clearance first, then distance."""
    prepared: list[tuple[dict, Tuple[float, float]]] = []
    for det in detections:
        coord5_xy = coord5_fn(det)
        if coord5_xy is None:
            continue
        prepared.append((det, (float(coord5_xy[0]), float(coord5_xy[1]))))

    candidates: list[TargetCandidate] = []
    for idx, (det, xy) in enumerate(prepared):
        nearest = float("inf")
        count = 0
        for jdx, (_other_det, other_xy) in enumerate(prepared):
            if idx == jdx:
                continue
            dist = hypot(xy[0] - other_xy[0], xy[1] - other_xy[1])
            nearest = min(nearest, dist)
            if dist <= neighbor_radius_mm:
                count += 1
        if nearest == float("inf"):
            nearest = 1e6
        candidates.append(
            TargetCandidate(
                detection=det,
                coord5_xy=xy,
                distance_to_robot_mm=hypot(xy[0] - robot_xy[0], xy[1] - robot_xy[1]),
                nearest_neighbor_mm=nearest,
                neighbor_count=count,
            )
        )

    return sorted(
        candidates,
        key=lambda item: (
            item.neighbor_count,
            -item.nearest_neighbor_mm,
            item.distance_to_robot_mm,
        ),
    )


def _is_orientation_safe(
    target_xy: Tuple[float, float],
    neighbors: Sequence[Tuple[float, float]],
    *,
    tool_c_deg: float,
    body_radius_mm: float,
    finger_reach_mm: float,
    clearance_mm: float,
) -> bool:
    """Check whether neighbors are outside a conservative finger/body footprint."""
    tool_angle = radians(_wrap_degrees(tool_c_deg))
    for neighbor_xy in neighbors:
        vec = np.array([neighbor_xy[0] - target_xy[0], neighbor_xy[1] - target_xy[1]], dtype=float)
        dist = float(np.linalg.norm(vec))
        if dist <= 0.0:
            return False
        ray_angle = atan2(vec[1], vec[0])
        alignment = abs(cos(ray_angle - tool_angle))
        reach = body_radius_mm + alignment * finger_reach_mm + clearance_mm
        if dist < reach:
            return False
    return True


def plan_pick_strategy(
    candidate: TargetCandidate,
    all_candidates: Sequence[TargetCandidate],
    *,
    current_c_deg: float,
    rotation_step_deg: float,
    rotation_max_delta_deg: float,
    body_radius_mm: float,
    finger_reach_mm: float,
    clearance_mm: float,
    separation_entry_mm: float,
    separation_push_mm: float,
) -> PickPlan:
    """Plan rotation-first pick strategy, else side-separation fallback."""
    target_xy = candidate.coord5_xy
    neighbors = [
        other.coord5_xy
        for other in all_candidates
        if other is not candidate
    ]

    for c_angle in _candidate_angles(current_c_deg, rotation_max_delta_deg, rotation_step_deg):
        if _is_orientation_safe(
            target_xy,
            neighbors,
            tool_c_deg=c_angle,
            body_radius_mm=body_radius_mm,
            finger_reach_mm=finger_reach_mm,
            clearance_mm=clearance_mm,
        ):
            return PickPlan(mode="rotate_pick", final_c=c_angle)

    if not neighbors:
        return PickPlan(mode="rotate_pick", final_c=_wrap_degrees(current_c_deg))

    blocker = min(neighbors, key=lambda xy: hypot(target_xy[0] - xy[0], target_xy[1] - xy[1]))
    push_dir = np.array(
        [target_xy[0] - blocker[0], target_xy[1] - blocker[1]],
        dtype=float,
    )
    norm = float(np.linalg.norm(push_dir))
    if norm <= 1e-6:
        push_dir = np.array([1.0, 0.0], dtype=float)
    else:
        push_dir /= norm

    entry_xy = (
        float(target_xy[0] - push_dir[0] * separation_entry_mm),
        float(target_xy[1] - push_dir[1] * separation_entry_mm),
    )
    push_end_xy = (
        float(target_xy[0] + push_dir[0] * separation_push_mm),
        float(target_xy[1] + push_dir[1] * separation_push_mm),
    )
    return PickPlan(
        mode="separate_then_rescan",
        final_c=_wrap_degrees(current_c_deg),
        separation=SeparationPlan(
            blocker_xy=(float(blocker[0]), float(blocker[1])),
            entry_xy=entry_xy,
            push_end_xy=push_end_xy,
        ),
    )
