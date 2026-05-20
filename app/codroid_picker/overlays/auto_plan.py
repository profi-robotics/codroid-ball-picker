"""Auto-planning overlay rendering for live camera frames."""

from __future__ import annotations

import logging
from math import cos, radians, sin
from typing import Optional, Sequence, Tuple

import cv2
import numpy as np

from app.codroid_picker.auto_plan_state import get_auto_plan_snapshot
from app.codroid_picker.procedures import _current_coordinate_id, _get_robot_session, _robot_to_workspace_mm
from app.codroid_picker.session import SESSION
from app.codroid_picker.transforms import _coord5_to_workspace_mm
from app.settings import SETTINGS
from .drawing import _draw_label

LOGGER = logging.getLogger(__name__)

_GREEN = (50, 220, 50)
_CYAN = (0, 190, 255)
_BLUE = (255, 180, 0)
_MAGENTA = (255, 0, 255)
_WHITE = (255, 255, 255)
_LIGHT_GRAY = (230, 230, 230)
_APPROACH = (255, 160, 80)
_DROP_ROUTE = (180, 180, 255)
_SEPARATION = (0, 140, 255)
_GRIPPER = (255, 200, 0)


def _css_color(bgr: tuple[int, int, int]) -> str:
    """Return a CSS color from an OpenCV BGR tuple."""
    b, g, r = bgr
    return f"rgb({int(r)}, {int(g)}, {int(b)})"


def _workspace_mm_to_pixel(workspace_xy: Sequence[float]) -> Optional[tuple[int, int]]:
    """Convert workspace mm coordinates into pixel coordinates."""
    if SESSION.workspace_definition is None or SESSION.calibration_bundle is None:
        return None
    scale = float(SESSION.calibration_bundle.pixel_to_mm)
    if scale <= 0:
        return None

    workspace_def = SESSION.workspace_definition
    try:
        camera_xyz = workspace_def.workspace_to_camera(
            float(workspace_xy[0]), float(workspace_xy[1]), 0.0
        )
    except Exception:  # noqa: BLE001
        return None

    px = int(round(float(camera_xyz[0]) / scale))
    py = int(round(float(camera_xyz[1]) / scale))
    return (px, py)


def _coord5_to_pixel(coord5_xy: Optional[Sequence[float]]) -> Optional[tuple[int, int]]:
    """Convert coord5 XY to pixel coordinates when transforms are available."""
    if coord5_xy is None or SESSION.workspace_definition is None:
        return None
    try:
        workspace_xy = _coord5_to_workspace_mm(
            SESSION.workspace_definition,
            (float(coord5_xy[0]), float(coord5_xy[1])),
        )
    except Exception:  # noqa: BLE001
        workspace_xy = None
    if workspace_xy is None:
        return None
    return _workspace_mm_to_pixel(workspace_xy)


def _robot_posture_pixel() -> Optional[tuple[int, int]]:
    """Project live robot posture into pixel coordinates."""
    if SESSION.workspace_definition is None:
        return None
    try:
        robot_session = _get_robot_session()
        if not robot_session.posture_seen():
            return None
        pos = robot_session.position_snapshot()
        coord_id = _current_coordinate_id(robot_session)
        if coord_id is None:
            return None
        if coord_id == SETTINGS.robot_coordinate_id:
            workspace_xy = _coord5_to_workspace_mm(
                SESSION.workspace_definition,
                (float(pos.x), float(pos.y)),
            )
        else:
            robot_xyz = np.array([pos.x, pos.y, 0.0], dtype=np.float64)
            workspace_xy = _robot_to_workspace_mm(SESSION.workspace_definition, robot_xyz)
        if workspace_xy is None:
            return None
        return _workspace_mm_to_pixel(workspace_xy)
    except Exception:  # noqa: BLE001
        return None


def _candidate_pixel(candidate: dict) -> Optional[tuple[int, int]]:
    """Resolve a candidate payload into image pixel coordinates."""
    center = candidate.get("center_px")
    if isinstance(center, Sequence) and len(center) >= 2:
        try:
            return (int(round(float(center[0]))), int(round(float(center[1]))))
        except (TypeError, ValueError):
            pass
    coord5_xy = candidate.get("coord5_xy")
    if isinstance(coord5_xy, Sequence) and len(coord5_xy) >= 2:
        return _coord5_to_pixel(coord5_xy)
    return None


def _draw_candidate(vis: np.ndarray, candidate: dict, color: tuple[int, int, int]) -> Optional[tuple[int, int]]:
    """Draw a ranked candidate marker and label."""
    px = _candidate_pixel(candidate)
    if px is None:
        return None

    rank = int(candidate.get("rank") or 0)
    label = str(candidate.get("label") or "target")
    distance_mm = candidate.get("distance_to_robot_mm")
    confidence = candidate.get("confidence")

    cv2.circle(vis, px, 12, color, 2, lineType=cv2.LINE_AA)
    cv2.circle(vis, px, 3, color, -1, lineType=cv2.LINE_AA)

    parts = [f"#{rank}", label]
    if isinstance(distance_mm, (int, float)):
        parts.append(f"d={distance_mm:.0f}mm")
    if isinstance(confidence, (int, float)):
        parts.append(f"{confidence:.2f}")
    _draw_label(vis, " ".join(parts), (px[0] + 10, px[1] - 10), color)
    return px


def _candidate_label(candidate: dict) -> str:
    """Build a compact candidate label for overlays."""
    rank = int(candidate.get("rank") or 0)
    label = str(candidate.get("label") or "target")
    distance_mm = candidate.get("distance_to_robot_mm")
    confidence = candidate.get("confidence")
    parts = [f"#{rank}", label]
    if isinstance(distance_mm, (int, float)):
        parts.append(f"d={distance_mm:.0f}mm")
    if isinstance(confidence, (int, float)):
        parts.append(f"{confidence:.2f}")
    return " ".join(parts)


def _gripper_footprint_lines(
    center_px: tuple[int, int],
    *,
    close_width_mm: float,
    angle_deg: float,
) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    """Return approximate two-finger footprint line segments."""
    scale_mm_per_px = None
    if SESSION.calibration_bundle is not None:
        try:
            scale_mm_per_px = float(SESSION.calibration_bundle.pixel_to_mm)
        except (TypeError, ValueError):
            scale_mm_per_px = None

    if scale_mm_per_px is None or scale_mm_per_px <= 0:
        scale_mm_per_px = 0.5

    gap_px = max(10.0, float(close_width_mm) / scale_mm_per_px)
    finger_len_px = max(16.0, 20.0 / scale_mm_per_px)

    theta = radians(float(angle_deg))
    ux, uy = cos(theta), sin(theta)
    vx, vy = -uy, ux

    half_gap = gap_px * 0.5
    half_len = finger_len_px * 0.5

    lines: list[tuple[tuple[int, int], tuple[int, int]]] = []
    for sign in (-1.0, 1.0):
        cx = center_px[0] + sign * vx * half_gap
        cy = center_px[1] + sign * vy * half_gap
        start = (int(round(cx - ux * half_len)), int(round(cy - uy * half_len)))
        end = (int(round(cx + ux * half_len)), int(round(cy + uy * half_len)))
        lines.append((start, end))
    return lines


def _draw_gripper_footprint(
    vis: np.ndarray,
    center_px: tuple[int, int],
    *,
    close_width_mm: float,
    angle_deg: float,
) -> None:
    """Draw an approximate two-finger footprint around target center."""
    for start, end in _gripper_footprint_lines(
        center_px,
        close_width_mm=close_width_mm,
        angle_deg=angle_deg,
    ):
        cv2.line(vis, start, end, _GRIPPER, 2, lineType=cv2.LINE_AA)

    cv2.circle(vis, center_px, 4, _GRIPPER, -1, lineType=cv2.LINE_AA)


def _point_payload(px: tuple[int, int]) -> dict:
    """Return a JSON-safe point object."""
    return {"x": int(px[0]), "y": int(px[1])}


def _circle_primitive(
    px: tuple[int, int],
    *,
    radius: int,
    color: tuple[int, int, int],
    label: str = "",
    fill: bool = False,
) -> dict:
    """Build a browser circle primitive."""
    primitive = {
        "type": "circle",
        "x": int(px[0]),
        "y": int(px[1]),
        "radius": int(radius),
        "color": _css_color(color),
        "lineWidth": 2,
        "fill": bool(fill),
    }
    if label:
        primitive["label"] = label
    return primitive


def _line_primitive(
    start: tuple[int, int],
    end: tuple[int, int],
    *,
    color: tuple[int, int, int],
    label: str = "",
    arrow: bool = False,
    label_at: str = "end",
) -> dict:
    """Build a browser line or arrow primitive."""
    primitive = {
        "type": "arrow" if arrow else "line",
        "from": _point_payload(start),
        "to": _point_payload(end),
        "color": _css_color(color),
        "lineWidth": 2,
    }
    if label:
        primitive["label"] = label
        primitive["label_at"] = label_at
    return primitive


def _point_in_frame(px: Optional[tuple[int, int]], frame_shape: Optional[Sequence[int]]) -> bool:
    """Return whether a point is inside the rendered frame."""
    if px is None:
        return False
    if frame_shape is None or len(frame_shape) < 2:
        return True
    height = int(frame_shape[0])
    width = int(frame_shape[1])
    return 0 <= int(px[0]) < width and 0 <= int(px[1]) < height


def _status_text(snapshot: dict) -> tuple[str, str, str, str]:
    """Return stage/message/status/metric strings for an auto-plan snapshot."""
    stage = str(snapshot.get("stage") or "idle")
    stale = bool(snapshot.get("stale"))
    message = str(snapshot.get("message") or "")
    stage_text = f"Auto plan: {stage}"
    if stale:
        stage_text += " (stale)"
    metric = f"Auto plan: {stage}"
    status = stage_text if not message else f"{stage_text} · {message}"
    return stage, message, status, metric


def build_auto_plan_primitives(
    frame_shape: Optional[Sequence[int]] = None,
) -> list[dict]:
    """Build semantic browser primitives for the auto-plan overlay."""
    snapshot = get_auto_plan_snapshot()
    if not snapshot.get("available"):
        return [
            {
                "type": "text",
                "x": 16,
                "y": 28,
                "text": "Auto plan unavailable.",
                "color": _css_color(_WHITE),
            }
        ]

    _stage, message, _status, _metric = _status_text(snapshot)
    stale = bool(snapshot.get("stale"))
    ranked = snapshot.get("ranked_candidates") or []
    selected = snapshot.get("selected_candidate") or {}
    pick_plan = snapshot.get("pick_plan") or {}
    primitives: list[dict] = []

    palette = [_GREEN, _CYAN, _BLUE]
    for idx, candidate in enumerate(ranked[:3]):
        px = _candidate_pixel(candidate)
        if px is None:
            continue
        color = palette[min(idx, len(palette) - 1)]
        primitives.append(
            _circle_primitive(px, radius=12, color=color, label=_candidate_label(candidate))
        )
        primitives.append(_circle_primitive(px, radius=3, color=color, fill=True))

    selected_px = _candidate_pixel(selected)
    if selected_px is not None:
        primitives.append(_circle_primitive(selected_px, radius=18, color=_MAGENTA))
        primitives.append(
            {
                "type": "cross",
                "x": int(selected_px[0]),
                "y": int(selected_px[1]),
                "size": 18,
                "color": _css_color(_MAGENTA),
                "lineWidth": 2,
                "label": "Selected",
                "label_dx": 8,
                "label_dy": 16,
            }
        )

    drop_target_mm = snapshot.get("drop_target_mm")
    drop_px = None
    if isinstance(drop_target_mm, Sequence) and len(drop_target_mm) >= 2:
        drop_px = _workspace_mm_to_pixel(drop_target_mm)
    if _point_in_frame(drop_px, frame_shape):
        primitives.append(_circle_primitive(drop_px, radius=10, color=_WHITE, label="Drop"))

    robot_px = _robot_posture_pixel()
    route_points_visible = (
        not stale
        and _point_in_frame(selected_px, frame_shape)
    )
    if route_points_visible and _point_in_frame(robot_px, frame_shape):
        primitives.append(
            _line_primitive(
                robot_px,
                selected_px,
                color=_APPROACH,
                label="Approach",
                arrow=True,
                label_at="start",
            )
        )
    if route_points_visible and _point_in_frame(drop_px, frame_shape):
        primitives.append(_line_primitive(selected_px, drop_px, color=_DROP_ROUTE, arrow=True))

    separation = pick_plan.get("separation") if isinstance(pick_plan, dict) else None
    if isinstance(separation, dict):
        entry_px = _coord5_to_pixel(separation.get("entry_xy"))
        push_px = _coord5_to_pixel(separation.get("push_end_xy"))
        blocker_px = _coord5_to_pixel(separation.get("blocker_xy"))
        if entry_px is not None and push_px is not None:
            primitives.append(
                _line_primitive(
                    entry_px,
                    push_px,
                    color=_SEPARATION,
                    label="Separation",
                    arrow=True,
                    label_at="start",
                )
            )
        if blocker_px is not None:
            primitives.append(_circle_primitive(blocker_px, radius=10, color=_SEPARATION))

    final_c = 0.0
    if isinstance(pick_plan, dict) and isinstance(pick_plan.get("final_c"), (int, float)):
        final_c = float(pick_plan["final_c"])
    close_width = snapshot.get("gripper_close_mm")
    if selected_px is not None and isinstance(close_width, (int, float)):
        for start, end in _gripper_footprint_lines(
            selected_px,
            close_width_mm=float(close_width),
            angle_deg=final_c,
        ):
            primitives.append(_line_primitive(start, end, color=_GRIPPER))
        primitives.append(_circle_primitive(selected_px, radius=4, color=_GRIPPER, fill=True))

    stage_text = f"Auto plan: {snapshot.get('stage') or 'idle'}"
    if snapshot.get("stale"):
        stage_text += " (stale)"
    primitives.append(
        {
            "type": "text",
            "x": 16,
            "y": 28,
            "text": stage_text,
            "color": _css_color(_WHITE),
        }
    )
    if message:
        primitives.append(
            {
                "type": "text",
                "x": 16,
                "y": 52,
                "text": message,
                "color": _css_color(_LIGHT_GRAY),
            }
        )
    return primitives


def draw_auto_plan_overlay(frame: np.ndarray) -> Tuple[np.ndarray, str, str]:
    """Render auto-planning intent overlay."""
    vis = frame.copy()
    snapshot = get_auto_plan_snapshot()

    if not snapshot.get("available"):
        return vis, "Auto plan unavailable.", "Auto plan: --"

    stage, message, status, metric = _status_text(snapshot)
    stale = bool(snapshot.get("stale"))
    ranked = snapshot.get("ranked_candidates") or []
    selected = snapshot.get("selected_candidate") or {}
    pick_plan = snapshot.get("pick_plan") or {}

    # Draw top ranked candidates.
    palette = [_GREEN, _CYAN, _BLUE]
    for idx, candidate in enumerate(ranked[:3]):
        _draw_candidate(vis, candidate, palette[min(idx, len(palette) - 1)])

    selected_px = _candidate_pixel(selected)
    if selected_px is not None:
        cv2.circle(vis, selected_px, 18, _MAGENTA, 2, lineType=cv2.LINE_AA)
        cv2.drawMarker(
            vis,
            selected_px,
            _MAGENTA,
            markerType=cv2.MARKER_CROSS,
            markerSize=18,
            thickness=2,
            line_type=cv2.LINE_AA,
        )
        _draw_label(vis, "Selected", (selected_px[0] + 8, selected_px[1] + 16), _MAGENTA)

    # Draw drop point and route.
    drop_target_mm = snapshot.get("drop_target_mm")
    drop_px = None
    if isinstance(drop_target_mm, Sequence) and len(drop_target_mm) >= 2:
        drop_px = _workspace_mm_to_pixel(drop_target_mm)
    if drop_px is not None:
        cv2.circle(vis, drop_px, 10, _WHITE, 2, lineType=cv2.LINE_AA)
        _draw_label(vis, "Drop", (drop_px[0] + 8, drop_px[1] - 8), _WHITE)
    robot_px = _robot_posture_pixel()
    if selected_px is not None and robot_px is not None:
        cv2.arrowedLine(vis, robot_px, selected_px, _APPROACH, 2, cv2.LINE_AA, 0, 0.04)
        _draw_label(vis, "Approach", (robot_px[0] + 8, robot_px[1] - 8), _APPROACH)
    if selected_px is not None and drop_px is not None:
        cv2.arrowedLine(vis, selected_px, drop_px, _DROP_ROUTE, 2, cv2.LINE_AA, 0, 0.03)

    # Draw separation vectors for crowded plans.
    separation = pick_plan.get("separation") if isinstance(pick_plan, dict) else None
    if isinstance(separation, dict):
        entry_px = _coord5_to_pixel(separation.get("entry_xy"))
        push_px = _coord5_to_pixel(separation.get("push_end_xy"))
        blocker_px = _coord5_to_pixel(separation.get("blocker_xy"))
        if entry_px is not None and push_px is not None:
            cv2.arrowedLine(vis, entry_px, push_px, _SEPARATION, 2, cv2.LINE_AA, 0, 0.08)
            _draw_label(vis, "Separation", (entry_px[0] + 6, entry_px[1] - 10), _SEPARATION)
        if blocker_px is not None:
            cv2.circle(vis, blocker_px, 10, _SEPARATION, 2, lineType=cv2.LINE_AA)

    # Draw approximate gripper finger placement.
    final_c = 0.0
    if isinstance(pick_plan, dict) and isinstance(pick_plan.get("final_c"), (int, float)):
        final_c = float(pick_plan["final_c"])
    close_width = snapshot.get("gripper_close_mm")
    if selected_px is not None and isinstance(close_width, (int, float)):
        _draw_gripper_footprint(
            vis,
            selected_px,
            close_width_mm=float(close_width),
            angle_deg=final_c,
        )

    stage_text = f"Auto plan: {stage}"
    if stale:
        stage_text += " (stale)"
    _draw_label(vis, stage_text, (16, 28), (255, 255, 255))
    if message:
        _draw_label(vis, message, (16, 52), (230, 230, 230))

    return vis, status, metric
