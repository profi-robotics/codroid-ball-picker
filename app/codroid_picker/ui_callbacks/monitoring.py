"""Monitoring and operation state callbacks."""

from __future__ import annotations

import time
from typing import Optional, Sequence, Tuple

import dash_mantine_components as dmc
import plotly.graph_objects as go
from dash import ALL, Dash, Input, Output, State, no_update

from app.codroid_picker.auto_plan_state import DEFAULT_PLAN_STALE_S, get_auto_plan_snapshot
from app.codroid_picker.i18n import t

DETECTION_HISTORY_WINDOW_S = 180.0


def _status_color(status: str) -> str:
    """Map status text to badge color."""
    normalized = (status or "").strip().lower()
    if normalized in {"live", "online", "on", "ready", "loaded"}:
        return "teal"
    if normalized in {"error", "offline", "missing"}:
        return "red"
    if normalized in {"pending", "probing"}:
        return "yellow"
    return "gray"


def _append_detection_history(
    session: object,
    detection_count: int,
    *,
    now_ts: float,
    window_s: float = DETECTION_HISTORY_WINDOW_S,
) -> list[dict]:
    """Append detection count and trim history window."""
    history = getattr(session, "monitor_detection_history", None)
    if history is None:
        return []

    history.append({"ts": float(now_ts), "count": int(detection_count)})
    cutoff = float(now_ts) - float(window_s)
    while history and float(history[0].get("ts", 0.0)) < cutoff:
        history.popleft()
    return list(history)


def _empty_trend_figure(title: str, yaxis_title: str) -> go.Figure:
    """Build an empty trend chart with consistent styling."""
    figure = go.Figure()
    figure.update_layout(
        template="plotly_white",
        title={"text": title, "x": 0.02, "xanchor": "left", "font": {"size": 12}},
        margin={"l": 36, "r": 20, "t": 38, "b": 32},
        xaxis={"title": "Samples", "showgrid": False},
        yaxis={"title": yaxis_title, "rangemode": "tozero", "showgrid": True},
        showlegend=False,
    )
    return figure


def _build_detection_trend_figure(history: Sequence[dict]) -> go.Figure:
    """Build detection trend line chart."""
    figure = _empty_trend_figure(t("Detections (Last ~3m)"), t("Detections"))
    if not history:
        return figure

    y_vals = [int(item.get("count", 0)) for item in history]
    x_vals = list(range(1, len(y_vals) + 1))
    figure.add_trace(
        go.Scatter(
            x=x_vals,
            y=y_vals,
            mode="lines+markers",
            line={"color": "#f15b5f", "width": 2},
            marker={"size": 5},
            fill="tozeroy",
            fillcolor="rgba(241, 91, 95, 0.12)",
        )
    )
    return figure


def _build_cycle_trend_figure(cycle_times: Sequence[float]) -> go.Figure:
    """Build cycle duration trend line chart."""
    figure = _empty_trend_figure(t("Cycle Time (Last 50)"), t("Seconds"))
    if not cycle_times:
        return figure

    y_vals = [float(value) for value in cycle_times]
    x_vals = list(range(1, len(y_vals) + 1))
    figure.add_trace(
        go.Scatter(
            x=x_vals,
            y=y_vals,
            mode="lines+markers",
            line={"color": "#1f8f4f", "width": 2},
            marker={"size": 5},
            fill="tozeroy",
            fillcolor="rgba(31, 143, 79, 0.10)",
        )
    )
    return figure


def _fmt_pair(value: Optional[Sequence[float]], *, precision: int = 1, suffix: str = "") -> str:
    """Format pair values for compact UI rendering."""
    if not value or len(value) < 2:
        return "--"
    return f"{value[0]:.{precision}f}, {value[1]:.{precision}f}{suffix}"


def _build_auto_plan_items(plan_snapshot: dict) -> list:
    """Build auto-plan summary UI elements."""
    if not plan_snapshot.get("available"):
        return [dmc.Text(t("No auto plan available."), c="dimmed", size="sm")]

    stage = str(plan_snapshot.get("stage") or "idle")
    stale = bool(plan_snapshot.get("stale"))
    age_s = plan_snapshot.get("age_s")
    message = str(plan_snapshot.get("message") or "")
    pick_plan = plan_snapshot.get("pick_plan") or {}
    selected = plan_snapshot.get("selected_candidate") or {}
    ranked = plan_snapshot.get("ranked_candidates") or []

    items = [
        dmc.Group(
            [
                dmc.Badge(t("Stage: {stage}", stage=stage.replace("_", " ")), color="blue", variant="light"),
                dmc.Badge(
                    t("Stale") if stale else t("Fresh"),
                    color="red" if stale else "teal",
                    variant="light",
                ),
                dmc.Text(
                    t("Age: {age:.1f}s", age=float(age_s)) if isinstance(age_s, (int, float)) else t("Age: --"),
                    size="xs",
                    c="dimmed",
                ),
            ],
            gap="xs",
            align="center",
        )
    ]

    if message:
        items.append(dmc.Text(message, size="sm"))

    selected_label = str(selected.get("label") or t("none"))
    selected_coord = selected.get("coord5_xy")
    items.append(
        dmc.Text(
            t("Selected target: {label} @ {coord}", label=selected_label, coord=_fmt_pair(selected_coord, suffix=" mm")),
            size="sm",
        )
    )

    mode_text = str(pick_plan.get("mode") or "--")
    final_c = pick_plan.get("final_c")
    c_text = f"{float(final_c):.1f}" if isinstance(final_c, (int, float)) else "--"
    items.append(dmc.Text(t("Strategy: {mode} · C={c}", mode=mode_text, c=c_text), size="sm"))

    gripper_close = plan_snapshot.get("gripper_close_mm")
    if isinstance(gripper_close, (int, float)):
        items.append(dmc.Text(t("Planned gripper close: {width:.1f} mm", width=float(gripper_close)), size="sm"))

    drop_target_mm = plan_snapshot.get("drop_target_mm")
    items.append(
        dmc.Text(
            t("Drop target: {coord}", coord=_fmt_pair(drop_target_mm, suffix=" mm")),
            size="sm",
        )
    )

    if ranked:
        ranked_lines = []
        for candidate in ranked:
            rank = int(candidate.get("rank") or 0)
            label = str(candidate.get("label") or t("target"))
            neighbor_count = int(candidate.get("neighbor_count") or 0)
            nearest = candidate.get("nearest_neighbor_mm")
            nearest_text = f"{float(nearest):.1f}" if isinstance(nearest, (int, float)) else "--"
            ranked_lines.append(
                dmc.Text(
                    t(
                        "#{rank} {label} · neighbors={neighbors} · nearest={nearest}mm",
                        rank=rank,
                        label=label,
                        neighbors=neighbor_count,
                        nearest=nearest_text,
                    ),
                    size="xs",
                    c="dimmed",
                )
            )
        items.append(dmc.Stack(ranked_lines, gap=2))

    return items


def register_monitoring_callbacks(app: Dash) -> None:
    """Register monitoring-related callbacks with the Dash app."""

    @app.callback(
        Output("auto-label-profile-editor", "children"),
        Input("auto-target-labels", "value"),
    )
    def _render_label_profiles(labels: Optional[list[str]]) -> list:
        """Render per-label pick/grip profile controls."""
        from app.codroid_picker.session import (
            SESSION,
            normalize_auto_label_profiles,
        )

        selected = [str(label).strip() for label in labels or [] if str(label).strip()]
        if not selected:
            return [
                dmc.Text(
                    t("Load a model and choose at least one pick label."),
                    size="sm",
                    c="dimmed",
                )
            ]

        profiles = normalize_auto_label_profiles(
            selected, SESSION.auto_config.get("label_profiles")
        )
        cards = []
        for label in selected:
            profile = profiles[label]
            cards.append(
                dmc.Card(
                    [
                        dmc.Text(label, fw=600, size="sm"),
                        dmc.Grid(
                            [
                                dmc.GridCol(
                                    dmc.NumberInput(
                                        id={
                                            "type": "auto-label-profile-field",
                                            "label": label,
                                            "field": "approach_mm",
                                        },
                                        label=t("Approach height (mm)"),
                                        value=profile["approach_mm"],
                                        min=-200,
                                        max=0,
                                        step=1,
                                        size="sm",
                                    ),
                                    span=4,
                                ),
                                dmc.GridCol(
                                    dmc.NumberInput(
                                        id={
                                            "type": "auto-label-profile-field",
                                            "label": label,
                                            "field": "pick_mm",
                                        },
                                        label=t("Pick height (mm)"),
                                        value=profile["pick_mm"],
                                        min=-50,
                                        max=200,
                                        step=1,
                                        size="sm",
                                    ),
                                    span=4,
                                ),
                                dmc.GridCol(
                                    dmc.NumberInput(
                                        id={
                                            "type": "auto-label-profile-field",
                                            "label": label,
                                            "field": "gripper_close_mm",
                                        },
                                        label=t("Close width (mm)"),
                                        value=profile["gripper_close_mm"],
                                        min=0,
                                        max=200,
                                        step=1,
                                        size="sm",
                                    ),
                                    span=4,
                                ),
                                dmc.GridCol(
                                    dmc.NumberInput(
                                        id={
                                            "type": "auto-label-profile-field",
                                            "label": label,
                                            "field": "gripper_force",
                                        },
                                        label=t("Grip force (N)"),
                                        value=profile["gripper_force"],
                                        min=0,
                                        max=200,
                                        step=1,
                                        size="sm",
                                    ),
                                    span=4,
                                ),
                                dmc.GridCol(
                                    dmc.NumberInput(
                                        id={
                                            "type": "auto-label-profile-field",
                                            "label": label,
                                            "field": "gripper_speed",
                                        },
                                        label=t("Grip speed (%)"),
                                        value=profile["gripper_speed"],
                                        min=1,
                                        max=100,
                                        step=1,
                                        size="sm",
                                    ),
                                    span=4,
                                ),
                                dmc.GridCol(
                                    dmc.NumberInput(
                                        id={
                                            "type": "auto-label-profile-field",
                                            "label": label,
                                            "field": "grip_timeout_s",
                                        },
                                        label=t("Grip timeout (s)"),
                                        value=profile["grip_timeout_s"],
                                        min=0.5,
                                        max=10,
                                        step=0.5,
                                        size="sm",
                                    ),
                                    span=4,
                                ),
                                dmc.GridCol(
                                    dmc.Switch(
                                        id={
                                            "type": "auto-label-profile-toggle",
                                            "label": label,
                                            "field": "soft_gentle",
                                        },
                                        label=t("Soft gentle grip"),
                                        checked=bool(profile["soft_gentle"]),
                                        size="sm",
                                    ),
                                    span=4,
                                ),
                                dmc.GridCol(
                                    dmc.NumberInput(
                                        id={
                                            "type": "auto-label-profile-field",
                                            "label": label,
                                            "field": "vacuum_setpoint",
                                        },
                                        label=t("Vacuum setpoint (kPa)"),
                                        value=profile["vacuum_setpoint"],
                                        min=1,
                                        max=80,
                                        step=1,
                                        size="sm",
                                    ),
                                    span=4,
                                ),
                                dmc.GridCol(
                                    dmc.NumberInput(
                                        id={
                                            "type": "auto-label-profile-field",
                                            "label": label,
                                            "field": "vacuum_threshold",
                                        },
                                        label=t("Vacuum threshold (kPa)"),
                                        value=profile["vacuum_threshold"],
                                        min=1,
                                        max=80,
                                        step=1,
                                        size="sm",
                                    ),
                                    span=4,
                                ),
                            ],
                            gutter="xs",
                        ),
                    ],
                    withBorder=True,
                    radius="md",
                    padding="sm",
                )
            )
        return cards

    @app.callback(
        Output("auto-config-state", "data"),
        Input("auto-target-labels", "value"),
        Input("auto-drop-ignore-radius", "value"),
        Input("plate-test-approach-height", "value"),
        Input("plate-test-pick-height", "value"),
        Input("plate-test-drop-height", "value"),
        Input("shared-tracking-mode", "value"),
        Input("shared-segment-min", "value"),
        Input("shared-segment-max", "value"),
        Input("shared-segment-fraction", "value"),
        Input("detection-frame-skip", "value"),
        Input(
            {
                "type": "auto-label-profile-field",
                "label": ALL,
                "field": ALL,
            },
            "value",
        ),
        Input(
            {
                "type": "auto-label-profile-toggle",
                "label": ALL,
                "field": ALL,
            },
            "checked",
        ),
        State(
            {
                "type": "auto-label-profile-field",
                "label": ALL,
                "field": ALL,
            },
            "id",
        ),
        State(
            {
                "type": "auto-label-profile-toggle",
                "label": ALL,
                "field": ALL,
            },
            "id",
        ),
    )
    def _update_auto_config(
        target_labels: Optional[list[str]],
        drop_ignore_radius: Optional[float],
        approach_height: Optional[float],
        pick_height: Optional[float],
        drop_height: Optional[float],
        tracking_mode: Optional[str],
        segment_min: Optional[float],
        segment_max: Optional[float],
        segment_fraction: Optional[float],
        frame_skip: Optional[float],
        profile_values: Optional[list[float]],
        profile_toggle_values: Optional[list[bool]],
        profile_ids: Optional[list[dict]],
        profile_toggle_ids: Optional[list[dict]],
    ) -> dict:
        """Update auto operation configuration from UI inputs."""
        import app.codroid_picker as codroid_picker_module
        import app.settings as app_settings
        from app.codroid_picker.session import (
            SESSION,
            normalize_auto_label_profiles,
        )

        selected_labels = [
            str(label).strip()
            for label in target_labels or []
            if str(label).strip()
        ]
        if not selected_labels:
            selected_labels = list(codroid_picker_module.SETTINGS.auto_target_labels)
        raw_profiles = dict(SESSION.auto_config.get("label_profiles") or {})
        for field_id, value in zip(profile_ids or [], profile_values or []):
            label = str(field_id.get("label", "")).strip()
            field = str(field_id.get("field", "")).strip()
            if not label or not field:
                continue
            raw_profiles.setdefault(label, {})[field] = value
        for field_id, value in zip(profile_toggle_ids or [], profile_toggle_values or []):
            label = str(field_id.get("label", "")).strip()
            field = str(field_id.get("field", "")).strip()
            if not label or not field:
                continue
            raw_profiles.setdefault(label, {})[field] = bool(value)
        profiles = normalize_auto_label_profiles(selected_labels, raw_profiles)
        primary_label = selected_labels[0] if selected_labels else (
            codroid_picker_module.SETTINGS.auto_target_label
        )
        frame_skip_value = max(codroid_picker_module._safe_int(frame_skip, 1), 1)

        config = {
            "target_label": primary_label,
            "target_labels": selected_labels,
            "label_profiles": profiles,
            "drop_ignore_radius_mm": codroid_picker_module._safe_float(
                drop_ignore_radius, codroid_picker_module.SETTINGS.auto_drop_ignore_radius_mm
            ),
            "approach_mm": codroid_picker_module._safe_float(
                approach_height, codroid_picker_module.SETTINGS.auto_approach_mm
            ),
            "pick_mm": codroid_picker_module._safe_float(
                pick_height, codroid_picker_module.SETTINGS.auto_pick_mm
            ),
            "drop_mm": codroid_picker_module._safe_float(
                drop_height, codroid_picker_module.SETTINGS.auto_drop_mm
            ),
            "estimate_width": bool(
                SESSION.auto_config.get(
                    "estimate_width", codroid_picker_module.SETTINGS.auto_estimate_width
                )
            ),
            "tracking_mode": (tracking_mode or codroid_picker_module.SETTINGS.tracking_mode).strip().lower(),
            "segment_min_mm": codroid_picker_module._safe_float(
                segment_min, codroid_picker_module.SETTINGS.segment_min
            ),
            "segment_max_mm": codroid_picker_module._safe_float(
                segment_max, codroid_picker_module.SETTINGS.segment_max
            ),
            "segment_fraction": codroid_picker_module._safe_float(
                segment_fraction, codroid_picker_module.SETTINGS.segment_fraction
            ),
            "detection_frame_skip": frame_skip_value,
        }
        SESSION.auto_config.update(config)
        SESSION.detection_frame_skip = frame_skip_value
        codroid_picker_module._commit_settings(
            auto_target_label=primary_label,
            auto_target_labels=selected_labels,
            auto_label_profiles=profiles,
            detection_frame_skip=frame_skip_value,
        )
        app_settings.reload_defaults()
        return config

    @app.callback(
        [
            Output("manual-pick-closest", "disabled"),
            Output("execute-drop", "disabled"),
            Output("manual-pick-place-target", "disabled"),
            Output("reset-operation-lock", "disabled"),
        ],
        Input("monitor-interval", "n_intervals"),
        prevent_initial_call=True,
    )
    def _update_button_states(_n: int) -> Tuple[bool, bool, bool, bool]:
        """Disable buttons during operations to prevent rapid clicking."""
        import logging
        from app.codroid_picker.session import SESSION

        if SESSION.operation_in_progress and SESSION.operation_started_at is not None:
            elapsed = time.monotonic() - SESSION.operation_started_at
            if elapsed > 120.0:
                logging.getLogger(__name__).warning(
                    "Auto-resetting stuck operation_in_progress flag after %.1f seconds",
                    elapsed,
                )
                SESSION.operation_in_progress = False
                SESSION.operation_started_at = None

        disabled = SESSION.operation_in_progress
        reset_disabled = not SESSION.operation_in_progress
        return disabled, disabled, disabled, reset_disabled

    @app.callback(
        Output("pick-place-status", "children", allow_duplicate=True),
        Input("reset-operation-lock", "n_clicks"),
        prevent_initial_call=True,
    )
    def _reset_operation_lock(_clicks: int) -> str:
        """Emergency reset for stuck operations."""
        import app.codroid_picker as codroid_picker_module
        from app.codroid_picker.session import SESSION

        SESSION.operation_in_progress = False
        SESSION.operation_started_at = None
        codroid_picker_module.LOGGER.info("Operation lock manually reset by user")
        return t("Operation lock reset.")

    @app.callback(
        Output("plate-test-status", "children", allow_duplicate=True),
        Input("monitor-interval", "n_intervals"),
        prevent_initial_call=True,
    )
    def _refresh_plate_test_status(_ticks: int) -> str:
        """Refresh plate test status from session state."""
        from app.codroid_picker.session import SESSION

        state = SESSION.plate_test_state or {}
        if state.get("running"):
            return state.get("message", "")
        return no_update

    @app.callback(
        Output("monitor-system-summary", "children"),
        Output("monitor-detection-summary", "children"),
        Output("monitor-pick-summary", "children"),
        Output("monitor-auto-plan-summary", "children"),
        Output("monitor-detection-trend", "figure"),
        Output("monitor-cycle-trend", "figure"),
        Output("use-existing-calibration", "style"),
        Output("use-existing-workspace", "style"),
        Output("use-existing-calibration", "disabled"),
        Output("use-existing-robot-calibration", "disabled"),
        Input("monitor-interval", "n_intervals"),
        State("camera-connection-state", "data"),
        State("robot-connection-state", "data"),
        State("gripper-connection-state", "data"),
        State("robot-power-state", "data"),
        State("robot-speed-state", "data"),
        State("settings-calib-path", "value"),
        State("settings-robot-calib-path", "value"),
        State("main-tabs", "value"),
        prevent_initial_call=True,
    )
    def _update_monitoring_summary(
        _monitor_ticks: int,
        camera_state: Optional[dict],
        robot_state: Optional[dict],
        gripper_state: Optional[dict],
        robot_power: Optional[dict],
        robot_speed: Optional[dict],
        bundle_path_value: Optional[str],
        robot_calib_path_value: Optional[str],
        active_tab: Optional[str],
    ) -> Tuple[list, list, list, list, go.Figure, go.Figure, dict, dict, bool, bool]:
        """Update monitoring panels, trends, and utility button states."""
        import app.codroid_picker as codroid_picker_module
        from app.codroid_picker.session import SESSION

        if active_tab not in {"monitoring", "monitor"}:
            return (no_update,) * 10

        camera_status = (camera_state or {}).get("status", "off")
        camera_message = (camera_state or {}).get("message") or ""
        robot_status = (robot_state or {}).get("status", "offline")
        gripper_status = (gripper_state or {}).get("status", "offline")
        calibration_state = "loaded" if SESSION.calibration_bundle is not None else "missing"
        workspace_state = "ready" if SESSION.workspace_definition is not None else "pending"
        plate_state = "ready" if SESSION.plate_calibration is not None else "pending"
        model_state = SESSION.active_model_path.name if SESSION.active_model_path else "none"
        detection_count = len(SESSION.detections)

        now_ts = time.time()
        detection_history = _append_detection_history(SESSION, detection_count, now_ts=now_ts)
        detection_trend_figure = _build_detection_trend_figure(detection_history)
        cycle_trend_figure = _build_cycle_trend_figure(list(SESSION.cycle_times))

        stream = SESSION.camera_stream
        capture_fps = stream.get_fps() if stream else 0.0
        stream_cfg_fps = stream.config.fps if stream else codroid_picker_module.DEFAULT_FPS
        res_w, res_h = stream.config.resolution if stream else codroid_picker_module.DEFAULT_RESOLUTION

        system_items = [
            dmc.Group(
                [
                    dmc.Badge(t("Camera: {status}", status=t(camera_status)), color=_status_color(camera_status), variant="light"),
                    dmc.Badge(t("Robot: {status}", status=t(robot_status)), color=_status_color(robot_status), variant="light"),
                    dmc.Badge(t("Gripper: {status}", status=t(gripper_status)), color=_status_color(gripper_status), variant="light"),
                    dmc.Badge(t("Power: {state}", state=t("on") if (robot_power or {}).get("value") else t("off")), color="teal" if (robot_power or {}).get("value") else "gray", variant="light"),
                ],
                gap="xs",
                wrap="wrap",
            ),
            dmc.Text(
                t("Camera: {status} · {details}", status=t(camera_status), details=camera_message)
                if camera_status != "live" and camera_message
                else t(
                    "Camera: {status} · {w}x{h} @ {fps:.1f}/{cfg:g} FPS",
                    status=t(camera_status),
                    w=res_w,
                    h=res_h,
                    fps=capture_fps,
                    cfg=stream_cfg_fps,
                ),
                size="sm",
            ),
            dmc.Text(
                t("Robot speed: {speed}%", speed=int((robot_speed or {}).get("value", SESSION.robot_speed_rate * 100))),
                size="sm",
            ),
            dmc.Text(t("Calibration bundle: {state}", state=t(calibration_state)), size="sm"),
            dmc.Text(t("Workspace: {state}", state=t(workspace_state)), size="sm"),
            dmc.Text(t("Plate calibration: {state}", state=t(plate_state)), size="sm"),
        ]

        detection_items = [
            dmc.SimpleGrid(
                [
                    dmc.Paper([dmc.Text(t("Model"), size="xs", c="dimmed"), dmc.Text(t(model_state), fw=600)], p="xs", radius="md", withBorder=True),
                    dmc.Paper([dmc.Text(t("Detections"), size="xs", c="dimmed"), dmc.Text(str(detection_count), fw=700)], p="xs", radius="md", withBorder=True),
                ],
                cols=2,
                spacing="xs",
            )
        ]
        if SESSION.last_detection_error:
            detection_items.append(
                dmc.Alert(
                    t("Last error: {error}", error=SESSION.last_detection_error),
                    color="red",
                    variant="light",
                )
            )

        attempts = SESSION.pick_attempts
        successes = SESSION.pick_successes
        failures = SESSION.pick_failures
        success_rate = (successes / attempts * 100.0) if attempts else None
        avg_cycle = (sum(SESSION.cycle_times) / len(SESSION.cycle_times)) if SESSION.cycle_times else None
        last_action = SESSION.last_action or t("No recent actions.")

        pick_items = [
            dmc.SimpleGrid(
                [
                    dmc.Paper([dmc.Text(t("Attempts"), size="xs", c="dimmed"), dmc.Text(str(attempts), fw=700)], p="xs", radius="md", withBorder=True),
                    dmc.Paper([dmc.Text(t("Success rate"), size="xs", c="dimmed"), dmc.Text(f"{success_rate:.1f}%" if success_rate is not None else t("n/a"), fw=700)], p="xs", radius="md", withBorder=True),
                    dmc.Paper([dmc.Text(t("Avg cycle"), size="xs", c="dimmed"), dmc.Text(f"{avg_cycle:.2f}s" if avg_cycle is not None else t("n/a"), fw=700)], p="xs", radius="md", withBorder=True),
                    dmc.Paper([dmc.Text(t("Holding"), size="xs", c="dimmed"), dmc.Text(t("yes") if SESSION.robot_has_object else t("no"), fw=700)], p="xs", radius="md", withBorder=True),
                ],
                cols=2,
                spacing="xs",
            ),
            dmc.Text(t("Last action: {action}", action=last_action), size="sm"),
            dmc.Text(t("Processed objects: {count}", count=successes), size="sm"),
            dmc.Text(t("Attempts: {attempts} · Failures: {failures}", attempts=attempts, failures=failures), size="sm"),
            dmc.Text(
                t("Last cycle: {cycle:.2f}s", cycle=SESSION.last_cycle_s)
                if SESSION.last_cycle_s is not None
                else t("Last cycle: n/a"),
                size="sm",
            ),
        ]

        if SESSION.last_pick_mm:
            pick_items.append(
                dmc.Text(
                    t("Last pick: {x:.1f}, {y:.1f} mm", x=SESSION.last_pick_mm[0], y=SESSION.last_pick_mm[1]),
                    size="sm",
                )
            )
        if SESSION.drop_target_mm:
            pick_items.append(
                dmc.Text(
                    t("Drop target: {x:.1f}, {y:.1f} mm", x=SESSION.drop_target_mm[0], y=SESSION.drop_target_mm[1]),
                    size="sm",
                )
            )

        plan_snapshot = get_auto_plan_snapshot(stale_after_s=DEFAULT_PLAN_STALE_S)
        auto_plan_items = _build_auto_plan_items(plan_snapshot)

        bundle_path = codroid_picker_module._safe_path(bundle_path_value, codroid_picker_module.DEFAULT_BUNDLE_PATH)
        bundle_style = {}
        workspace_style = {} if codroid_picker_module.DEFAULT_WORKSPACE_PATH.exists() else {"display": "none"}
        bundle_disabled = not bundle_path.exists()
        robot_path = codroid_picker_module._safe_path(
            robot_calib_path_value, codroid_picker_module.DEFAULT_ROBOT_CALIBRATION_PATH
        )
        robot_disabled = not robot_path.exists()

        return (
            system_items,
            detection_items,
            pick_items,
            auto_plan_items,
            detection_trend_figure,
            cycle_trend_figure,
            bundle_style,
            workspace_style,
            bundle_disabled,
            robot_disabled,
        )

    @app.callback(
        Output("plate-detection-summary", "children"),
        Output("plate-detection-summary", "color"),
        Output("start-robot-calibration", "disabled"),
        Output("start-robot-calibration", "variant"),
        Input("plate-detection-state", "data"),
        Input("robot-connection-state", "data"),
        State("robot-calibration-state", "data"),
    )
    def _update_plate_summary(
        plate_state: Optional[dict],
        robot_conn_state: Optional[dict],
        calib_state: Optional[dict],
    ) -> Tuple[str, str, bool, str]:
        """Update plate detection summary and calibration button state."""
        import app.codroid_picker as codroid_picker_module
        from app.codroid_picker import state

        plate_data = plate_state or {}
        count = plate_data.get("count", 0)
        ok = bool(plate_data.get("ok"))
        stable = bool(plate_data.get("stable", ok))
        calib_status = (calib_state or {}).get("status")
        in_progress = calib_status in {"running", "awaiting_verification", "verifying"}
        if state.get_calibration_guidance_snapshot().get("status") in {
            "running",
            "awaiting_verification",
            "verifying",
        }:
            in_progress = True
        robot_online = (robot_conn_state or {}).get("status") == "online"
        posture_ready = robot_online
        posture_message = t("Robot not connected.")
        if robot_online:
            posture_ready, posture_message = codroid_picker_module._robot_posture_ready()

        if in_progress:
            if calib_status == "awaiting_verification":
                message = t("Awaiting verification move.")
                color = "yellow"
            elif calib_status == "verifying":
                message = t("Verification in progress...")
                color = "blue"
            else:
                message = t("Calibration in progress...")
                color = "blue"
        elif not robot_online:
            message = t("Robot not connected. Click 'Test robot connection' to establish connection.")
            color = "red"
        elif not ok:
            message = plate_data.get("message") or t("Detected {count} marker(s).", count=count)
            color = "yellow"
        elif not stable:
            message = plate_data.get("message") or t("Calibration plate moved; hold it steady.")
            color = "yellow"
        elif not posture_ready:
            message = posture_message
            color = "yellow"
        else:
            message = t("Plate markers stable. Ready to calibrate.")
            color = "green"

        disabled = not ok or not stable or not posture_ready or in_progress or not robot_online
        variant = "filled" if not disabled else "light"
        return message, color, disabled, variant

    @app.callback(
        Output("run-verification-inline", "disabled"),
        Input("plate-detection-state", "data"),
        Input("robot-calibration-state", "data"),
        Input("robot-connection-state", "data"),
    )
    def _sync_verification_inline_button(
        plate_state: Optional[dict],
        calibration_state: Optional[dict],
        robot_state: Optional[dict],
    ) -> bool:
        """Sync inline verification button disabled state."""
        from app.codroid_picker.session import SESSION

        plate_ok = bool((plate_state or {}).get("ok"))
        plate_stable = bool((plate_state or {}).get("stable", plate_ok))
        guidance = calibration_state or {}
        status = guidance.get("status")
        in_progress = status in {"running", "awaiting_verification", "verifying"}
        robot_online = (robot_state or {}).get("status") == "online"
        has_calibration_points = len(SESSION.robot_calibration_points) >= 3
        return not (
            plate_ok
            and plate_stable
            and robot_online
            and has_calibration_points
            and not in_progress
        )
