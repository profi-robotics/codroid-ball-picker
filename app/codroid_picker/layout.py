"""
Main application layout builders.

This module contains the setup drawer and main layout construction function.
"""

from __future__ import annotations

import dash_mantine_components as dmc
from dash import dcc, html

import app.settings as app_settings
from app.codroid_picker.session import SESSION
from app.settings import (
    SETTINGS,
    DEFAULT_BUNDLE_PATH,
    DEFAULT_WORKSPACE_PATH,
    DEFAULT_ROBOT_CALIBRATION_PATH,
    DEFAULT_RESOLUTION,
    DEFAULT_FPS,
)
from app.codroid_picker.state import _operation_snapshot, get_error_snapshot
from app.codroid_picker.components import (
    build_header,
    build_footer,
    build_camera_panel,
)
from app.codroid_picker.modals import (
    build_robot_calibration_modal,
    build_emergency_stop_modal,
    build_overspeed_modal,
    build_joint_protection_modal,
    build_wrong_tool_modal,
    build_drag_not_allowed_modal,
)
from app.codroid_picker.i18n import current_locale, t


def build_setup_drawer() -> dmc.Drawer:
    """Build the setup and connections drawer."""
    return dmc.Drawer(
        id="setup-drawer",
        opened=False,
        title=t("setup.title"),
        padding="md",
        size="lg",
        zIndex=2000,
        withOverlay=True,
        overlayProps={"opacity": 0.55, "color": "#000"},
        withinPortal=True,
        children=dmc.Stack(
            [
                dmc.Select(
                    id="language-select-setup",
                    label=t("language.label"),
                    data=[
                        {"value": "en", "label": t("language.english")},
                        {"value": "uk", "label": t("language.ukrainian")},
                    ],
                    value=current_locale(),
                    clearable=False,
                    allowDeselect=False,
                    comboboxProps={"withinPortal": False},
                    size="sm",
                ),
                dmc.Text(t("Camera connection"), fw=600),
                dmc.Group(
                    [
                        dmc.Button(t("Scan for camera"),
                                   id="camera-scan", variant="light"),
                        dmc.Switch(
                            id="camera-enable",
                            label=t("Enable live camera stream"),
                            checked=SETTINGS.camera_auto_start,
                            color="teal",
                        ),
                    ],
                    gap="md",
                ),
                dmc.Group(
                    [
                        dmc.NumberInput(
                            id="camera-width",
                            label=t("Width"),
                            value=DEFAULT_RESOLUTION[0],
                            min=320,
                            max=3840,
                        ),
                        dmc.NumberInput(
                            id="camera-height",
                            label=t("Height"),
                            value=DEFAULT_RESOLUTION[1],
                            min=240,
                            max=2160,
                        ),
                        dmc.NumberInput(
                            id="camera-fps-input",
                            label=t("FPS"),
                            value=DEFAULT_FPS,
                            min=1,
                            max=120,
                            step=1,
                        ),
                    ],
                    grow=True,
                ),
                dmc.Switch(
                    id="camera-lock-af",
                    label=t("Lock autofocus"),
                    checked=False,
                    color="teal",
                    size="md",
                ),
                dmc.Text(id="camera-status-text", size="sm", c="dimmed"),
                html.Div(id="camera-device-info"),
                dmc.Divider(),
                dmc.Text(t("Robot connection"), fw=600),
                dmc.TextInput(
                    id="robot-uri",
                    label=t("WebSocket URI"),
                    value=app_settings.DEFAULT_ROBOT_URI,
                ),
                dmc.NumberInput(
                    id="settings-robot-coordinate-id",
                    label=t("Robot coordinate ID"),
                    value=SETTINGS.robot_coordinate_id,
                    description=t("Coordinate system used for plate + pick controls."),
                    min=0,
                    max=10,
                    size="sm",
                ),
                dmc.Group(
                    [
                        dmc.Button(t("Test robot connection"),
                                   id="robot-test", variant="light"),
                        dmc.Button(
                            t("Acquire robot control"),
                            id="robot-acquire-control",
                            variant="light",
                            color="teal",
                        ),
                        dmc.Button(
                            t("Release robot control"),
                            id="robot-release-control",
                            variant="outline",
                            color="red",
                        ),
                    ],
                    gap="sm",
                ),
                dmc.Text(id="robot-status-text", size="sm", c="dimmed"),
                dmc.Divider(),
                dmc.Text(t("OnRobot 2FG7 Gripper"), fw=600),
                dmc.TextInput(
                    id="gripper-cb-ip",
                    label=t("Compute Box IP Address"),
                    value=app_settings.DEFAULT_GRIPPER_IP,
                    placeholder="192.168.101.95",
                    description=t("IP address of the OnRobot Compute Box"),
                ),
                dmc.Group(
                    [
                        dmc.Switch(
                            id="gripper-enable",
                            label=t("Enable gripper connection"),
                            checked=True,
                            color="teal",
                        ),
                        dmc.Button(t("Test gripper"),
                                   id="gripper-test", variant="light"),
                    ],
                    gap="md",
                ),
                dmc.Text(id="gripper-status-text", size="sm", c="dimmed"),
                html.Div(id="gripper-device-info"),
                dmc.Stack(
                    [
                        dmc.Text(t("Finger orientation"), fw=600, size="sm"),
                        dmc.SegmentedControl(
                            id="gripper-orientation",
                            data=[
                                {"label": t("Inward"), "value": "inward"},
                                {"label": t("Outward"), "value": "outward"},
                            ],
                            value="inward",
                            size="sm",
                            color="brand",
                            radius="md",
                        ),
                        dmc.Text(
                            id="gripper-orientation-status",
                            size="xs",
                            c="dimmed",
                        ),
                    ],
                    gap=4,
                ),
                dmc.Divider(),
                dmc.Text(t("Calibration & settings"), fw=600),
                dmc.Grid(
                    [
                        dmc.GridCol(
                            dmc.NumberInput(
                                id="settings-marker-a",
                                label=t("Marker ID A"),
                                value=SETTINGS.marker_ids[0],
                                min=0,
                                max=49,
                                size="sm",
                            ),
                            span=4,
                        ),
                        dmc.GridCol(
                            dmc.NumberInput(
                                id="settings-marker-b",
                                label=t("Marker ID B"),
                                value=SETTINGS.marker_ids[1],
                                min=0,
                                max=49,
                                size="sm",
                            ),
                            span=4,
                        ),
                        dmc.GridCol(
                            dmc.NumberInput(
                                id="settings-marker-size",
                                label=t("Marker size (mm)"),
                                value=SETTINGS.marker_size_mm,
                                min=1,
                                max=200,
                                size="sm",
                            ),
                            span=4,
                        ),
                        dmc.GridCol(
                            dmc.TextInput(
                                id="settings-calib-path",
                                label=t("Calibration bundle path"),
                                value=str(DEFAULT_BUNDLE_PATH),
                                size="sm",
                            ),
                            span=12,
                        ),
                        dmc.GridCol(
                            dmc.TextInput(
                                id="settings-workspace-path",
                                label=t("Workspace path"),
                                value=str(DEFAULT_WORKSPACE_PATH),
                                size="sm",
                            ),
                            span=12,
                        ),
                        dmc.GridCol(
                            dmc.TextInput(
                                id="settings-robot-calib-path",
                                label=t("Robot calibration path"),
                                value=str(DEFAULT_ROBOT_CALIBRATION_PATH),
                                size="sm",
                            ),
                            span=12,
                        ),
                    ],
                    gutter="sm",
                ),
                dmc.Text(t("Auto pick & place defaults"), fw=600),
                dmc.Grid(
                    [
                        dmc.GridCol(
                            dmc.NumberInput(
                                id="settings-auto-drop-ignore-radius",
                                label=t("Drop ignore radius (mm)"),
                                value=SETTINGS.auto_drop_ignore_radius_mm,
                                min=0,
                                max=200,
                                step=5,
                                size="sm",
                            ),
                            span=4,
                        ),
                        dmc.GridCol(
                            dmc.NumberInput(
                                id="settings-auto-intrusion-match",
                                label=t("Intrusion match (mm)"),
                                value=SETTINGS.auto_intrusion_match_mm,
                                min=10,
                                max=200,
                                step=1,
                                size="sm",
                            ),
                            span=4,
                        ),
                        dmc.GridCol(
                            dmc.NumberInput(
                                id="settings-auto-detection-stale",
                                label=t("Detection stale (s)"),
                                value=SETTINGS.auto_detection_stale_s,
                                min=0.05,
                                max=2.0,
                                step=0.05,
                                size="sm",
                            ),
                            span=4,
                        ),
                        dmc.GridCol(
                            dmc.NumberInput(
                                id="settings-auto-idle-sleep",
                                label=t("Idle sleep (s)"),
                                value=SETTINGS.auto_idle_sleep_s,
                                min=0.05,
                                max=1.0,
                                step=0.05,
                                size="sm",
                            ),
                            span=4,
                        ),
                        dmc.GridCol(
                            dmc.Switch(
                                id="settings-auto-estimate-width",
                                label=t("Estimate width"),
                                checked=SETTINGS.auto_estimate_width,
                                size="sm",
                            ),
                            span=4,
                        ),
                        dmc.GridCol(
                            dmc.NumberInput(
                                id="settings-auto-approach-height",
                                label=t("Approach height (mm)"),
                                value=SETTINGS.auto_approach_mm,
                                min=-200,
                                max=0,
                                step=1,
                                size="sm",
                            ),
                            span=4,
                        ),
                        dmc.GridCol(
                            dmc.NumberInput(
                                id="settings-auto-pick-height",
                                label=t("Pick height (mm)"),
                                value=SETTINGS.auto_pick_mm,
                                min=-50,
                                max=200,
                                step=1,
                                size="sm",
                            ),
                            span=4,
                        ),
                        dmc.GridCol(
                            dmc.NumberInput(
                                id="settings-auto-drop-height",
                                label=t("Drop height (mm)"),
                                value=SETTINGS.auto_drop_mm,
                                min=-50,
                                max=200,
                                step=1,
                                size="sm",
                            ),
                            span=4,
                        ),
                        dmc.GridCol(
                            dmc.NumberInput(
                                id="settings-auto-transit-clearance",
                                label=t("Transit clearance (mm)"),
                                description=t("Extra upward clearance for XY travel above approach/drop."),
                                value=SETTINGS.auto_transit_clearance_mm,
                                min=0,
                                max=100,
                                step=1,
                                size="sm",
                            ),
                            span=4,
                        ),
                    ],
                    gutter="sm",
                ),
                dmc.Text(t("Pick follow tuning"), fw=600),
                dmc.Grid(
                    [
                        dmc.GridCol(
                            dmc.NumberInput(
                                id="settings-pick-follow-timeout",
                                label=t("Timeout (s)"),
                                description=t("Max follow loop duration."),
                                value=SETTINGS.pick_follow_timeout_s,
                                min=1,
                                max=60,
                                step=1,
                                size="sm",
                            ),
                            span=4,
                        ),
                        dmc.GridCol(
                            dmc.NumberInput(
                                id="settings-pick-follow-interval",
                                label=t("Update interval (s)"),
                                value=SETTINGS.pick_follow_update_interval_s,
                                min=0.01,
                                max=0.5,
                                step=0.01,
                                size="sm",
                            ),
                            span=4,
                        ),
                        dmc.GridCol(
                            dmc.NumberInput(
                                id="settings-pick-follow-segment",
                                label=t("Segment hold (s)"),
                                value=SETTINGS.pick_follow_segment_hold_s,
                                min=0.01,
                                max=1.0,
                                step=0.01,
                                size="sm",
                            ),
                            span=4,
                        ),
                        dmc.GridCol(
                            dmc.NumberInput(
                                id="settings-pick-follow-settle",
                                label=t("Settle time (s)"),
                                value=SETTINGS.pick_follow_settle_s,
                                min=0.01,
                                max=5.0,
                                step=0.05,
                                size="sm",
                            ),
                            span=4,
                        ),
                        dmc.GridCol(
                            dmc.NumberInput(
                                id="settings-pick-follow-tolerance",
                                label=t("Tolerance (mm)"),
                                value=SETTINGS.pick_follow_tolerance_mm,
                                min=0.1,
                                max=20.0,
                                step=0.1,
                                size="sm",
                            ),
                            span=4,
                        ),
                    ],
                    gutter="sm",
                ),
                dmc.Text(t("Safety guards"), fw=600),
                dmc.Grid(
                    [
                        dmc.GridCol(
                            dmc.Select(
                                id="settings-camera-safety-mode",
                                label=t("Camera safety mode"),
                                data=[
                                    {"label": t("Safety off"), "value": "off"},
                                    {"label": t("Model safety"), "value": "model"},
                                    {"label": t("Heuristic safety"), "value": "heuristic"},
                                ],
                                value=getattr(SETTINGS, "camera_safety_mode", "model"),
                                description=t("Select how camera safety should pause robot motion."),
                                comboboxProps={"withinPortal": False},
                                clearable=False,
                                size="sm",
                            ),
                            span=4,
                        ),
                        dmc.GridCol(
                            dmc.Select(
                                id="settings-camera-safety-zone",
                                label=t("Camera safety zone"),
                                data=[
                                    {"label": t("Workspace only"), "value": "workspace"},
                                    {"label": t("Full frame"), "value": "full_frame"},
                                ],
                                value=SETTINGS.camera_safety_zone,
                                comboboxProps={"withinPortal": False},
                                clearable=False,
                                size="sm",
                            ),
                            span=4,
                        ),
                        dmc.GridCol(
                            dmc.TagsInput(
                                id="settings-camera-safety-labels",
                                label=t("Safety stop labels"),
                                description=t("YOLO labels that should pause motion."),
                                data=list(SETTINGS.camera_safety_labels),
                                value=list(SETTINGS.camera_safety_labels),
                                allowDuplicates=False,
                                acceptValueOnBlur=True,
                                clearable=False,
                                size="sm",
                            ),
                            span=4,
                        ),
                        dmc.GridCol(
                            dmc.Box(
                                [
                                    dmc.NumberInput(
                                        id="settings-camera-safety-heuristic-threshold",
                                        label=t("Heuristic diff threshold"),
                                        description=t("Normalized changed-pixel ratio to trigger pause."),
                                        value=SETTINGS.camera_safety_heuristic_threshold,
                                        min=0.001,
                                        max=1.0,
                                        step=0.001,
                                        decimalScale=3,
                                        size="sm",
                                    ),
                                    dmc.NumberInput(
                                        id="settings-camera-safety-heuristic-min-area",
                                        label=t("Heuristic min area ratio"),
                                        description=t("Small-change filter ratio inside monitored zone."),
                                        value=SETTINGS.camera_safety_heuristic_min_area_ratio,
                                        min=0.001,
                                        max=1.0,
                                        step=0.001,
                                        decimalScale=3,
                                        size="sm",
                                        mt="sm",
                                    ),
                                    dmc.NumberInput(
                                        id="settings-camera-safety-baseline-refresh-ms",
                                        label=t("Baseline refresh (ms)"),
                                        description=t("How often heuristic baseline can refresh while clear."),
                                        value=SETTINGS.camera_safety_baseline_refresh_ms,
                                        min=100,
                                        max=10000,
                                        step=50,
                                        size="sm",
                                        mt="sm",
                                    ),
                                ],
                                id="settings-camera-safety-heuristic-box",
                                style={
                                    "display": "block"
                                    if getattr(SETTINGS, "camera_safety_mode", "model") == "heuristic"
                                    else "none"
                                },
                            ),
                            span=8,
                        ),
                        dmc.GridCol(
                            dmc.NumberInput(
                                id="settings-rescue-jog-hold",
                                label=t("Rescue jog hold (s)"),
                                value=SETTINGS.rescue_jog_hold_s,
                                min=0.1,
                                max=1.5,
                                step=0.05,
                                size="sm",
                            ),
                            span=4,
                        ),
                        dmc.GridCol(
                            dmc.NumberInput(
                                id="settings-coord5-max-z",
                                label=t("Coord5 max Z (mm)"),
                                value=SETTINGS.coord5_max_z_mm,
                                description=t("Positive Z moves downward; clamp to safe range."),
                                min=-50,
                                max=50,
                                step=1,
                                size="sm",
                            ),
                            span=4,
                        ),
                    ],
                    gutter="sm",
                ),
                dmc.Group(
                    [
                        dmc.Button(t("Save settings"),
                                   id="settings-save", color="teal"),
                        dmc.Button(t("Reset to defaults"), id="settings-reset",
                                   variant="outline", color="red"),
                        dmc.Text(id="settings-status", size="sm", c="dimmed"),
                    ],
                    justify="space-between",
                    align="center",
                ),
                dmc.Group(
                    [
                        dmc.Button(t("setup.close"), id="setup-close",
                                   variant="subtle"),
                    ],
                    justify="flex-end",
                ),
            ],
            gap="sm",
        ),
    )


def build_layout() -> dmc.MantineProvider:
    """Build the complete application layout."""
    font_family = (
        'system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif'
    )
    brand_palette = [
        "#ffe8de",
        "#ffcdbd",
        "#ffac91",
        "#ff8a61",
        "#ff6b36",
        "#f04100",
        "#d23a00",
        "#b03200",
        "#8a2700",
        "#611a00",
    ]
    theme = {
        "fontFamily": font_family,
        "headings": {"fontFamily": font_family},
        "primaryColor": "brand",
        "colors": {"brand": brand_palette},
    }
    return dmc.MantineProvider(
        theme=theme,
        withGlobalClasses=True,
        withCssVariables=True,
        children=html.Div(
            [
                dmc.NotificationContainer(
                    id="app-notifications", position="top-right", limit=3),
                dcc.Location(id="app-location", refresh=True),
                dcc.Store(id="camera-connection-state"),
                dcc.Store(id="robot-connection-state"),
                dcc.Store(
                    id="robot-control-state",
                    data={"mode": "acquired", "released_at": 0.0, "message": ""},
                ),
                dcc.Store(id="ui-language-state", data={"value": current_locale()}),
                dcc.Store(
                    id="gripper-connection-state",
                    data={"status": "offline", "message": ""},
                ),
                dcc.Store(
                    id="notification-state",
                    data={"camera": None, "robot": None},
                ),
                dcc.Store(id="calibration-progress-store",
                          data={"progress": 0}),
                dcc.Store(
                    id="calibration-menu-open-store",
                    data=SESSION.calibration_menu_open,
                ),
                dcc.Store(
                    id="application-tab-state",
                    data={"value": "pick-place"},
                ),
                dcc.Store(
                    id="calibration-step-state",
                    data={"active": 0},
                ),
                dcc.Store(
                    id="camera-ui-metrics-state",
                    data={
                        "camera_calibration_metrics": "",
                        "workspace_metrics": "",
                        "plate_metrics": "",
                        "detection_metrics": "",
                        "calibrate_camera_disabled": True,
                        "markers_detected_color": "gray",
                    },
                ),
                dcc.Store(
                    id="model-load-state",
                    data={"status": "idle", "message": "", "model": None},
                ),
                dcc.Store(
                    id="pick-label-config-state",
                    data=[
                        {
                            "label": SETTINGS.auto_target_label,
                            "confidence": 0.1,
                        }
                    ],
                ),
                dcc.Store(
                    id="sidebar-visibility",
                    data={"collapsed": False},
                ),
                dcc.Store(
                    id="plate-detection-state",
                    data={
                        "count": 0,
                        "ok": False,
                        "stable": False,
                        "center_px": None,
                        "movement_px": None,
                        "timestamp": 0.0,
                        "message": "",
                    },
                ),
                dcc.Store(
                    id="robot-calibration-state",
                    data={
                        "status": "idle",
                        "message": "",
                        "step": 0,
                        "verification_ready": False,
                        "accuracy": None,
                        "accuracy_ready": False,
                        "accuracy_passed": False,
                        "modal_suppressed": False,
                    },
                ),
                dcc.Store(
                    id="auto-operation-state",
                    data={"running": False, "message": ""},
                ),
                dcc.Store(
                    id="auto-config-state",
                    data={
                        "target_label": SETTINGS.auto_target_label,
                        "target_labels": list(SETTINGS.auto_target_labels),
                        "drop_ignore_radius_mm": SETTINGS.auto_drop_ignore_radius_mm,
                        "approach_mm": SETTINGS.auto_approach_mm,
                        "pick_mm": SETTINGS.auto_pick_mm,
                        "drop_mm": SETTINGS.auto_drop_mm,
                        "estimate_width": SETTINGS.auto_estimate_width,
                        "tracking_mode": SETTINGS.tracking_mode,
                        "segment_min_mm": SETTINGS.segment_min,
                        "segment_max_mm": SETTINGS.segment_max,
                        "segment_fraction": SETTINGS.segment_fraction,
                    },
                ),
                dcc.Store(
                    id="robot-operation-state",
                    data=_operation_snapshot(),
                ),
                dcc.Store(
                    id="robot-error-state",
                    data=get_error_snapshot(),
                ),
                dcc.Store(
                    id="robot-rescue-state",
                    data={"active": False, "timestamp": 0.0},
                ),
                dcc.Store(
                    id="prerequisites-state",
                    data={
                        "camera_live": False,
                        "camera_calibrated": False,
                        "workspace_defined": False,
                        "robot_calibrated": False,
                        "robot_connected": False,
                        "model_loaded": False,
                        "gripper_ready": False,
                        "drop_target_defined": False,
                    },
                ),
                dcc.Store(
                    id="footer-visibility",
                    data={"collapsed": True},
                ),
                dcc.Store(
                    id="robot-speed-state",
                    data={"value": 25},
                ),
                dcc.Store(
                    id="robot-power-state",
                    data={"value": None, "timestamp": 0.0},
                ),
                dcc.Store(
                    id="camera-safety-state",
                    data={"blocked": False, "matches": 0, "action": "none", "active": False},
                ),
                dcc.Interval(id="ui-ready", interval=150,
                             n_intervals=0, max_intervals=1),
                dcc.Interval(id="hardware-init", interval=900,
                             n_intervals=0, max_intervals=1),
                dcc.Interval(id="model-auto-load", interval=5000,
                             n_intervals=0, max_intervals=1),
                dcc.Interval(id="robot-status-interval",
                             interval=1000, n_intervals=0),
                dcc.Interval(
                    id="monitor-interval",
                    interval=3000,
                    n_intervals=0,
                ),
                dcc.Interval(
                    id="application-ui-interval",
                    interval=2500,
                    n_intervals=0,
                    disabled=True,
                ),
                dcc.Interval(
                    id="calibration-ui-interval",
                    interval=1000,
                    n_intervals=0,
                    disabled=True,
                ),
                dcc.Interval(
                    id="monitoring-graph-interval",
                    interval=5000,
                    n_intervals=0,
                    disabled=True,
                ),
                dcc.Interval(
                    id="metrics-interval",
                    # Keep UI metrics updates light for embedded devices (Jetson Nano).
                    interval=1000,
                    n_intervals=0,
                    disabled=True,
                ),
                dcc.Interval(
                    id="camera-safety-interval",
                    interval=SETTINGS.camera_safety_check_interval_ms,
                    n_intervals=0,
                    disabled=True,
                ),
                html.Div(id="camera-refresh-container",
                         style={"display": "none"}),
                build_setup_drawer(),
                build_robot_calibration_modal(),
                build_emergency_stop_modal(),
                build_overspeed_modal(),
                build_joint_protection_modal(),
                build_wrong_tool_modal(),
                build_drag_not_allowed_modal(),
                dmc.AppShell(
                    id="app-shell",
                    padding="xs",
                    className="app-shell",
                    header={"height": 76},
                    footer={"height": 174},
                    children=[
                        build_header(),
                        dmc.AppShellMain(
                            dmc.Container(
                                [
                                    dmc.Group(
                                        [
                                            dmc.Button(
                                                t("Hide controls"),
                                                id="sidebar-toggle",
                                                variant="light",
                                                color="gray",
                                                size="xs",
                                                radius="md",
                                            ),
                                        ],
                                        justify="flex-start",
                                        align="center",
                                        className="sidebar-toolbar",
                                    ),
                                    dmc.Grid(
                                        [
                                            dmc.GridCol(
                                                html.Div(
                                                    [
                                                        dmc.Tabs(
                                                            [
                                                                dmc.TabsList(
                                                                    [
                                                                        dmc.TabsTab(
                                                                            t("tabs.applications"),
                                                                            value="applications",
                                                                        ),
                                                                        dmc.TabsTab(
                                                                            t("tabs.monitoring"),
                                                                            value="monitoring",
                                                                        ),
                                                                        dmc.TabsTab(
                                                                            t("tabs.calibration"),
                                                                            value="calibration",
                                                                        ),
                                                                    ],
                                                                    style={
                                                                        "width": "100%",
                                                                        "justifyContent": "flex-start",
                                                                        "marginTop": "0px",
                                                                        "marginBottom": "8px",
                                                                    },
                                                                ),
                                                            ],
                                                            id="main-tabs",
                                                            value="applications",
                                                            variant="pills",
                                                            color="brand",
                                                            className="main-tabs",
                                                            style={
                                                                "width": "100%",
                                                                "marginBottom": "4px",
                                                            },
                                                        ),
                                                        html.Div(id="main-tab-content"),
                                                    ],
                                                    className="tabs-scroll-area",
                                                ),
                                                id="tabs-sidebar",
                                                span={"base": 12, "md": 4, "lg": 3},
                                                className="tabs-column",
                                            ),
                                            html.Div(
                                                id="sidebar-resizer",
                                                className="sidebar-resizer",
                                                role="separator",
                                                **{
                                                    "aria-label": t("Resize controls"),
                                                    "aria-orientation": "vertical",
                                                },
                                            ),
                                            dmc.GridCol(
                                                dmc.Stack(
                                                    [build_camera_panel()],
                                                    gap="xs",
                                                    className="camera-stack",
                                                ),
                                                id="camera-grid-column",
                                                span={"base": 12, "md": 8, "lg": 9},
                                                className="camera-column",
                                            ),
                                        ],
                                        gutter="xs",
                                        id="main-layout-grid",
                                        className="layout-grid",
                                    ),
                                ],
                                fluid=True,
                                px=0,
                                mx=0,
                                mt=0,
                                className="main-panel",
                            ),
                            id="app-main",
                            className="app-main footer-collapsed",
                            style={"overflow": "hidden"},
                        ),
                        build_footer(),
                    ],
                ),
            ]
        ),
    )
