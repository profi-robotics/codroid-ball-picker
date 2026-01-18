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
    build_application_tab,
    build_monitoring_tab,
    build_calibration_tab,
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


def build_setup_drawer() -> dmc.Drawer:
    """Build the setup and connections drawer."""
    return dmc.Drawer(
        id="setup-drawer",
        opened=False,
        title="Setup & Connections",
        padding="md",
        size="lg",
        zIndex=2000,
        withOverlay=True,
        overlayProps={"opacity": 0.55, "color": "#000"},
        withinPortal=True,
        children=dmc.Stack(
            [
                dmc.Text("Camera connection", fw=600),
                dmc.Group(
                    [
                        dmc.Button("Scan for camera",
                                   id="camera-scan", variant="light"),
                        dmc.Switch(
                            id="camera-enable",
                            label="Enable live camera stream",
                            checked=False,
                            color="teal",
                        ),
                    ],
                    gap="md",
                ),
                dmc.Group(
                    [
                        dmc.NumberInput(
                            id="camera-width",
                            label="Width",
                            value=DEFAULT_RESOLUTION[0],
                            min=320,
                            max=3840,
                        ),
                        dmc.NumberInput(
                            id="camera-height",
                            label="Height",
                            value=DEFAULT_RESOLUTION[1],
                            min=240,
                            max=2160,
                        ),
                        dmc.NumberInput(
                            id="camera-fps-input",
                            label="FPS",
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
                    label="Lock autofocus",
                    checked=False,
                    color="teal",
                    size="md",
                ),
                dmc.Text(id="camera-status-text", size="sm", c="dimmed"),
                html.Div(id="camera-device-info"),
                dmc.Divider(),
                dmc.Text("Robot connection", fw=600),
                dmc.TextInput(
                    id="robot-uri",
                    label="WebSocket URI",
                    value=app_settings.DEFAULT_ROBOT_URI,
                ),
                dmc.NumberInput(
                    id="settings-robot-coordinate-id",
                    label="Robot coordinate ID",
                    value=SETTINGS.robot_coordinate_id,
                    description="Coordinate system used for plate + pick controls.",
                    min=0,
                    max=10,
                    size="sm",
                ),
                dmc.Group(
                    [
                        dmc.Button("Test robot connection",
                                   id="robot-test", variant="light"),
                        dmc.Button(
                            "Disconnect robot",
                            id="robot-disconnect",
                            variant="outline",
                            color="red",
                        ),
                    ],
                    gap="sm",
                ),
                dmc.Text(id="robot-status-text", size="sm", c="dimmed"),
                dmc.Divider(),
                dmc.Text("OnRobot 2FG7 Gripper", fw=600),
                dmc.TextInput(
                    id="gripper-cb-ip",
                    label="Compute Box IP Address",
                    value=app_settings.DEFAULT_GRIPPER_IP,
                    placeholder="192.168.101.95",
                    description="IP address of the OnRobot Compute Box",
                ),
                dmc.Group(
                    [
                        dmc.Switch(
                            id="gripper-enable",
                            label="Enable gripper connection",
                            checked=True,
                            color="teal",
                        ),
                        dmc.Button("Test gripper",
                                   id="gripper-test", variant="light"),
                    ],
                    gap="md",
                ),
                dmc.Text(id="gripper-status-text", size="sm", c="dimmed"),
                html.Div(id="gripper-device-info"),
                dmc.Stack(
                    [
                        dmc.Text("Finger orientation", fw=600, size="sm"),
                        dmc.SegmentedControl(
                            id="gripper-orientation",
                            data=[
                                {"label": "Inward", "value": "inward"},
                                {"label": "Outward", "value": "outward"},
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
                dmc.Text("Calibration & settings", fw=600),
                dmc.Grid(
                    [
                        dmc.GridCol(
                            dmc.NumberInput(
                                id="settings-marker-a",
                                label="Marker ID A",
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
                                label="Marker ID B",
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
                                label="Marker size (mm)",
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
                                label="Calibration bundle path",
                                value=str(DEFAULT_BUNDLE_PATH),
                                size="sm",
                            ),
                            span=12,
                        ),
                        dmc.GridCol(
                            dmc.TextInput(
                                id="settings-workspace-path",
                                label="Workspace path",
                                value=str(DEFAULT_WORKSPACE_PATH),
                                size="sm",
                            ),
                            span=12,
                        ),
                        dmc.GridCol(
                            dmc.TextInput(
                                id="settings-robot-calib-path",
                                label="Robot calibration path",
                                value=str(DEFAULT_ROBOT_CALIBRATION_PATH),
                                size="sm",
                            ),
                            span=12,
                        ),
                    ],
                    gutter="sm",
                ),
                dmc.Text("Auto pick & place defaults", fw=600),
                dmc.Grid(
                    [
                        dmc.GridCol(
                            dmc.TextInput(
                                id="settings-auto-target-label",
                                label="Target label",
                                value=SETTINGS.auto_target_label,
                                size="sm",
                                description="YOLO label the auto routine should pick.",
                            ),
                            span=12,
                        ),
                        dmc.GridCol(
                            dmc.NumberInput(
                                id="settings-auto-drop-ignore-radius",
                                label="Drop ignore radius (mm)",
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
                                label="Intrusion match (mm)",
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
                                label="Detection stale (s)",
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
                                label="Idle sleep (s)",
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
                                label="Estimate width",
                                checked=SETTINGS.auto_estimate_width,
                                size="sm",
                            ),
                            span=4,
                        ),
                        dmc.GridCol(
                            dmc.NumberInput(
                                id="settings-auto-approach-height",
                                label="Approach height (mm)",
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
                                label="Pick height (mm)",
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
                                label="Drop height (mm)",
                                value=SETTINGS.auto_drop_mm,
                                min=-50,
                                max=200,
                                step=1,
                                size="sm",
                            ),
                            span=4,
                        ),
                    ],
                    gutter="sm",
                ),
                dmc.Text("Pick follow tuning", fw=600),
                dmc.Grid(
                    [
                        dmc.GridCol(
                            dmc.NumberInput(
                                id="settings-pick-follow-timeout",
                                label="Timeout (s)",
                                description="Max follow loop duration.",
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
                                label="Update interval (s)",
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
                                label="Segment hold (s)",
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
                                label="Settle time (s)",
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
                                label="Tolerance (mm)",
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
                dmc.Text("Safety guards", fw=600),
                dmc.Grid(
                    [
                        dmc.GridCol(
                            dmc.NumberInput(
                                id="settings-rescue-jog-hold",
                                label="Rescue jog hold (s)",
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
                                label="Coord5 max Z (mm)",
                                value=SETTINGS.coord5_max_z_mm,
                                description="Positive Z moves downward; clamp to safe range.",
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
                        dmc.Button("Save settings",
                                   id="settings-save", color="teal"),
                        dmc.Button("Reset to defaults", id="settings-reset",
                                   variant="outline", color="red"),
                        dmc.Text(id="settings-status", size="sm", c="dimmed"),
                    ],
                    justify="space-between",
                    align="center",
                ),
                dmc.Group(
                    [
                        dmc.Button("Close", id="setup-close",
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
        "fontFamily": "Space Grotesk, IBM Plex Sans, sans-serif",
        "headings": {"fontFamily": "Space Grotesk, IBM Plex Sans, sans-serif"},
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
                dcc.Store(id="camera-connection-state"),
                dcc.Store(id="robot-connection-state"),
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
                        "drop_ignore_radius_mm": SETTINGS.auto_drop_ignore_radius_mm,
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
                        "camera_calibrated": False,
                        "workspace_defined": False,
                        "robot_calibrated": False,
                        "model_loaded": False,
                        "gripper_ready": False,
                        "drop_target_defined": False,
                    },
                ),
                dcc.Store(
                    id="footer-visibility",
                    data={"collapsed": False},
                ),
                dcc.Store(
                    id="robot-speed-state",
                    data={"value": 25},
                ),
                dcc.Store(
                    id="robot-power-state",
                    data={"value": None, "timestamp": 0.0},
                ),
                dcc.Interval(id="auto-init", interval=400,
                             n_intervals=0, max_intervals=1),
                dcc.Interval(id="robot-status-interval",
                             interval=400, n_intervals=0),
                dcc.Interval(
                    id="monitor-interval",
                    interval=1000,
                    n_intervals=0,
                ),
                dcc.Interval(
                    id="camera-interval",
                    interval=33,  # ~30 FPS UI poll; stream push happens each tick
                    n_intervals=0,
                    disabled=True,
                ),
                dcc.Interval(id="calibration-progress-interval",
                             interval=100, n_intervals=0),
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
                    padding="md",
                    className="app-shell",
                    header={"height": 90},
                    footer={"height": 340},
                    children=[
                        build_header(),
                        dmc.AppShellMain(
                            dmc.Container(
                                dmc.Grid(
                                    [
                                        dmc.GridCol(
                                            html.Div(
                                                dmc.Tabs(
                                                    [
                                                        dmc.TabsList(
                                                            [
                                                                dmc.TabsTab(
                                                                    "Applications", value="applications"),
                                                                dmc.TabsTab(
                                                                    "Monitoring", value="monitoring"),
                                                                dmc.TabsTab(
                                                                    "Calibration", value="calibration"),
                                                            ],
                                                            style={
                                                                "width": "100%",
                                                                "justifyContent": "flex-start",
                                                                "marginTop": "0px",
                                                                "marginBottom": "12px",
                                                            },
                                                        ),
                                                        dmc.TabsPanel(
                                                            build_application_tab(), value="applications"),
                                                        dmc.TabsPanel(
                                                            build_monitoring_tab(), value="monitoring"),
                                                        dmc.TabsPanel(
                                                            build_calibration_tab(), value="calibration"),
                                                    ],
                                                    id="main-tabs",
                                                    value="applications",
                                                    variant="pills",
                                                    color="brand",
                                                    className="main-tabs",
                                                    style={
                                                        "width": "100%",
                                                        "marginBottom": "6px",
                                                    },
                                                ),
                                                className="tabs-scroll-area",
                                            ),
                                            span={"base": 12, "md": 4},
                                            className="tabs-column",
                                        ),
                                        dmc.GridCol(
                                            dmc.Stack(
                                                [build_camera_panel()], gap="md"),
                                            span={"base": 12, "md": 8},
                                            className="camera-column",
                                        ),
                                    ],
                                    gutter="md",
                                    className="layout-grid",
                                ),
                                fluid=True,
                                px=0,
                                mx=0,
                                mt="sm",
                                className="main-panel",
                            ),
                            id="app-main",
                            className="app-main footer-expanded",
                            style={"overflow": "hidden"},
                        ),
                        build_footer(),
                    ],
                ),
            ]
        ),
    )
