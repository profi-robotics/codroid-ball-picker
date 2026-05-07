"""Calibration tab UI components.

This module contains component builders for the calibration workflow:
- Calibration stepper/wizard
- Camera panel with overlay controls
- Workspace step content
- Settings card
"""

from __future__ import annotations

import dash_mantine_components as dmc
from dash import html

from app.codroid_picker.components.common import _encode_frame, placeholder_frame
from app.codroid_picker.i18n import t
from app.codroid_picker.session import SESSION
from app.settings import (
    DEFAULT_BUNDLE_PATH,
    DEFAULT_ROBOT_CALIBRATION_PATH,
    DEFAULT_WORKSPACE_PATH,
    GRIPPER_PROFILE_OPTIONS,
    SETTINGS,
)

__all__ = [
    "build_calibration_tab",
    "build_stepper_card",
    "build_workspace_step_content",
    "build_camera_panel",
    "build_settings_card",
]


def _calibration_stepper_stack() -> dmc.Stepper:
    """Build the calibration wizard stepper stack."""
    return dmc.Stepper(
        id="calibration-stepper",
        active=0,
        orientation="vertical",
        mt="sm",
        children=[
            dmc.StepperStep(
                label=t("Camera calibration"),
                description=t("Intrinsics + workspace plane"),
                children=dmc.Stack(
                    [
                        dmc.Text(
                            t("Capture the ChArUco board frames to calibrate the camera."),
                            size="sm",
                            c="dimmed",
                        ),
                        dmc.Card(
                            dmc.Group(
                                [
                                    dmc.ThemeIcon(
                                        dmc.Text("✕", size="lg"),
                                        size="lg",
                                        radius="md",
                                        color="red",
                                        id="markers-detected-icon",
                                    ),
                                    dmc.Stack(
                                        [
                                            dmc.Text(t("Markers detected"),
                                                     size="sm", fw=600),
                                            dmc.Text(
                                                t("Position the ChArUco board in view"),
                                                size="xs",
                                                c="dimmed",
                                            ),
                                        ],
                                        gap="xs",
                                    ),
                                ],
                                gap="md",
                            ),
                            p="md",
                            withBorder=True,
                        ),
                        dmc.Button(
                            t("Calibrate"),
                            id="calibrate-camera",
                            fullWidth=True,
                            disabled=True,
                            color="teal",
                        ),
                        dmc.Button(
                            t("Use saved calibration"),
                            id="use-existing-calibration",
                            fullWidth=True,
                            variant="outline",
                            disabled=True,
                        ),
                        dmc.Box(
                            id="camera-calibration-action",
                            style={"minHeight": "0px"},
                        ),
                        dmc.Text(
                            id="calibration-progress-text",
                            size="sm",
                            c="dimmed",
                            style={"display": "none",
                                   "marginBottom": "8px"},
                        ),
                        dmc.Progress(
                            id="calibration-progress",
                            value=0,
                            size="sm",
                            striped=True,
                            animated=True,
                            style={"display": "none"},
                        ),
                        dmc.Text(
                            id="camera-calibration-metrics", size="sm"),
                    ],
                    gap="md",
                ),
            ),
            dmc.StepperStep(
                label=t("Robot calibration"),
                description=t("Detect plate markers"),
                children=dmc.Stack(
                    [
                        dmc.Text(
                            t("Detect the calibration plate and attach the homography."),
                            size="sm",
                            c="dimmed",
                        ),
                        dmc.Alert(
                            id="plate-detection-summary",
                            title=t("Plate visibility"),
                            color="yellow",
                            children=t("Waiting for plate markers..."),
                            variant="light",
                        ),
                        dmc.Button(
                            t("Begin calibrating robot"),
                            id="start-robot-calibration",
                            variant="filled",
                            color="teal",
                            disabled=True,
                        ),
                        dmc.Button(
                            t("Use saved calibration"),
                            id="use-existing-robot-calibration",
                            fullWidth=True,
                            variant="outline",
                            color="brand",
                            disabled=True,
                        ),
                        dmc.Button(
                            t("Run verification move"),
                            id="run-verification-inline",
                            fullWidth=True,
                            variant="outline",
                            color="teal",
                            disabled=True,
                        ),
                        dmc.Text(id="current-robot-position",
                                 size="xs", c="dimmed"),
                        dmc.Text(id="plate-action-status",
                                 size="sm", c="dimmed"),
                        dmc.Text(id="plate-metrics", size="sm"),
                        dmc.Text(
                            id="robot-calibration-status",
                            size="sm",
                            c="teal",
                        ),
                    ],
                    gap="xs",
                ),
            ),
        ],
    )


def _calibration_loaded_body() -> dmc.Stack:
    """Build the 'calibration loaded' view when calibration data exists."""
    return dmc.Stack(
        [
            dmc.Text(t("Calibration data loaded"), fw=600, size="lg"),
            dmc.Text(
                t("Camera and robot calibration data already exist. Reset calibration to start over."),
                size="sm",
                c="dimmed",
            ),
            dmc.Text(
                t("Loading new calibration artifacts will delete the existing bundle and plate data."),
                size="xs",
                c="dimmed",
            ),
            dmc.Button(
                t("Reset calibration"),
                id="calibration-reset-button",
                variant="outline",
                color="red",
            ),
            html.Div(id="calibration-metadata"),
        ],
        gap="sm",
    )


def build_stepper_card() -> dmc.Card:
    """Build the calibration stepper/wizard card."""
    stepper_visible = SESSION.calibration_menu_open
    stepper_style = {} if stepper_visible else {"display": "none"}
    loaded_style = {} if not stepper_visible else {"display": "none"}
    nav_style = stepper_style
    return dmc.Card(
        [
            dmc.Group(
                [
                    dmc.Text(t("Calibration Steps"), fw=600, size="lg"),
                    dmc.Group(
                        [
                            dmc.Button(t("Back"), id="step-prev",
                                       variant="light", size="sm"),
                            dmc.Button(t("Next"), id="step-next",
                                       size="sm"),
                        ],
                        gap="xs",
                        id="calibration-nav-actions",
                        style=nav_style,
                    ),
                ],
                justify="space-between",
                align="center",
            ),
            dmc.Divider(),
            html.Div(
                id="calibration-stepper-body",
                children=_calibration_stepper_stack(),
                style=stepper_style,
            ),
            html.Div(
                id="calibration-loaded-body",
                children=_calibration_loaded_body(),
                style=loaded_style,
            ),
        ],
        withBorder=True,
        radius="lg",
        className="panel-card",
    )


def build_workspace_step_content() -> dmc.Stack:
    """Build workspace calibration step content."""
    return dmc.Stack(
        [
            dmc.Text(
                t("Use the two corner markers to define the workarea size."),
                size="sm",
                c="dimmed",
            ),
            dmc.Button(
                t("Compute workspace"), id="compute-workspace", variant="light"),
            dmc.Button(
                t("Use saved workspace"),
                id="use-existing-workspace",
                variant="outline",
                style={"display": "none"},
            ),
            dmc.Button(
                t("Clear workspace"),
                id="clear-workspace",
                variant="outline",
                color="red",
                size="xs",
            ),
            dmc.Text(id="workspace-action-status", size="sm", c="dimmed"),
            dmc.Text(id="workspace-metrics", size="sm"),
        ],
        gap="xs",
    )


def build_camera_panel(title: str | None = None) -> dmc.Card:
    """Build the camera feed panel with overlay controls."""
    placeholder_src = _encode_frame(
        placeholder_frame(t("Camera stream offline")))
    panel_title = title or t("Camera View")
    return dmc.Card(
        [
            dmc.Group(
                [
                    dmc.Text(panel_title, fw=600, size="lg"),
                    dmc.Group(
                        [
                            dmc.SegmentedControl(
                                id="overlay-mode-select",
                                data=[
                                    {"label": t("Auto"), "value": "auto"},
                                    {"label": t("Calib"), "value": "calibration"},
                                    {"label": t("Workspace"), "value": "workspace"},
                                    {"label": t("Plate"), "value": "plate"},
                                    {"label": t("Detections"), "value": "detection"},
                                    {"label": t("Auto Plan"), "value": "auto-plan"},
                                    {"label": t("Off"), "value": "none"},
                                ],
                                value="auto",
                                size="xs",
                                color="brand",
                                radius="md",
                            ),
                            dmc.ActionIcon(
                                html.Span(
                                    "⛶", style={"fontSize": "1.2rem"}, title=t("Fullscreen")),
                                id="fullscreen-btn",
                                variant="light",
                            ),
                        ],
                        gap="sm",
                    ),
                ],
                justify="space-between",
                align="center",
            ),
            dmc.Text(
                t("Overlay: waiting for camera frames..."),
                id="overlay-status",
                size="sm",
                c="dimmed",
            ),
            dmc.Text(
                t("Camera: -- FPS"),
                id="camera-fps",
                size="sm",
                c="dark",
                fw=500,
            ),
            html.Div(
                id="camera-feed-container",
                children=[],  # Will be populated by callback based on camera connection state
                style={"position": "relative",
                       "height": "100%", "overflow": "hidden"}
            ),
        ],
        withBorder=True,
        radius="lg",
        padding="sm",
        className="panel-card camera-panel",
    )


def build_settings_card() -> dmc.Card:
    """Build the settings configuration card."""
    return dmc.Card(
        [
            dmc.Group(
                [
                    dmc.Text(t("Settings"), fw=600, size="lg"),
                    dmc.Text(id="settings-status", size="sm", c="dimmed"),
                ],
                justify="space-between",
                align="center",
            ),
            dmc.Divider(),
            dmc.Grid(
                [
                    dmc.GridCol(
                        dmc.TextInput(
                            id="settings-robot-host",
                            label=t("Robot host"),
                            placeholder=t("192.168.101.100"),
                            value=SETTINGS.robot_host,
                            size="sm",
                        ),
                        span=3,
                    ),
                    dmc.GridCol(
                        dmc.NumberInput(
                            id="settings-robot-port",
                            label=t("Robot port"),
                            value=SETTINGS.robot_port,
                            size="sm",
                            min=1,
                        ),
                        span=3,
                    ),
                    dmc.GridCol(
                        dmc.NumberInput(
                            id="settings-user-port",
                            label=t("User WS port"),
                            value=SETTINGS.user_port,
                            size="sm",
                            min=1,
                        ),
                        span=3,
                    ),
                    dmc.GridCol(
                        dmc.TextInput(
                            id="settings-gripper-ip",
                            label=t("Gripper IP"),
                            placeholder=t("192.168.101.95"),
                            value=SETTINGS.gripper_ip,
                            size="sm",
                        ),
                        span=3,
                    ),
                    dmc.GridCol(
                        dmc.Select(
                            id="settings-gripper-type",
                            label=t("Gripper type"),
                            data=GRIPPER_PROFILE_OPTIONS,
                            value=SETTINGS.gripper_type,
                            size="sm",
                            clearable=False,
                        ),
                        span=3,
                    ),
                    dmc.GridCol(
                        dmc.NumberInput(
                            id="settings-marker-a",
                            label=t("Marker ID A"),
                            value=SETTINGS.marker_ids[0],
                            min=0,
                            max=49,
                            size="sm",
                        ),
                        span=3,
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
                        span=3,
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
                        span=3,
                    ),
                    dmc.GridCol(
                        dmc.TextInput(
                            id="settings-calib-path",
                            label=t("Calibration bundle path"),
                            value=str(DEFAULT_BUNDLE_PATH),
                            size="sm",
                        ),
                        span=6,
                    ),
                    dmc.GridCol(
                        dmc.TextInput(
                            id="settings-workspace-path",
                            label=t("Workspace path"),
                            value=str(DEFAULT_WORKSPACE_PATH),
                            size="sm",
                        ),
                        span=6,
                    ),
                    dmc.GridCol(
                        dmc.TextInput(
                            id="settings-robot-calib-path",
                            label=t("Robot calibration path"),
                            value=str(DEFAULT_ROBOT_CALIBRATION_PATH),
                            size="sm",
                        ),
                        span=6,
                    ),
                    dmc.GridCol(
                        dmc.NumberInput(
                            id="settings-camera-calib-frames",
                            label=t("Camera calib frames"),
                            value=SETTINGS.camera_calibration_frame_count,
                            min=1,
                            max=120,
                            step=1,
                            size="sm",
                            description=t("Frames captured for each camera calibration run."),
                        ),
                        span=4,
                    ),
                    dmc.GridCol(
                        dmc.NumberInput(
                            id="settings-camera-calib-delay",
                            label=t("Frame delay (s)"),
                            value=SETTINGS.camera_calibration_frame_delay_s,
                            min=0.0,
                            max=1.0,
                            step=0.05,
                            size="sm",
                            description=t("Wait between captures to reduce motion blur."),
                        ),
                        span=4,
                    ),
                    dmc.GridCol(
                        dmc.NumberInput(
                            id="settings-camera-calib-min-usable",
                            label=t("Min usable frames"),
                            value=SETTINGS.camera_calibration_min_usable_frames,
                            min=1,
                            max=30,
                            step=1,
                            size="sm",
                            description=t("Calibration failures when usable frames below this."),
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
                ],
                justify="flex-end",
            ),
        ],
        withBorder=True,
        radius="lg",
        className="panel-card",
    )


def build_calibration_tab() -> dmc.Stack:
    """Build the calibration tab content."""
    return dmc.Stack([build_stepper_card()], gap="md")
