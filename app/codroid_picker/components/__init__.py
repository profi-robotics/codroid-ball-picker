"""
UI builder components for the Calibration UI.

This module contains all the build_* functions that construct the UI layout.
"""

from __future__ import annotations

import dash_mantine_components as dmc
from dash import dcc, html

import app.settings as app_settings
from app.codroid_picker.components.common import _encode_frame, _placeholder_frame
from app.codroid_picker.session import SESSION

# Re-export settings constants used by builders
SETTINGS = app_settings.SETTINGS
DEFAULT_RESOLUTION = app_settings.DEFAULT_RESOLUTION
DEFAULT_FPS = app_settings.DEFAULT_FPS
DEFAULT_BUNDLE_PATH = app_settings.DEFAULT_BUNDLE_PATH
DEFAULT_WORKSPACE_PATH = app_settings.DEFAULT_WORKSPACE_PATH
DEFAULT_ROBOT_CALIBRATION_PATH = app_settings.DEFAULT_ROBOT_CALIBRATION_PATH
DEFAULT_ROBOT_URI = app_settings.DEFAULT_ROBOT_URI
DEFAULT_GRIPPER_IP = app_settings.DEFAULT_GRIPPER_IP
GRIPPER_PROFILE_OPTIONS = app_settings.GRIPPER_PROFILE_OPTIONS

# Gripper constants - refreshed dynamically
GRIPPER_PROFILE = SETTINGS.gripper_profile
GRIPPER_FORCE_MIN = GRIPPER_PROFILE.force_min
GRIPPER_FORCE_MAX = GRIPPER_PROFILE.force_max
GRIPPER_FORCE_DEFAULT = GRIPPER_PROFILE.force_default
GRIPPER_SPEED_MIN = GRIPPER_PROFILE.speed_min

# Import from common module for backward compatibility


# These will be set by main.py after import
format_gpu_status = None
_model_options = None


def set_builder_dependencies(gpu_status_fn, model_options_fn):
    """Set external dependencies for builders that need them."""
    global format_gpu_status, _model_options
    format_gpu_status = gpu_status_fn
    _model_options = model_options_fn


def build_tracking_mode_controls() -> dmc.Group:
    """Build shared tracking mode controls used in both plate test and dynamic pick & place."""
    return dmc.Group(
        [
            dmc.Select(
                id="shared-tracking-mode",
                label="Tracking mode",
                data=[
                    {"label": "Move target (default)", "value": "move"},
                    {"label": "Segmented tracking", "value": "segment"},
                ],
                value=SETTINGS.tracking_mode,
                clearable=False,
            ),
            dmc.NumberInput(
                id="shared-segment-min",
                label="Segment min (mm)",
                description="Minimum step size for segmented tracking.",
                value=SETTINGS.segment_min,
                min=1.0,
                max=200.0,
                step=1.0,
            ),
            dmc.NumberInput(
                id="shared-segment-max",
                label="Segment max (mm)",
                description="Maximum step size for segmented tracking.",
                value=SETTINGS.segment_max,
                min=5.0,
                max=500.0,
                step=5.0,
            ),
            dmc.NumberInput(
                id="shared-segment-fraction",
                label="Segment fraction",
                description="Fraction of remaining XY distance each step; higher = fewer, longer segments.",
                value=SETTINGS.segment_fraction,
                min=0.1,
                max=1.0,
                step=0.1,
            ),
        ],
        grow=True,
    )


def build_header() -> dmc.AppShellHeader:
    branding = dmc.Text("Estun Codroid Pick-and-Place", className="app-title")
    badges = dmc.Group(
        [
            dmc.Badge("Camera: off", id="header-camera-badge",
                      color="gray", variant="light"),
            dmc.Badge("Robot: idle", id="header-robot-badge",
                      color="gray", variant="light"),
            dmc.Badge("Gripper: off", id="header-gripper-badge",
                      color="gray", variant="light"),
        ],
        className="header-badges",
    )
    actions = dmc.Group(
        [
            badges,
            dmc.Button("Setup", id="open-setup", variant="filled",
                       color="brand", radius="xl", size="md"),
        ],
        className="header-actions",
        align="center",
    )
    return dmc.AppShellHeader(
        dmc.Group(
            [
                dmc.Group(
                    [
                        dmc.Image(
                            src="/assets/profi-robotics.webp",
                            alt="Profi Robotics",
                            w=64,
                            h=64,
                            className="header-logo",
                        ),
                        branding,
                    ],
                    align="center",
                    gap="sm",
                ),
                actions,
            ],
            justify="space-between",
            align="center",
            className="header-bar",
        ),
        className="app-header",
        px="lg",
        h=84,
    )


def build_footer() -> dmc.AppShellFooter:
    stop_button = dmc.Button(
        "STOP",
        id="safety-stop",
        color="red",
        size="md",
        radius="xl",
        variant="filled",
        fullWidth=False,
        style={
            "fontWeight": 800,
            "letterSpacing": "0.06em",
            "padding": "14px 26px",
            "minWidth": "210px",
            "fontSize": "16px",
            "backgroundImage": "repeating-linear-gradient(45deg, #ff4747 0, #ff4747 12px, #ffd447 12px, #ffd447 24px)",
            "color": "#1f1f1f",
            "border": "2px solid #8b0000",
        },
    )
    power_toggle = dmc.Switch(
        id="robot-power-toggle",
        label="Robot power",
        checked=False,
        color="red",
        size="md",
    )
    rescue_button = dmc.Button(
        "Enable rescue mode",
        id="rescue-mode-enable",
        color="orange",
        variant="outline",
        size="sm",
        radius="md",
    )
    rescue_exit_button = dmc.Button(
        "Exit rescue mode",
        id="rescue-mode-exit",
        color="orange",
        variant="light",
        size="sm",
        radius="md",
        style={"display": "none"},
    )
    rescue_status = dmc.Text(
        id="rescue-mode-status",
        size="xs",
        c="dimmed",
    )

    robot_speed_slider = dmc.Slider(
        id="robot-speed-slider",
        min=0,
        max=100,
        step=5,
        value=25,
        labelAlwaysOn=False,
        size="sm",
        marks=[
            {"value": 0, "label": "0"},
            {"value": 50, "label": "50"},
            {"value": 100, "label": "100"},
        ],
        style={"flexGrow": 1, "minWidth": "220px"},
    )

    robot_speed_controls = dmc.Group(
        [
            dmc.ActionIcon(
                "-",
                id="robot-speed-decrease",
                variant="filled",
                color="gray",
                radius="xl",
                size="md",
            ),
            robot_speed_slider,
            dmc.ActionIcon(
                "+",
                id="robot-speed-increase",
                variant="filled",
                color="gray",
                radius="xl",
                size="md",
            ),
        ],
        gap="sm",
        align="center",
        justify="flex-start",
        style={"flexGrow": 1},
    )

    auto_start_button = dmc.Button(
        "Start Auto",
        id="auto-start",
        color="brand",
        size="sm",
        radius="md",
        variant="filled",
        leftSection=dmc.ActionIcon(
            "▶",
            size="sm",
            variant="transparent",
            color="white",
        ),
        style={
            "fontWeight": 700,
            "minWidth": "110px",
        },
    )

    auto_stop_button = dmc.Button(
        "Stop Auto",
        id="auto-stop",
        color="orange",
        size="sm",
        radius="md",
        variant="filled",
        leftSection=dmc.ActionIcon(
            "■",
            size="sm",
            variant="transparent",
            color="white",
        ),
        style={
            "fontWeight": 700,
            "minWidth": "110px",
        },
    )

    program_pause_button = dmc.Button(
        "Pause",
        id="program-pause",
        color="yellow",
        size="sm",
        radius="md",
        variant="filled",
        disabled=True,
        leftSection=dmc.ActionIcon(
            "⏸",
            size="sm",
            variant="transparent",
            color="black",
        ),
        style={
            "fontWeight": 700,
            "minWidth": "110px",
        },
    )

    program_stop_button = dmc.Button(
        "Stop",
        id="program-stop",
        color="red",
        size="sm",
        radius="md",
        variant="outline",
        disabled=True,
        style={
            "fontWeight": 700,
            "minWidth": "90px",
        },
    )

    program_reset_button = dmc.Button(
        "Reset",
        id="program-reset",
        color="gray",
        size="sm",
        radius="md",
        variant="light",
        disabled=True,
        style={
            "fontWeight": 700,
            "minWidth": "90px",
        },
    )

    stop_column = dmc.Stack(
        [
            dmc.Text("Emergency stop", fw=700, size="sm", c="red"),
            stop_button,
            power_toggle,
            rescue_button,
            rescue_exit_button,
            rescue_status,
        ],
        gap=6,
        align="center",
    )

    auto_control_column = dmc.Stack(
        [
            dmc.Text("Auto pick & place", fw=700, size="sm", c="blue"),
            dmc.Group(
                [auto_start_button, auto_stop_button],
                gap="xs",
            ),
        ],
        gap=6,
        align="center",
    )

    program_control_column = dmc.Stack(
        [
            dmc.Text("Program control", fw=700, size="sm", c="orange"),
            dmc.Group(
                [program_pause_button, program_stop_button, program_reset_button],
                gap="xs",
            ),
            dmc.Text(id="program-status-text", size="xs", c="dimmed"),
        ],
        gap=4,
        align="center",
    )

    robot_speed_block = dmc.Stack(
        [
            dmc.Text("Robot speed", fw=600, size="sm"),
            robot_speed_controls,
        ],
        gap=4,
        align="flex-start",
        style={"minWidth": "280px"},
    )

    fine_width_slider = dmc.Slider(
        id="footer-gripper-width",
        min=0,
        max=40,
        step=0.5,
        value=10.0,
        labelAlwaysOn=False,
        size="sm",
        marks=[],
        style={"minWidth": "220px"},
    )

    fine_force_slider = dmc.Slider(
        id="footer-gripper-force",
        min=GRIPPER_FORCE_MIN,
        max=GRIPPER_FORCE_MAX,
        step=5,
        value=GRIPPER_FORCE_DEFAULT,
        labelAlwaysOn=False,
        size="sm",
        marks=[],
        style={"minWidth": "220px"},
    )

    fine_speed_slider = dmc.Slider(
        id="footer-gripper-speed",
        min=10,
        max=100,
        step=5,
        value=60,
        labelAlwaysOn=False,
        size="sm",
        marks=[],
        style={"minWidth": "220px"},
    )

    width_controls = dmc.Group(
        [
            dmc.ActionIcon("-", id="footer-gripper-width-decrease",
                           variant="light", color="gray", radius="xl", size="md"),
            fine_width_slider,
            dmc.ActionIcon("+", id="footer-gripper-width-increase",
                           variant="light", color="gray", radius="xl", size="md"),
        ],
        gap="xs",
        align="center",
        justify="flex-start",
        className="footer-slider-controls",
    )

    force_controls = dmc.Group(
        [
            dmc.ActionIcon("-", id="footer-gripper-force-decrease",
                           variant="light", color="gray", radius="xl", size="md"),
            fine_force_slider,
            dmc.ActionIcon("+", id="footer-gripper-force-increase",
                           variant="light", color="gray", radius="xl", size="md"),
        ],
        gap="xs",
        align="center",
        justify="flex-start",
        className="footer-slider-controls",
    )

    speed_controls = dmc.Group(
        [
            dmc.ActionIcon("-", id="footer-gripper-speed-decrease",
                           variant="light", color="gray", radius="xl", size="md"),
            fine_speed_slider,
            dmc.ActionIcon("+", id="footer-gripper-speed-increase",
                           variant="light", color="gray", radius="xl", size="md"),
        ],
        gap="xs",
        align="center",
        justify="flex-start",
        className="footer-slider-controls",
    )

    gripper_block = dmc.Stack(
        [
            dmc.Text("Gripper control", fw=600, size="sm"),
            dmc.Text(id="footer-gripper-status", size="sm", c="dimmed"),
            dmc.Text(id="footer-gripper-meta", size="xs", c="dimmed"),
            html.Div(
                [
                    dmc.Text("Width (mm)", size="xs", fw=600),
                    width_controls,
                ],
                className="footer-fine-row",
            ),
            html.Div(
                [
                    dmc.Text("Force (N)", size="xs", fw=600),
                    force_controls,
                ],
                className="footer-fine-row",
            ),
            html.Div(
                [
                    dmc.Text("Speed (%)", size="xs", fw=600),
                    speed_controls,
                ],
                className="footer-fine-row",
            ),
        ],
        gap=6,
        align="flex-start",
        style={"flexGrow": 1, "minWidth": "320px"},
    )

    robot_position_text = dmc.Text(
        id="footer-robot-position",
        size="sm",
        className="footer-robot-position",
    )

    flange_indicators = html.Div(
        [
            html.Div(id="footer-flange-indicator-1",
                     className="footer-flange-indicator"),
            html.Div(id="footer-flange-indicator-2",
                     className="footer-flange-indicator"),
            html.Div(id="footer-flange-indicator-3",
                     className="footer-flange-indicator"),
            html.Div(id="footer-flange-indicator-4",
                     className="footer-flange-indicator"),
        ],
        className="footer-flange-indicators",
    )

    flange_block = html.Div(
        [
            dmc.Text("Flange press", fw=600, size="sm"),
            flange_indicators,
        ],
        className="footer-flange-block",
    )

    footer_toggle = dmc.Button(
        "Hide footer",
        id="footer-toggle",
        variant="light",
        color="gray",
        size="sm",
        radius="xl",
    )

    footer_grid = html.Div(
        [robot_speed_block, gripper_block],
        className="footer-body-grid",
    )

    telemetry_block = dmc.Stack(
        [
            dmc.Text("Robot telemetry", fw=600, size="xs"),
            robot_position_text,
        ],
        gap=2,
    )

    status_group = dmc.Group(
        [telemetry_block, flange_block],
        gap="md",
        align="center",
        className="footer-status-group",
        wrap="wrap",
    )

    rescue_joint_rows = []
    for joint_idx in range(1, 7):
        rescue_joint_rows.append(
            dmc.Group(
                [
                    dmc.Text(f"J{joint_idx}", size="xs", fw=600),
                    dmc.Button(
                        "-",
                        id=f"rescue-joint-{joint_idx}-minus",
                        variant="outline",
                        color="gray",
                        size="xs",
                    ),
                    dmc.Button(
                        "+",
                        id=f"rescue-joint-{joint_idx}-plus",
                        variant="outline",
                        color="gray",
                        size="xs",
                    ),
                ],
                gap="xs",
                align="center",
            )
        )

    rescue_controls = dmc.Stack(
        [
            dmc.Text("Rescue joint control", fw=700, size="sm", c="orange"),
            dmc.Stack(rescue_joint_rows, gap=4),
            dmc.Text(id="rescue-jog-status", size="xs", c="dimmed"),
        ],
        id="robot-controls-rescue",
        gap=6,
        align="flex-start",
        style={"display": "none"},
    )

    standard_controls = dmc.Group(
        [auto_control_column, program_control_column],
        id="robot-controls-standard",
        gap="lg",
        align="center",
        wrap="wrap",
    )

    footer_header = html.Div(
        [
            dmc.Group(
                [stop_column, standard_controls, rescue_controls],
                gap="lg",
                align="center",
                wrap="wrap",
            ),
            dmc.Group(
                [status_group, footer_toggle],
                gap="md",
                align="center",
                wrap="wrap",
            ),
        ],
        className="footer-header",
    )

    footer_body = html.Div(
        footer_grid,
        className="footer-body",
        id="footer-body",
    )

    return dmc.AppShellFooter(
        dmc.Container(
            dmc.Card(
                [footer_header, footer_body],
                className="footer-panel",
                padding="sm",
            ),
            fluid=True,
            px="lg",
            py="sm",
        ),
        id="app-footer",
        className="app-footer",
    )


def build_robot_calibration_modal() -> dmc.Modal:
    return dmc.Modal(
        id="robot-calibration-modal",
        opened=False,
        title="Robot calibration guide",
        centered=True,
        size="lg",
        withCloseButton=False,
        closeOnClickOutside=False,
        children=dmc.Stack(
            [
                dmc.Stepper(
                    id="robot-calibration-modal-stepper",
                    active=0,
                    orientation="vertical",
                    children=[
                        dmc.StepperStep(
                            label="Point 1",
                            description="Indentation 1 (origin)",
                        ),
                        dmc.StepperStep(
                            label="Point 2",
                            description="Indentation 2 (+X)",
                        ),
                        dmc.StepperStep(
                            label="Point 3",
                            description="Indentation 3 (+X/+Y)",
                        ),
                        dmc.StepperStep(
                            label="Verification",
                            description="Safe motion check",
                        ),
                    ],
                ),
                dmc.Card(
                    id="robot-calibration-accuracy-panel",
                    style={"display": "none"},
                    withBorder=True,
                    radius="md",
                    padding="sm",
                    children=dmc.Stack(
                        [
                            dmc.Text(
                                "",
                                id="robot-calibration-accuracy-text",
                                size="sm",
                                c="dimmed",
                            ),
                            dmc.Group(
                                [
                                    dmc.Button(
                                        "Repeat calibration",
                                        id="robot-calibration-retry",
                                        variant="outline",
                                        color="red",
                                        size="xs",
                                        disabled=True,
                                    ),
                                    dmc.Button(
                                        "Proceed to verification",
                                        id="robot-calibration-continue",
                                        color="teal",
                                        size="xs",
                                        disabled=True,
                                    ),
                                ],
                                justify="space-between",
                            ),
                        ],
                        gap="sm",
                    ),
                ),
                dmc.Alert(
                    id="robot-calibration-modal-alert",
                    title="Calibration status",
                    color="blue",
                    variant="light",
                    children="Waiting for calibration to start.",
                ),
                dmc.Group(
                    [
                        dmc.Button(
                            "Run verification move",
                            id="robot-calibration-verify",
                            color="teal",
                            disabled=True,
                        ),
                        dmc.Button(
                            "Close",
                            id="robot-calibration-modal-close",
                            variant="outline",
                        ),
                    ],
                    justify="flex-end",
                ),
            ],
            gap="md",
        ),
    )


def build_emergency_stop_modal() -> dmc.Modal:
    return dmc.Modal(
        id="emergency-stop-modal",
        opened=False,
        centered=True,
        size="xl",
        withCloseButton=False,
        closeOnClickOutside=False,
        overlayProps={"opacity": 0.65, "color": "#000"},
        children=dmc.Stack(
            [
                dmc.Text(
                    "Emergency stop triggered",
                    fw=700,
                    size="xl",
                    c="red",
                ),
                dmc.Text(
                    "The current program has been paused. Clear the physical E-stop "
                    "and acknowledge to continue.",
                    size="sm",
                    c="dimmed",
                ),
                dmc.Alert(
                    id="emergency-stop-modal-text",
                    title="Emergency stop status",
                    color="red",
                    variant="light",
                    children="Emergency stop detected.",
                ),
                dmc.Group(
                    [
                        dmc.Button(
                            "Clear",
                            id="emergency-ack",
                            color="red",
                            size="md",
                            disabled=True,
                        )
                    ],
                    justify="flex-end",
                ),
            ],
            gap="md",
        ),
    )


def build_overspeed_modal() -> dmc.Modal:
    return dmc.Modal(
        id="overspeed-modal",
        opened=False,
        centered=True,
        size="xl",
        withCloseButton=False,
        closeOnClickOutside=False,
        overlayProps={"opacity": 0.65, "color": "#000"},
        children=dmc.Stack(
            [
                dmc.Text(
                    "Overspeed detected",
                    fw=700,
                    size="xl",
                    c="orange",
                ),
                dmc.Text(
                    "The program has been paused. Choose how to proceed.",
                    size="sm",
                    c="dimmed",
                ),
                dmc.Alert(
                    id="overspeed-modal-text",
                    title="Overspeed status",
                    color="orange",
                    variant="light",
                    children="Overspeed alarm detected.",
                ),
                dmc.Group(
                    [
                        dmc.Button(
                            "Clear",
                            id="overspeed-clear",
                            color="orange",
                            size="md",
                        ),
                        dmc.Button(
                            "Stay paused",
                            id="overspeed-stay-paused",
                            color="yellow",
                            size="md",
                            variant="outline",
                        ),
                        dmc.Button(
                            "Clear & resume",
                            id="overspeed-continue",
                            color="teal",
                            size="md",
                            variant="outline",
                        ),
                        dmc.Button(
                            "Cancel & reset",
                            id="overspeed-reset",
                            color="red",
                            size="md",
                            variant="outline",
                        ),
                    ],
                    justify="flex-end",
                ),
            ],
            gap="md",
        ),
    )


def build_joint_protection_modal() -> dmc.Modal:
    return dmc.Modal(
        id="joint-protection-modal",
        opened=False,
        centered=True,
        size="xl",
        withCloseButton=False,
        closeOnClickOutside=False,
        overlayProps={"opacity": 0.65, "color": "#000"},
        children=dmc.Stack(
            [
                dmc.Text(
                    "Joint protection detected",
                    fw=700,
                    size="xl",
                    c="orange",
                ),
                dmc.Text(
                    "The program has been paused. Clear the protection alarm to continue.",
                    size="sm",
                    c="dimmed",
                ),
                dmc.Alert(
                    id="joint-protection-modal-text",
                    title="Joint protection status",
                    color="orange",
                    variant="light",
                    children="Joint protection warning detected.",
                ),
                dmc.Group(
                    [
                        dmc.Button(
                            "Clear",
                            id="joint-protection-clear",
                            color="orange",
                            size="md",
                        ),
                        dmc.Button(
                            "Stay paused",
                            id="joint-protection-stay-paused",
                            color="gray",
                            size="md",
                            variant="outline",
                        ),
                    ],
                    justify="flex-end",
                ),
            ],
            gap="md",
        ),
    )


def build_wrong_tool_modal() -> dmc.Modal:
    return dmc.Modal(
        id="wrong-tool-modal",
        opened=False,
        centered=True,
        size="xl",
        withCloseButton=False,
        closeOnClickOutside=False,
        overlayProps={"opacity": 0.65, "color": "#000"},
        children=dmc.Stack(
            [
                dmc.Text(
                    "Tool/payload mismatch",
                    fw=700,
                    size="xl",
                    c="red",
                ),
                dmc.Text(
                    "The program has been paused. Clear the tool error or enter rescue mode.",
                    size="sm",
                    c="dimmed",
                ),
                dmc.Alert(
                    id="wrong-tool-modal-text",
                    title="Tool error status",
                    color="red",
                    variant="light",
                    children="Tool/payload mismatch detected.",
                ),
                dmc.Group(
                    [
                        dmc.Button(
                            "Enter rescue mode",
                            id="wrong-tool-rescue",
                            color="orange",
                            size="md",
                            variant="outline",
                        ),
                        dmc.Button(
                            "Clear",
                            id="wrong-tool-clear",
                            color="red",
                            size="md",
                        ),
                    ],
                    justify="flex-end",
                ),
            ],
            gap="md",
        ),
    )


def build_drag_not_allowed_modal() -> dmc.Modal:
    return dmc.Modal(
        id="drag-not-allowed-modal",
        opened=False,
        centered=True,
        size="xl",
        withCloseButton=False,
        closeOnClickOutside=False,
        overlayProps={"opacity": 0.65, "color": "#000"},
        children=dmc.Stack(
            [
                dmc.Text(
                    "Drag not allowed",
                    fw=700,
                    size="xl",
                    c="orange",
                ),
                dmc.Text(
                    "Manual dragging is blocked. Clear the warning to continue.",
                    size="sm",
                    c="dimmed",
                ),
                dmc.Alert(
                    id="drag-not-allowed-modal-text",
                    title="Drag warning status",
                    color="orange",
                    variant="light",
                    children="Drag not allowed warning detected.",
                ),
                dmc.Group(
                    [
                        dmc.Button(
                            "Clear",
                            id="drag-not-allowed-clear",
                            color="orange",
                            size="md",
                        ),
                        dmc.Button(
                            "Stay paused",
                            id="drag-not-allowed-stay-paused",
                            color="gray",
                            size="md",
                            variant="outline",
                        ),
                    ],
                    justify="flex-end",
                ),
            ],
            gap="md",
        ),
    )


def _calibration_stepper_stack() -> dmc.Stepper:
    return dmc.Stepper(
        id="calibration-stepper",
        active=0,
        orientation="vertical",
        mt="sm",
        children=[
            dmc.StepperStep(
                label="Camera calibration",
                description="Intrinsics + workspace plane",
                children=dmc.Stack(
                    [
                        dmc.Text(
                            "Capture the ChArUco board frames to calibrate the camera.",
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
                                            dmc.Text("Markers detected",
                                                     size="sm", fw=600),
                                            dmc.Text(
                                                "Position the ChArUco board in view",
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
                            "Calibrate",
                            id="calibrate-camera",
                            fullWidth=True,
                            disabled=True,
                            color="teal",
                        ),
                        dmc.Button(
                            "Use saved calibration",
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
                label="Robot calibration",
                description="Detect plate markers",
                children=dmc.Stack(
                    [
                        dmc.Text(
                            "Detect the calibration plate and attach the homography.",
                            size="sm",
                            c="dimmed",
                        ),
                        dmc.Alert(
                            id="plate-detection-summary",
                            title="Plate visibility",
                            color="yellow",
                            children="Waiting for plate markers...",
                            variant="light",
                        ),
                        dmc.Button(
                            "Begin calibrating robot",
                            id="start-robot-calibration",
                            variant="filled",
                            color="teal",
                            disabled=True,
                        ),
                        dmc.Button(
                            "Use saved calibration",
                            id="use-existing-robot-calibration",
                            fullWidth=True,
                            variant="outline",
                            color="brand",
                            disabled=True,
                        ),
                        dmc.Button(
                            "Run verification move",
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
    return dmc.Stack(
        [
            dmc.Text("Calibration data loaded", fw=600, size="lg"),
            dmc.Text(
                "Camera and robot calibration data already exist. Reset calibration to start over.",
                size="sm",
                c="dimmed",
            ),
            dmc.Text(
                "Loading new calibration artifacts will delete the existing bundle and plate data.",
                size="xs",
                c="dimmed",
            ),
            dmc.Button(
                "Reset calibration",
                id="calibration-reset-button",
                variant="outline",
                color="red",
            ),
            html.Div(id="calibration-metadata"),
        ],
        gap="sm",
    )


def build_workspace_step_content() -> dmc.Stack:
    return dmc.Stack(
        [
            dmc.Text(
                "Use the two corner markers to define the workarea size.",
                size="sm",
                c="dimmed",
            ),
            dmc.Button(
                "Compute workspace", id="compute-workspace", variant="light"),
            dmc.Button(
                "Use saved workspace",
                id="use-existing-workspace",
                variant="outline",
                style={"display": "none"},
            ),
            dmc.Button(
                "Clear workspace",
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


def build_camera_panel(title: str = "Camera View") -> dmc.Card:
    placeholder_src = _encode_frame(
        _placeholder_frame("Camera stream offline"))
    return dmc.Card(
        [
            dmc.Group(
                [
                    dmc.Text(title, fw=600, size="lg"),
                    dmc.Group(
                        [
                            dmc.SegmentedControl(
                                id="overlay-mode-select",
                                data=[
                                    {"label": "Auto", "value": "auto"},
                                    {"label": "Calib", "value": "calibration"},
                                    {"label": "Workspace", "value": "workspace"},
                                    {"label": "Plate", "value": "plate"},
                                    {"label": "Detections", "value": "detection"},
                                    {"label": "Off", "value": "none"},
                                ],
                                value="auto",
                                size="xs",
                                color="brand",
                                radius="md",
                            ),
                            dmc.ActionIcon(
                                html.Span(
                                    "⛶", style={"fontSize": "1.2rem"}, title="Fullscreen"),
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
            dmc.Text(id="overlay-status", size="sm", c="dimmed"),
            dmc.Text(id="camera-fps", size="sm", c="dimmed"),
            html.Div(
                id="camera-feed-container",
                children=html.Img(
                    id="camera-feed",
                    src="/video_feed",  # Use MJPEG stream to avoid per-frame HTTP polling
                    style={
                        "width": "100%",
                        "height": "auto",
                        "maxHeight": "100%",
                        "objectFit": "contain",
                        "borderRadius": "16px",
                        "border": "1px solid rgba(240, 65, 0, 0.25)",
                        "marginTop": "8px",
                        "backgroundColor": "#0b0f14",
                    },
                ),
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
    return dmc.Card(
        [
            dmc.Group(
                [
                    dmc.Text("Settings", fw=600, size="lg"),
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
                            label="Robot host",
                            placeholder="192.168.101.100",
                            value=SETTINGS.robot_host,
                            size="sm",
                        ),
                        span=3,
                    ),
                    dmc.GridCol(
                        dmc.NumberInput(
                            id="settings-robot-port",
                            label="Robot port",
                            value=SETTINGS.robot_port,
                            size="sm",
                            min=1,
                        ),
                        span=3,
                    ),
                    dmc.GridCol(
                        dmc.NumberInput(
                            id="settings-user-port",
                            label="User WS port",
                            value=SETTINGS.user_port,
                            size="sm",
                            min=1,
                        ),
                        span=3,
                    ),
                    dmc.GridCol(
                        dmc.TextInput(
                            id="settings-gripper-ip",
                            label="Gripper IP",
                            placeholder="192.168.101.180",
                            value=SETTINGS.gripper_ip,
                            size="sm",
                        ),
                        span=3,
                    ),
                    dmc.GridCol(
                        dmc.Select(
                            id="settings-gripper-type",
                            label="Gripper type",
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
                            label="Marker ID A",
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
                            label="Marker ID B",
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
                            label="Marker size (mm)",
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
                            label="Calibration bundle path",
                            value=str(DEFAULT_BUNDLE_PATH),
                            size="sm",
                        ),
                        span=6,
                    ),
                    dmc.GridCol(
                        dmc.TextInput(
                            id="settings-workspace-path",
                            label="Workspace path",
                            value=str(DEFAULT_WORKSPACE_PATH),
                            size="sm",
                        ),
                        span=6,
                    ),
                    dmc.GridCol(
                        dmc.TextInput(
                            id="settings-robot-calib-path",
                            label="Robot calibration path",
                            value=str(DEFAULT_ROBOT_CALIBRATION_PATH),
                            size="sm",
                        ),
                        span=6,
                    ),
                    dmc.GridCol(
                        dmc.NumberInput(
                            id="settings-camera-calib-frames",
                            label="Camera calib frames",
                            value=SETTINGS.camera_calibration_frame_count,
                            min=1,
                            max=120,
                            step=1,
                            size="sm",
                            description="Frames captured for each camera calibration run.",
                        ),
                        span=4,
                    ),
                    dmc.GridCol(
                        dmc.NumberInput(
                            id="settings-camera-calib-delay",
                            label="Frame delay (s)",
                            value=SETTINGS.camera_calibration_frame_delay_s,
                            min=0.0,
                            max=1.0,
                            step=0.05,
                            size="sm",
                            description="Wait between captures to reduce motion blur.",
                        ),
                        span=4,
                    ),
                    dmc.GridCol(
                        dmc.NumberInput(
                            id="settings-camera-calib-min-usable",
                            label="Min usable frames",
                            value=SETTINGS.camera_calibration_min_usable_frames,
                            min=1,
                            max=30,
                            step=1,
                            size="sm",
                            description="Calibration failures when usable frames below this.",
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
                ],
                justify="flex-end",
            ),
        ],
        withBorder=True,
        radius="lg",
        className="panel-card",
    )


def build_calibration_tab() -> dmc.Stack:
    return dmc.Stack([build_stepper_card()], gap="md")


def build_detection_tab() -> dmc.Stack:
    gpu_card = dmc.Card(
        [
            dmc.Text("GPU Status", fw=600, size="lg"),
            dmc.Stack(id="gpu-status-display", gap="xs",
                      children=format_gpu_status()),
        ],
        withBorder=True,
        radius="lg",
        className="panel-card",
    )

    model_options = _model_options()
    has_models = len(model_options) > 0

    if has_models:
        model_select_placeholder = "Select a model"
        upload_button_variant = "light"
    else:
        model_select_placeholder = "No models available - upload a model first"
        upload_button_variant = "gradient"  # Highlighted

    model_card = dmc.Card(
        [
            dmc.Text("Model Loader", fw=600, size="lg"),
            dmc.Text(
                "Upload a YOLO model (.pt/.onnx) to enable detection." if not has_models else "Detection starts automatically when a model is loaded.", size="sm", c="dimmed"),
            dcc.Upload(
                id="model-upload",
                children=dmc.Button("Upload YOLO model",
                                    variant=upload_button_variant),
                multiple=False,
            ),
            dmc.Select(
                id="model-select",
                label="Active model",
                data=model_options,
                value=str(
                    SESSION.active_model_path) if SESSION.active_model_path else None,
                placeholder=model_select_placeholder,
                searchable=True,
                clearable=True,
                disabled=not has_models,
            ),
            dmc.Button("Load selected model",
                       id="load-model", variant="outline", disabled=not has_models),
            dmc.Text(id="model-status", size="sm", c="dimmed"),
        ],
        withBorder=True,
        radius="lg",
        className="panel-card",
    )

    detection_card = dmc.Card(
        [
            dmc.Text("Detection Controls", fw=600, size="lg"),
            dmc.Text("Detection runs automatically when a model is loaded.",
                     size="sm", c="dimmed", style={"marginBottom": "10px"}),
            dmc.Group(
                [
                    dmc.NumberInput(
                        id="detection-confidence",
                        label="Confidence",
                        value=0.4,
                        min=0.0,
                        max=1.0,
                        step=0.05,
                    ),
                    dmc.NumberInput(
                        id="detection-iou",
                        label="IOU",
                        value=0.45,
                        min=0.0,
                        max=1.0,
                        step=0.05,
                    ),
                ],
                grow=True,
            ),
            dmc.Group(
                [
                    dmc.NumberInput(
                        id="detection-max",
                        label="Max detections",
                        value=10,
                        min=1,
                        max=200,
                    ),
                    dmc.TextInput(
                        id="detection-classes",
                        label="Class filter (comma IDs)",
                        placeholder="e.g. 0, 32",
                    ),
                ],
                grow=True,
            ),
            dmc.Text(id="detection-status", size="sm", c="dimmed"),
            dmc.Text(id="detection-metrics", size="sm"),
        ],
        withBorder=True,
        radius="lg",
        className="panel-card",
    )

    detections_card = dmc.Card(
        [
            dmc.Text("Detected Objects", fw=600, size="lg"),
            dmc.Stack(id="detection-list", gap="xs"),
        ],
        withBorder=True,
        radius="lg",
        className="panel-card",
    )

    return dmc.Stack([gpu_card, model_card, detection_card, detections_card], gap="md")


def build_pick_place_tab() -> dmc.Stack:
    # Improved drop location controls with better button styling
    drop_card = dmc.Card(
        [
            dmc.Text("Drop Target", fw=600, size="lg"),
            dmc.Text("First teach a drop position, then use return to go back to it.",
                     size="sm", c="dimmed"),
            dmc.Button(
                "Teach drop position (use current robot pose)",
                id="teach-drop-position",
                variant="filled",
                className="action-btn",
                color="green",
            ),
            dmc.Button(
                "↩ Return to drop position",
                id="return-to-drop-position",
                variant="outline",
                className="action-btn-outline",
            ),
            dmc.Text(id="drop-position-metrics", size="sm"),
            dmc.NumberInput(
                id="drop-height",
                label="Drop height offset (mm)",
                value=10.0,
                min=-50,
                max=200,
                description="Height above taught drop position (positive = higher)",
            ),
        ],
        withBorder=True,
        radius="lg",
        className="panel-card",
    )

    # Merged Pick/Drop Actions (combines manual control with action card)
    action_card = dmc.Card(
        [
            dmc.Text("Pick & Drop Controls", fw=600, size="lg"),
            build_tracking_mode_controls(),  # Add shared tracking mode controls
            # Hidden select kept for callbacks that expect a pick target selector
            dmc.Select(
                id="pick-detection-select",
                label="Detection to pick",
                data=[],
                value=None,
                placeholder="(hidden) Pick detection",
                clearable=True,
                style={"display": "none"},
            ),
            dmc.Group(
                [
                    dmc.Button("Pick closest object", id="manual-pick-closest",
                               variant="filled", color="teal"),
                    dmc.Button("Drop", id="execute-drop", variant="light"),
                ],
                grow=True,
            ),
            dmc.Space(h="md"),  # Add space for visual separation
            dmc.Button("🔄 Pick & Drop closest",
                       id="manual-pick-place-target",
                       variant="filled",
                       color="blue",
                       size="md"),
            dmc.Group([
                dmc.Button("Reset if stuck",
                           id="reset-operation-lock",
                           variant="subtle",
                           color="red",
                           size="xs"),
            ], justify="center", style={"marginTop": "8px"}),
            dmc.Text(id="pick-place-status", size="sm", c="dimmed"),
        ],
        withBorder=True,
        radius="lg",
        className="panel-card",
    )

    plate_test_card = dmc.Card(
        [
            dmc.Text("Plate Pick & Place Test", fw=600, size="lg"),
            dmc.Text(
                "Uses calibration plate targets 1-3 and re-tracks the plate pose before running.",
                size="sm",
                c="dimmed",
            ),
            build_tracking_mode_controls(),  # Use shared tracking controls
            dmc.Group(
                [
                    dmc.NumberInput(
                        id="plate-test-approach-height",
                        label="Approach height (mm)",
                        description="Negative = UP from table (safe in coord5)",
                        value=-30.0,
                        min=-200,
                        max=0,
                    ),
                    dmc.NumberInput(
                        id="plate-test-pick-height",
                        label="Pick height (mm)",
                        value=0.0,
                        min=-50,
                        max=200,
                    ),
                ],
                grow=True,
            ),
            dmc.Group(
                [
                    dmc.NumberInput(
                        id="plate-test-drop-height",
                        label="Drop height offset (mm)",
                        value=10.0,
                        min=-50,
                        max=200,
                        description="Height above taught drop position",
                    ),
                    dmc.NumberInput(
                        id="plate-test-grip-timeout",
                        label="Grip timeout (s)",
                        value=SETTINGS.plate_test_grip_timeout_s,
                        min=0.5,
                        max=10.0,
                        step=0.5,
                    ),
                ],
                grow=True,
            ),
            dmc.Text(
                f"Gripper close uses {GRIPPER_FORCE_MIN} N and {GRIPPER_SPEED_MIN}% speed.",
                size="xs",
                c="dimmed",
            ),
            dmc.Button(
                "Run plate test",
                id="plate-test-run",
                color="teal",
            ),
            dmc.Text(id="plate-test-status", size="sm", c="dimmed"),
            dcc.Textarea(
                id="plate-test-analysis",
                value="",
                readOnly=True,
                style={
                    "width": "100%",
                    "minHeight": "120px",
                    "marginTop": "6px",
                },
            ),
        ],
        withBorder=True,
        radius="lg",
        className="panel-card",
    )

    auto_card = dmc.Card(
        [
            dmc.Text("Auto pick & place", fw=600, size="lg"),
            dmc.Text(
                "Auto picks the nearest target and pauses if another object enters the workspace.",
                size="sm",
                c="dimmed",
            ),
            dmc.Group(
                [
                    dmc.TextInput(
                        id="auto-target-class",
                        label="Target class",
                        value=SETTINGS.auto_target_label,
                        description="YOLO label to pick (case-insensitive).",
                    ),
                    dmc.NumberInput(
                        id="auto-drop-ignore-radius",
                        label="Drop ignore radius (mm)",
                        value=SETTINGS.auto_drop_ignore_radius_mm,
                        min=0,
                        max=200,
                        step=5,
                        description="Detections inside this radius around the drop zone are ignored.",
                    ),
                ],
                grow=True,
            ),
        ],
        withBorder=True,
        radius="lg",
        className="panel-card",
    )

    accordion_items = [
        dmc.AccordionItem(
            [
                dmc.AccordionControl("Detect"),
                dmc.AccordionPanel(build_detection_tab()),
            ],
            value="detect",
        ),
        dmc.AccordionItem(
            [
                dmc.AccordionControl("Drop Location"),
                dmc.AccordionPanel(drop_card),
            ],
            value="drop-target",
        ),
        dmc.AccordionItem(
            [
                dmc.AccordionControl("Auto pick & place"),
                dmc.AccordionPanel(auto_card),
            ],
            value="auto",
        ),
        dmc.AccordionItem(
            [
                dmc.AccordionControl("Plate Pick & Place Test"),
                dmc.AccordionPanel(plate_test_card),
            ],
            value="plate-test",
        ),
    ]

    return dmc.Stack(
        [
            dmc.Accordion(
                accordion_items,
                multiple=True,
                value=[],
                variant="separated",
            ),
            action_card,  # Pick & Drop Controls (merged from manual control)
        ],
        gap="md",
    )


def build_application_tab() -> dmc.Stack:
    return dmc.Stack(
        [
            dmc.Tabs(
                [
                    dmc.TabsList(
                        [
                            dmc.TabsTab("Workspace", value="workspace"),
                            dmc.TabsTab("Pick & Place", value="pick-place"),
                        ],
                        style={
                            "width": "100%",
                            "justifyContent": "flex-start",
                            "marginBottom": "12px",
                        },
                    ),
                    dmc.TabsPanel(
                        build_workspace_step_content(), value="workspace"
                    ),
                    dmc.TabsPanel(
                        build_pick_place_tab(), value="pick-place"
                    ),
                ],
                id="application-tabs",
                value="pick-place",
                variant="pills",
                color="brand",
                style={"width": "100%"},
            )
        ],
        gap="md",
    )


def build_monitoring_tab() -> dmc.Stack:
    summary_card = dmc.Card(
        [
            dmc.Text("System Summary", fw=600, size="lg"),
            dmc.Stack(
                [
                    dmc.Text("Camera: off", size="sm"),
                    dmc.Text("Robot: offline", size="sm"),
                    dmc.Text("Gripper: offline", size="sm"),
                    dmc.Text("Calibration bundle: missing", size="sm"),
                    dmc.Text("Workspace: pending", size="sm"),
                ],
                id="monitor-system-summary",
                gap="xs",
            ),
        ],
        withBorder=True,
        radius="lg",
        className="panel-card",
    )
    detection_card = dmc.Card(
        [
            dmc.Text("Detection Summary", fw=600, size="lg"),
            dmc.Stack(
                [
                    dmc.Text("Model: none", size="sm"),
                    dmc.Text("Detections: 0", size="sm"),
                ],
                id="monitor-detection-summary",
                gap="xs",
            ),
        ],
        withBorder=True,
        radius="lg",
        className="panel-card",
    )
    pick_card = dmc.Card(
        [
            dmc.Text("Pick & Drop Summary", fw=600, size="lg"),
            dmc.Stack(
                [
                    dmc.Text("Holding object: no", size="sm"),
                    dmc.Text("Last action: No recent actions.", size="sm"),
                    dmc.Text("Processed objects: 0", size="sm"),
                    dmc.Text("Attempts: 0 · Failures: 0", size="sm"),
                    dmc.Text("Success rate: n/a", size="sm"),
                    dmc.Text("Avg cycle: n/a", size="sm"),
                    dmc.Text("Last cycle: n/a", size="sm"),
                ],
                id="monitor-pick-summary",
                gap="xs",
            ),
        ],
        withBorder=True,
        radius="lg",
        className="panel-card",
    )

    return dmc.Stack([summary_card, detection_card, pick_card], gap="md")
