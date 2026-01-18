"""Application layout components - header and footer.

This module contains the main application shell components:
- Header with branding and connection badges
- Footer with robot controls, telemetry, and gripper controls
"""

from __future__ import annotations

from dash import html
import dash_mantine_components as dmc

from app.codroid_picker.components.common import (
    GRIPPER_FORCE_MIN,
    GRIPPER_FORCE_MAX,
    GRIPPER_FORCE_DEFAULT,
)

__all__ = [
    "build_header",
    "build_footer",
]


def build_header() -> dmc.AppShellHeader:
    """Build the application header with branding and connection badges."""
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


def _build_stop_column() -> dmc.Stack:
    """Build the emergency stop and power controls column."""
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
    return dmc.Stack(
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


def _build_auto_control_column() -> dmc.Stack:
    """Build the auto pick & place control column."""
    auto_start_button = dmc.Button(
        "Start Auto",
        id="auto-start",
        color="brand",
        size="sm",
        radius="md",
        variant="filled",
        disabled=True,
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
        disabled=True,
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
    return dmc.Stack(
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


def _build_program_control_column() -> dmc.Stack:
    """Build the program control column with pause/stop/reset."""
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
    return dmc.Stack(
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


def _build_robot_speed_block() -> dmc.Stack:
    """Build the robot speed control block."""
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
    return dmc.Stack(
        [
            dmc.Text("Robot speed", fw=600, size="sm"),
            robot_speed_controls,
        ],
        gap=4,
        align="flex-start",
        style={"minWidth": "280px"},
    )


def _build_gripper_block() -> dmc.Stack:
    """Build the gripper control block with width/force/speed sliders."""
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
    return dmc.Stack(
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


def _build_rescue_controls() -> dmc.Stack:
    """Build the rescue mode joint control panel."""
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
    return dmc.Stack(
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


def _build_status_group() -> dmc.Group:
    """Build the telemetry and flange indicator status group."""
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
    telemetry_block = dmc.Stack(
        [
            dmc.Text("Robot telemetry", fw=600, size="xs"),
            robot_position_text,
        ],
        gap=2,
    )
    return dmc.Group(
        [telemetry_block, flange_block],
        gap="md",
        align="center",
        className="footer-status-group",
        wrap="wrap",
    )


def build_footer() -> dmc.AppShellFooter:
    """Build the application footer with robot controls and telemetry."""
    stop_column = _build_stop_column()
    auto_control_column = _build_auto_control_column()
    program_control_column = _build_program_control_column()
    robot_speed_block = _build_robot_speed_block()
    gripper_block = _build_gripper_block()
    rescue_controls = _build_rescue_controls()
    status_group = _build_status_group()

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
