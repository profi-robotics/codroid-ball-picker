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
from app.codroid_picker.i18n import current_locale, t

__all__ = [
    "build_header",
    "build_footer",
]


# =============================================================================
# UI Component Factories
# =============================================================================


def _create_slider_with_controls(
    slider_id: str,
    min_val: float,
    max_val: float,
    step: float,
    default_val: float,
    decrease_id: str,
    increase_id: str,
) -> dmc.Group:
    """Create a slider with decrease/increase button controls.

    This consolidates the repeated slider+controls pattern used for
    gripper width, force, and speed controls.

    Args:
        slider_id: ID for the slider component
        min_val: Minimum slider value
        max_val: Maximum slider value
        step: Slider step size
        default_val: Default slider value
        decrease_id: ID for the decrease button
        increase_id: ID for the increase button

    Returns:
        A dmc.Group containing the slider and control buttons
    """
    slider = dmc.Slider(
        id=slider_id,
        min=min_val,
        max=max_val,
        step=step,
        value=default_val,
        labelAlwaysOn=False,
        size="xs",
        marks=[],
        style={"flexGrow": 1, "minWidth": "140px", "maxWidth": "180px"},
    )
    return dmc.Group(
        [
            dmc.ActionIcon("-", id=decrease_id,
                           variant="light", color="gray", radius="xl", size="sm"),
            slider,
            dmc.ActionIcon("+", id=increase_id,
                           variant="light", color="gray", radius="xl", size="sm"),
        ],
        gap="xs",
        align="center",
        justify="flex-start",
        className="footer-slider-controls",
    )


def _create_control_row(
    label: str,
    controls: dmc.Group,
) -> html.Div:
    """Create a labeled control row for the footer.

    Args:
        label: Label text for the control row
        controls: The controls group to display

    Returns:
        A div containing the label and controls
    """
    return html.Div(
        [
            dmc.Text(label, size="xs", fw=600),
            controls,
        ],
        className="footer-fine-row",
    )


def build_header() -> dmc.AppShellHeader:
    """Build the application header with branding and connection badges."""
    branding = dmc.Text(t("header.title"), className="app-title")
    badges = dmc.Group(
        [
            dmc.Badge(t("header.badge.camera_off"), id="header-camera-badge",
                      color="gray", variant="light", size="sm"),
            dmc.Badge(t("header.badge.robot_idle"), id="header-robot-badge",
                      color="gray", variant="light", size="sm"),
            dmc.Badge(t("header.badge.gripper_off"), id="header-gripper-badge",
                      color="gray", variant="light", size="sm"),
        ],
        className="header-badges",
    )
    language_select = dmc.Select(
        id="language-select-header",
        size="xs",
        w=140,
        value=current_locale(),
        clearable=False,
        allowDeselect=False,
        data=[
            {"value": "en", "label": t("language.english")},
            {"value": "uk", "label": t("language.ukrainian")},
        ],
        placeholder=t("language.label"),
    )
    actions = dmc.Group(
        [
            language_select,
            badges,
            dmc.Button(t("header.setup"), id="open-setup", variant="filled",
                       color="brand", radius="md", size="xs"),
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
                            w=44,
                            h=44,
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
        px="md",
        h=64,
    )


def _build_stop_column() -> dmc.Stack:
    """Build the emergency stop and power controls column."""
    stop_button = dmc.Button(
        children="STOP",
        id="safety-stop",
        color="red",
        size="md",
        radius="md",
        variant="filled",
        fullWidth=False,
        style={
            "fontWeight": 900,
            "letterSpacing": "0.08em",
            "padding": "12px 18px",
            "minWidth": "132px",
            "fontSize": "17px",
            "height": "52px",
            "backgroundImage": "repeating-linear-gradient(45deg, #ff4747 0, #ff4747 10px, #ffd447 10px, #ffd447 20px)",
            "color": "#000000",  # Pure black for maximum contrast
            "border": "2px solid #8b0000",
            "textShadow": "0 0 2px rgba(255,255,255,0.5)",  # Subtle text outline for visibility
        },
    )
    power_toggle = dmc.Switch(
        id="robot-power-toggle",
        label=t("Robot power"),
        checked=False,
        color="red",
        size="xs",
    )
    return dmc.Stack(
        [
            dmc.Text(t("Emergency stop"), fw=700, size="xs", c="red"),
            stop_button,
            power_toggle,
        ],
        gap=2,
        align="center",
    )


def _build_auto_control_column() -> dmc.Stack:
    """Build the auto pick & place control column.

    Note: Pause/Resume functionality is now handled by the program control buttons.
    """
    auto_start_button = dmc.Button(
        t("Start Auto"),
        id="auto-start",
        color="brand",
        size="xs",
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
            "minWidth": "86px",
        },
    )
    auto_stop_button = dmc.Button(
        t("Stop Auto"),
        id="auto-stop",
        color="red",
        size="xs",
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
            "minWidth": "86px",
        },
    )
    prereq_summary = dmc.Stack(
        [],
        id="auto-prereq-badges",
        gap=2,
        align="center",
    )
    return dmc.Stack(
        [
            dmc.Text(t("Auto pick & place"), fw=700, size="xs", c="blue"),
            dmc.Group(
                [auto_start_button, auto_stop_button],
                gap="xs",
            ),
            prereq_summary,
        ],
        gap=3,
        align="center",
    )


def _build_program_control_column() -> dmc.Stack:
    """Build the program control column with pause/stop/reset."""
    program_pause_button = dmc.Button(
        t("Pause"),
        id="program-pause",
        color="yellow",
        size="xs",
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
            "minWidth": "72px",
        },
    )
    program_stop_button = dmc.Button(
        t("Stop"),
        id="program-stop",
        color="red",
        size="xs",
        radius="md",
        variant="outline",
        disabled=True,
        style={
            "fontWeight": 700,
            "minWidth": "64px",
        },
    )
    program_reset_button = dmc.Button(
        t("Reset"),
        id="program-reset",
        color="gray",
        size="xs",
        radius="md",
        variant="light",
        disabled=True,
        style={
            "fontWeight": 700,
            "minWidth": "64px",
        },
    )
    return dmc.Stack(
        [
            dmc.Text(t("Program control"), fw=700, size="xs", c="orange"),
            dmc.Group(
                [program_pause_button, program_stop_button, program_reset_button],
                gap="xs",
            ),
            dmc.Text(id="program-status-text", size="xs", c="dimmed"),
        ],
        gap=3,
        align="center",
    )


def _build_robot_speed_block() -> dmc.Stack:
    """Build the compact robot speed slider for the visible footer."""
    robot_speed_slider = dmc.Slider(
        id="robot-speed-slider",
        min=0,
        max=100,
        step=5,
        value=65,
        labelAlwaysOn=False,
        size="xs",
        marks=[],
        style={"flexGrow": 1, "minWidth": "140px", "maxWidth": "180px"},
    )
    robot_speed_controls = dmc.Group(
        [
            dmc.ActionIcon(
                "-",
                id="robot-speed-decrease",
                variant="filled",
                color="gray",
                radius="xl",
                size="sm",
            ),
            robot_speed_slider,
            dmc.ActionIcon(
                "+",
                id="robot-speed-increase",
                variant="filled",
                color="gray",
                radius="xl",
                size="sm",
            ),
        ],
        gap="xs",
        align="center",
        justify="flex-start",
        style={"flexGrow": 1},
    )

    return dmc.Stack(
        [
            dmc.Text(t("Robot speed"), fw=600, size="xs"),
            robot_speed_controls,
        ],
        gap=3,
        align="flex-start",
        className="footer-speed-block",
    )


def _build_rescue_mode_block() -> dmc.Stack:
    """Build rescue mode controls for the expandable footer details."""
    rescue_button = dmc.Button(
        t("Enable rescue mode"),
        id="rescue-mode-enable",
        color="orange",
        variant="outline",
        size="xs",
        radius="md",
    )
    rescue_exit_button = dmc.Button(
        t("Exit rescue mode"),
        id="rescue-mode-exit",
        color="orange",
        variant="light",
        size="xs",
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
            dmc.Text(t("Rescue mode"), fw=600, size="xs", c="orange"),
            dmc.Group([rescue_button, rescue_exit_button], gap="xs"),
            rescue_status,
        ],
        gap=4,
        align="flex-start",
    )


def _build_auto_motion_test_block() -> dmc.Stack:
    """Build the explicit socket motion smoke-test control."""
    auto_motion_test_button = dmc.Button(
        t("Motion Test"),
        id="auto-motion-test-start",
        color="indigo",
        size="xs",
        radius="md",
        variant="light",
        disabled=True,
        leftSection=dmc.ActionIcon(
            "↕",
            size="sm",
            variant="transparent",
            color="indigo",
        ),
        style={
            "fontWeight": 700,
            "minWidth": "96px",
        },
    )
    return dmc.Stack(
        [
            dmc.Text(t("Auto socket test"), fw=600, size="xs", c="indigo"),
            dmc.Group([auto_motion_test_button], gap="xs"),
            dmc.Text(
                t("Small vertical socket move at the current robot pose."),
                size="xs",
                c="dimmed",
            ),
        ],
        gap=4,
        align="flex-start",
    )


def _build_gripper_width_control() -> dmc.Group:
    """Build the gripper width control for the footer header (most frequently used)."""
    width_controls = _create_slider_with_controls(
        slider_id="footer-gripper-width",
        min_val=0,
        max_val=40,
        step=0.5,
        default_val=10.0,
        decrease_id="footer-gripper-width-decrease",
        increase_id="footer-gripper-width-increase",
    )
    return dmc.Group(
        [
            dmc.Stack(
                [
                    dmc.Text(t("Gripper width"), fw=700, size="xs", c="teal"),
                    dmc.Text(id="footer-gripper-status", size="xs", c="dimmed"),
                ],
                gap=1,
                align="flex-start",
                className="footer-gripper-label",
            ),
            width_controls,
        ],
        gap="xs",
        align="center",
        wrap="nowrap",
        className="footer-gripper-width-control",
    )


def _build_gripper_advanced_block() -> dmc.Stack:
    """Build the gripper advanced controls block (Force & Speed) for footer body."""
    force_controls = _create_slider_with_controls(
        slider_id="footer-gripper-force",
        min_val=GRIPPER_FORCE_MIN,
        max_val=GRIPPER_FORCE_MAX,
        step=5,
        default_val=GRIPPER_FORCE_DEFAULT,
        decrease_id="footer-gripper-force-decrease",
        increase_id="footer-gripper-force-increase",
    )
    speed_controls = _create_slider_with_controls(
        slider_id="footer-gripper-speed",
        min_val=10,
        max_val=100,
        step=5,
        default_val=60,
        decrease_id="footer-gripper-speed-decrease",
        increase_id="footer-gripper-speed-increase",
    )
    return dmc.Stack(
        [
            dmc.Group(
                [
                    dmc.Text(t("Gripper advanced"), fw=600, size="xs"),
                    dmc.Text(id="footer-gripper-meta", size="xs", c="dimmed"),
                ],
                gap="xs",
                align="center",
            ),
            dmc.Grid(
                [
                    dmc.GridCol(
                        _create_control_row(t("Force (N)"), force_controls),
                        span=6,
                    ),
                    dmc.GridCol(
                        _create_control_row(t("Speed (%)"), speed_controls),
                        span=6,
                    ),
                ],
                gutter="xs",
            ),
        ],
        gap=4,
        align="flex-start",
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
            dmc.Text(t("Rescue joint control"), fw=700, size="sm", c="orange"),
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
        size="xs",
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
            dmc.Text(t("Flange press"), fw=600, size="xs"),
            flange_indicators,
        ],
        className="footer-flange-block",
    )
    telemetry_block = dmc.Stack(
        [
            dmc.Text(t("Robot telemetry"), fw=600, size="xs"),
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
    gripper_width_control = _build_gripper_width_control()
    robot_speed_block = _build_robot_speed_block()
    gripper_advanced_block = _build_gripper_advanced_block()
    rescue_mode_block = _build_rescue_mode_block()
    auto_motion_test_block = _build_auto_motion_test_block()
    rescue_controls = _build_rescue_controls()
    status_group = _build_status_group()

    footer_toggle = dmc.Button(
        t("Show details"),
        id="footer-toggle",
        variant="light",
        color="gray",
        size="xs",
        radius="xl",
    )

    footer_grid = html.Div(
        [
            gripper_advanced_block,
            auto_motion_test_block,
            rescue_mode_block,
            rescue_controls,
        ],
        className="footer-body-grid",
    )

    footer_header = html.Div(
        [
            html.Div(stop_column, className="footer-stop-area"),
            html.Div(auto_control_column, className="footer-auto-area"),
            html.Div(program_control_column, className="footer-program-area"),
            html.Div(gripper_width_control, className="footer-gripper-area"),
            html.Div(robot_speed_block, className="footer-speed-area"),
            dmc.Group(
                [status_group, footer_toggle],
                gap="xs",
                align="center",
                wrap="wrap",
                className="footer-status-actions",
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
                padding="xs",
            ),
            fluid=True,
            px="sm",
            py=4,
        ),
        id="app-footer",
        className="app-footer is-collapsed",
    )
