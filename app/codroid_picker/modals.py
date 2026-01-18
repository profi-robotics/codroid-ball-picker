"""
Modal dialog components for calibration application.

This module contains all modal dialog builders for error handling,
robot calibration workflows, and user interactions.
"""

from __future__ import annotations

import dash_mantine_components as dmc


def _modal_shell(
    *,
    modal_id: str,
    title: str | None = None,
    children=None,
    size: str = "lg",
    overlay: dict | None = None,
    close_on_outside: bool = False,
) -> dmc.Modal:
    """Build a modal with common defaults used across the app."""
    return dmc.Modal(
        id=modal_id,
        opened=False,
        centered=True,
        size=size,
        title=title,
        withCloseButton=False,
        closeOnClickOutside=close_on_outside,
        overlayProps=overlay or {"opacity": 0.65, "color": "#000"},
        children=children,
    )


def build_robot_calibration_modal() -> dmc.Modal:
    """Build the robot calibration workflow modal."""
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
    """Build the emergency stop error modal."""
    return _modal_shell(
        modal_id="emergency-stop-modal",
        size="xl",
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
    """Build the overspeed error modal."""
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


# Helper for simple error modals that share the same structure but differ in strings/ids
def _build_simple_error_modal(
    modal_id: str,
    alert_id: str,
    clear_id: str,
    stay_id: str,
    title_text: str,
    subtitle_text: str,
    alert_title: str,
    alert_children: str,
    clear_label: str = "Clear",
    stay_label: str = "Stay paused",
    title_color: str = "orange",
) -> dmc.Modal:
    """Build a simple error modal with a title, subtitle, alert, and two buttons.

    This consolidates the repeated structure used for joint protection and drag warnings.
    """
    return dmc.Modal(
        id=modal_id,
        opened=False,
        centered=True,
        size="xl",
        withCloseButton=False,
        closeOnClickOutside=False,
        overlayProps={"opacity": 0.65, "color": "#000"},
        children=dmc.Stack(
            [
                dmc.Text(
                    title_text,
                    fw=700,
                    size="xl",
                    c=title_color,
                ),
                dmc.Text(
                    subtitle_text,
                    size="sm",
                    c="dimmed",
                ),
                dmc.Alert(
                    id=alert_id,
                    title=alert_title,
                    color=title_color,
                    variant="light",
                    children=alert_children,
                ),
                dmc.Group(
                    [
                        dmc.Button(
                            clear_label,
                            id=clear_id,
                            color=title_color,
                            size="md",
                        ),
                        dmc.Button(
                            stay_label,
                            id=stay_id,
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


def build_joint_protection_modal() -> dmc.Modal:
    """Build the joint protection error modal."""
    return _build_simple_error_modal(
        modal_id="joint-protection-modal",
        alert_id="joint-protection-modal-text",
        clear_id="joint-protection-clear",
        stay_id="joint-protection-stay-paused",
        title_text="Joint protection detected",
        subtitle_text="The program has been paused. Clear the protection alarm to continue.",
        alert_title="Joint protection status",
        alert_children="Joint protection warning detected.",
        title_color="orange",
    )


def build_wrong_tool_modal() -> dmc.Modal:
    """Build the wrong tool/payload error modal."""
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
    """Build the drag-not-allowed error modal."""
    return _build_simple_error_modal(
        modal_id="drag-not-allowed-modal",
        alert_id="drag-not-allowed-modal-text",
        clear_id="drag-not-allowed-clear",
        stay_id="drag-not-allowed-stay-paused",
        title_text="Drag not allowed",
        subtitle_text="Manual dragging is blocked. Clear the warning to continue.",
        alert_title="Drag warning status",
        alert_children="Drag not allowed warning detected.",
        title_color="orange",
    )
