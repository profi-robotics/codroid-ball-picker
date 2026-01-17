"""Tab content builders for the calibration UI.

This module contains builders for the non-calibration tabs:
- Detection tab
- Pick & Place tab  
- Application/settings tab
- Monitoring tab
"""

from __future__ import annotations

from app.codroid_picker.components.calibration_tab import (
    build_settings_card,
)
from app.codroid_picker.components.common import dmc
from app.settings import SETTINGS

__all__ = [
    "build_detection_tab",
    "build_pick_place_tab",
    "build_application_tab",
    "build_monitoring_tab",
    "build_tracking_mode_controls",
]


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


def build_detection_tab() -> dmc.Stack:
    """Build the detection tab with ball detection controls."""
    return dmc.Stack(
        [
            dmc.Card(
                [
                    dmc.Group(
                        [
                            dmc.Text("Ball Detection", fw=600, size="lg"),
                            dmc.Text(id="detection-status",
                                     size="sm", c="dimmed"),
                        ],
                        justify="space-between",
                        align="center",
                    ),
                    dmc.Divider(),
                    dmc.Stack(
                        [
                            dmc.Text(
                                "Run YOLO detection model to identify ping-pong balls.",
                                size="sm",
                                c="dimmed",
                            ),
                            dmc.Group(
                                [
                                    dmc.Button(
                                        "Load model",
                                        id="load-model",
                                        variant="outline",
                                    ),
                                    dmc.Button(
                                        "Run inference",
                                        id="run-inference",
                                        color="teal",
                                        disabled=True,
                                    ),
                                ],
                                gap="sm",
                            ),
                            dmc.Alert(
                                id="model-load-status",
                                title="Model Status",
                                color="gray",
                                children="No model loaded",
                                variant="light",
                            ),
                            dmc.Text(id="inference-results", size="sm"),
                        ],
                        gap="sm",
                    ),
                ],
                withBorder=True,
                radius="lg",
                className="panel-card",
            )
        ],
        gap="md",
    )


def build_pick_place_tab() -> dmc.Stack:
    """Build the pick & place operations tab."""
    return dmc.Stack(
        [
            dmc.Card(
                [
                    dmc.Group(
                        [
                            dmc.Text("Pick & Place", fw=600, size="lg"),
                            dmc.Text(id="pickplace-status",
                                     size="sm", c="dimmed"),
                        ],
                        justify="space-between",
                        align="center",
                    ),
                    dmc.Divider(),
                    dmc.Stack(
                        [
                            dmc.Text(
                                "Automated pick-and-place operations.",
                                size="sm",
                                c="dimmed",
                            ),
                            dmc.Group(
                                [
                                    dmc.Button(
                                        "Start cycle",
                                        id="start-pick-cycle",
                                        color="teal",
                                    ),
                                    dmc.Button(
                                        "Stop",
                                        id="stop-pick-cycle",
                                        color="red",
                                        variant="outline",
                                    ),
                                ],
                                gap="sm",
                            ),
                            build_tracking_mode_controls(),
                            dmc.NumberInput(
                                id="pick-height-offset",
                                label="Pick height offset (mm)",
                                description="Z-offset above detected ball position",
                                value=SETTINGS.pick_height_offset_mm,
                                min=-50.0,
                                max=50.0,
                                step=1.0,
                            ),
                            dmc.NumberInput(
                                id="place-height-offset",
                                label="Place height offset (mm)",
                                description="Z-offset for drop target",
                                value=SETTINGS.place_height_offset_mm,
                                min=-50.0,
                                max=50.0,
                                step=1.0,
                            ),
                            dmc.Text(id="cycle-metrics", size="sm"),
                        ],
                        gap="sm",
                    ),
                ],
                withBorder=True,
                radius="lg",
                className="panel-card",
            )
        ],
        gap="md",
    )


def build_application_tab() -> dmc.Stack:
    """Build the application/configuration tab."""
    return dmc.Stack([build_settings_card()], gap="md")


def build_monitoring_tab() -> dmc.Stack:
    """Build the monitoring/diagnostics tab."""
    return dmc.Stack(
        [
            dmc.Card(
                [
                    dmc.Text("System Monitoring", fw=600, size="lg"),
                    dmc.Divider(),
                    dmc.Stack(
                        [
                            dmc.Text("System diagnostics and performance metrics.",
                                     size="sm", c="dimmed"),
                            dmc.Text(id="system-stats", size="sm"),
                            dmc.Text(id="robot-stats", size="sm"),
                            dmc.Text(id="gripper-stats", size="sm"),
                        ],
                        gap="sm",
                    ),
                ],
                withBorder=True,
                radius="lg",
                className="panel-card",
            )
        ],
        gap="md",
    )
