"""Tab content builders for the calibration UI.

This module contains builders for the non-calibration tabs:
- Detection tab
- Pick & Place tab
- Application/settings tab
- Monitoring tab
"""

from __future__ import annotations

from typing import Callable, List, Optional

import dash_mantine_components as dmc
from dash import dcc, html

from app.codroid_picker.components.calibration_tab import (
    build_workspace_step_content,
)
from app.codroid_picker.session import SESSION
from app.settings import SETTINGS


def _panel_card(title: str, children, *, color: str = "dimmed") -> dmc.Card:
    """Create a standard panel card with a title and stacked body."""
    return dmc.Card(
        [
            dmc.Text(title, fw=600, size="lg", c=color),
            dmc.Stack(children=children, gap="xs"),
        ],
        withBorder=True,
        radius="lg",
        className="panel-card",
    )

# Gripper constants - refreshed dynamically
GRIPPER_PROFILE = SETTINGS.gripper_profile
GRIPPER_FORCE_MIN = GRIPPER_PROFILE.force_min
GRIPPER_SPEED_MIN = GRIPPER_PROFILE.speed_min

__all__ = [
    "build_detection_tab",
    "build_pick_place_tab",
    "build_application_tab",
    "build_monitoring_tab",
    "build_tracking_mode_controls",
    "set_builder_dependencies",
]


# These will be set by main.py after import via set_builder_dependencies
_format_gpu_status: Optional[Callable[[], List]] = None
_model_options_fn: Optional[Callable[[], List]] = None


def set_builder_dependencies(gpu_status_fn: Callable, model_options_fn: Callable) -> None:
    """Set external dependencies for builders that need them.

    Args:
        gpu_status_fn: Function that returns GPU status display elements.
        model_options_fn: Function that returns available model options.
    """
    global _format_gpu_status, _model_options_fn
    _format_gpu_status = gpu_status_fn
    _model_options_fn = model_options_fn


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
    """Build the detection tab with GPU status, model loader, and detection controls."""
    gpu_card = _panel_card(
        "GPU Status",
        children=[dmc.Stack(id="gpu-status-display", gap="xs",
                      children=_format_gpu_status() if _format_gpu_status else [])],
    )

    model_options = _model_options_fn() if _model_options_fn else []
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
    """Build the pick & place operations tab."""
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

    tracking_controls_card = dmc.Card(
        [
            dmc.Text("Tracking Controls", fw=600, size="lg"),
            build_tracking_mode_controls(),
        ],
        withBorder=True,
        radius="lg",
        className="panel-card",
    )

    return dmc.Stack(
        [
            tracking_controls_card,
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
    """Build the application/configuration tab."""
    return dmc.Stack(
        [
            dmc.Tabs(
                [
                    dmc.TabsList(
                        [
                            dmc.TabsTab("Pick & Place", value="pick-place"),
                            dmc.TabsTab("Workspace", value="workspace"),
                        ],
                        style={
                            "width": "100%",
                            "justifyContent": "flex-start",
                            "marginBottom": "12px",
                        },
                    ),
                    dmc.TabsPanel(
                        build_pick_place_tab(), value="pick-place"
                    ),
                    dmc.TabsPanel(
                        build_workspace_step_content(), value="workspace"
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
    """Build the monitoring/diagnostics tab."""
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
