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
from app.codroid_picker.i18n import t
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
                label=t("Tracking mode"),
                data=[
                    {"label": t("Move target (default)"), "value": "move"},
                    {"label": t("Segmented tracking"), "value": "segment"},
                ],
                value=SETTINGS.tracking_mode,
                clearable=False,
            ),
            dmc.NumberInput(
                id="shared-segment-min",
                label=t("Segment min (mm)"),
                description=t("Minimum step size for segmented tracking."),
                value=SETTINGS.segment_min,
                min=1.0,
                max=200.0,
                step=1.0,
            ),
            dmc.NumberInput(
                id="shared-segment-max",
                label=t("Segment max (mm)"),
                description=t("Maximum step size for segmented tracking."),
                value=SETTINGS.segment_max,
                min=5.0,
                max=500.0,
                step=5.0,
            ),
            dmc.NumberInput(
                id="shared-segment-fraction",
                label=t("Segment fraction"),
                description=t("Fraction of remaining XY distance each step; higher = fewer, longer segments."),
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
        t("GPU Status"),
        children=[dmc.Stack(id="gpu-status-display", gap="xs",
                      children=_format_gpu_status() if _format_gpu_status else [])],
    )

    model_options = _model_options_fn() if _model_options_fn else []
    has_models = len(model_options) > 0

    if has_models:
        model_select_placeholder = t("Select a model")
        upload_button_variant = "light"
    else:
        model_select_placeholder = t("No models available - upload a model first")
        upload_button_variant = "gradient"  # Highlighted

    model_card = dmc.Card(
        [
            dmc.Text(t("Model Loader"), fw=600, size="lg"),
            dmc.Text(
                t("Upload a YOLO model (.pt/.onnx) to enable detection.") if not has_models else t("Detection starts automatically when a model is loaded."), size="sm", c="dimmed"),
            dcc.Upload(
                id="model-upload",
                children=dmc.Button(t("Upload YOLO model"),
                                    variant=upload_button_variant),
                multiple=False,
            ),
            dmc.Select(
                id="model-select",
                label=t("Active model"),
                data=model_options,
                value=str(
                    SESSION.active_model_path) if SESSION.active_model_path else None,
                placeholder=model_select_placeholder,
                searchable=True,
                clearable=True,
                disabled=not has_models,
            ),
            dmc.Button(t("Load selected model"),
                       id="load-model", variant="outline", disabled=not has_models),
            dmc.Text(id="model-status", size="sm", c="dimmed"),
        ],
        withBorder=True,
        radius="lg",
        className="panel-card",
    )

    detection_card = dmc.Card(
        [
            dmc.Text(t("Detection Controls"), fw=600, size="lg"),
            dmc.Text(t("Detection runs automatically when a model is loaded."),
                     size="sm", c="dimmed", style={"marginBottom": "10px"}),
            dmc.Group(
                [
                    dmc.NumberInput(
                        id="detection-confidence",
                        label=t("Confidence"),
                        value=0.4,
                        min=0.0,
                        max=1.0,
                        step=0.05,
                    ),
                    dmc.NumberInput(
                        id="detection-iou",
                        label=t("IOU"),
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
                        label=t("Max detections"),
                        value=10,
                        min=1,
                        max=200,
                    ),
                    dmc.TextInput(
                        id="detection-classes",
                        label=t("Class filter (comma IDs)"),
                        placeholder=t("e.g. 0, 32"),
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
            dmc.Text(t("Detected Objects"), fw=600, size="lg"),
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
            dmc.Text(t("Drop Target"), fw=600, size="lg"),
            dmc.Text(t("First teach a drop position, then use return to go back to it."),
                     size="sm", c="dimmed"),
            dmc.Button(
                t("Teach drop position (use current robot pose)"),
                id="teach-drop-position",
                variant="filled",
                className="action-btn",
                color="green",
            ),
            dmc.Button(
                t("↩ Return to drop position"),
                id="return-to-drop-position",
                variant="outline",
                className="action-btn-outline",
            ),
            dmc.Text(id="drop-position-metrics", size="sm"),
            dmc.NumberInput(
                id="drop-height",
                label=t("Drop height offset (mm)"),
                value=10.0,
                min=-50,
                max=200,
                description=t("Height above taught drop position (positive = higher)"),
            ),
        ],
        withBorder=True,
        radius="lg",
        className="panel-card",
    )

    # Merged Pick/Drop Actions (combines manual control with action card)
    action_card = dmc.Card(
        [
            dmc.Text(t("Pick & Drop Controls"), fw=600, size="lg"),
            # Hidden select kept for callbacks that expect a pick target selector
            dmc.Select(
                id="pick-detection-select",
                label=t("Detection to pick"),
                data=[],
                value=None,
                placeholder=t("(hidden) Pick detection"),
                clearable=True,
                style={"display": "none"},
            ),
            dmc.Group(
                [
                    dmc.Button(t("Pick closest object"), id="manual-pick-closest",
                               variant="filled", color="teal"),
                    dmc.Button(t("Drop"), id="execute-drop", variant="light"),
                ],
                grow=True,
            ),
            dmc.Space(h="md"),  # Add space for visual separation
            dmc.Button(t("🔄 Pick & Drop closest"),
                       id="manual-pick-place-target",
                       variant="filled",
                       color="blue",
                       size="md"),
            dmc.Group([
                dmc.Button(t("Reset if stuck"),
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
            dmc.Text(t("Plate Pick & Place Test"), fw=600, size="lg"),
            dmc.Text(
                t("Uses calibration plate targets 1-3 and re-tracks the plate pose before running."),
                size="sm",
                c="dimmed",
            ),
            dmc.Group(
                [
                    dmc.NumberInput(
                        id="plate-test-approach-height",
                        label=t("Approach height (mm)"),
                        description=t("Negative = UP from table (safe in coord5)"),
                        value=-30.0,
                        min=-200,
                        max=0,
                    ),
                    dmc.NumberInput(
                        id="plate-test-pick-height",
                        label=t("Pick height (mm)"),
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
                        label=t("Drop height offset (mm)"),
                        value=10.0,
                        min=-50,
                        max=200,
                        description=t("Height above taught drop position"),
                    ),
                    dmc.NumberInput(
                        id="plate-test-grip-timeout",
                        label=t("Grip timeout (s)"),
                        value=SETTINGS.plate_test_grip_timeout_s,
                        min=0.5,
                        max=10.0,
                        step=0.5,
                    ),
                ],
                grow=True,
            ),
            dmc.Text(
                t("Gripper close uses {force} N and {speed}% speed.", force=GRIPPER_FORCE_MIN, speed=GRIPPER_SPEED_MIN),
                size="xs",
                c="dimmed",
            ),
            dmc.Button(
                t("Run plate test"),
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
            dmc.Text(t("Auto pick & place"), fw=600, size="lg"),
            dmc.Text(
                t("Auto picks the nearest target and pauses if another object enters the workspace."),
                size="sm",
                c="dimmed",
            ),
            dmc.Group(
                [
                    dmc.TextInput(
                        id="auto-target-class",
                        label=t("Target class"),
                        value=SETTINGS.auto_target_label,
                        description=t("YOLO label to pick (case-insensitive)."),
                    ),
                    dmc.NumberInput(
                        id="auto-drop-ignore-radius",
                        label=t("Drop ignore radius (mm)"),
                        value=SETTINGS.auto_drop_ignore_radius_mm,
                        min=0,
                        max=200,
                        step=5,
                        description=t("Detections inside this radius around the drop zone are ignored."),
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
                dmc.AccordionControl(t("Detect")),
                dmc.AccordionPanel(build_detection_tab()),
            ],
            value="detect",
        ),
        dmc.AccordionItem(
            [
                dmc.AccordionControl(t("Drop Location")),
                dmc.AccordionPanel(drop_card),
            ],
            value="drop-target",
        ),
        dmc.AccordionItem(
            [
                dmc.AccordionControl(t("Auto pick & place")),
                dmc.AccordionPanel(auto_card),
            ],
            value="auto",
        ),
        dmc.AccordionItem(
            [
                dmc.AccordionControl(t("Plate Pick & Place Test")),
                dmc.AccordionPanel(plate_test_card),
            ],
            value="plate-test",
        ),
    ]

    tracking_controls_card = dmc.Card(
        [
            dmc.Text(t("Tracking Controls"), fw=600, size="lg"),
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
                            dmc.TabsTab(t("Pick & Place"), value="pick-place"),
                            dmc.TabsTab(t("Workspace"), value="workspace"),
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
            dmc.Text(t("System Summary"), fw=600, size="lg"),
            dmc.Stack(
                [
                    dmc.Text(t("Camera: off"), size="sm"),
                    dmc.Text(t("Robot: offline"), size="sm"),
                    dmc.Text(t("Gripper: offline"), size="sm"),
                    dmc.Text(t("Calibration bundle: missing"), size="sm"),
                    dmc.Text(t("Workspace: pending"), size="sm"),
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
            dmc.Text(t("Detection Summary"), fw=600, size="lg"),
            dmc.Stack(
                [
                    dmc.Text(t("Model: none"), size="sm"),
                    dmc.Text(t("Detections: 0"), size="sm"),
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
            dmc.Text(t("Pick & Drop Summary"), fw=600, size="lg"),
            dmc.Stack(
                [
                    dmc.Text(t("Holding object: no"), size="sm"),
                    dmc.Text(t("Last action: No recent actions."), size="sm"),
                    dmc.Text(t("Processed objects: 0"), size="sm"),
                    dmc.Text(t("Attempts: 0 · Failures: 0"), size="sm"),
                    dmc.Text(t("Success rate: n/a"), size="sm"),
                    dmc.Text(t("Avg cycle: n/a"), size="sm"),
                    dmc.Text(t("Last cycle: n/a"), size="sm"),
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
