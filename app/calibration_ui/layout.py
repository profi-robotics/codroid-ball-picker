"""Helper module for building the Dash layout components."""

from __future__ import annotations

from dash import dcc, html
import dash_mantine_components as dmc

from .state import SESSION, SETTINGS
from .constants import (
    DEFAULT_RESOLUTION,
    DEFAULT_FPS,
    DEFAULT_BUNDLE_PATH,
    DEFAULT_WORKSPACE_PATH,
    DEFAULT_ROBOT_CALIBRATION_PATH,
    DEFAULT_ROBOT_URI,
    DEFAULT_GRIPPER_IP,
    GRIPPER_FORCE_MIN,
    GRIPPER_FORCE_MAX,
    GRIPPER_FORCE_DEFAULT,
    GRIPPER_SPEED_MIN,
    PLATE_TEST_GRIP_TIMEOUT_S,
)
from .ui_helpers import (
    _encode_frame,
    _format_device_list,
    _model_options,
    _placeholder_frame,
    format_gpu_status,
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

    stop_column = dmc.Stack(
        [
            dmc.Text("Emergency stop", fw=700, size="sm", c="red"),
            stop_button,
            power_toggle,
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

    footer_header = html.Div(
        [
            dmc.Group(
                [stop_column, auto_control_column],
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


def build_setup_drawer() -> dmc.Drawer:
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
                    value=DEFAULT_ROBOT_URI,
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
                    value=DEFAULT_GRIPPER_IP,
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
                dmc.Group(
                    [
                        dmc.Button("Save settings", id="settings-save", color="teal"),
                        dmc.Button("Reset to defaults", id="settings-reset", variant="outline", color="red"),
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


def build_stepper_card() -> dmc.Card:
    return dmc.Card(
        [
            dmc.Group(
                [
                    dmc.Text("Calibration Steps", fw=600, size="lg"),
                    dmc.Group(
                        [
                            dmc.Button("Back", id="step-prev",
                                       variant="light", size="sm"),
                            dmc.Button("Next", id="step-next", size="sm"),
                        ],
                        gap="xs",
                    ),
                ],
                justify="space-between",
                align="center",
            ),
            dmc.Divider(),
            dmc.Stepper(
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
                                                    dmc.Text(
                                                        "Markers detected", size="sm", fw=600),
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
                                    style={"display": "none"},
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
                                dmc.Text(id="current-robot-position",
                                         size="xs", c="dimmed"),
                                dmc.Text(id="plate-action-status",
                                         size="sm", c="dimmed"),
                                dmc.Text(id="plate-metrics", size="sm"),
                                dmc.Text(
                                    "Step-by-step calibration guide:\n"
                                    "1. Start the camera stream and confirm the plate markers are visible (>= 2)\n"
                                    "2. Keep the plate steady and click 'Begin calibrating robot'\n"
                                    "3. Follow the modal instructions for points 1, 2, and 3\n"
                                    "4. Press flange button 1 at each indentation\n"
                                    "5. Run the verification move once prompted",
                                    size="xs",
                                    c="dimmed",
                                    style={"whiteSpace": "pre-line"},
                                ),
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
            ),
        ],
        withBorder=True,
        radius="lg",
        className="panel-card",
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
                        "height": "560px",
                        "objectFit": "contain",
                        "borderRadius": "16px",
                        "border": "1px solid rgba(240, 65, 0, 0.25)",
                        "marginTop": "8px",
                        "backgroundColor": "#0b0f14",
                    },
                ),
                style={"position": "relative"}
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
                ],
                gutter="sm",
            ),
            dmc.Group(
                [
                    dmc.Button("Save settings", id="settings-save", color="teal"),
                    dmc.Button("Reset to defaults", id="settings-reset", variant="outline", color="red"),
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

    model_card = dmc.Card(
        [
            dmc.Text("Model Loader", fw=600, size="lg"),
            dmc.Text(
                "Upload a YOLO model (.pt/.onnx) to enable detection.", size="sm", c="dimmed"),
            dcc.Upload(
                id="model-upload",
                children=dmc.Button("Upload YOLO model", variant="light"),
                multiple=False,
            ),
            dmc.Select(
                id="model-select",
                label="Active model",
                data=_model_options(),
                value=str(
                    SESSION.active_model_path) if SESSION.active_model_path else None,
                searchable=True,
                clearable=True,
            ),
            dmc.Button("Load selected model",
                       id="load-model", variant="outline"),
            dmc.Text(id="model-status", size="sm", c="dimmed"),
        ],
        withBorder=True,
        radius="lg",
        className="panel-card",
    )

    detection_card = dmc.Card(
        [
            dmc.Text("Detection Controls", fw=600, size="lg"),
            dmc.Switch(
                id="continuous-detection-toggle",
                label="Continuous detection",
                checked=False,
                color="teal",
                size="md",
            ),
            dmc.Group(
                [
                    dmc.NumberInput(
                        id="detection-confidence",
                        label="Confidence",
                        value=0.7,
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
            dmc.Button("Run detection once",
                       id="run-detection", variant="light"),
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
    target_card = dmc.Card(
        [
            dmc.Text("Pick Targets", fw=600, size="lg"),
            dmc.Select(
                id="pick-detection-select",
                label="Detection to pick",
                data=[],
                placeholder="Run detection first",
                clearable=True,
            ),
            dmc.Group(
                [
                    dmc.NumberInput(
                        id="pick-approach-height",
                        label="Approach height (mm)",
                        value=30.0,
                        min=0,
                        max=200,
                    ),
                    dmc.NumberInput(
                        id="pick-height",
                        label="Pick height (mm)",
                        value=0.0,
                        min=-50,
                        max=200,
                    ),
                ],
                grow=True,
            ),
        ],
        withBorder=True,
        radius="lg",
        className="panel-card",
    )

    drop_card = dmc.Card(
        [
            dmc.Text("Drop Target", fw=600, size="lg"),
            dmc.Button(
                "Teach drop position (use current robot pose)",
                id="teach-drop-position",
                variant="light",
            ),
            dmc.Text(id="drop-position-metrics", size="sm"),
            dmc.NumberInput(
                id="drop-height",
                label="Drop height offset (mm)",
                value=0.0,
                min=-50,
                max=200,
            ),
        ],
        withBorder=True,
        radius="lg",
        className="panel-card",
    )

    gripper_card = dmc.Card(
        [
            dmc.Text("Gripper Control", fw=600, size="lg"),
            dmc.Switch(
                id="estimate-object-width-toggle",
                label="Estimate object width from detection",
                checked=False,
                color="teal",
                size="md",
            ),
            dmc.Group(
                [
                    dmc.Button("Open gripper", id="gripper-open",
                               variant="light"),
                    dmc.Button("Close gripper", id="gripper-close",
                               variant="light"),
                ],
                grow=True,
            ),
            dmc.Text(id="holding-status", size="sm", c="dimmed"),
        ],
        withBorder=True,
        radius="lg",
        className="panel-card",
    )

    action_card = dmc.Card(
        [
            dmc.Text("Pick/Drop Actions", fw=600, size="lg"),
            dmc.Group(
                [
                    dmc.Button("Pick", id="execute-pick"),
                    dmc.Button("Drop", id="execute-drop", variant="light"),
                ],
                grow=True,
            ),
            dmc.Button("Pick & Drop", id="execute-pick-drop",
                       variant="outline"),
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
                        value=30.0,
                        min=0,
                        max=200,
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
                        value=0.0,
                        min=-50,
                        max=200,
                    ),
                    dmc.NumberInput(
                        id="plate-test-grip-timeout",
                        label="Grip timeout (s)",
                        value=PLATE_TEST_GRIP_TIMEOUT_S,
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

    manual_card = dmc.Card(
        [
            dmc.Text("Manual Control", fw=600, size="lg"),
            dmc.Group(
                [
                    dmc.Button(
                        "Pick closest object",
                        id="manual-pick-closest",
                        variant="light",
                    ),
                    dmc.Button(
                        "Release object",
                        id="manual-release",
                        variant="outline",
                    ),
                    dmc.Button(
                        "Pick closest & place at target",
                        id="manual-pick-place-target",
                        variant="filled",
                        color="teal",
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
                dmc.AccordionControl("Acquire target"),
                dmc.AccordionPanel(target_card),
            ],
            value="acquire-target",
        ),
        dmc.AccordionItem(
            [
                dmc.AccordionControl("Grab target"),
                dmc.AccordionPanel(gripper_card),
            ],
            value="grab-target",
        ),
        dmc.AccordionItem(
            [
                dmc.AccordionControl("Acquire drop location"),
                dmc.AccordionPanel(drop_card),
            ],
            value="drop-target",
        ),
        dmc.AccordionItem(
            [
                dmc.AccordionControl("Release target"),
                dmc.AccordionPanel(
                    dmc.Text("Configure release options (TBD).",
                             size="sm", c="dimmed")
                ),
            ],
            value="release-target",
        ),
        dmc.AccordionItem(
            [
                dmc.AccordionControl("Manual control"),
                dmc.AccordionPanel(manual_card),
            ],
            value="manual",
        ),
    ]

    return dmc.Stack(
        [
            dmc.Accordion(
                accordion_items,
                multiple=True,
                value=["detect"],
                variant="separated",
            ),
            action_card,
            plate_test_card,
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
                value="workspace",
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


def build_layout() -> dmc.MantineProvider:
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
                    },
                ),
                dcc.Store(
                    id="auto-operation-state",
                    data={"running": False, "message": ""},
                ),
                dcc.Store(
                    id="prerequisites-state",
                    data={
                        "camera_calibrated": False,
                        "workspace_defined": False,
                        "robot_calibrated": False,
                        "model_loaded": False,
                        "gripper_ready": False,
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
                                            dmc.Tabs(
                                                [
                                                    dmc.TabsList(
                                                        [
                                                            dmc.TabsTab(
                                                                "Calibration", value="calibration"),
                                                            dmc.TabsTab(
                                                                "Applications", value="applications"),
                                                            dmc.TabsTab(
                                                                "Monitoring", value="monitoring"),
                                                        ],
                                                        style={
                                                            "width": "100%",
                                                            "justifyContent": "flex-start",
                                                            "marginTop": "0px",
                                                            "marginBottom": "12px",
                                                        },
                                                    ),
                                                    dmc.TabsPanel(
                                                        build_calibration_tab(), value="calibration"),
                                                    dmc.TabsPanel(
                                                        build_application_tab(), value="applications"),
                                                    dmc.TabsPanel(
                                                        build_monitoring_tab(), value="monitoring"),
                                                ],
                                                id="main-tabs",
                                                value="calibration",
                                                variant="pills",
                                                color="brand",
                                                className="main-tabs",
                                                style={"width": "100%",
                                                       "marginBottom": "6px"},
                                            ),
                                            span={"base": 12, "md": 4},
                                        ),
                                        dmc.GridCol(
                                            dmc.Stack(
                                                [build_camera_panel()], gap="md"),
                                            span={"base": 12, "md": 8},
                                        ),
                                    ],
                                    gutter="md",
                                ),
                                fluid=True,
                                px=0,
                                mx=0,
                                mt="sm",
                                className="main-panel",
                            ),
                            id="app-main",
                            className="app-main footer-expanded",
                        ),
                        build_footer(),
                    ],
                ),
            ]
        ),
    )

