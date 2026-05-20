# Codroid Vision-Guided Bootstrap Demo

This repository contains a hardware-agnostic scaffold for bringing up the Estun
Codroid + OAK-D pick-and-place pipeline. The focus is on repeatable calibration,
marker-driven workspace definition, coordinate mapping, and `codroid-api` for
robot motion commands.

## Layout
- `vision/` – camera calibration, marker detection, workspace mapping helpers.
- `robot/` – WebSocket client, gripper control (OnRobot 2FG7), and command catalog extracted from browser traffic.
- `app/` – orchestration logic that ties vision outputs to robot commands, now powering the Dash UI.
- `run_codroid_picker.py` – entry point for the Dash-based bring-up console (camera, calibration, workspace, and robots all flow through this script).
- `pyproject.toml` – project metadata and dependency manifest used by `uv`.
- `scripts/capture_calibration_data.py` – CLI helper that streams an OAK-D, saves ChArUco board frames, and records a depth snapshot.
- `docs/` – integration guides for gripper and other peripherals.

## Quickstart
1. Keep sibling checkouts of `../codroid-api` and `../onrobot-api` next to this
   repository, then install dependencies via `uv sync`—the manifest pins
   DepthAI to the 3.2.x series used by the capture helpers and installs YOLO
   inference dependencies. Jetson Linux ARM deployments use Python 3.10 because
   the configured GPU PyTorch wheels are built for `cp310`; ARM Python 3.11+
   is supported for robot/API-only scripts, and non-Jetson deployments use
   Python 3.11+:
   ```bash
   uv sync
   ```
   Jetson ARM Python 3.10 hosts use the Jetson GPU PyTorch wheel index;
   non-Jetson hosts use CPU PyTorch wheels from the PyTorch CPU index.
   See [`docs/deploy_non_jetson.md`](docs/deploy_non_jetson.md) for
   non-Jetson deployment notes.
2. Capture calibration data (ChArUco board images + one depth frame) and place them somewhere accessible to the Jetson. The helper script can automate the capture. For the default ChArUco board layout run:
   ```bash
   python scripts/capture_calibration_data.py \
     --output-dir data/calibration \
     --count 15 \
     --preview
   ```
3. Configure robot connectivity via `CODROID_*` env vars (defaults come from `codroid-api`). `AppSettings` in `app/settings.py` groups the robot/network, marker, and asset-path preferences and persists edits to `data/app_settings.json`. Add a `.env` file with `CODROID_`-prefixed keys (for example `CODROID_ROBOT_HOST=192.168.1.5`) to override those values in a particular environment.
4. Verify your OAK-D is visible:
   ```bash
   python -m vision.camera_check
   ```
5. Launch the Dash calibration console:
   ```bash
   python run_codroid_picker.py --debug
   ```
Use the Setup drawer to review camera status, verify the robot connection, and work through the Calibration, Applications, and Monitoring tabs. The UI shell is rendered before camera, robot, gripper, or YOLO startup so Raspberry Pi Chromium clients get a fast first paint. By default, camera/robot/gripper and the remembered YOLO model auto-start after first paint; set `camera_auto_start`, `model_auto_load`, `robot_auto_connect_after_first_paint`, or `gripper_auto_connect_after_first_paint` to `false` in `data/app_settings.json` or via `CODROID_` environment overrides to keep hardware fully manual. Calibration assets are read/written under `data/` by default; once a workspace is computed the UI saves it to `data/workspace_definition.npz` and reloads it on restart. Runtime camera startup enables depth so camera calibration can capture the required depth snapshot from the live stream without stopping or restarting the DepthAI pipeline. Stability knobs: `CODROID_DEPTH_MAX_FPS` (default `10`), `CODROID_RGB_MAX_FPS` (default `30`), `CODROID_RGB_MAX_FPS_WITH_DEPTH` (default `15`), and `CODROID_MJPEG_MAX_FPS` (default `30`). Color transport defaults to `NV12` (set `CODROID_COLOR_OUTPUT_FORMAT=bgr` to force BGR transport).
The robot and gripper connections auto-start after first paint when enabled. Use **Test robot connection** or **Acquire robot control** in Setup to retry manually after a connection failure or after releasing robot control. Robot connection attempts show a loading state and time out after 10 seconds so a missing or unreachable controller does not leave the UI looking idle.
YOLO models are remembered across restarts and loaded from `active_model`/`last_model` after the initial UI paint when `model_auto_load` is enabled. Open the Detection/Pick view and click **Load selected model** to retry manually after a load failure.
When you need to hand control back to the native robot console, use **Release robot control** in Setup. This sends a stop command, powers the robot off, and closes the websocket session, then keeps codroid-ball-picker in a released mode that does not auto-reconnect in the background. Use **Acquire robot control** (or **Test robot connection**) to reconnect from the UI.
The Dash app supports UI localization for English (`en`) and Ukrainian (`uk`). Use the language selector in the header or in **Setup & Connections**. The selection is persisted to `data/app_settings.json` (`ui_language`) and applies immediately via automatic page refresh.

For Jetson runtime FPS benchmarking, use the Jetson deployment flow below, then
run without debug mode:
```bash
python run_codroid_picker.py
```
The UI keeps Monitoring graphs lazy and uses lightweight styling/callbacks by default so slow browser clients stay responsive without reducing the configured camera or MJPEG FPS.

## User Manual (UK)

An operator-focused Ukrainian user manual is available at:

- `docs/manuals/user_manual_uk.md`

Generate a PDF from the Markdown source with:

```bash
bash scripts/build_user_manual_pdf.sh
```

Optional: choose a specific PDF engine (`pdflatex`, `xelatex`, `lualatex`, or `weasyprint`):

```bash
bash scripts/build_user_manual_pdf.sh --engine xelatex
```

Example with a Python-only engine from the project environment:

```bash
bash scripts/build_user_manual_pdf.sh --engine weasyprint
```

### PDF build prerequisites

- `pandoc` available either from system `PATH` or from project dependencies (`pypandoc-binary` after `uv sync`)
- A PDF engine available for the selected `--engine`:
  - TeX engine (`pdflatex`, `xelatex`, `lualatex`) for LaTeX-based export, or
  - `weasyprint` (provided by project dependencies after `uv sync`)

Default output path:

- `docs/manuals/user_manual_uk.pdf`

## Application Settings

`app/settings.py` defines an `AppSettings` model backed by `pydantic-settings`. It groups robot connectivity, marker/workspace preferences, and persisted metadata (gripper IP, last/active model, drop marker info) into a single object that is written to `data/app_settings.json` whenever the UI saves settings, and it now exposes auto pick-and-place defaults (target class, drop-ignore radius, approach/pick/drop offsets, estimate width, etc.). The same model loads `.env` files automatically, so drop a file with `CODROID_`-prefixed keys (for example `CODROID_ROBOT_PORT=9000`, `CODROID_MARKER_IDS=[1,2]`) to override the defaults for a specific deployment without touching the UI.

## Calibration UI (Dash)
The calibration console is a Dash + Dash Mantine Components UI that stitches the
camera, detection, and robot flows together. It provides:
- A tabbed layout for Calibration, Applications (workspace definition + per-app
  panels like Pick & Place), and Monitoring.
- A live camera view with overlays for ChArUco detection, workspace markers,
  YOLO detections, drop marker, and an auto-planning overlay that shows the
  next target/approach/drop intent during Auto Pick & Place.
- A Monitoring dashboard with KPI cards and trend charts (detections + cycle
  timing), plus a live auto-plan summary panel.
- A guided robot calibration modal with a verification move; the calibration
  points and safe Z floor are persisted in `data/robot_calibration.json` and
  loaded on startup when available.
- An Auto Motion Test footer action that enters the same auto-socket motion
  mode as Auto Pick & Place, sends a small vertical move from the current
  robot pose, then returns to the starting pose.
- A program safety banner (warning border) that highlights when robot motion
  sequences are running, plus pause/stop/reset controls and emergency stop /
  overspeed acknowledgment modals that pause active programs. The compact
  footer keeps primary motion controls visible and places advanced footer
  controls behind **Show details**.
- If `data/calibration_bundle.npz` or `data/workspace_definition.npz` already
  exist, the Calibration and Application workspace steps surface “Use saved …”
  buttons to load them without recomputing.
- A Setup drawer to verify camera connectivity and test the robot connection.

Run it locally with:
```bash
python run_codroid_picker.py --debug
```
Then open `http://127.0.0.1:8050` in your browser. The UI defaults for robot
connectivity follow the `codroid-api` settings (env `CODROID_HOST` and
`CODROID_ROBOT_PORT`, or the factory defaults).
Note: the debug reloader is disabled to keep camera/robot streams in a single
process. Restart the app manually to pick up code changes, or pass `--reload`
when you explicitly want Werkzeug file watching during development.

### Docker on Jetson Orin Nano
Pull and run with:
```bash
docker compose pull
docker compose up
```
See `docs/deploy_jetson_docker.md` for Jetson prerequisites, device mappings, and GPU/runtime notes.
The provided compose file uses host networking so the UI can be opened via a
Tailscale IP (for example `100.x.x.x:8050`) while backend services still reach
local robot/gripper IPs such as `192.168.101.95`.

For native Jetson bring-up without Docker, run the base `uv sync`/`uv run`
workflow with Python 3.10 when YOLO inference is needed. Python 3.11+ is valid
for robot/API-only utilities, but does not install the Jetson GPU PyTorch/YOLO
stack.

## CI/CD
GitHub Actions publishes releases and container images automatically for pushes to `master`:
- Computes the next `vX.Y.Z` tag from existing git tags (patch increment).
- Creates and pushes the new tag.
- Builds and publishes the ARM64 image (`linux/arm64/v8`) on a native ARM GitHub runner (`ubuntu-24.04-arm`) to `ghcr.io/profi-robotics/codroid-ball-picker` with `<tag>`, `latest`, and `sha-<commit>` tags.
- Creates a GitHub release for the new tag.

If `codroid-api` and `onrobot-api` are private in your organization, add a repository secret named `CI_REPO_READ_TOKEN` with read access to those repos.

YOLO inference is part of the base dependency set. Run `uv sync` after pulling
dependency changes so uploaded `.pt`/`.onnx` models can load.

## Corner markers
Printable ArUco fiducials for the default `(1, 2)` marker IDs live under `data/markers/corner_markers/`:
- `data/markers/corner_markers/top_left/aruco_4x4_id1.png`
- `data/markers/corner_markers/bottom_right/aruco_4x4_id2.png`

Use `data/markers/markers_40mm_sheet.pdf` or `data/markers/markers_40mm_sheet.svg` (A4, 300 DPI) to print both markers on a single page, then print at 100% scale to get 40 mm edges. Mount marker 1 so it defines the workspace origin and marker 2 at the opposite corner, both fully within the RGB frame.

For 3D printing, only the multicolor 3MFs are kept (single object with base + inlay components) to avoid slicer path conflicts:

Corner marker 3MFs:
- `data/markers/corner_markers/top_left/workspace_marker_id1_multicolor.3mf`
- `data/markers/corner_markers/bottom_right/workspace_marker_id2_multicolor.3mf`

OpenSCAD sources for the holder geometry:
- `data/markers/corner_markers/top_left/workspace_marker_id1.scad`
- `data/markers/corner_markers/bottom_right/workspace_marker_id2.scad`

## Camera calibration board (ChArUco)
The camera intrinsics are calibrated with an OpenCV ChArUco board defined in `data/markers/camera_calibration/camera_calibration_board.json`. Print the matching SVG/PDF at 100% scale so the marker size and spacing match the JSON values.

Assets:
- `data/markers/camera_calibration/camera_calibration_board.svg` - SVG layout for the ChArUco board.
- `data/markers/camera_calibration/camera_calibration_board.json` - board definition (square size, marker size, dictionary).
- `data/markers/camera_calibration/camera_calibration_board.scad` - parametric 3D printable plate with marker inlays.
- `data/markers/camera_calibration/camera_calibration_board_multicolor.3mf` - multicolor 3MF export (single object with base + inlay components).
- `data/markers/camera_calibration/camera_calibration_board.pdf` - 300 DPI printable PDF with footer text.
The 3D plate adds rounded corners and a footer label below the marker grid; the grid dimensions still match the OpenCV board spec.

## Robot calibration plate
The robot calibration plate uses edge markers for camera tracking so the center stays clear for gripper alignment grooves. The layout uses DICT_4X4_50 markers at 20 mm edge size on a 300 x 280 mm plate, with three numbered rectangular groove targets for robot calibration.

Assets:
- `data/markers/robot_calibration/calibration_plate.svg` - full-size 300 x 280 mm layout with edge markers and groove targets.
- `data/markers/robot_calibration/calibration_plate.scad` - parametric plate model with marker inlays and groove targets.
- `data/markers/robot_calibration/calibration_plate_multicolor.3mf` - multicolor 3MF export (base + inlay objects).
- `data/markers/robot_calibration/calibration_plate.pdf` - printable PDF layout.
- `data/markers/robot_calibration/calibration_plate_layout.json` - edge marker IDs plus calibration point coordinates in mm.
- `data/markers/robot_calibration/aruco_4x4_id20.png` through `data/markers/robot_calibration/aruco_4x4_id27.png` - edge markers.

Edge marker centers (mm, origin at plate top-left, +x right, +y down):
- ID 20: (20, 20)
- ID 21: (150, 20)
- ID 22: (280, 20)
- ID 23: (20, 140)
- ID 24: (280, 140)
- ID 25: (20, 260)
- ID 26: (150, 260)
- ID 27: (280, 260)

Robot calibration points (rectangular groove centers, 20 x 33 mm):
- Point 1: (70, 70)
- Point 2: (230, 70)
- Point 3: (230, 210)

Robot 3-point calibration: teach the robot to the groove intersections for Points 1–3 in sequence. Point 1 is the start, Point 2 is +X relative to Point 1, and Point 3 is +X/+Y relative to Point 1. The edge markers are for camera tracking only. Calibration starts only when CoordinateId 0 (base) is active so posture telemetry is accurate, and the UI powers the robot on automatically when you begin the calibration flow.

The script will:
1. Load calibration (or raise if missing),
2. Detect the two workspace markers,
3. Compute workspace size and target poses (center + bottom-left),
4. Enable the robot and execute two linear moves via `codroid-api`,
5. Disable the robot.

## WebSocket Protocol Notes
A concise command catalog derived from DevTools captures is available at
`robot/command_catalog.md`. If new commands are discovered, extend that file and
mirror the behaviour in `codroid-api` (the legacy `robot/websocket_client.py`
remains for reference only).

## OnRobot 2FG7 Gripper Integration

The system supports the **OnRobot 2FG7** two-finger gripper via the OnRobot Compute Box using XML-RPC. The gripper integration is available in the calibration UI for manual control and pick-place operations.

**Quick Setup:**
1. Connect the 2FG7 gripper to the OnRobot Compute Box
2. Note the Compute Box IP address (default: `192.168.101.95`)
3. In the calibration UI, click **Setup** → **OnRobot 2FG7 Gripper**
4. Enter the Compute Box IP and enable the connection
5. Click **Test gripper** to verify

**Features:**
- Real-time gripper status monitoring (width, force, grip detection)
- Manual open/close control from Applications > Pick & Place
- Automatic grip/release during pick-drop sequences
- No authentication required (direct TCP/IP connection)

For detailed documentation, see [`docs/gripper_integration.md`](docs/gripper_integration.md) and
[`docs/pick_place_test_procedure.md`](docs/pick_place_test_procedure.md).

## Auto Pick & Place (YOLO)

Auto mode uses YOLO detections to continuously pick the nearest target:

- Start/stop from the **Auto pick & place** controls in the footer.
- Auto Pick & Place uses `codroid-api` auto-socket motion by default, injecting
  and updating the managed controller project `pjcodroidautosock`; the listener
  port defaults to `auto_socket_port` / `CODROID_AUTO_SOCKET_PORT=19093`.
- Fixed auto pick/drop moves use the same `move_to_cartesian_target_linear`
  auto-socket path as the Auto Motion Test; segmented tracking keeps its live
  target update loop with socket health checks.
- After robot control is acquired, codroid-ball-picker releases the controller
  web/admin websocket so the native controller web console can be opened while
  robot posture/control stays connected on the robot websocket.
- The auto-socket controller read timeout defaults to
  `auto_socket_read_timeout_ms` / `CODROID_AUTO_SOCKET_READ_TIMEOUT_MS=120000`
  for long pick/place cycles.
- Set the **Target class** (default `orange`) and **Drop ignore radius** in the Pick & Place tab.
- Teach a drop position first; auto waits at the drop approach when no targets are visible.
- Camera safety is selectable in **Setup → Safety guards**:
  - `Off`: disable camera-driven pauses.
  - `Model safety`: pause on configured YOLO labels in workspace/full-frame zone.
  - `Heuristic safety`: pause on large frame-difference intrusions against a baseline image.
- Auto transit uses a configurable **Transit clearance (mm)** so XY travel stays above drop/approach heights.
- Auto uses the shared **Tracking mode** controls; segmented tracking keeps the last known target when it is obscured.
- Coordinate mapping uses the saved plate homography from robot calibration; re-run plate calibration if coord5 overlays drift.
- Overlay selector includes **Auto Plan** mode. When overlay is set to **Auto**
  and Auto Pick & Place is running in Applications → Pick & Place, the camera
  view auto-switches to planning overlay rendering.
