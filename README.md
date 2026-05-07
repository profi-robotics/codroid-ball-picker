# Codroid Vision-Guided Bootstrap Demo

This repository contains a hardware-agnostic scaffold for bringing up the Estun
Codroid + OAK-D pick-and-place pipeline. The focus is on repeatable calibration,
marker-driven workspace definition, coordinate mapping, and `codroid-api` for
robot motion commands.

## Layout
- `vision/` – camera calibration, marker detection, workspace mapping helpers.
- `app/` – orchestration logic that ties vision outputs to robot commands, now powering the Dash UI.
- `codroid-api` (external dependency) – robot WebSocket/HTTP control client.
- `onrobot-api` (external dependency) – OnRobot gripper control client.
- `run_codroid_picker.py` – entry point for the Dash-based bring-up console (camera, calibration, workspace, and robots all flow through this script).
- `pyproject.toml` – project metadata and dependency manifest used by `uv`.
- `scripts/capture_calibration_data.py` – CLI helper that streams an OAK-D, saves ChArUco board frames, and records a depth snapshot.
- `docs/` – integration guides for gripper and other peripherals.

## Quickstart
1. Install dependencies (Python 3.11+ recommended) via `uv sync`; the manifest pins DepthAI to the tested runtime version:
   ```bash
   uv sync
   ```
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
Use the Setup drawer to review camera status (the stream auto-starts and retries if the device is slow to appear), verify the robot connection, and work through the Calibration, Applications, and Monitoring tabs. Calibration assets are read/written under `data/` by default; once a workspace is computed the UI saves it to `data/workspace_definition.npz` and reloads it on restart. Runtime OAK-D startup currently forces RGB-only mode (depth disabled) to prioritize stream stability on flaky USB links. During camera calibration, the app uses an isolated helper process to capture the depth snapshot while the live stream is paused; click **Enable camera** after calibration to resume RGB streaming. For local testing without an OAK-D, choose **Host camera** in Setup and set the OpenCV camera index, or start with `CODROID_CAMERA_SOURCE=host CODROID_HOST_CAMERA_INDEX=0`. Host cameras provide RGB frames only; depth capture and OAK-D calibration helpers still require DepthAI hardware. Color transport defaults to `NV12` (set `CODROID_COLOR_OUTPUT_FORMAT=bgr` to force BGR transport).

The live camera preview uses WebRTC only. WebRTC sends raw camera frames from the latest-frame store, caps output with `CODROID_WEBRTC_MAX_FPS` (default `30`), limits software-encoded width with `CODROID_WEBRTC_MAX_WIDTH` (default `640`), and limits concurrent browser peers with `CODROID_WEBRTC_MAX_PEERS` (default `2`). Other stability knobs: `CODROID_DEPTH_MAX_FPS` (default `10`), `CODROID_RGB_MAX_FPS` (default `30`), `CODROID_OVERLAY_MAX_FPS` (default `10`), and `CODROID_CAMERA_RETRY_MAX_BACKOFF_S` (default `30`).
To keep startup responsive on Jetson-class hardware, robot and gripper network probes are not forced during initial page load; use **Test robot connection** and **Test gripper** in Setup when needed.
When you need to hand control back to the native robot console, use **Release robot control** in Setup. This sends a stop command, powers the robot off, and closes the websocket session, then keeps codroid-ball-picker in a released mode that does not auto-reconnect in the background. Use **Acquire robot control** (or **Test robot connection**) to reconnect from the UI.
The Dash app supports UI localization for English (`en`) and Ukrainian (`uk`). Use the language selector in the header or in **Setup & Connections**. The selection is persisted to `data/app_settings.json` (`ui_language`) and applies immediately via automatic page refresh.

For Jetson runtime FPS benchmarking, run without debug mode:
```bash
python run_codroid_picker.py
```
On Jetson-class devices, the UI uses slower polling intervals by default to reduce callback load and keep camera/robot pipelines responsive. Override with `CODROID_UI_STATUS_INTERVAL_MS` and `CODROID_UI_MONITOR_INTERVAL_MS` when a deployment needs different UI update cadence.
The Dash shell also opens `/ui_events` as a Server-Sent Events stream. That stream pushes cached camera, robot, gripper, operation, safety, model, detection, and monitoring snapshots into Dash stores so display updates do not poll hardware. If the stream is disconnected for more than 3 seconds, the existing low-rate interval callbacks resume as a fallback until SSE reconnects.

## Run checks locally

```bash
uv sync --dev
uv run python -m compileall .
uv run ruff check .
uv run pytest -q
uv build
```

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

`app/settings.py` defines an `AppSettings` model backed by `pydantic-settings`. It groups robot connectivity, marker/workspace preferences, and persisted metadata (gripper IP, last/active model, drop marker info) into a single object that is written to `data/app_settings.json` whenever the UI saves settings. It also stores auto pick defaults, selected YOLO pick labels, per-label pick/grip profiles, and the YOLO frame-skip cadence used on Jetson-class hardware. The same model loads `.env` files automatically, so drop a file with `CODROID_`-prefixed keys (for example `CODROID_ROBOT_PORT=9000`, `CODROID_MARKER_IDS=[1,2]`) to override the defaults for a specific deployment without touching the UI.

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
- A program safety banner (warning border) that highlights when robot motion
  sequences are running, plus pause/stop/reset controls and emergency stop /
  overspeed acknowledgment modals that pause active programs.
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
process. Restart the app manually to pick up code changes.

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

## CI/CD
Use branch protection rules to require the `CI` workflow to pass before merging pull requests.

GitHub Actions publishes releases and container images automatically for pushes to `master`:
- Computes the next `vX.Y.Z` tag from existing git tags (patch increment).
- Creates and pushes the new tag.
- Builds and publishes the ARM64 image (`linux/arm64/v8`) on a native ARM GitHub runner (`ubuntu-24.04-arm`) to `ghcr.io/profi-robotics/codroid-ball-picker` with `<tag>`, `latest`, and `sha-<commit>` tags.
- Creates a GitHub release for the new tag.

If `codroid-api` and `onrobot-api` are private in your organization, add a repository secret named `CI_REPO_READ_TOKEN` with read access to those repos.

YOLO inference relies on the `ultralytics` package. Install it with `uv add ultralytics`
if you want to run detections from uploaded `.pt`/`.onnx` models.

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
A concise command catalog derived from DevTools captures lives in the
`codroid-api` repository. If new commands are discovered, extend the command
catalog and API behaviour there.

## OnRobot Gripper Integration

The system supports **OnRobot 2FG7**, **OnRobot Soft Gripper**, and **OnRobot VGC10** through the OnRobot Compute Box using XML-RPC. The gripper integration is available in the calibration UI for manual control and pick-place operations.

**Quick Setup:**
1. Connect the gripper to the OnRobot Compute Box.
2. Note the Compute Box IP address (default: `192.168.101.95`)
3. In the calibration UI, click **Setup** → **OnRobot Gripper**.
4. Leave **Gripper type** on **Auto detect** or choose `2FG7`, `Soft Gripper`, or `VGC10` manually.
5. Enter the Compute Box IP, enable the connection, and click **Test gripper**.

**Features:**
- Auto detection using `onrobot-api` product codes.
- Real-time status monitoring for the connected gripper: width/force for 2FG7, width/depth/init state for Soft Gripper, and vacuum A/B for VGC10.
- Manual open/close control from Applications > Pick & Place
- Automatic grip/release during pick-drop sequences
- Per-label auto-pick grip profiles: 2FG7 uses close width/force/speed, Soft Gripper uses close width/gentle grip, and VGC10 uses a vacuum setpoint/hold threshold.
- No authentication required (direct TCP/IP connection)

For detailed documentation, see [`docs/gripper_integration.md`](docs/gripper_integration.md) and
[`docs/pick_place_test_procedure.md`](docs/pick_place_test_procedure.md).

## Auto Pick & Place (YOLO)

Auto mode uses YOLO detections to continuously pick the nearest target:

- Start/stop from the **Auto pick & place** controls in the footer.
- Upload or select any Ultralytics YOLO-compatible `.pt` model in **Detect**.
- After load, the UI extracts the model class names and uses them for the **Pick labels** multi-select.
- Use the per-label profile editor to tune approach height, pick height, grip timeout, and the connected gripper's pick parameters. 2FG7 uses close width/force/speed, Soft Gripper uses close width/gentle grip, and VGC10 uses vacuum setpoint/threshold.
- Run the model benchmark in **Detect** to measure inference latency/FPS on the current hardware. The rating is a practical Jetson Nano signal: `usable`, `warning`, or `poor`.
- Use **YOLO frame skip** to run inference once every N frames (`1` = every frame). Skipped frames reuse the latest successful detections so Jetson Nano deployments can trade responsiveness for stability.
- Set the **Drop ignore radius** in the Pick & Place tab.
- Teach a drop position first; auto waits at the drop when no targets are visible.
- The robot pauses if any other detection enters the workspace and resumes once clear.
- Auto uses the shared **Tracking mode** controls; segmented tracking keeps the last known target when it is obscured.
- Coordinate mapping uses the saved plate homography from robot calibration; re-run plate calibration if coord5 overlays drift.
- Overlay selector includes **Auto Plan** mode. When overlay is set to **Auto**
  and Auto Pick & Place is running in Applications → Pick & Place, the camera
  view auto-switches to planning overlay rendering.
