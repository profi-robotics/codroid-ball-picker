# Codroid Vision-Guided Bootstrap Demo

This repository contains a hardware-agnostic scaffold for bringing up the Estun
Codroid + OAK-D pick-and-place pipeline. The runtime UI is now a FastAPI
operator console with server-rendered HTML, Server-Sent Events status updates,
and MJPEG camera streaming. The focus remains repeatable calibration,
marker-driven workspace definition, coordinate mapping, and `codroid-api` for
robot motion commands.

## Layout
- `vision/` – camera calibration, marker detection, workspace mapping helpers.
- `codroid_picker/` – FastAPI application, service layer, state store, hardware adapters, frontend assets, and no-build operator UI.
- `app/` – legacy Dash-era orchestration modules retained during migration and for compatibility tests.
- `codroid-api` (external dependency) – robot WebSocket/HTTP control client.
- `onrobot-api` (external dependency) – OnRobot gripper control client.
- `run_codroid_picker.py` – entry point for the FastAPI operator console.
- `pyproject.toml` – project metadata and dependency manifest used by `uv`.
- `scripts/capture_calibration_data.py` – CLI helper that streams an OAK-D, saves ChArUco board frames, and records a depth snapshot.
- `docs/` – integration guides for gripper and other peripherals.

## Quickstart
1. Install dependencies with Python 3.11+ via `uv sync`; the manifest pins DepthAI to the tested runtime version:
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
3. Configure robot connectivity via `CODROID_*` env vars. `AppSettings` in `codroid_picker/config.py` merges environment values, `data/app_settings.json`, and defaults. Add a `.env` file with `CODROID_`-prefixed keys, for example `CODROID_ROBOT_HOST=192.168.1.5`, to override values in a specific environment.
4. Verify your OAK-D is visible:
   ```bash
   uv run python -m vision.camera_check
   ```
5. Launch the FastAPI operator console:
   ```bash
   uv run python run_codroid_picker.py --debug
   ```
Then open `http://127.0.0.1:8050`. The root page is a single server-rendered operator panel with Manual, Auto, Calibration, Setup, and Monitor tabs. The camera preview is served through `GET /video/stream` as MJPEG and falls back to `GET /video/snapshot.jpg` polling if the stream disconnects. UI state is pushed through `GET /events/status` as SSE snapshots and deltas, with `GET /state` available for diagnostics.

For local testing without physical hardware, start with fake adapters:

```bash
CODROID_USE_FAKE_HARDWARE=1 uv run python run_codroid_picker.py --debug
```

Fake hardware does not override an explicit camera source: `CODROID_CAMERA_SOURCE=generic` or a **Start generic camera** command still opens the OpenCV host camera path, while robot and gripper adapters can remain fake.

For a generic OpenCV camera, use an index or an explicit device path. OAK-D remains the default when `CODROID_CAMERA_SOURCE` is unset.

```bash
CODROID_CAMERA_SOURCE=generic CODROID_HOST_CAMERA_INDEX=0 uv run python run_codroid_picker.py
CODROID_CAMERA_SOURCE=generic CODROID_HOST_CAMERA_DEVICE=/dev/video0 uv run python run_codroid_picker.py
CODROID_CAMERA_SOURCE=generic CODROID_HOST_CAMERA_DEVICE=/dev/v4l/by-id/<camera-id> uv run python run_codroid_picker.py
```

Set `CODROID_CAMERA_AUTOSTART=1` to open the configured camera during server startup instead of pressing **Setup -> Start generic camera** in the UI.

For Azure Kinect testing, first confirm that the color stream is exposed to OpenCV, for example as `/dev/video*` or through a GStreamer pipeline. If the Azure Kinect Sensor SDK is the only available interface on the host, install/configure a V4L2/GStreamer bridge first or add a dedicated `k4a` adapter; the generic path intentionally only depends on OpenCV.

The FastAPI server intentionally runs as one process and one uvicorn worker so the camera, robot session, gripper session, state store, and detection worker are not duplicated. Robot and gripper network probes are not forced during initial page load; use the Setup tab to connect or acquire control. The UI supports English (`en`) and Ukrainian (`uk`) through locale JSON files in `codroid_picker/i18n/`; `ui_language` is persisted in `data/app_settings.json`.

For Jetson runtime FPS benchmarking, run without debug mode:
```bash
uv run python run_codroid_picker.py
```
Useful video knobs are `CODROID_VIDEO_MAX_FPS`, `CODROID_VIDEO_MAX_WIDTH`, and `CODROID_VIDEO_JPEG_QUALITY`. Detection is isolated behind `DetectionService`; install the optional ML stack with `uv sync --extra ml` when YOLO model loading is needed.

## Run checks locally

```bash
uv sync --dev
uv run python -m compileall .
uv run ruff check .
uv run mypy --strict codroid_picker
uv run pytest -q
uv build
```

For no-hardware development, use `docs/offline_fastapi_development.md`.
Before retiring the legacy Dash package, run the FastAPI hardware acceptance
checklist in `docs/fastapi_hardware_acceptance.md` on the real cell.

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

- `pandoc` available from system `PATH`.
- A PDF engine available outside the runtime dependency set:
  - TeX engine (`pdflatex`, `xelatex`, `lualatex`) for LaTeX-based export, or
  - `weasyprint` installed manually in a documentation environment.

Default output path:

- `docs/manuals/user_manual_uk.pdf`

## Application Settings

`codroid_picker/config.py` defines an `AppSettings` model backed by
`pydantic-settings`. It groups robot connectivity, camera/video defaults,
marker/workspace preferences, detection settings, safety settings, auto-pick
defaults, and persisted metadata into one effective model. Runtime-editable
settings are saved atomically to `data/app_settings.json`; environment variables
with the `CODROID_` prefix continue to override deployment-specific values.

## Operator UI (FastAPI)
The operator console is a FastAPI + Jinja2 UI that stitches the camera,
detection, and robot flows together without Dash, React, WebRTC, or an npm build.
It provides:
- Manual, Auto, Calibration, Setup, and Monitor tabs.
- A live MJPEG camera view with overlay selection and snapshot fallback.
- SSE-driven status badges for camera, robot, gripper, safety, detection,
  operation, calibration, and monitoring state.
- Command endpoints for jog, move, gripper open/close, auto start/stop,
  calibration workflow steps, settings, model activation, and locale changes.
- Bilingual EN/UK strings from vendored JSON catalogs.

Run it locally with:
```bash
uv run python run_codroid_picker.py --debug
```
Then open `http://127.0.0.1:8050` in your browser. The UI defaults for robot
connectivity follow `CODROID_ROBOT_HOST`, `CODROID_ROBOT_PORT`, and
`CODROID_USER_PORT`. The debug reloader is disabled to keep camera/robot streams
in a single process; restart the app manually to pick up code changes.

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

`codroid-api` and `onrobot-api` are resolved from the `profi-robotics` GitHub
repositories through `uv` source entries in `pyproject.toml`. For active
multi-repo development you may still clone the sibling repositories beside this
checkout and temporarily switch the sources to local editable paths.

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

Robot 3-point calibration: teach the robot to the groove intersections for Points 1–3 in sequence. Point 1 is the start, Point 2 is +X relative to Point 1, and Point 3 is +X/+Y relative to Point 1. The edge markers are for camera tracking only. The FastAPI UI switches the robot to CoordinateId 0 before capturing taught poses, then **Apply calibration** sends `CoordinateCalibration` through `codroid-api`, persists the returned frame with `ChangeCoordinateParameter`, and selects the configured user coordinate slot (`CODROID_ROBOT_COORDINATE_ID`, default `5`) for verification and pick-place moves.

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
