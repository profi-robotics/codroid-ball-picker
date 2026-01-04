# Codroid Vision-Guided Bootstrap Demo

This repository contains a hardware-agnostic scaffold for bringing up the Estun
Codroid + OAK-D pick-and-place pipeline. The focus is on repeatable calibration,
marker-driven workspace definition, coordinate mapping, and a minimal WebSocket
client for motion commands.

## Layout
- `vision/` – camera calibration, marker detection, workspace mapping helpers.
- `robot/` – WebSocket client and command catalog extracted from browser traffic.
- `app/` – orchestration logic that ties vision outputs to robot commands.
- `run_bootstrap_demo.py` – entry point that runs the end-to-end flow using a
  frame provider stub.
- `pyproject.toml` – project metadata and dependency manifest used by `uv`.
- `scripts/capture_calibration_data.py` – CLI helper that streams an OAK-D, saves ChArUco board frames, and records a depth snapshot.

## Quickstart
1. Install dependencies (Python 3.11+ recommended) via `uv install`—the manifest pins DepthAI to the 2.x series so the capture helpers use the classic API:
   ```bash
   uv install
   ```
2. Capture calibration data (ChArUco board images + one depth frame) and place them
   somewhere accessible to the Jetson. The helper script can automate the capture.
   For the default ChArUco board layout you’d run:
   ```bash
   python scripts/capture_calibration_data.py \
     --output-dir data/calibration \
     --count 15 \
     --preview
   ```
3. Update `DemoConfig` in `app/orchestrator.py` with your WebSocket URI and
   marker parameters.
4. Verify your OAK-D is visible:
   ```bash
   python -m vision.camera_check
   ```
5. Provide a real camera feed. You can now launch the demo with `--use-depthai-camera`
   (optionally overriding `--camera-resolution`/`--camera-fps`) to stream live RGB frames
   from the attached OAK-D, or replace the `frame_provider_stub` with your own camera hook.
6. Run the demo (adjust the ChArUco board layout path if needed, e.g.
   `python run_bootstrap_demo.py --aruco-board-layout data/markers/camera_calibration/camera_calibration_board.json`). To capture the calibration
   data as part of this step, add `--auto-calibrate` (you can tweak resolution,
   count, and preview mode):
   ```bash
   python run_bootstrap_demo.py --auto-calibrate --capture-count 15 --capture-resolution 1920x1080 --capture-preview
   ```
   Add `--use-depthai-camera --camera-resolution 1920x1080 --camera-fps 30` when you want the script
   to grab an actual frame from the OAK-D before detecting the workspace markers. Include
   `--marker-retry-attempts 20` (auto-enabled when using the live camera) to keep sampling frames until both
   ArUco markers are detected instead of exiting immediately. Use `--marker-preview` to view the live feed and
   press 'm' to lock a frame before detection, and inspect `data/marker_debug_frame.png` if detection still fails.
   For a stable workspace-only run (no robot motion), this command has worked well in practice:
   ```bash
   python run_bootstrap_demo.py \
     --use-depthai-camera --camera-resolution 1920x1080 --camera-fps 30 \
     --track-workspace --marker-preview \
     --workspace-scale 0.97
   ```
7. Launch the calibration UI when you want a visual, step-by-step bring-up:
   ```bash
   python run_calibration_ui.py --debug
   ```
   Use the Setup drawer to start the camera stream, verify the robot connection, and
   work through the three calibration steps. The UI reads/writes calibration assets
   under `data/` by default.

## Calibration UI (Dash)
The calibration console is a Dash + Dash Mantine Components UI that stitches the
camera, workspace, and robot calibration flows together. It provides:
- A tabbed layout (Calibration tab) with a vertical stepper for camera, workarea,
  and robot calibration.
- A live camera view with overlays for ChArUco detection, workspace markers, and
  calibration plate markers.
- A Setup drawer to verify camera connectivity and test the robot WebSocket link.

Run it locally with:
```bash
python run_calibration_ui.py --debug
```
Then open `http://127.0.0.1:8050` in your browser.

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

Robot 3-point calibration: teach the robot to the groove intersections for Points 1–3 in sequence. Point 1 is the start, Point 2 is +X relative to Point 1, and Point 3 is +X/+Y relative to Point 1. The edge markers are for camera tracking only.

The script will:
1. Load calibration (or raise if missing),
2. Detect the two workspace markers,
3. Compute workspace size and target poses (center + bottom-left),
4. Enable the robot and execute two linear moves via WebSocket,
5. Disable the robot.

## WebSocket Protocol Notes
A concise command catalog derived from DevTools captures is available at
`robot/command_catalog.md`. If new commands are discovered, extend that file and
mirror the behaviour in `robot/websocket_client.py`.
