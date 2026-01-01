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
- `scripts/capture_calibration_data.py` – CLI helper that streams an OAK-D, saves ArUco board frames, and records a depth snapshot.

## Quickstart
1. Install dependencies (Python 3.11+ recommended) via `uv install`—the manifest pins DepthAI to the 2.x series so the capture helpers use the classic API:
   ```bash
   uv install
   ```
2. Capture calibration data (ArUco board images + one depth frame) and place them
   somewhere accessible to the Jetson. The helper script can automate the capture.
   For the default ArUco board layout you’d run:
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
6. Run the demo (adjust the ArUco board layout path if needed, e.g.
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

## Camera calibration board (ArUco)
The camera intrinsics are calibrated with an OpenCV ArUco grid board defined in `data/markers/camera_calibration/camera_calibration_board.json`. Print the matching SVG at 100% scale so the marker size and spacing match the JSON values.

Assets:
- `data/markers/camera_calibration/camera_calibration_board.svg` - SVG layout for the ArUco grid board.
- `data/markers/camera_calibration/camera_calibration_board.json` - board definition (marker size, spacing, dictionary).

## Robot calibration plate
The robot calibration plate contains only the three robot calibration markers so the camera/robot coordinate systems can be aligned without the workspace markers. The layout uses DICT_4X4_50 markers at 40 mm edge size and is designed to be easy to turn into a 3D printable plate.

Assets:
- `data/markers/robot_calibration/calibration_plate.svg` - full-size 300 x 200 mm layout with three robot markers.
- `data/markers/robot_calibration/calibration_plate.scad` - parametric plate model with 1 mm marker recesses and labels.
- `data/markers/robot_calibration/calibration_plate_layout.json` - marker IDs and center coordinates in mm.
- `data/markers/robot_calibration/aruco_4x4_id10.png`, `data/markers/robot_calibration/aruco_4x4_id11.png`, `data/markers/robot_calibration/aruco_4x4_id12.png` - robot calibration markers.

Marker centers (mm, origin at plate top-left, +x right, +y down):
- ID 10 (robot calibration, Point 1): (100, 80)
- ID 11 (robot calibration, Point 2): (200, 80)
- ID 12 (robot calibration, Point 3): (200, 150)

Robot 3-point calibration: teach the robot to the centers of IDs 10, 11, and 12 in sequence. Point 1 is the start, Point 2 is +X relative to Point 1, and Point 3 is +X/+Y relative to Point 1. The SVG/SCAD files include numeric labels next to each robot marker. Keep `marker_size_mm` at 40.0 in `app/orchestrator.py` to match the printed markers.

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
