# Agent Instructions

- Use `uv` for Python dependency management; install deps via `uv install` so `pyproject.toml` defines the manifest.
- Follow PEP 8 style for Python code and keep functions small with descriptive names.
- When altering Python sources, run `python -m compileall .` to ensure files compile.
- Update relevant documentation when behavior or setup steps change.

## Marker assets

- Folder layout:
  - `data/markers/corner_markers/top_left/` holds the top-left workspace marker assets (ID 1).
  - `data/markers/corner_markers/bottom_right/` holds the bottom-right workspace marker assets (ID 2).
  - `data/markers/camera_calibration/` holds the ArUco camera calibration board.
  - `data/markers/robot_calibration/` holds the robot calibration plate and marker PNGs.
- Workspace corner markers keep the OpenSCAD sources plus the multicolor 3MFs (single object with base + inlay components). Base/inlay 3MFs are not tracked; generate them temporarily if you need to rebuild the multicolor export.
- Corner marker PNGs (IDs 1/2) are generated from the OpenCV DICT_4X4_50 dictionary via `cv2.aruco.generateImageMarker` at 600x600 px. The A4 sheet (`data/markers/markers_40mm_sheet.svg`/`.pdf`) places both markers at 40 mm edges.
- Camera calibration board: `camera_calibration_board.json` stores the OpenCV ArUco grid board spec; the SVG is generated from the same spec so it prints at true scale.
- Robot calibration marker PNGs (IDs 10/11/12) are generated from the OpenCV DICT_4X4_50 dictionary via `cv2.aruco.generateImageMarker` at 500x500 px.
- Robot calibration plate: `calibration_plate_layout.json` is the source of marker IDs and centers; `calibration_plate.svg` is the printable layout and `calibration_plate.scad` is the 3D plate model derived from the same measurements.
