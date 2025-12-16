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
- `requirements.txt` – Python dependencies for the stack.

## Quickstart
1. Install dependencies (Python 3.10+ recommended) using `uv`:
   ```bash
   uv pip install -r requirements.txt
   ```
2. Capture calibration data (chessboard images + one depth frame) and place them
   somewhere accessible to the Jetson.
3. Update `DemoConfig` in `app/orchestrator.py` with your WebSocket URI and
   marker parameters.
4. Verify your OAK-D is visible:
   ```bash
   python -m vision.camera_check
   ```
5. Replace the `frame_provider_stub` in `run_bootstrap_demo.py` with an actual
   OAK-D frame acquisition function.
6. Run the demo:
   ```bash
   python run_bootstrap_demo.py
   ```

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
