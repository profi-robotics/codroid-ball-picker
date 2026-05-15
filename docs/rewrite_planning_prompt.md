# Planning Prompt: codroid-ball-picker Rewrite

## Task

You are a senior software architect. Your task is to produce a **detailed, actionable rewrite plan** for the `codroid-ball-picker` application. Do **not** implement anything yet — your output is a plan document that another engineer will follow. The plan must be comprehensive enough to execute without further clarification.

---

## Context: Current System

`codroid-ball-picker` is a Python application that provides a web UI for a vision-guided pick-and-place robot cell. Here is what it currently does and how it is built.

### Hardware

- **Robot:** Estun Codroid arm, controlled via WebSocket through `codroid-api` (local Python library).
- **Gripper:** OnRobot (2FG7, SG, VGC10) via XML-RPC through `onrobot-api` (local Python library).
- **Camera:** Luxonis OAK-D (DepthAI), or a host USB camera. Used for ArUco marker detection and YOLO ball detection.
- **Compute targets:**
  - **Jetson Nano** (4 GB RAM, 128-core Maxwell GPU, no hardware H.264 encoder with aiortc, CUDA available for ML).
  - **Raspberry Pi** (4 or 5, 4–8 GB RAM, no discrete GPU, software rendering only in browser).
  - Browser access from any LAN device (laptop, tablet).

### Current Tech Stack

| Layer | Technology | Problem |
|---|---|---|
| Web framework | Plotly Dash (Flask + React renderer) | Ships 40+ MB of JS React/Dash bundles; each callback round-trips over HTTP; heavy JS runtime in browser on resource-limited devices |
| UI component library | dash-mantine-components (Mantine v7 React) | Large JS bundle; over-engineering for an industrial operator panel |
| Video streaming | aiortc WebRTC (Python, H.264 SW codec) | CPU-intensive; SDP negotiation fragile on NAT-free LAN; SW H.264 encoding burns CPU on Jetson |
| Server-side events | Custom SSE route (`/ui_events`) + JS batching | Works reasonably but is bolted on top of Dash's callback model |
| YOLO inference | ultralytics YOLOv8 via PyTorch | Already frame-skipped (every 3rd frame) but still high latency |
| PDF export | weasyprint + pypandoc | Heavyweight deps for one-time manual export; irrelevant to runtime |
| Python framework | Flask (synchronous) + `dash[async]` | Mix of sync/async creates subtle bugs; Werkzeug dev server not suitable for production |
| File/module layout | Flat `app/codroid_picker/` with 50+ files and 14+ callback modules | Hard to navigate; business logic co-mingled with UI wiring |

### Key Functional Modules to Preserve

All business logic must be preserved verbatim in behaviour. These modules contain the core algorithms:

| Module | What it does | Keep or rewrite? |
|---|---|---|
| `vision/calibration.py` | ChArUco camera calibration, workspace plane computation | Keep, minor refactor |
| `vision/markers.py` | ArUco marker detection | Keep |
| `vision/workspace.py` | Pixel→mm→robot coordinate transforms | Keep |
| `vision/capture.py` / `stream.py` | OAK-D DepthAI pipeline, host camera, background reader threads | Keep, minor refactor |
| `vision/plate.py` | Calibration plate homography | Keep |
| `app/codroid_picker/target_planner.py` | Collision-aware pick ranking | Keep |
| `app/codroid_picker/pick_place_helpers.py` | Motion trajectory helpers | Keep |
| `app/codroid_picker/auto_operation.py` | Auto pick-and-place loop | Keep |
| `app/codroid_picker/gripper_adapters.py` | Normalised OnRobot adapter layer | Keep |
| `app/codroid_picker/transforms.py` | Coordinate system transforms | Keep |
| `app/codroid_picker/safety_guard.py` | Human detection safety checks | Keep |
| `app/codroid_picker/camera_calibration.py` | ChArUco calibration workflow | Keep |
| `app/codroid_picker/state.py` / `session.py` | Thread-safe operation and session state | Redesign |
| `app/codroid_picker/persistence.py` | Save/load calibration, pick records | Keep |
| `app/codroid_picker/models.py` | YOLO model loading and selection | Keep |
| `app/codroid_picker/settings.py` | `pydantic-settings` AppSettings | Keep, minor refactor |

### Existing Tests

`tests/` has 14 test files using pytest with three marks: `unit`, `integration`, `hardware`.  
`conftest.py` auto-marks unmarked tests as `unit`.  
All tests must keep passing after the rewrite (adapt their import paths).

### External APIs (do not rewrite these)

- `codroid-api` — WebSocket robot client. The picker is a consumer only.
- `onrobot-api` — XML-RPC gripper client. The picker is a consumer only.

---

## Goals for the Rewrite

1. **Performance on constrained hardware.** The web UI must be fast on Raspberry Pi 4 (browser client) and Jetson Nano (server). The browser page should load in < 2 s on the Pi's Chromium.
2. **Reliable video delivery.** Video must display without stalling on both platforms. Hardware-accelerated paths should be preferred where available.
3. **Clean architecture.** Clear separation: hardware drivers → core business logic → API layer → UI/frontend. No circular imports. No global singletons spread across 50 files.
4. **Testability.** Hardware adapters must be behind interfaces so all core logic is testable without physical hardware. Target ≥ 80 % unit test coverage on core logic.
5. **Maintainability.** Smaller, well-named files. One concern per file. Public surface documented with type annotations.
6. **Operator UX.** Same features as today: camera view, calibration wizard, manual jog, auto pick-and-place loop, gripper control, safety monitoring, status indicators. Bilingual (EN/UK).

---

## What the Plan Must Cover

Your plan document must address every section below with enough specificity to execute. Where you recommend a specific library or pattern, justify it in one sentence.

### 1. Technology Selection

For each layer, select the new technology and justify it given the hardware constraints:

**a) Python web framework + ASGI server**
- Candidates: FastAPI + uvicorn, Starlette bare, Flask 3 with gevent, aiohttp.
- Explain how async routes map to the existing callback model.

**b) Frontend architecture**
- The current UI is a Dash React SPA loaded by the browser.
- Recommend the replacement: pure server-rendered HTML + HTMX, Alpine.js micro-reactivity, or a lightweight SPA (Preact, Svelte). Justify based on JS bundle size, rendering cost on Pi, and maintainability without a Node.js toolchain.
- No React, no Webpack, no npm build pipeline unless explicitly justified.

**c) Video streaming**
- The current stack uses aiortc (Python WebRTC, SW H.264).
- Candidates: MJPEG-over-HTTP (simplest, universal), WebRTC with GStreamer H.264 HW encoding (Jetson), WebSocket binary JPEG frames, RTSP via GStreamer + browser HLS/DASH.
- Provide a primary strategy and a fallback. Consider both Jetson (NVENC) and Pi (no HW encoder).

**d) Real-time UI state push**
- Dash currently uses SSE + JS batching for status updates (robot state, camera FPS, gripper state, safety status).
- Recommend: pure SSE, WebSocket, or long-polling. Explain event schema.

**e) YOLO inference process isolation**
- Currently runs in the main Python process with frame skipping.
- Recommend a subprocess/worker architecture to isolate GIL/memory pressure. Include inter-process communication method (multiprocessing.Queue, ZeroMQ, shared memory).

**f) PyTorch / TensorRT on Jetson**
- Current: PyTorch via NVIDIA JetPack wheels, frame-skipped ultralytics.
- Recommend: Keep PyTorch path + add TensorRT export option for Jetson; or switch to ONNX Runtime with TensorRT EP. Explain the trade-off.

### 2. Package and File Structure

Provide the exact proposed directory tree, with a one-line description of each file or directory. The structure must:
- Separate hardware adapters, core business logic, API routes, and frontend assets into distinct top-level packages or directories.
- Have a `tests/` hierarchy that mirrors the source tree (`tests/unit/`, `tests/integration/`, `tests/hardware/`).
- Locate configuration (settings, env parsing) in one place.
- Keep `vision/` as an independent package importable without the web layer.

### 3. Architecture Design

Provide a layered architecture diagram (ASCII or Mermaid) showing:
- Hardware adapter interfaces and their concrete implementations.
- Core service objects (CameraService, RobotService, GripperService, DetectionService, CalibrationService, AutoPickService).
- API layer (routes and WebSocket/SSE handlers).
- Frontend (browser events → API calls → SSE updates).
- Data flow for the auto pick-and-place loop (camera frame → detection → target planning → robot motion → gripper → state update → UI push).

Specify the concurrency model: which components run in threads vs. asyncio tasks vs. subprocesses.

### 4. State Management

The current system uses a mix of module-level singletons (`SESSION`, `AppRegistry`, `OperationState`, `ErrorState`) scattered across files.

Design a replacement:
- Define state domains (e.g., `CameraState`, `RobotState`, `OperationState`, `CalibrationState`).
- Specify whether state lives in memory only or is also persisted, and where.
- Define the thread/async safety strategy for each state domain.
- Show how UI clients receive state updates (push vs. pull).

### 5. Video Pipeline Design

Draw the frame lifecycle from camera sensor to browser pixel:

```
OAK-D / host camera
  → background reader thread (DepthAI SDK / OpenCV)
  → RawFrameBuffer (latest frame, thread-safe)
  → [overlay renderer — async task, rate-limited]
  → [YOLO detection — subprocess]
  → video server endpoint
  → browser decoder
```

For each stage specify:
- Buffer depth and drop policy (newest-wins vs. queue).
- Threading model (OS thread, asyncio task, subprocess).
- Format at each stage (BGR numpy, JPEG bytes, YUV420, H.264 NAL).
- Maximum latency budget to hit 15–30 FPS at the browser on Pi.

### 6. API Design

List all HTTP/WebSocket/SSE endpoints the new server must expose. For each endpoint:
- Method, path, description.
- Request schema (JSON or form) if applicable.
- Response schema.
- Whether it replaces an existing Dash callback or video route, and which one.

At minimum include endpoints for:
- Video stream (GET /video/stream)
- SSE status stream (GET /events/status)
- Robot jog (POST /robot/jog)
- Robot move to pose (POST /robot/move)
- Gripper open/close (POST /gripper/open, /gripper/close)
- Auto pick start/stop (POST /auto/start, /auto/stop)
- Calibration workflow (POST /calibration/...) 
- Settings read/write (GET/PUT /settings)
- Model management (GET /models, POST /models/activate)

### 7. Frontend HTML/JS/CSS Design

Describe the page layout and component breakdown:
- Single-page HTML template served by the server.
- Sections: header (branding, connection status), camera view (full-width video), control panel (tabs: Manual, Auto, Calibration, Setup, Monitor), footer (status bar).
- For each tab, list the controls and their interaction model (what they call on the server, what SSE event updates them).
- CSS approach: custom properties for theming, no CSS-in-JS, no Mantine/Tailwind (unless justified as zero-runtime).
- JS approach: vanilla JS + HTMX for declarative server interactions, Alpine.js (3 KB) for reactive bindings (show/hide, disable, text binding). No React.
- Fonts: keep Space Grotesk + IBM Plex Sans, loaded from Google Fonts or bundled.

### 8. Settings and Configuration

Map every current `AppSettings` field to the new system:
- Which fields remain as env vars (12-factor).
- Which fields are persisted to JSON in `data/` and editable at runtime via the UI.
- How settings are validated (keep pydantic-settings, or switch to).
- How settings changes propagate to running components without restart.

### 9. Testing Strategy

Define the test plan:
- **Unit tests:** What can be tested with no hardware and no network. List the modules. Target ≥ 80 % line coverage. Specify any new fixtures needed.
- **Integration tests:** What can be tested with fake/mock hardware adapters. List the scenarios (auto pick loop with fake camera+robot, calibration workflow with synthetic frames, SSE state push, video endpoint MJPEG frame delivery).
- **Hardware tests:** Minimal set of smoke tests requiring real robot + camera + gripper. Must be idempotent and safe (no unexpected motion).
- **Performance tests:** Specify a benchmark for MJPEG throughput (frames/second) and SSE event latency, runnable on CI without hardware.
- Specify how to mock `codroid-api` and `onrobot-api` in tests (protocol-level fake vs. adapter interface mock).

### 10. Migration Plan

Provide a phased migration path:
- **Phase 0:** Establish new project skeleton — directory structure, pyproject.toml, CI config, linting (ruff), type checking (mypy strict), test runner. No functional code yet.
- **Phase 1:** Port `vision/` and core business logic (target_planner, pick_place_helpers, safety_guard, transforms, persistence) with full unit tests passing. No web layer.
- **Phase 2:** Implement hardware adapter interfaces and fakes. Write integration tests against fakes. Port auto_operation and calibration workflows to new service objects.
- **Phase 3:** Implement FastAPI backend: routes, SSE, video endpoint (MJPEG first). Port settings.
- **Phase 4:** Implement frontend HTML+JS. Replicate all current UI features tab by tab.
- **Phase 5:** Replace Dash-based Docker image with new stack. Run against real hardware. Validate all hardware tests.
- **Phase 6:** Performance tuning — MJPEG throughput, YOLO subprocess isolation, SSE event rate tuning on Jetson and Pi.

For each phase, list: inputs, outputs, acceptance criteria.

### 11. Dependency Constraints

- `uv` must remain the package manager. `pyproject.toml` is the manifest.
- `depthai==3.2.1` must remain pinned (SDK stability).
- `codroid-api` and `onrobot-api` remain as editable local path dependencies.
- `torch` must remain an optional extra, with the Jetson JetPack NVIDIA wheel install documented.
- Remove: `dash`, `dash-mantine-components`, `aiortc`, `weasyprint`, `pypandoc-binary`.
- Add: `fastapi`, `uvicorn[standard]`, `python-multipart` (form uploads), `httpx` (async HTTP for tests), `anyio` (async compat).
- Optional adds (justify each): `htmx` is frontend-only (no Python dep), `sse-starlette` (SSE helper), `opencv-python-headless` instead of `opencv-python` (saves ~50 MB).

### 12. Docker and Deployment

Update the Docker and Compose setup:
- Base image for Jetson: `nvcr.io/nvidia/l4t-pytorch:r36.x.x-pth2.x.x-py3` or Ubuntu 22.04 + JetPack manual install.
- Base image for Pi: `python:3.11-slim-bookworm` or `arm64v8/python:3.11-slim`.
- Replace Werkzeug dev server with `uvicorn --workers 1` (single worker to share camera singleton).
- Keep: network_mode=host (robot/gripper LAN), USB device passthrough, data volume mount.
- Add: health-check endpoint (`GET /health` returning 200 + JSON).
- Document GStreamer installation for Jetson NVENC path if WebRTC HW encoding is chosen.

---

## Constraints and Non-Goals

- Do **not** change `codroid-api` or `onrobot-api` as part of this rewrite.
- Do **not** change the physical calibration workflow logic — only its wiring to the UI.
- Do **not** introduce a Node.js/npm build step unless the plan explicitly justifies it and provides a fallback for environments without Node.
- Do **not** use a database (SQLite, Redis, etc.) — file-based persistence in `data/` is sufficient.
- All new Python code must be typed (PEP 484) and pass `mypy --strict`.
- All new Python code must pass `ruff check` with at minimum `E9, F63, F7` rules (keep existing baseline, optionally add more).
- The application must still support bilingual UI (EN/UK) — carry over the `i18n.py` locale system or replace it with an equivalent.

---

## Deliverable

Return a single Markdown document structured exactly as:

```
# Rewrite Plan: codroid-ball-picker

## 1. Technology Selection
...

## 2. Package and File Structure
...

## 3. Architecture Design
...

## 4. State Management
...

## 5. Video Pipeline Design
...

## 6. API Design
...

## 7. Frontend HTML/JS/CSS Design
...

## 8. Settings and Configuration
...

## 9. Testing Strategy
...

## 10. Migration Plan
...

## 11. Dependency Constraints
...

## 12. Docker and Deployment
...
```

Each section must be specific, not generic. Where two options are equally valid, pick one and state why. The plan should be immediately actionable by a developer who has read this prompt and the codebase.
