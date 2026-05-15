# Offline FastAPI Development Runbook

Use this runbook when robot, camera, and gripper hardware are not connected.
The goal is to keep the FastAPI migration moving while leaving physical
acceptance to `docs/real_cell_test_items.md`.

## Environment

```bash
uv sync --dev
export CODROID_USE_FAKE_HARDWARE=true
export CODROID_CAMERA_SOURCE=fake
export CODROID_SETTINGS_PATH=/tmp/codroid_picker_settings.json
export CODROID_ROBOT_CALIBRATION_PATH=/tmp/codroid_picker_robot_calibration.json
export CODROID_DROP_TARGET_PATH=/tmp/codroid_picker_drop_target.json
export CODROID_PLATE_TEST_CAPTURE_DIR=/tmp/codroid_picker_plate_runs
```

## Run The App

```bash
uv run python run_codroid_picker.py --host 127.0.0.1 --port 8050
```

Open `http://127.0.0.1:8050/`, start the fake camera, and use fake robot and
gripper routes through the operator UI.

## Validation

Run the default migration checks:

```bash
uv run python -m compileall -q .
uv run ruff check .
uv run mypy --strict codroid_picker
uv run pytest -q
```

Run only the FastAPI confidence slice:

```bash
uv run pytest -q -m "not legacy"
```

Hardware tests are collected but skipped unless `CODROID_HARDWARE_SMOKE=1` and
the specific `CODROID_HARDWARE_*` device flags are set.

## Offline Workflows

- Camera: start the fake camera, verify `/video/snapshot.jpg`, and switch every
  overlay mode.
- Detection: use model-less paths to verify missing-ML messages, model listing,
  parameter updates, frame-skip behavior, and benchmark unavailable state.
- Robot: use fake connect, speed, jog, move, calibration teaching, verification,
  stale posture simulation, DI state simulation, and error state simulation.
- Gripper: use fake connect/open/close/orientation plus busy and failed-grip
  simulation in tests.
- Pick/place: seed detections and drop targets in tests, then verify manual
  pick, pick-and-drop, auto-pick, pause ownership, stop handling, and monitoring
  counters.
- Plate test: run dry-run and fake-hardware paths, assert persisted summaries,
  per-target poses, skipped/failed counts, and monitoring state.

## Migration Rule

Do not remove legacy Dash/WebRTC code from offline results alone. Complete the
real-cell checklist in `docs/real_cell_test_items.md` before runtime retirement.
