# FastAPI Hardware Acceptance Checklist

Run this checklist on the Jetson/OAK-D/Codroid/OnRobot cell before removing the
legacy Dash package from the repository.

## Preflight

```bash
uv sync --dev
uv run python -m compileall -q .
uv run ruff check .
uv run mypy --strict codroid_picker
uv run pytest -q
```

The default suite includes skipped hardware smoke tests. Run them on the cell
with:

```bash
CODROID_HARDWARE_SMOKE=1 uv run pytest -q -m hardware
```

Detailed per-device flags and physical acceptance items live in
`docs/real_cell_test_items.md`.

## Runtime Smoke Tests

1. Start the FastAPI console with the deployment `.env`:
   ```bash
   uv run python run_codroid_picker.py --host 0.0.0.0 --port 8050
   ```
2. Open `/health` and verify `ok=true`, detection worker status, and no stale
   robot posture after robot connection.
3. In Setup, start the configured OAK-D or host camera and verify MJPEG video,
   snapshots, and each overlay mode.
4. Connect/acquire the Codroid robot, toggle power, set speed, jog TCP axes, jog
   rescue joints `j1` through `j6`, then release control.
5. Connect the OnRobot gripper, open/close, and apply orientation if supported.
6. Load or capture camera calibration, compute/load workspace, detect the robot
   calibration plate, and run robot calibration verification.
7. Teach a drop target and run plate test in **Dry run** mode first; verify
   per-target approach/pick poses, coordinate mode, skipped count, and saved
   JSON summary.
8. Run one real plate-test cycle only after the dry-run poses look correct.
9. Activate a YOLO model, run benchmark, start auto pick, verify safety pause
   on configured labels, then stop/reset.
10. Trigger or simulate each recoverable robot alarm type and confirm the UI
    modal plus `/robot/errors/ack` behavior for emergency, overspeed, joint
    protection, wrong tool, and drag-not-allowed.

Record failures with the `/state` JSON payload and the latest
`data/plate_test_runs/plate_test_*.json` result.
