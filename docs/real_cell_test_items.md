# Real-Cell Test Items

Use this checklist when the OAK-D or host camera, Estun Codroid robot, and
OnRobot gripper are physically connected. Do not remove legacy Dash/WebRTC code
or declare migration parity complete until these items have been run against the
FastAPI runtime.

## Command

Run all hardware-gated tests:

```bash
CODROID_HARDWARE_SMOKE=1 \
CODROID_HARDWARE_CAMERA=1 \
CODROID_HARDWARE_ROBOT=1 \
CODROID_HARDWARE_GRIPPER=1 \
CODROID_HARDWARE_PLATE_DRY_RUN=1 \
uv run pytest -q -m hardware
```

Run motion-sensitive tests only in a cleared cell:

```bash
CODROID_HARDWARE_SMOKE=1 \
CODROID_HARDWARE_ROBOT_JOG=1 \
CODROID_HARDWARE_ROBOT_RESCUE=1 \
uv run pytest -q -m hardware
```

Run recovery tests only after deliberately triggering or confirming a matching
recoverable robot alarm:

```bash
CODROID_HARDWARE_SMOKE=1 \
CODROID_HARDWARE_ROBOT_RECOVERY=1 \
uv run pytest -q -m hardware
```

## Camera

- Start the configured OAK-D camera and confirm `/video/snapshot.jpg` returns a
  current JPEG with overlays enabled.
- Switch overlays through `auto`, `calibration`, `workspace`, `plate`,
  `detection`, and `auto_plan`; confirm the stream remains live.
- Confirm camera stop releases the device and a subsequent start succeeds.

## Robot

- Connect, acquire, set speed to 25%, issue safe stop, and release.
- Confirm `/state.robot.poll_age_s` stays under the configured stale timeout
  while connected.
- Confirm flange DI/button state updates in `/state.robot.di_state`.
- In a cleared cell, jog TCP axes `x`, `y`, `z`, `a`, `b`, and `c` with a short
  hold and verify expected small motion.
- In rescue mode, jog `j1` through `j6` with a short hold and verify the robot
  accepts each command.

## Robot Recovery

- Trigger or simulate each recoverable alarm and verify `/state.robot.active_error`
  plus the operator modal:
  - emergency stop
  - overspeed
  - joint protection
  - wrong tool
  - drag-not-allowed
- Run `/robot/errors/ack` for the matching type and confirm the real controller
  clears or reports the expected state.
- Record the exact `codroid-api` method behavior if any recovery path differs
  from `clear_emergency_stop`, `clear_overspeed`, `clear_joint_protection`,
  `clear_tool_error`, `clear_drag_not_allowed`, or `clear_robot_error`.

## Gripper

- Connect to the OnRobot Compute Box with `type=auto`.
- Confirm detected type, display name, capabilities, telemetry, busy state, and
  holding state publish through `/state.gripper`.
- Open and close with conservative width/force/speed values.
- Apply orientation on grippers that support it; confirm unsupported models do
  not fail unsafe.
- Disconnect and reconnect without restarting the FastAPI process.

## Plate Test

- Load or apply robot calibration.
- Run `/plate-test/run` with `dry_run=true`; inspect per-target `approach_pose`,
  `pick_pose`, `coordinate_mode`, `plate_check`, `skipped`, and saved JSON under
  `data/plate_test_runs/`.
- Confirm plate re-detection blocks execution when motion exceeds
  `plate_movement_threshold_px`.
- Teach a drop target and run one real plate-test cycle only after dry-run poses
  are correct and the cell is clear.

## Migration Gate

Only after the above passes should a follow-up remove Dash/WebRTC runtime
dependencies and quarantine remaining legacy tests.
