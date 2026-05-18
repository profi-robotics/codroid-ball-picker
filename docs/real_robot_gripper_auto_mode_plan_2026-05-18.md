# Finished Pick And Place Bring-Up Plan - 2026-05-18

This is the preferred work order for the Monday, May 18, 2026 real-cell
session with the Codroid robot, OnRobot gripper, camera, and controller browser
interface.

The goal is not only to test pieces or collect HAR files. The goal is to use
the robot access to finish the missing implementation, validate it immediately
on real hardware, and leave the robot cell with a usable pick-and-place system
that can boot into an auto operation state, run at production speed when safe,
and support different grippers, target objects, and YOLO detection models
through configuration.

## Current Reality

Do not assume the finished system already exists at the start of the session.
The first part of the day is to identify and validate the behavior that is
already implemented. The second part is implementation against real hardware
feedback. The final part is validation, speed ramp, and reboot proof.

Currently expected to exist:

- FastAPI runtime and operator UI.
- Camera, robot, gripper, calibration, manual pick/drop, plate-test, and
  app-side auto-pick service paths.
- `codroid-api` primitives for robot mode, direct target motion, project
  run/stop, and project JSON edit.
- `onrobot-api` gripper detection and model-specific commands.

Expected to be missing or incomplete until implemented on location:

- Production auto operation as a first-class ball-picker switch/state.
- Auto boot policy: `manual`, `armed`, `auto`.
- Startup orchestration that connects camera/robot/gripper, loads the selected
  model, verifies calibration/drop target, and arms or starts auto.
- Persisted production profiles for YOLO model, target labels/objects, and
  gripper-specific pick behavior.
- UI/API state for `booting`, `armed`, `auto_running`, `blocked`, `faulted`,
  and `manual_override`.
- Controller-auto executor selection and readiness reporting.
- Reboot validation.

Therefore, the working rule is: validate existing functions first, implement
one missing production slice at a time, then immediately prove that slice on the
real cell before moving deeper into auto mode.

## Definition Of Done

The session is successful when all of these are true:

- Ball picker starts as the single operator/runtime application.
- Auto operation is a switch/state in ball picker, not a separate operator
  executable.
- The robot can auto boot into the configured production mode:
  - app service starts
  - camera starts
  - selected YOLO model loads
  - robot connects
  - gripper connects or is auto-detected
  - controller auto mode is selected when safe
  - auto pick loop starts or waits armed, depending on the boot policy
- A complete pick/drop cycle runs from YOLO detection through robot motion,
  gripper close/open, drop, and next-target selection.
- At least one low-speed cycle and one ramped-speed cycle are validated.
- The system supports multiple production profiles:
  - gripper type: `auto`, 2FG7, Soft Gripper, VGC10/VG10/RG where available
  - target labels/objects
  - per-object pick heights, drop heights, grip parameters, and ignore radius
  - selectable YOLO model and confidence settings
- Operator stop, controller stop, heartbeat/stale-state handling, and recovery
  behavior are verified.
- All final settings, calibration files, controller project data, logs, and
  known limitations are saved.

If the full definition cannot be completed, the acceptable fallback is a
working low-speed pick/drop system plus a committed, tested implementation slice
and a precise blocker list backed by logs, `/state` payloads, HAR captures, or
controller screenshots.

## Operating Model

Auto operation must be implemented as a switch in the ball-picker system.

There should be one deployed ball-picker runtime. Auto is a mode/state with a
boot policy, for example:

- `manual`: boot idle, operator starts auto from UI/API.
- `armed`: boot, connect hardware, load model, then wait for operator start.
- `auto`: boot, connect hardware, load model, switch robot/controller to auto
  mode, and start picking once safety gates are clear.

The controller-side program, if used, is an internal managed controller artifact,
not a separate operator executable. Ball picker owns the switch, state machine,
configuration, target selection, gripper behavior, stop handling, and boot
policy. The controller project only executes the motion primitive needed to get
full controller speed.

## Key Findings From Sibling Repos

`codroid-api` is linked from GitHub in `pyproject.toml`; for live multi-repo
development it can also be cloned at `../codroid-api`. It already exposes useful
controller primitives:

- WebSocket login for user and robot channels.
- Robot mode commands through `Robot/Control/command`:
  - manual mode: `3`
  - rescue mode: `4`
  - auto mode: `5`
- Speed multiplier through `Robot/Control/manualMoveRate`.
- Direct target moves by setting `Robot/Control/targetPosType`,
  `Robot/Control/targetCPos`, then command `106` for linear or `107` for
  optimal-path motion.
- Coordinate tagging on CPOS payloads with `attach_coordinate_to_cpos`.
- Project execution through `projexecute/run` and `projexecute/stop`, requiring
  project, task, and label IDs.
- HTTP project editing through `/robot/project/read`,
  `/robot/project/edit`, and `/robot/project/del`.
- Creation of a minimal project document with a task and points container.
- `VariableState`, but not yet general variable value read/write helpers for a
  running controller-side wait loop.

`onrobot-api` is linked from GitHub in `pyproject.toml`; for live multi-repo
development it can also be cloned at `../onrobot-api`. It confirms the gripper
layer can support profile-based production operation:

- Detection order: `twofg7`, `rg2`, `vgc10`, `vg10`, `sg`.
- 2FG7: width, force, speed, busy state, grip detection, orientation.
- Soft Gripper: initialization, width/gentle grip, grip detection.
- VGC10/VG10: vacuum setpoint and vacuum readings.
- Socket.IO status streams can improve telemetry, but XML-RPC commands are
  sufficient for conservative validation.

## Preferred Work Order

| Order | Timebox | Work | Exit Criteria |
| --- | --- | --- | --- |
| 0 | Before access | Sync repos, run offline checks, confirm app starts with fake hardware. | Local build is trustworthy before using robot time. |
| 1 | 15 min | Safety baseline: E-stop, controller stop, robot speed low, gripper clearance, current project backup. | Operator can stop motion immediately and restore controller state. |
| 2 | 20 min | Validate current app bring-up only. Connect camera, robot, and gripper through FastAPI. Check `/state`, posture freshness, DI state, gripper telemetry. | Hardware state is live and reliable enough to develop against. |
| 3 | 30 min | Validate current low-speed robot and gripper primitives. Jog tiny TCP moves and open/close gripper with conservative parameters. | Axis directions, coordinate mode, and gripper behavior are understood. |
| 4 | 45 min | Validate current calibration stack. Load or redo camera/workspace/robot calibration. Teach/verify drop target. Run plate dry-run and inspect poses. | Coordinate transform, Z values, and drop frame are correct. |
| 5 | 30 min | Validate current manual/known-target pick/drop without YOLO ambiguity. | One complete low-speed pick, lift, drop, release, and retreat cycle works. |
| 6 | 30 min | Validate current YOLO/model path. Load a model, confirm detections, target labels, confidence settings, and frame rate. | Detection data is good enough to drive auto implementation. |
| 7 | 30 min | Run current app-side auto if available. Start/stop/pause, safety block, and retry behavior must be observed, not assumed. | Known auto behavior and gaps are written down before coding. |
| 8 | 60-120 min | Implement the production auto switch slice. Add `manual`/`armed`/`auto` boot policy, persisted setting, state fields, and UI/API controls. | Auto is represented as a ball-picker state switch, even before full controller speed work. |
| 9 | 60-120 min | Implement production profiles. Persist selected YOLO model, target labels, per-label motion/grip parameters, and gripper policy. | Different grippers/objects/models are configurable without code edits. |
| 10 | 60-120 min | Implement startup orchestration. On boot, load settings, start camera, load model, connect hardware, load calibration/drop target, and arm/start based on policy. | Reboot can reach `manual`, `armed`, or blocked state predictably. |
| 11 | 30 min | Probe known `codroid-api` auto primitives. Set speed low, switch robot auto mode, send one safe CPOS target, stop, return manual. | Direct auto-mode target motion is proven or ruled out. |
| 12 | 45-90 min | Fill controller gaps only if they block speed. Use HAR/browser inspection for project list/select, task/label IDs, program block JSON, and variable/register channels. | Missing APIs are identified and implemented in `codroid-api` only as needed. |
| 13 | 60-120 min | Implement controller-auto executor if direct CPOS cannot deliver required speed. Keep it behind the auto switch. | Ball picker can select direct or controller-project executor internally. |
| 14 | 45 min | Validate the finished auto switch at low speed. Use known target first, then YOLO-selected target. | Full pick/drop cycle works through the new auto state and profiles. |
| 15 | 45 min | Ramped validation. Repeat at 10%, 25%, 50%, then full speed only when safe. | Production-speed path works with stop and recovery verified. |
| 16 | 30 min | Reboot validation. Restart the app or host service and prove boot policy behavior. | System returns to configured production state after restart. |
| 17 | 15 min | Closeout. Save configs, calibration, code changes, controller project/export, HARs if used, logs, `/state`, test notes. | Work can be reproduced without guessing. |

## On-Location Start Guide

Start with a terminal and browser, but do not start robot motion immediately.

1. Sync code and dependencies:

```bash
cd /workspace/codroid-ball-picker
git pull --ff-only
test -d ../codroid-api || git clone https://github.com/profi-robotics/codroid-api.git ../codroid-api
test -d ../onrobot-api || git clone https://github.com/profi-robotics/onrobot-api.git ../onrobot-api
uv sync --dev
```

2. Run quick static checks:

```bash
uv run python -m compileall -q .
uv run pytest -q
```

If this fails because of robot-cell environment issues, record the error and
continue only after deciding whether it affects the hardware path.

3. Start the current app:

```bash
uv run python run_codroid_picker.py --host 0.0.0.0 --port 8050
```

4. In the UI, validate current implemented behavior in this exact order:
   - camera starts and shows a live frame
   - robot connects and `/state.robot.poll_age_s` is fresh
   - gripper connects or auto-detects and telemetry is live
   - gripper opens/closes at conservative settings
   - tiny robot jogs move in expected directions
   - calibration loads or is redone
   - drop target is taught/verified
   - plate dry-run returns plausible approach/pick/drop poses
   - one known-target low-speed pick/drop works

5. Only after step 4 passes, start implementation work:
   - add or fix auto switch state
   - add or fix boot policy persistence
   - add or fix profile persistence for model/object/gripper settings
   - add or fix startup orchestration
   - validate each change immediately against the real robot state

6. Keep a running notes file during the session:

```text
data/real_cell_notes_2026-05-18.md
```

Record each pass/fail, exact settings used, and any `/state` or controller
evidence needed to continue later.

## Auto Boot Requirements

Implement or verify these before considering the system finished:

- Persistent settings for `auto_boot_policy`: `manual`, `armed`, or `auto`.
- Persistent selected YOLO model path and model confidence/IoU thresholds.
- Persistent target labels and per-label profiles.
- Persistent gripper type policy: `auto` or a specific gripper.
- Persistent drop target and robot coordinate ID.
- Startup sequence:
  1. load settings
  2. start camera
  3. load selected model
  4. connect robot
  5. connect or auto-detect gripper
  6. load calibration and drop target
  7. verify safety gates
  8. switch robot/controller to auto mode only when allowed
  9. start auto loop or arm it based on boot policy
- Clear UI/API state for `booting`, `armed`, `auto_running`, `blocked`,
  `faulted`, and `manual_override`.

Do not auto-start motion after boot if calibration, model, gripper, robot
posture, drop target, or safety status is stale.

## Profile Support

Production behavior must come from configuration, not hard-coded assumptions.

Each target profile should include:

- label name or model class ID
- enabled/disabled
- approach height
- pick height
- drop height
- gripper open setting
- gripper close/vacuum setting
- force, speed, gentle grip, or vacuum threshold as applicable
- grip timeout
- minimum confidence
- optional object width/radius estimate
- drop ignore radius or workspace exclusion radius

Each gripper profile should map to the capabilities exposed by `onrobot-api`:

- 2FG7/RG: width, force, speed, grip detection, busy state.
- Soft Gripper: initialization, width, gentle grip, grip detection.
- VGC10/VG10: vacuum setpoint, hold threshold, channel behavior.

Each model profile should include:

- model file path
- expected class names
- confidence and IoU defaults
- frame skip / inference cadence
- target labels enabled for production
- benchmark result on the deployment hardware

## Controller Auto Strategy

Use the simplest controller path that gets full speed while preserving
ball-picker control.

Preferred order:

1. Direct CPOS executor through existing `codroid-api` primitives if it reaches
   full speed and stop behavior is acceptable.
2. Controller-side project as an internal artifact selected by the auto switch
   if direct CPOS commands cannot reach required speed.
3. Persistent project loop only if a variable/register channel can be discovered
   and validated.
4. One-shot project fallback: update a named point in the project and start the
   task with `projexecute/run(proj, task, label)`.

The operator should never need to launch a second executable. If a controller
project is required, ball picker installs/verifies/selects/starts it as part of
the auto switch and boot policy.

## HAR Capture Scope

HAR capture is a tool for closing controller protocol gaps, not a session goal.
Capture only what is needed to finish the system:

- project list and active project selection
- project export/import if different from HTTP read/edit
- creating a point
- creating a move-to-point block
- creating a variable wait/read/write block if available
- start/stop/pause/reset task
- robot auto/manual mode switch if different from known command `3`/`5`
- status frames while idle, running, stopped, and faulted

Save captures under:

```text
data/controller_captures/2026-05-18/
  notes.md
  *.har
```

Do not commit raw HAR files if they contain credentials, cookies, private IPs,
or proprietary controller project contents.

## Validation Commands

Baseline real-cell hardware slice:

```bash
CODROID_HARDWARE_SMOKE=1 \
CODROID_HARDWARE_CAMERA=1 \
CODROID_HARDWARE_ROBOT=1 \
CODROID_HARDWARE_GRIPPER=1 \
CODROID_HARDWARE_PLATE_DRY_RUN=1 \
uv run pytest -q -m hardware
```

Motion-sensitive tests only in a cleared cell:

```bash
CODROID_HARDWARE_SMOKE=1 \
CODROID_HARDWARE_ROBOT_JOG=1 \
CODROID_HARDWARE_ROBOT_RESCUE=1 \
uv run pytest -q -m hardware
```

After Python changes:

```bash
uv run python -m compileall .
uv run pytest -q
```

## Safety Gates

Do not allow auto motion unless all are true:

- Current robot coordinate ID matches the saved ball-picker calibration.
- Drop target was taught in the same coordinate frame used for motion.
- `/state.robot.poll_age_s` remains below the stale timeout.
- Camera stream is live and detection state is not stale.
- Selected YOLO model is loaded and target labels are configured.
- Gripper telemetry is live and `busy` clears between commands.
- Object/gripper profile exists for the selected target label.
- Operator stop and controller stop both work.
- Auto boot policy is explicit; no motion starts from an unset default.
- A controller project backup/export exists before installing anything new.

Immediately stop auto testing if:

- controller mode, active project, or coordinate frame changes unexpectedly
- any command executes with stale target data
- ball picker loses robot/controller status while motion can continue
- gripper state is unknown before pick or drop
- the robot reports emergency stop, overspeed, joint protection, wrong tool, or
  drag-not-allowed

## If Time Runs Short

Cut scope in this order while preserving a finished usable system:

1. Keep one complete, low-speed, known-target pick/drop path.
2. Keep auto as a ball-picker switch with manual start, even if auto boot is
   left as `armed`.
3. Keep configurable gripper/object/model profiles, even if only one profile is
   fully validated.
4. Use direct `codroid-api` CPOS auto motion before building a controller
   project loop.
5. Use HAR only for the controller gap that blocks production operation.
6. Defer full-speed YOLO only if safety, calibration, or stop behavior is not
   repeatable.

## Closeout Notes To Save

At the end of the session, save:

- final app settings and boot policy
- selected model file and model profile
- target/object profiles
- gripper profile and detected gripper type
- `data/drop_target.json`
- `data/robot_calibration.json` if present
- latest `data/plate_test_runs/plate_test_*.json`
- controller project backup/export if used
- controller HARs and scrubbed notes if used
- ball-picker server log
- `/state` JSON before and after representative cycles
- exact pass/fail notes for each gripper/object/model combination tested
