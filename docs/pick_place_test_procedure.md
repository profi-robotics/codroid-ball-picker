# Pick & Place Test Procedure (No YOLO)

This procedure validates robot motion from known targets to the drop-off point
without using dynamic detection. Targets are the three calibration plate
indentations labeled **1**, **2**, **3**.

## Prerequisites

1. Camera calibrated (intrinsics + workspace plane).
2. Robot calibrated (plate homography + robot transform).
3. Drop-off position defined.

## Setup

1. Place the calibration plate in the workspace.
2. Place **three orange balls** in indentations **1**, **2**, **3**.
3. Keep the plate markers visible; the test continuously re-tracks the plate
   pose while approaching and picking each target. If detection fails, re-run
   the plate detection step to re-initialize tracking.
4. Verify the drop-off position is still reachable and safe.

## UI Control

Use **Pick & Place → Plate Pick & Place Test** in the calibration UI. Set the
approach/pick/drop heights and the grip timeout, then run the test. The run
returns to the drop position and fully opens the gripper before any targets,
then re-tracks the plate pose continuously during each approach/pick move. Use
**Tracking mode** to switch between standard move updates and segmented
tracking (longer steps when far away, finer steps near the target). The
segment min/max and fraction controls let you tune how many steps the robot
takes during tracking.

After each run, the UI shows a short analysis of commanded vs. reported
positions and saves a capture log under `data/plate_test_runs/` for deeper
inspection.

## Target List

Use the three calibration indentation targets in order: **1 → 2 → 3**. These are
the same three points used during robot calibration.

## Procedure

For each target (1, 2, 3):

1. Move to a safe **approach pose** above the target (Z + approach offset).
2. Descend to the **pick pose** at the target indentation.
3. Close the gripper using **minimum force (20 N)** and **minimum speed (10%)**
   until grip detection is reported or a timeout is reached.
4. If grip detection succeeds:
   - Lift to the approach pose.
   - Move to the drop-off approach pose.
   - Descend to the drop-off pose.
   - Open the gripper fully.
   - Return to the drop-off approach pose.
5. If grip detection fails (no object):
   - Lift to the approach pose.
   - Continue to the next target.

After all targets are processed:

1. Return to the **drop-off pose**.
2. Fully open (extend) the gripper.

## Notes

- Keep the plate markers in view; the test refreshes the plate pose during
  approach/pick moves, but you may need to re-run detection if tracking drops.
- The plate test compares the live pose to the reference pose captured during
  robot calibration. Re-run robot calibration if the plate baseline changes
  significantly.
- The plate test uses base-frame poses derived from the saved robot calibration
  points; re-run robot calibration if the robot controller resets its frames.
- If the drop-off pose was taught under a different robot coordinate frame or
  before recalibrating the robot, re-teach the drop-off position.
- Keep approach heights conservative during testing to avoid collisions.
- If the gripper reports **busy**, wait until it clears before issuing another
  grip command.
- Plate test captures include robot postures, commanded poses, and plate marker
  positions in Coord5 space for offline analysis.
