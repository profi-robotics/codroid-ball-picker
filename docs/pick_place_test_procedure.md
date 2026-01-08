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
3. Re-detect the plate (run the plate detection step) to refresh the plate pose
   if the plate moved. This updates the target offsets used by the test.
4. Verify the drop-off position is still reachable and safe.

## UI Control

Use **Pick & Place → Plate Pick & Place Test** in the calibration UI. Set the
approach/pick/drop heights and the grip timeout, then run the test. The run
always re-tracks the plate pose before moving.

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

- Re-run plate detection any time the plate is repositioned to avoid offset
  errors.
- Keep approach heights conservative during testing to avoid collisions.
- If the gripper reports **busy**, wait until it clears before issuing another
  grip command.
