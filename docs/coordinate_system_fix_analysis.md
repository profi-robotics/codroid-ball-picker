# Coordinate System Axis Flip Fix - Analysis and Implementation

**Date:** January 9, 2026  
**Issue:** Robot axes appearing flipped during plate pick test  
**Status:** Fixed

## Problem Statement

During the plate pick test, the robot's movement trajectory was incorrect - axes appeared flipped or inverted. Instead of picking orange balls from calibration points 1, 2, and 3 in the expected sequence, the robot followed an incorrect trajectory (shown as blue line in the attached image).

## Root Cause Analysis

### Coordinate System Architecture

The system uses multiple coordinate frames:

1. **Plate Coordinate System** (Reference Frame)
   - Origin: Top-left of plate
   - +X: Right
   - +Y: Down
   - Calibration points:
     - Point 1: (70, 70) mm - Origin for robot calibration
     - Point 2: (230, 70) mm - Defines +X axis (160mm right from P1)
     - Point 3: (230, 210) mm - Defines XY plane (160mm right, 140mm down from P1)

2. **Robot Base Coordinate System** (Coordinate ID 0)
   - Fixed robot base frame
   - Points captured during calibration are in this frame

3. **Robot User Coordinate System 5** (Coordinate ID 5)
   - Created during robot calibration via three-point calibration
   - Origin: Robot position at Point 1
   - +X: Direction from Point 1 to Point 2
   - +Y: Perpendicular direction, forming right-handed system
   - **CRITICAL**: This is a STATIC frame - it doesn't move/rotate with the plate

### The Bug

The code had three related issues:

1. **Incorrect Default Behavior**: Used base coordinates by default (`PLATE_TEST_USE_BASE_COORDINATES = True`) instead of coordinate system 5 as required.

2. **Mixed Coordinate Frames**: The code applied 2D rotation/translation adjustments (`r2, t2`) to compensate for plate movement, then used these adjusted coordinates in coordinate system 5. This is incorrect because:
   - Coordinate system 5 is STATIC (aligned with reference plate orientation at calibration time)
   - Applying plate rotation adjustments creates coordinates in a CURRENT plate frame
   - Using current-frame coordinates in a static coordinate system causes axis misalignment

3. **Transformation Confusion**: The `baseline_transform` (3D rigid transform from reference plate to robot base) was computed at calibration time but the code was mixing:
   - Reference plate coordinates (no rotation adjustment)
   - Adjusted plate coordinates (with rotation `r2, t2`)
   - Without properly accounting for which frame each belongs to

### Visual Explanation

```
CALIBRATION TIME:
Reference Plate Frame ----baseline_transform----> Robot Base Frame
        |                                                |
        +------ Coord System 5 is aligned here          |
                                                         |
RUNTIME (plate rotated):                                |
Current Plate Frame -----r2, t2 adjustment----------> [should map to base]
        |
        ❌ WRONG: Using adjusted coords in static coord system 5
        ✓ CORRECT: Use reference coords in coord system 5
```

## Solution Implemented

### Changes Made

#### 1. Default Coordinate System (`app/calibration_ui.py:95`)
```python
# Changed from True to False
PLATE_TEST_USE_BASE_COORDINATES = False
```
Now defaults to coordinate system 5 as required.

#### 2. Target Coordinate Storage (`app/calibration_ui.py:1343-1360`)
Modified `_resolve_plate_test_coord5_targets()` to store BOTH coordinate frames:

```python
targets.append({
    "sequence": int(sequence),
    "reference_xy": (float(base_xy[0]), float(base_xy[1])),  # For coord system 5
    "adjusted_xy": (float(adj_xy[0]), float(adj_xy[1])),     # For base system
    "plate_mm": (float(point.center_mm[0]), float(point.center_mm[1])),
})
```

**Rationale:**
- `reference_xy`: Coordinates in the reference (calibration-time) plate frame, no rotation adjustment
- `adjusted_xy`: Coordinates accounting for current plate position/rotation
- Use `reference_xy` with coordinate system 5 (static frame)
- Use `adjusted_xy` with base coordinates (dynamic tracking)

#### 3. Plate Pick Test Coordinate Selection (`app/calibration_ui.py:6378-6383`)
```python
# Use reference coordinates for coord system 5 (static frame)
# Use adjusted coordinates for base system (accounts for plate movement)
coord_xy = target["reference_xy"] if not use_base_coordinates else target["adjusted_xy"]
```

#### 4. Robot Calibration Verification (`app/calibration_ui.py:8020-8155`)
Updated verification to use coordinate system 5:

- Loads plate layout to compute coordinate system 5 positions
- Converts calibration points to positions relative to Point 1 (origin)
- Performs verification moves in coordinate system 5
- Falls back to base coordinates if plate layout unavailable

**Key changes:**
- Switch to coordinate system 5 at start (not base)
- Compute relative positions: `(pt.center_mm - origin_mm)`
- Use `attach_coordinate_to_cpos()` for coordinate system 5 moves
- Move using static reference frame coordinates

## Technical Details

### Coordinate System 5 Definition

Coordinate system 5 is created by the robot controller using three-point calibration:
1. **Point 1**: Defines origin (0, 0, 0)
2. **Point 2**: Defines +X axis direction (vector P1→P2)
3. **Point 3**: Defines XY plane and +Y direction (perpendicular component of P1→P3)

Given the calibration plate layout:
- P1→P2 = (160, 0) in plate coords → Pure +X direction
- P1→P3 = (160, 140) in plate coords → +X and +Y components
- +Y component: (0, 140) after removing parallel component

### Baseline Transform

The `baseline_transform` is a 3D rigid transform computed during calibration:
```python
baseline_transform = _rigid_transform(
    plate_points_relative_to_P1,  # Reference plate coordinates
    robot_points_base              # Robot base coordinates at calibration
)
```

This maps: Reference Plate Frame → Robot Base Frame

### Plate Movement Tracking

When the plate moves/rotates:
1. Detect current marker positions via camera
2. Compute 2D rigid transform `(r2, t2)`: Reference → Current
3. Apply to target positions: `adj_xy = (r2 @ base_xy) + t2`

This adjustment is ONLY valid for base coordinate movements, not coordinate system 5.

## Why the Fix Works

### For Coordinate System 5 (Default)
1. Use reference coordinates (no rotation adjustment)
2. Coordinate system 5 is aligned with reference plate frame
3. Plate can rotate physically, but coord system 5 stays static
4. Reference coordinates map correctly to static frame

### For Base Coordinates (Fallback)
1. Use adjusted coordinates (with rotation)
2. Transform through `baseline_transform` to base frame
3. Accounts for plate movement in robot base space
4. Correct for dynamic plate tracking

## Testing Recommendations

1. **Verification Test**: After calibration, run verification to confirm coordinate system 5 movements hit points 1, 2, 3 correctly

2. **Static Plate Test**: Run plate pick test with plate in original calibration orientation - should work perfectly

3. **Rotated Plate Test**: Rotate plate slightly, run plate pick test - with coordinate system 5, should use reference positions (may miss if plate moved significantly; consider switching to base coordinates for dynamic tracking)

4. **Coordinate System Consistency**: All post-calibration movements should report coordinate system 5 as active

## Related Files

- `/Users/levko/repos/codroid-ball-picker/app/calibration_ui.py`: Main implementation
- `/Users/levko/repos/codroid-ball-picker/data/markers/robot_calibration/calibration_plate_layout.json`: Plate layout definition
- `/Users/levko/repos/codroid-ball-picker/vision/plate.py`: Plate detection and coordinate computation
- `/Users/levko/repos/codroid-ball-picker/vision/workspace.py`: Workspace coordinate definitions

## Key Takeaways

1. **Coordinate System 5 is Static**: Once calibrated, it doesn't track plate movement
2. **Separate Reference from Current**: Always distinguish between reference (calibration-time) and current (runtime) plate frames
3. **Match Coordinates to Frame**: Reference coordinates → Coordinate system 5; Adjusted coordinates → Base system
4. **User Requirement Met**: System now uses coordinate system 5 for all post-calibration movements as specified

## Future Considerations

If dynamic plate tracking is needed (plate moves between calibration and pick operations):
- Option A: Stick with coordinate system 5, accept that it uses reference positions (requires re-calibration if plate moves significantly)
- Option B: Use base coordinates with full dynamic tracking (current adjusted coordinate approach)
- Option C: Implement coordinate system re-calibration when plate movement exceeds threshold

Current implementation: Option A (coordinate system 5 with reference positions)
