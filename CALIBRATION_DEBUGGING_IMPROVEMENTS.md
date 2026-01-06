# Robot Calibration Debugging Improvements

## Summary
Enhanced logging and validation for the robot plate calibration procedure to better diagnose communication and initialization issues.
Added a persistent robot connection monitor and guided, step-by-step calibration status updates in the UI.

## Changes Made

### 1. Enhanced Listener Logging ([app/calibration_ui.py](app/calibration_ui.py#L1070-L1110))
- Added message counting to track listener activity (logs every 50th message to avoid spam)
- Added detailed logging for RobotPosture updates with coordinate values
- Added error handling and exc_info logging for listener exceptions
- Added warning when flange button is pressed but no posture data available yet

**Key Logs:**
```
Listener task started, waiting for robot messages...
Received 50 listener messages so far
RobotPosture: x=..., y=..., z=...
Flange button pressed with posture: {...}
Flange button pressed but no posture data received yet
```

### 2. Plate Visibility Validation ([app/calibration_ui.py](app/calibration_ui.py#L838-L849))
- Added pre-calibration check to ensure camera stream is running and plate is visible
- Added detailed logging showing PLATE_STATUS at calibration start
- Clear error message if camera not active or plate not visible
- Added logging in _ensure_plate_visible() with current status and age of markers

**Key Logs:**
```
Pre-calibration check: camera stream status: PLATE_STATUS={...}
Checking plate visibility... PLATE_STATUS={...}
Plate markers are stale: age=X.XXs (max=2.00s)
```

### 3. Updated Calibration Instructions ([app/calibration_ui.py](app/calibration_ui.py#L2431-L2438))
- Updated step-by-step guide to include starting camera stream as first step
- Now emphasizes that plate must be visible before clicking "Begin calibrating robot"

**New Instructions:**
```
1. Start the camera stream and confirm the plate markers are visible (>= 2)
2. Click 'Begin calibrating robot'
3. Follow the status line below for each point prompt
4. Move the gripper to each groove target (1, 2, then 3)
5. Press the flange button when aligned
6. The robot will calculate the coordinate transform
```

### 4. Persistent Robot Connection + Guided Status ([app/calibration_ui.py](app/calibration_ui.py#L140-L520))
- Robot connection is established once and reused across calibration and pick/place operations
- A single listener/poller updates live robot posture and flange button state
- Calibration status text now updates with step-by-step guidance as points are captured
- Added a manual "Disconnect robot" control to explicitly close the session when needed
- Header now shows live robot connection/position status at all times
- The UI shows "Awaiting posture stream" until the first RobotPosture event is received
- A persistent user websocket login is opened to allow RobotPosture publish events
- Disconnecting now sends best-effort logout messages on both user and robot websockets
- Robot status + calibration guidance refresh on a dedicated interval so posture updates appear immediately
- Robot connection + monitor logic now lives in `codroid_api.RobotSession` to keep the UI thin

## Root Cause Analysis

The "Calibration plate markers are not visible" error was likely caused by:

1. **Camera stream not running**: Users were clicking "Begin calibrating robot" without first starting the camera stream
2. **Plate markers not detected**: The camera stream may have been running but had stale data (> 2 seconds old)
3. **Poor listener initialization**: Listener task was created but messages might not have been flowing yet

## How to Use (Revised Procedure)

1. **Open the Calibration UI** - Navigate to the calibration tab
2. **Start the camera stream**
   - Set resolution, FPS, and autofocus options
   - Click "Start camera stream"
   - Wait for the plate to appear in the video and show stable marker detection
3. **Begin calibration**
   - Once markers are detected and stable, click "Begin calibrating robot"
   - The UI will display detailed status messages at each step
4. **Follow the prompts**
   - Move gripper to Point 1, press flange button
   - Move gripper to Point 2, press flange button
   - Move gripper to Point 3, press flange button
5. **Monitor the logs**
   - Watch the terminal output for detailed debugging information
   - Look for messages about robot position updates and flange button presses

## Debugging Output to Watch For

### Success Indicators
```
Robot API connected, logging in...
Robot login successful
Setting current coordinate ID to 5...
Coordinate ID 5 set
Listener task started, waiting for robot messages...
IO poll task started, requesting IO values...
Flange button pressed with posture: {...}
✓ Point 1 captured. Position: X=..., Y=..., Z=...
```

### Common Failure Points
```
Pre-calibration check: camera stream status: PLATE_STATUS={'count': 0, 'ok': False, ...}
  → Camera stream not running or plate not visible - start camera stream first!

Plate markers are stale: age=3.25s (max=2.00s)
  → Plate visibility data is too old - ensure camera stream is actively running

Flange button pressed but no posture data received yet
  → Listener task not receiving RobotPosture messages - check robot connection

Timed out waiting for flange button 1 press at Point X
  → Button press not detected - ensure flange button is functional and connected
```

## Key Improvements Made in Previous Sessions

### ArUco Marker Detection ([vision/markers.py](vision/markers.py))
- Optimized detector parameters for small 20mm calibration markers
- `adaptiveThreshWinSizeMin = 3`, `adaptiveThreshWinSizeMax = 23`
- `polygonalApproxAccuracyRate = 0.05` for more lenient corner detection
- `cornerRefinementMethod = CORNER_REFINE_SUBPIXEL` for accuracy

### Gripper Integration ([onrobot/twofg.py](onrobot/twofg.py))
- Thread-safe XML-RPC wrapper for concurrent access
- Uses `move()` method for positioning (no grip detection)
- Uses `grip()` method for gripping operations (with detection)
- Prevents "grip detection timeout" errors on slider adjustments

### Workspace Calibration ([vision/workspace.py](vision/workspace.py))
- Fixed Y-axis computation using perpendicular rotation
- `y_axis_px = np.array([-x_axis_px[1], x_axis_px[0]])`
- Matches live overlay coordinate system

## Next Steps if Issues Persist

1. **Enable DEBUG logging** - Set LOGGER level to DEBUG to see more detailed messages
2. **Check robot connectivity** - Verify websocket connection to robot API
3. **Test listener separately** - Create a standalone test script to verify RobotPosture message flow
4. **Validate flange button** - Manually test that flange button triggers DI state changes
5. **Check plate layout** - Verify calibration_plate_layout.json has valid marker IDs and calibration points

## Files Modified
- [app/calibration_ui.py](app/calibration_ui.py) - Enhanced logging, plate visibility checks, updated instructions
- [vision/markers.py](vision/markers.py) - ArUco detector parameter optimization (previous session)
- [onrobot/twofg.py](onrobot/twofg.py) - Thread-safe XML-RPC wrapper (previous session)
- [vision/workspace.py](vision/workspace.py) - Y-axis computation fix (previous session)
