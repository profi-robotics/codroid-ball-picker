# OnRobot 2FG7 Gripper Integration Summary

## Changes Made

### 1. New Files Created

#### `robot/gripper_client.py`
Complete OnRobot 2FG7 gripper client implementation:
- **GripperDevice** class: Manages XML-RPC connection to OnRobot Compute Box
- **TwoFG7Gripper** class: High-level gripper control API
- Methods: `grip_ext()`, `move()`, `stop()`, `is_connected()`, `get_info()`
- Full error handling and logging
- No authentication required (direct TCP/IP connection)

#### `docs/gripper_integration.md`
Comprehensive documentation including:
- Hardware setup instructions
- Network configuration
- API reference with examples
- Troubleshooting guide
- Pick & place workflow examples

### 2. Updated Files

#### `robot/__init__.py`
- Added exports for `GripperDevice` and `TwoFG7Gripper`

#### `app/calibration_ui.py`
Multiple enhancements:

**Imports:**
- Added `from robot.gripper_client import GripperDevice, TwoFG7Gripper`

**Configuration:**
- Added `_load_gripper_defaults()` function
- Added `GRIPPER_DEFAULTS` and `DEFAULT_GRIPPER_IP` constants
- Supports `ONROBOT_CB_IP` and `ONROBOT_T_INDEX` environment variables

**UISession:**
- Added `gripper_device: Optional[GripperDevice]` field
- Added `gripper: Optional[TwoFG7Gripper]` field

**UI Components:**
- **Header:** Added "Gripper: off" badge (id: `header-gripper-badge`)
- **Setup Drawer:** Added OnRobot 2FG7 configuration section with:
  - Compute Box IP input (`gripper-cb-ip`)
  - Enable toggle (`gripper-enable`)
  - Test button (`gripper-test`)
  - Status text (`gripper-status-text`)
  - Device info card (`gripper-device-info`)

**Callbacks:**
- Added `_handle_gripper()` callback for connection management and testing
  - Inputs: `gripper-test` clicks, `gripper-enable` state, `gripper-cb-ip` value
  - Outputs: status text, device info card, header badge text/color
  - Features: connection validation, live status monitoring, grip detection
  
- Updated `_handle_pick_place()` gripper control:
  - Replaced generic robot API parameter control with direct gripper client calls
  - Uses `SESSION.gripper.move()` for open/close commands
  - Returns detailed error messages if gripper not connected

#### `README.md`
- Updated `robot/` layout description to mention gripper control
- Added `docs/` entry for integration guides
- Added "OnRobot 2FG7 Gripper Integration" section with quick setup guide

## Configuration

### Environment Variables

```bash
# OnRobot Compute Box IP address
export ONROBOT_CB_IP="192.168.101.95"

# Tool index (0=single, 1=dual primary, 2=dual secondary)
export ONROBOT_T_INDEX="0"
```

### Default Values

- **Compute Box IP:** `192.168.101.95`
- **XML-RPC Port:** `41414` (hardcoded, OnRobot standard)
- **Tool Index:** `0` (single gripper configuration)

## User Workflow

### Setup Phase

1. Launch calibration UI: `python run_calibration_ui.py`
2. Click **Setup** button in header
3. Scroll to **OnRobot 2FG7 Gripper** section
4. Enter Compute Box IP address (default: 192.168.101.95)
5. Toggle **Enable gripper connection**
6. Click **Test gripper** to verify connection
7. Observe:
   - Status text shows connection result
   - Device info card displays current width, force, and grip state
   - Header badge updates: "Gripper: online" (teal) or "Gripper: error" (red/yellow)

### Operation Phase

**Manual Control (Pick & Place Tab):**
1. Set **Gripper open** value (e.g., 60.0 mm for full open)
2. Set **Gripper close** value (e.g., 20.0 mm for gripping)
3. Click **Open gripper** or **Close gripper** to test
4. Status updates show command results

**Automated Pick & Place:**
- Gripper automatically opens/closes during pick-drop sequences
- Grip detection confirms successful object acquisition
- Errors display if gripper not connected

## Technical Details

### Communication Protocol

- **Protocol:** XML-RPC over TCP/IP
- **Port:** 41414
- **Authentication:** None (direct connection)
- **Device ID:** `0xC0` (TWOFG_ID constant)

### API Methods

**Connection:**
- `GripperDevice.get_cb()` → ServerProxy connection
- `TwoFG7Gripper.is_connected()` → bool

**Control:**
- `move(width, wait)` → Move to width (no grip detection)
- `grip_ext(width, force, speed, wait)` → Grip with force control
- `stop()` → Emergency stop

**Monitoring:**
- `get_ext_width()` → Current width (mm)
- `get_force()` → Current force (N)
- `is_gripped()` → Object detection (bool)
- `is_busy()` → Command execution status (bool)
- `get_info()` → Complete state dictionary

### Return Codes

- `0` (RET_OK) - Success
- `-1` (RET_FAIL) - Command failed
- `-2` (CONN_ERR) - Connection error

## Error Handling

**No Gripper Detected:**
- Yellow badge: "Gripper: error"
- Alert displayed with troubleshooting suggestions
- Possible causes: wrong IP, wrong tool index, gripper not powered

**Connection Timeout:**
- Red badge: "Gripper: error"
- Status text: "Failed to connect to Compute Box"
- Possible causes: network routing, firewall, Compute Box offline

**Command Failures:**
- Status text shows error code
- Logged to console with full exception traceback
- Gripper remains connected for retry

## Testing

All Python files compile successfully:
```bash
python -c "import ast; ast.parse(open('robot/gripper_client.py').read())"
# ✓ Syntax validated
```

No authentication is required, so testing only needs:
1. Valid Compute Box IP address
2. Network connectivity to Compute Box
3. 2FG7 gripper connected and powered

## Future Enhancements

Potential improvements:
- [ ] Add gripper calibration UI (fingertip offset, finger length)
- [ ] Implement grip force monitoring with visual feedback
- [ ] Add gripper preset positions (fully open, pick, place, etc.)
- [ ] Support for dual gripper configurations (tool index 1/2)
- [ ] Integration with DepthAI depth data for adaptive grip force
- [ ] Gripper state logging for pick-place cycle analysis

## References

- OnRobot API: https://github.com/profi-robotics/onrobot-api
- 2FG7 Specification: https://onrobot.com/en/products/2fg7-gripper
- Compute Box: https://onrobot.com/en/compute-box
