# OnRobot 2FG7 Gripper Integration

This document describes the OnRobot 2FG7 gripper integration for the Codroid ball-picker system using the [onrobot-api](https://github.com/profi-robotics/onrobot-api) Python package.

## Hardware Setup

### OnRobot Compute Box

The 2FG7 gripper communicates via an **OnRobot Compute Box** using XML-RPC protocol over TCP/IP port 41414.

1. Connect the 2FG7 gripper to the Compute Box
2. Power on the Compute Box
3. Note the Compute Box IP address (default: `192.168.101.180`)
4. Ensure the Compute Box is accessible from your network

### Network Configuration

The Compute Box must be on the same network as the computer running the calibration UI, or routable between networks.

**Default Configuration:**
- **Compute Box IP:** `192.168.101.180`
- **XML-RPC Port:** `41414`
- **Tool Index:** `0` (for single gripper setup)

## Software Configuration

### Installation

The onrobot-api package is included in the `codroid-ball-picker` dependencies via `pyproject.toml`:

```toml
dependencies = [
    "onrobot-api",
    # ... other dependencies
]

[tool.uv.sources]
onrobot-api = { path = "../onrobot-api", editable = true }
```

To install dependencies:
```bash
uv install
```

1. Launch the calibration UI: `python run_calibration_ui.py`
2. Click **Setup** in the header
3. Scroll to the **OnRobot 2FG7 Gripper** section
4. Enter the Compute Box IP address
5. Toggle **Enable gripper connection**
6. Click **Test gripper** to verify connection

## API Reference

### Device Connection

The onrobot-api provides `Device` and `TWOFG` classes for gripper control:

```python
from onrobot.device import Device
from onrobot.twofg import TWOFG

device = Device(Global_cbip="192.168.101.180")
device.getCB()  # Establish XML-RPC connection

gripper = TWOFG(device)
```

### TWOFG Class

High-level controller for 2FG7 gripper operations.

```python
from onrobot.device import Device
from onrobot.twofg import TWOFG

device = Device(Global_cbip="192.168.101.180")
device.getCB()

gripper = TWOFG(device)

# Check connection
if gripper.isConnected(t_index=0):
    print("Gripper connected!")

# Get current state
width = gripper.get_ext_width(t_index=0)
force = gripper.get_force(t_index=0)
is_gripped = gripper.isGripped(t_index=0)

print(f"Width: {width} mm")
print(f"Force: {force} N")
print(f"Gripped: {is_gripped}")

# Move to specific width (no grip detection)
gripper.move(t_index=0, t_width=50.0, f_wait=True)

# Close gripper to grip object (with force control)
gripper.grip(t_index=0, t_width=20.0, n_force=15, p_speed=10, f_wait=True)

# Stop motion
gripper.halt(t_index=0)
```

### Methods

#### Connection & Status

- **`isConnected(t_index) -> bool`** - Check if gripper is connected
- **`getStatus(t_index) -> int`** - Get status code (0=stable, -1=error, -2=no connection)

#### Gripper Control

- **`release(t_index, t_width, p_speed=10, f_wait=True) -> int`**  
  Move to target width without grip detection
  - `t_index`: Tool index (0=single, 1=dual primary, 2=dual secondary)
  - `t_width`: Target width in mm
  - `p_speed`: Speed as percentage (default: 10%)
  - `f_wait`: Block until complete (default: True)
  
- **`grip_ext(t_index, t_width, n_force=20, p_speed=10, f_wait=True) -> int`**  
  Grip with force control and detection
  - `t_index`: Tool index
  - `t_width`: Target width in mm
  - `n_force`: Grip force in Newtons (default: 20 N)
  - `p_speed`: Speed as percentage (default: 10%)
  - `f_wait`: Block until complete (default: True)

- **`halt(t_index) -> int`**  
  Emergency stop gripper motion

#### State Monitoring

- **`get_ext_width(t_index) -> float`** - Current external width in mm
- **`get_force(t_index) -> float`** - Current force in Newtons
- **`isBusy(t_index) -> bool`** - Check if executing command
- **`isGripped(t_index) -> bool`** - Check if object detected

### Return Codes

- `0` (RET_OK) - Success
- `-1` (RET_FAIL) - Command failed
- `-2` (CONN_ERR) - Connection error

## Pick & Place Integration

The gripper is integrated into Applications > Pick & Place:

1. **Configure Gripper Widths:**
   - **Gripper open:** Width in mm when releasing (e.g., 60.0 mm)
   - **Gripper close:** Width in mm when gripping (e.g., 20.0 mm)

2. **Test Gripper:**
   - Click **Open gripper** to move to open width
   - Click **Close gripper** to grip to close width

3. **Automated Pick & Place:**
   - The gripper automatically opens/closes during pick-drop sequences
   - Grip detection confirms successful object acquisition

## Authentication

**Note:** The OnRobot XML-RPC API does **not require authentication** (no username/password). The Compute Box is accessed directly via TCP/IP at the configured IP address.

If your network requires authentication/VPN to reach the Compute Box, configure network-level access separately.

## Troubleshooting

### Gripper Not Detected

**Symptoms:**
- "No 2FG7 gripper detected" error
- Gripper badge shows "error" (yellow/red)

**Solutions:**
1. Verify physical connection between gripper and Compute Box
2. Check Compute Box power and network connection
3. Confirm Compute Box IP address is correct
4. Verify tool index (0 for single gripper)
5. Test XML-RPC connection manually:
   ```python
   import xmlrpc.client
   cb = xmlrpc.client.ServerProxy("http://192.168.101.180:41414/")
   print(cb.cb_is_device_connected(0, 0xC0))  # Should return True for 2FG7
   ```

### Connection Timeout

**Symptoms:**
- Long delay before error message
- "Connection to Compute Box failed" error

**Solutions:**
1. Check network routing (ping Compute Box IP)
2. Verify firewall rules allow port 41414
3. Ensure Compute Box is powered on
4. Verify Compute Box network configuration

### Gripper Commands Fail

**Symptoms:**
- "Grip command timeout"
- Gripper doesn't move

**Solutions:**
1. Check if gripper is in error state (red LED on Compute Box)
2. Power cycle the Compute Box
3. Verify width values are within valid range (check with `get_min_ext_width` / `get_max_ext_width`)
4. Ensure gripper is not mechanically obstructed

## Example Workflow

```python
from onrobot.device import Device
from onrobot.twofg import TWOFG

# Initialize
device = Device(Global_cbip="192.168.101.180")
device.getCB()
gripper = TWOFG(device)

t_index = 0  # Single gripper

# Open gripper
print("Opening gripper...")
gripper.move(t_index=t_index, t_width=60.0, f_wait=True)

# Move to pick position (robot motion)
# ...

# Close gripper to grip object
print("Gripping object...")
gripper.grip(t_index=t_index, t_width=25.0, n_force=15, p_speed=10, f_wait=True)

if gripper.isGripped(t_index=t_index):
    print("Object gripped successfully!")
    
    # Move to drop position (robot motion)
    # ...
    
    # Release object
    print("Releasing object...")
    gripper.move(t_index=t_index, t_width=60.0, f_wait=True)
else:
    print("Failed to grip object")
```

## References

- [OnRobot API Repository](https://github.com/profi-robotics/onrobot-api)
- [OnRobot 2FG7 Product Page](https://onrobot.com/en/products/2fg7-gripper)
- [OnRobot Compute Box Documentation](https://onrobot.com/en/compute-box)
