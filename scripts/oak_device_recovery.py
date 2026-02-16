#!/usr/bin/env python3
"""
OAK-D device reset and recovery script.

Clears device locks and properly releases the camera for use.
"""

import subprocess
import time
import sys

try:
    import depthai as dai
except ImportError:
    print("ERROR: depthai module not found")
    sys.exit(1)


def kill_depthai_processes():
    """Kill any processes that might be holding the device."""
    print("🔄 Killing conflicting processes...")
    
    commands = [
        "pkill -f depthai",
        "pkill -f 'python.*oak'",
        "pkill -f 'python.*codroid'", 
        "pkill -f run_codroid_picker",
    ]
    
    for cmd in commands:
        try:
            result = subprocess.run(cmd, shell=True, capture_output=True)
            if result.returncode == 0:
                print(f"  ✅ {cmd}")
        except:
            pass


def reset_usb_device():
    """Reset the USB device to clear locks."""
    print("🔌 Resetting USB device...")
    
    try:
        # Find OAK device USB path
        result = subprocess.run(
            "lsusb | grep '03e7:' | head -1", 
            shell=True, capture_output=True, text=True
        )
        
        if result.stdout.strip():
            # Extract bus and device numbers
            parts = result.stdout.split()
            bus = parts[1]
            device = parts[3].rstrip(':')
            
            print(f"  Found OAK device: Bus {bus} Device {device}")
            
            # Reset using usbutils if available
            reset_cmd = f"sudo usbreset /dev/bus/usb/{bus}/{device} 2>/dev/null || echo 'usbreset not available'"
            subprocess.run(reset_cmd, shell=True)
            
            print("  ✅ USB reset attempted")
            return True
            
    except Exception as e:
        print(f"  ❌ USB reset failed: {e}")
        
    return False


def clear_depthai_locks():
    """Clear any DepthAI specific locks."""
    print("🔓 Clearing DepthAI locks...")
    
    try:
        # Get all available devices to trigger cleanup
        devices = dai.Device.getAllAvailableDevices()
        print(f"  Found {len(devices)} device(s)")
        
        # Try to connect briefly to each device to clear locks
        for i, device_info in enumerate(devices):
            try:
                print(f"  Clearing lock on device {i+1}: {device_info.name}")
                
                # Brief connection to release any locks
                bootloader = dai.DeviceBootloader(device_info)
                bootloader.close()
                time.sleep(0.5)
                
            except Exception as e:
                print(f"    ⚠️  Could not clear lock: {e}")
        
        print("  ✅ Lock clearing complete")
        return True
        
    except Exception as e:
        print(f"  ❌ Lock clearing failed: {e}")
        return False


def test_clean_connection():
    """Test if we can now connect cleanly."""
    print("🧪 Testing clean connection...")
    
    try:
        devices = dai.Device.getAllAvailableDevices()
        if not devices:
            print("  ❌ No devices found")
            return False
        
        device_info = devices[0]
        
        # Try minimal connection
        device = dai.Device(device_info, dai.UsbSpeed.HIGH)
        device_id = device.getDeviceInfo().getDeviceId()
        usb_speed = device.getUsbSpeed()
        
        print(f"  ✅ Connected: {device_id} at {usb_speed}")
        
        # Test basic pipeline
        with dai.Pipeline() as pipeline:
            cam = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)
            rgb_out = cam.requestOutput((640, 480), dai.ImgFrame.Type.RGB888p)
            queue = rgb_out.createOutputQueue(maxSize=1, blocking=False)
            
            pipeline.start()
            
            # Brief test
            time.sleep(2)
            frames = 0
            for _ in range(10):
                if queue.tryGet() is not None:
                    frames += 1
                time.sleep(0.1)
            
            print(f"  📷 Received {frames} frames")
            
        device.close()
        
        if frames > 0:
            print("  ✅ Camera is working!")
            return True
        else:
            print("  ⚠️  No frames received")
            return False
            
    except Exception as e:
        print(f"  ❌ Connection test failed: {e}")
        return False


def main():
    print("🚑 OAK-D Device Recovery")
    print("=" * 24)
    print("Clearing device locks and resetting connections\n")
    
    # Step 1: Kill conflicting processes
    kill_depthai_processes()
    time.sleep(2)
    
    # Step 2: Clear DepthAI locks
    clear_depthai_locks()
    time.sleep(2)
    
    # Step 3: Reset USB device
    reset_usb_device()
    time.sleep(3)
    
    # Step 4: Test connection
    if test_clean_connection():
        print("\n✅ SUCCESS: Camera is ready for use!")
        print("\nTo prevent future issues:")
        print("• Always properly close DepthAI devices")
        print("• Use context managers: 'with dai.Pipeline() as pipeline:'")
        print("• Call device.close() explicitly")
        print("• Don't force-quit applications using the camera")
        
        print("\nYou can now run:")
        print("  uv run run_codroid_picker.py --debug --host 0.0.0.0")
        
    else:
        print("\n❌ Recovery failed. Additional steps needed:")
        print("• Unplug camera for 10 seconds, then reconnect")
        print("• Reboot the Jetson")
        print("• Try a powered USB hub")
        print("• Check for hardware issues")


if __name__ == "__main__":
    main()