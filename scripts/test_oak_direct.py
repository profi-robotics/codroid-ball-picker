#!/usr/bin/env python3
"""
Direct OAK-D camera test script to diagnose USB extension cable compatibility issues.

This script bypasses the full application stack and directly tests:
1. Device enumeration
2. Pipeline creation and connection
3. Stream stability
4. Error recovery
5. USB link health monitoring

Run this to isolate whether issues are due to USB extension cables or application-level problems.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from typing import Optional

try:
    import depthai as dai
except ImportError:
    print("ERROR: depthai module not found. Run 'uv install' to install dependencies.")
    sys.exit(1)


def setup_logging(debug: bool = False) -> None:
    """Configure logging with appropriate level."""
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )


def list_devices() -> list:
    """List all available OAK devices."""
    print("\n=== Device Enumeration ===")
    devices = dai.Device.getAllAvailableDevices()
    
    if not devices:
        print("❌ No OAK-D devices detected.")
        print("Check:")
        print("  - USB cable connection")
        print("  - Device power")
        print("  - USB permissions (try running with sudo)")
        print("  - Remove any extension cables and test directly")
        return []
    
    print(f"✅ Found {len(devices)} device(s):")
    for idx, device_info in enumerate(devices, 1):
        mxid = getattr(device_info, 'mxid', 'unknown')
        state = getattr(device_info, 'state', 'unknown')
        name = getattr(device_info, 'name', 'unknown')
        print(f"  {idx}) MXID: {mxid} | State: {state} | Name: {name}")
    
    return devices


def test_device_connection(timeout_seconds: int = 10) -> bool:
    """Test basic device connection and pipeline creation."""
    print("\n=== Device Connection Test ===")
    
    try:
        print("Creating pipeline...")
        pipeline = dai.Pipeline()
        
        print("Testing device connection...")
        with pipeline:
            device = pipeline.getDefaultDevice()
            if device is None:
                print("❌ Failed to get default device")
                return False
                
            device_info = device.getDeviceInfo()
            device_id = device_info.getDeviceId()
            print(f"✅ Connected to device: {device_id}")
        
        return True
        
    except Exception as e:
        print(f"❌ Device connection failed: {e}")
        return False


def create_minimal_pipeline() -> Optional[dai.Pipeline]:
    """Create a minimal RGB-only pipeline for testing."""
    try:
        pipeline = dai.Pipeline()
        
        # Create RGB camera node using new v3 API
        cam_rgb = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)
        
        # Request RGB output at 720p - in v3, no XLinkOut needed, direct queue creation
        # The pipeline will handle the output internally
        return pipeline
        
    except Exception as e:
        logging.error(f"Failed to create minimal pipeline: {e}")
        return None


def test_pipeline_stability(duration_seconds: int = 30, test_reconnects: bool = True) -> bool:
    """Test pipeline stability and USB link health."""
    print(f"\n=== Pipeline Stability Test ({duration_seconds}s) ===")
    
    try:
        print("Creating pipeline...")
        with dai.Pipeline() as pipeline:
            print("Creating camera node...")
            cam_rgb = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)
            rgb_output = cam_rgb.requestOutput((1280, 720), dai.ImgFrame.Type.RGB888p)
            
            print("Creating output queue...")
            q_rgb = rgb_output.createOutputQueue(maxSize=4, blocking=False)
            
            print("Starting pipeline...")
            pipeline.start()
            
            device_info = pipeline.getDefaultDevice().getDeviceInfo()
            device_id = device_info.getDeviceId()
            usb_speed = pipeline.getDefaultDevice().getUsbSpeed()
            
            print(f"✅ Connected to {device_id} at {usb_speed}")
            
            print(f"Testing stability for {duration_seconds} seconds...")
            start_time = time.time()
            frame_count = 0
            error_count = 0
            last_error_time = 0
            
            while time.time() - start_time < duration_seconds:
                try:
                    # Try to get frame with timeout
                    msg = q_rgb.tryGet()
                    if msg is not None:
                        frame_count += 1
                        # Every 100 frames, show progress
                        if frame_count % 100 == 0:
                            elapsed = time.time() - start_time
                            fps = frame_count / elapsed if elapsed > 0 else 0
                            print(f"  📷 Frames: {frame_count}, FPS: {fps:.1f}, Errors: {error_count}")
                    
                    time.sleep(0.01)  # Small delay
                    
                except Exception as e:
                    error_count += 1
                    current_time = time.time()
                    
                    # Only log errors every 2 seconds to avoid spam
                    if current_time - last_error_time > 2.0:
                        print(f"  ⚠️  Frame read error: {e}")
                        last_error_time = current_time
                    
                    # If too many errors, exit the test
                    if error_count > 50:
                        print("  ❌ Too many errors, stopping test")
                        break
            
            elapsed = time.time() - start_time
            fps = frame_count / elapsed if elapsed > 0 else 0
            
            print(f"\n📊 Test Results:")
            print(f"  Duration: {elapsed:.1f}s")
            print(f"  Total frames: {frame_count}")
            print(f"  Average FPS: {fps:.1f}")
            print(f"  Total errors: {error_count}")
            print(f"  Error rate: {(error_count/max(frame_count,1))*100:.2f}%")
            
            # Determine if test passed
            if error_count == 0:
                print("✅ Perfect stability - no errors detected")
                return True
            elif error_count < frame_count * 0.05:  # Less than 5% error rate
                print("✅ Good stability - low error rate")
                return True
            else:
                print("❌ Poor stability - high error rate indicates USB issues")
                return False
                
    except Exception as e:
        print(f"❌ Pipeline test failed: {e}")
        return False


def test_usb_link_health() -> bool:
    """Test USB link health with various configurations."""
    print("\n=== USB Link Health Test ===")
    
    # Test different USB speed configurations
    speeds_to_test = [
        (dai.UsbSpeed.SUPER, "USB 3.0 SuperSpeed"),
        (dai.UsbSpeed.HIGH, "USB 2.0 HighSpeed"),
    ]
    
    for usb_speed, description in speeds_to_test:
        print(f"\n--- Testing {description} ---")
        try:
            with dai.Pipeline() as pipeline:
                # Create camera node
                cam_rgb = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)
                rgb_output = cam_rgb.requestOutput((1280, 720), dai.ImgFrame.Type.RGB888p)
                q_rgb = rgb_output.createOutputQueue(maxSize=4, blocking=False)
                
                # Connect with specific USB speed
                device = dai.Device(pipeline, maxUsbSpeed=usb_speed)
                
                device_info = device.getDeviceInfo()
                device_id = device_info.getDeviceId()
                actual_speed = device.getUsbSpeed()
                print(f"✅ Connected at {description}: {device_id} (actual: {actual_speed})")
                
                # Start pipeline
                pipeline.start()
                
                frames_received = 0
                for _ in range(50):  # Try to get 50 frames
                    msg = q_rgb.tryGet()
                    if msg is not None:
                        frames_received += 1
                    time.sleep(0.02)
                
                print(f"  📷 Received {frames_received}/50 frames")
                
                if frames_received < 10:
                    print(f"⚠️  Low frame rate at {description} - potential USB issue")
                else:
                    print(f"✅ Good performance at {description}")
                
        except Exception as e:
            print(f"❌ {description} test failed: {e}")
    
    return True


def main() -> None:
    """Main test function."""
    parser = argparse.ArgumentParser(description="Direct OAK-D camera test for USB extension cable issues")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument("--duration", type=int, default=30, help="Stability test duration in seconds")
    parser.add_argument("--skip-stability", action="store_true", help="Skip the stability test")
    parser.add_argument("--skip-usb-test", action="store_true", help="Skip USB speed tests")
    
    args = parser.parse_args()
    
    setup_logging(args.debug)
    
    print("🔍 OAK-D Direct Connection Test")
    print("================================")
    print("This script tests direct camera connectivity to diagnose USB extension cable issues.")
    print()
    
    # Step 1: List devices
    devices = list_devices()
    if not devices:
        print("\n❌ No devices found. Cannot continue with tests.")
        return
    
    # Step 2: Test basic connection
    if not test_device_connection():
        print("\n❌ Basic connection test failed.")
        return
    
    # Step 3: USB link health test
    if not args.skip_usb_test:
        test_usb_link_health()
    
    # Step 4: Stability test
    if not args.skip_stability:
        success = test_pipeline_stability(args.duration, test_reconnects=True)
        if not success:
            print("\n❌ Stability test failed - this suggests USB extension cable issues.")
            print("\nTroubleshooting recommendations:")
            print("1. Try connecting the camera directly without any extension cables")
            print("2. Use a shorter, higher-quality USB extension cable")
            print("3. Ensure the extension cable supports USB 3.0 SuperSpeed")
            print("4. Check power supply - some extension cables don't provide enough power")
            print("5. Try a powered USB hub if using long cables")
        else:
            print("\n✅ All tests passed - camera connection is stable.")
    
    print("\nTest complete.")


if __name__ == "__main__":
    main()