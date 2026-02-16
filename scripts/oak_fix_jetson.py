#!/usr/bin/env python3
"""
Targeted OAK-D connection fix for Jetson power/USB issues.

This script tries specific workarounds for the Jetson Orin Nano power delivery issues.
"""

import time
import sys

try:
    import depthai as dai
except ImportError:
    print("ERROR: depthai module not found")
    sys.exit(1)


def test_reduced_power_mode():
    """Test with reduced power/performance settings."""
    print("🔋 Testing Reduced Power Mode")
    print("=" * 29)
    
    try:
        print("Attempting connection with USB 2.0 speed limit...")
        devices = dai.Device.getAllAvailableDevices()
        if not devices:
            print("❌ No devices found")
            return False
        
        device_info = devices[0]
        
        # Try USB 2.0 speed
        device = dai.Device(device_info, dai.UsbSpeed.HIGH)
        device_id = device.getDeviceInfo().getDeviceId()
        usb_speed = device.getUsbSpeed()
        
        print(f"✅ Connected: {device_id} at {usb_speed}")
        
        # Test with minimal pipeline
        with dai.Pipeline() as pipeline:
            # Create very simple camera setup
            cam = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)
            
            # Use lower resolution to reduce bandwidth/power
            rgb_out = cam.requestOutput((320, 240), dai.ImgFrame.Type.RGB888p)
            queue = rgb_out.createOutputQueue(maxSize=1, blocking=False)
            
            print("Starting minimal pipeline...")
            pipeline.start()
            
            print("Testing frame capture (5 seconds)...")
            start_time = time.time()
            frame_count = 0
            
            while time.time() - start_time < 5:
                try:
                    frame = queue.tryGet()
                    if frame is not None:
                        frame_count += 1
                except Exception as e:
                    print(f"Frame error: {e}")
                    break
                
                time.sleep(0.1)  # Slower polling
            
            elapsed = time.time() - start_time
            fps = frame_count / elapsed
            
            print(f"Results: {frame_count} frames in {elapsed:.1f}s ({fps:.1f} FPS)")
            
            device.close()
            
            if frame_count > 5:  # At least 1 FPS
                print("✅ SUCCESS: Camera works in reduced power mode!")
                return True
            else:
                print("❌ Still failing in reduced power mode")
                return False
                
    except Exception as e:
        print(f"❌ Reduced power test failed: {e}")
        return False


def test_bootloader_firmware():
    """Test if we can work with the bootloader directly."""
    print("\n🛠️  Testing Bootloader Mode")
    print("=" * 26)
    
    try:
        devices = dai.Device.getAllAvailableDevices()
        if not devices:
            print("❌ No devices found")
            return False
        
        device_info = devices[0]
        print(f"Connecting to bootloader: {device_info.name}")
        
        bootloader = dai.DeviceBootloader(device_info)
        version = bootloader.getVersion()
        print(f"✅ Bootloader version: {version}")
        
        # Check if we can get device info
        bl_device_info = bootloader.getDeviceInfo()
        print(f"Device ID: {bl_device_info.getDeviceId()}")
        
        bootloader.close()
        print("✅ Bootloader communication successful")
        return True
        
    except Exception as e:
        print(f"❌ Bootloader test failed: {e}")
        return False


def test_device_config_options():
    """Test different device configuration options."""
    print("\n⚙️  Testing Device Configurations")
    print("=" * 32)
    
    try:
        devices = dai.Device.getAllAvailableDevices()
        if not devices:
            print("❌ No devices found")
            return False
        
        device_info = devices[0]
        
        # Test 1: Minimal config
        print("1. Testing minimal config...")
        try:
            config = dai.Device.Config()
            device = dai.Device(config, device_info)
            print(f"   ✅ Minimal config works: {device.getDeviceInfo().getDeviceId()}")
            device.close()
            time.sleep(1)
        except Exception as e:
            print(f"   ❌ Minimal config failed: {e}")
        
        # Test 2: USB 2.0 only
        print("2. Testing USB 2.0 mode...")
        try:
            device = dai.Device(device_info, dai.UsbSpeed.HIGH)
            print(f"   ✅ USB 2.0 works: {device.getUsbSpeed()}")
            device.close()
            time.sleep(1)
        except Exception as e:
            print(f"   ❌ USB 2.0 failed: {e}")
        
        # Test 3: Full USB 3.0
        print("3. Testing USB 3.0 mode...")
        try:
            device = dai.Device(device_info, dai.UsbSpeed.SUPER)
            print(f"   ✅ USB 3.0 works: {device.getUsbSpeed()}")
            device.close()
            time.sleep(1)
        except Exception as e:
            print(f"   ❌ USB 3.0 failed: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False


def main():
    print("🎯 OAK-D Connection Fix Attempts")
    print("=" * 33)
    print("Targeting Jetson power delivery and USB stability issues\n")
    
    # Run tests
    success_count = 0
    
    if test_bootloader_firmware():
        success_count += 1
    
    if test_device_config_options():
        success_count += 1
    
    if test_reduced_power_mode():
        success_count += 1
    
    print(f"\n📊 SUMMARY")
    print("=" * 10)
    print(f"Successful tests: {success_count}/3")
    
    if success_count >= 2:
        print("✅ Camera appears to be working!")
        print("\nRECOMMENDED SETTINGS:")
        print("• Use USB 2.0 mode (UsbSpeed.HIGH)")
        print("• Use lower resolution (320x240 or 640x480)")
        print("• Reduce frame rate (10-15 FPS)")
        print("• Use smaller queue sizes (maxSize=1)")
        print("\nTo apply these settings in your main app:")
        print("1. Force USB 2.0: dai.Device(device_info, dai.UsbSpeed.HIGH)")
        print("2. Lower resolution: cam.requestOutput((640, 480), ...)")
        print("3. Smaller queues: createOutputQueue(maxSize=1)")
    elif success_count >= 1:
        print("⚠️  Partial success - camera has limited functionality")
        print("Consider using a powered USB hub or OAK-D-PoE variant")
    else:
        print("❌ All tests failed - hardware issue likely")
        print("Contact Luxonis support with diagnostic results")


if __name__ == "__main__":
    main()