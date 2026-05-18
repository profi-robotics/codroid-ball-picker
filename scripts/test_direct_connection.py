#!/usr/bin/env python3
"""
Simple direct camera test to verify OAK-D functionality without extension cables.

Run this test after connecting the camera directly to confirm the camera itself works.
"""

import time
import sys

try:
    import depthai as dai
except ImportError:
    print("ERROR: depthai module not found. Run 'uv sync' to install dependencies.")
    sys.exit(1)

def test_direct_connection():
    print("🔍 OAK-D Direct Connection Test")
    print("==============================")
    print("This test verifies the camera works when connected directly (no extension cables)")
    print()
    
    try:
        print("1. Checking for available devices...")
        devices = dai.Device.getAllAvailableDevices()
        if not devices:
            print("❌ No OAK-D devices found")
            print("   • Check USB connection")
            print("   • Try a different USB port")
            print("   • Check device power")
            return False
        
        device_info = devices[0]
        print(f"✅ Found device: {device_info.name}")
        
        print("2. Creating minimal test pipeline...")
        with dai.Pipeline() as pipeline:
            # Create simple RGB camera pipeline
            cam = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)
            rgb_out = cam.requestOutput((640, 360), dai.ImgFrame.Type.RGB888p)
            queue = rgb_out.createOutputQueue(maxSize=2, blocking=False)
            
            print("3. Starting pipeline...")
            pipeline.start()
            
            device = pipeline.getDefaultDevice()
            device_id = device.getDeviceInfo().getDeviceId()
            usb_speed = device.getUsbSpeed()
            
            print(f"✅ Connected: {device_id}")
            print(f"✅ USB Speed: {usb_speed}")
            
            print("4. Testing frame capture (10 seconds)...")
            start_time = time.time()
            frames_received = 0
            errors = 0
            
            while time.time() - start_time < 10:
                try:
                    frame = queue.tryGet()
                    if frame is not None:
                        frames_received += 1
                        if frames_received % 50 == 0:
                            print(f"   📷 Received {frames_received} frames")
                except Exception as e:
                    errors += 1
                    if errors == 1:  # Only print first error
                        print(f"   ❌ Error: {e}")
                
                time.sleep(0.02)
            
            elapsed = time.time() - start_time
            fps = frames_received / elapsed
            
            print(f"\n📊 Results:")
            print(f"   Duration: {elapsed:.1f}s")
            print(f"   Frames: {frames_received}")
            print(f"   FPS: {fps:.1f}")
            print(f"   Errors: {errors}")
            
            if frames_received > 100 and errors == 0:
                print("✅ EXCELLENT: Camera works perfectly when connected directly")
                print("   The issue is definitely with the USB extension cable.")
                return True
            elif frames_received > 50:
                print("✅ GOOD: Camera works with minor issues when connected directly")
                print("   Extension cable would likely make issues worse.")
                return True
            else:
                print("❌ POOR: Camera has issues even when connected directly")
                print("   This suggests a problem with the camera or USB port.")
                return False
                
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

def main():
    success = test_direct_connection()
    print()
    
    if success:
        print("🔧 NEXT STEPS:")
        print("1. Keep camera connected directly for reliable operation")
        print("2. Purchase a high-quality USB 3.0 active extension cable")
        print("3. Test the new extension cable with this script")
        print("4. Consider a powered USB hub for very long distances")
    else:
        print("🔧 TROUBLESHOOTING:")
        print("1. Try a different USB 3.0 port on your computer")
        print("2. Check if the camera needs a firmware update")
        print("3. Test with a different USB cable (if removable)")
        print("4. Contact Luxonis support if issues persist")

if __name__ == "__main__":
    main()
