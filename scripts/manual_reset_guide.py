#!/usr/bin/env python3
"""
Manual OAK-D reset procedure.

Provides step-by-step manual reset instructions.
"""

def main():
    print("🔧 MANUAL OAK-D RESET PROCEDURE")
    print("=" * 35)
    print()
    
    print("The camera is stuck in an unrecoverable state.")
    print("Please follow these steps in order:")
    print()
    
    print("STEP 1: Physical Reset")
    print("=" * 18)
    print("1. Unplug the OAK-D camera from USB completely")
    print("2. Wait 10 seconds for capacitors to discharge")
    print("3. Plug it back into a DIFFERENT USB port")
    print("4. Wait 5 seconds for it to enumerate")
    print()
    
    print("STEP 2: USB Port Selection")
    print("=" * 23)
    print("• Use a direct USB 3.0 port (blue port if available)")
    print("• AVOID the USB hub (Realtek 4-Port hub detected)")
    print("• Try the USB-C port with an adapter if available")
    print("• Do NOT use any extension cables")
    print()
    
    print("STEP 3: Power Considerations") 
    print("=" * 25)
    print("• The OAK-D needs 2.5W (500mA @ 5V) peak power")
    print("• Jetson USB ports may be power-limited")
    print("• Consider using a powered USB 3.0 hub")
    print("• Or try a USB-C to USB-A cable for better power")
    print()
    
    print("STEP 4: Test After Reset")
    print("=" * 21)
    print("After physical reset, run:")
    print("  uv run scripts/test_direct_connection.py")
    print()
    
    print("STEP 5: Alternative Solutions")
    print("=" * 26)
    print("If manual reset doesn't work:")
    print("• Reboot the entire Jetson")
    print("• Try a powered USB 3.0 hub")
    print("• Consider OAK-D-PoE for network power")
    print("• Contact Luxonis support")
    print()
    
    print("⚠️  WARNING: The USB hub is likely causing power issues")
    print("   Connect directly to Jetson USB ports for best results")


if __name__ == "__main__":
    main()