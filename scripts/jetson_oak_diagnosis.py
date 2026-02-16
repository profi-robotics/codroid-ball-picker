#!/usr/bin/env python3
"""
Comprehensive Jetson + OAK-D diagnostic script for hardware-level issues.

This script diagnoses deeper hardware problems that persist even with direct connections.
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

try:
    import depthai as dai
except ImportError:
    print("ERROR: depthai module not found. Run 'uv install' to install dependencies.")
    sys.exit(1)


def run_command(cmd: str) -> tuple[int, str]:
    """Run shell command and return exit code and output."""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=10)
        return result.returncode, result.stdout + result.stderr
    except subprocess.TimeoutExpired:
        return -1, "Command timed out"
    except Exception as e:
        return -1, f"Command failed: {e}"


def check_jetson_info() -> None:
    """Check Jetson hardware information."""
    print("🤖 JETSON HARDWARE DIAGNOSTICS")
    print("=" * 30)
    
    # Jetson model
    exit_code, output = run_command("cat /proc/device-tree/model 2>/dev/null || echo 'Unknown'")
    model = output.strip()
    print(f"Model: {model}")
    
    # Jetpack version
    exit_code, output = run_command("dpkg -l | grep nvidia-jetpack | awk '{print $3}' | head -1")
    jetpack_version = output.strip() if output.strip() else "Unknown"
    print(f"JetPack: {jetpack_version}")
    
    # L4T version
    exit_code, output = run_command("cat /etc/nv_tegra_release 2>/dev/null")
    l4t_info = output.strip() if output.strip() else "Unknown"
    print(f"L4T: {l4t_info}")
    
    # Power mode
    exit_code, output = run_command("nvpmodel -q 2>/dev/null")
    if "NV Power Mode" in output:
        power_mode = output.split("NV Power Mode: ")[1].split("\n")[0]
        print(f"Power Mode: {power_mode}")
    else:
        print("Power Mode: Unknown")
    
    # Temperature
    exit_code, output = run_command("cat /sys/devices/virtual/thermal/thermal_zone*/temp 2>/dev/null")
    if output.strip():
        temps = [int(t) // 1000 for t in output.strip().split("\n") if t.isdigit()]
        if temps:
            print(f"CPU Temp: {max(temps)}°C")
    
    print()


def check_usb_system() -> None:
    """Check USB subsystem health."""
    print("🔌 USB SYSTEM DIAGNOSTICS")
    print("=" * 25)
    
    # USB controllers
    exit_code, output = run_command("lspci | grep -i usb")
    if exit_code == 0 and output.strip():
        print("USB Controllers:")
        for line in output.strip().split("\n"):
            print(f"  • {line}")
    else:
        print("❌ No USB controllers found")
    
    # USB devices
    exit_code, output = run_command("lsusb")
    if exit_code == 0:
        print("\nUSB Devices:")
        luxonis_found = False
        for line in output.strip().split("\n"):
            if "03e7" in line.lower():  # Luxonis vendor ID
                print(f"  🎥 {line}")
                luxonis_found = True
            else:
                print(f"  • {line}")
        
        if not luxonis_found:
            print("  ❌ No Luxonis/OAK devices found in USB list")
    
    # USB power management
    exit_code, output = run_command("find /sys/bus/usb/devices -name 'power/control' -exec sh -c 'echo \"{}: $(cat {})\"' \\; 2>/dev/null")
    if exit_code == 0 and output.strip():
        print("\nUSB Power Management:")
        for line in output.strip().split("\n")[:5]:  # Show first 5
            print(f"  {line}")
    
    # Check if device is in unrecoverable state
    exit_code, output = run_command("dmesg | grep -i 'usb.*disconnect\\|usb.*error' | tail -10")
    if output.strip():
        print("\nRecent USB errors:")
        for line in output.strip().split("\n"):
            print(f"  ⚠️  {line}")
    
    print()


def check_power_supply() -> None:
    """Check power supply capabilities."""
    print("⚡ POWER SUPPLY DIAGNOSTICS") 
    print("=" * 28)
    
    # Check power supply info
    exit_code, output = run_command("cat /sys/class/power_supply/*/type /sys/class/power_supply/*/status 2>/dev/null")
    if output.strip():
        print("Power Supply Status:")
        for line in output.strip().split("\n"):
            print(f"  {line}")
    
    # USB port power capability
    exit_code, output = run_command("cat /sys/bus/usb/devices/*/bMaxPower 2>/dev/null | sort -u")
    if output.strip():
        print("\nUSB Port Max Power:")
        for power in output.strip().split("\n"):
            if power.isdigit():
                power_mw = int(power) * 2  # Convert from 2mA units to mW (assuming 5V)
                print(f"  {power_mw}mW ({power} units)")
    
    # Voltage/current monitoring if available
    exit_code, output = run_command("cat /sys/class/hwmon/*/in*_input 2>/dev/null | head -5")
    if output.strip():
        print("\nVoltage readings (if available):")
        for line in output.strip().split("\n"):
            if line.isdigit():
                voltage_mv = int(line)
                print(f"  {voltage_mv}mV")
    
    print()


def test_device_recovery() -> bool:
    """Attempt device recovery procedures."""
    print("🔧 DEVICE RECOVERY ATTEMPTS")
    print("=" * 27)
    
    try:
        # Check if any devices are available
        devices = dai.Device.getAllAvailableDevices()
        if not devices:
            print("❌ No devices found for recovery")
            return False
        
        device_info = devices[0]
        print(f"🎯 Attempting recovery on: {device_info.name}")
        
        # Try bootloader mode recovery
        try:
            print("1. Attempting bootloader recovery...")
            bootloader = dai.DeviceBootloader(device_info)
            print("   ✅ Bootloader accessible")
            
            # Get bootloader info
            version = bootloader.getVersion()
            print(f"   Bootloader version: {version}")
            
            # Try to close cleanly
            bootloader.close()
            time.sleep(2)
            
        except Exception as e:
            print(f"   ❌ Bootloader recovery failed: {e}")
        
        # Try minimal device connection
        try:
            print("2. Attempting minimal device connection...")
            config = dai.Device.Config()
            device = dai.Device(config, device_info)
            
            device_id = device.getDeviceInfo().getDeviceId()
            print(f"   ✅ Minimal connection successful: {device_id}")
            
            device.close()
            time.sleep(2)
            return True
            
        except Exception as e:
            print(f"   ❌ Minimal connection failed: {e}")
        
        # Try different USB speeds
        for speed, name in [(dai.UsbSpeed.HIGH, "USB 2.0"), (dai.UsbSpeed.SUPER, "USB 3.0")]:
            try:
                print(f"3. Attempting {name} connection...")
                device = dai.Device(device_info, speed)
                print(f"   ✅ {name} connection successful")
                device.close()
                time.sleep(2)
                return True
            except Exception as e:
                print(f"   ❌ {name} connection failed: {e}")
        
        return False
        
    except Exception as e:
        print(f"❌ Recovery procedure failed: {e}")
        return False


def check_depthai_installation() -> None:
    """Check DepthAI installation and environment."""
    print("📦 DEPTHAI INSTALLATION CHECK")
    print("=" * 29)
    
    # DepthAI version
    try:
        version = dai.__version__
        print(f"DepthAI version: {version}")
    except:
        print("❌ Cannot determine DepthAI version")
    
    # Check for conflicting installations
    exit_code, output = run_command("pip list | grep -i depthai")
    if output.strip():
        print("Python DepthAI packages:")
        for line in output.strip().split("\n"):
            print(f"  {line}")
    
    # Check library paths
    try:
        import depthai
        lib_path = depthai.__file__
        print(f"Library path: {lib_path}")
    except:
        pass
    
    print()


def provide_jetson_recommendations() -> None:
    """Provide Jetson-specific recommendations."""
    print("💡 JETSON-SPECIFIC RECOMMENDATIONS")
    print("=" * 35)
    print()
    
    print("IMMEDIATE ACTIONS:")
    print("1. 🔋 Check power supply - Jetson may not provide enough USB power")
    print("   sudo nvpmodel -m 0  # Set max performance mode")
    print("   sudo jetson_clocks   # Enable maximum clocks")
    print()
    
    print("2. 📏 Try different USB 3.0 ports")
    print("   - Use blue USB 3.0 ports if available")
    print("   - Avoid USB hubs")
    print("   - Try USB-C to USB-A adapter if needed")
    print()
    
    print("3. 🔧 Reset USB subsystem:")
    print("   sudo modprobe -r xhci_hcd")
    print("   sleep 5")
    print("   sudo modprobe xhci_hcd")
    print()
    
    print("4. ⚡ Use powered USB hub")
    print("   - OAK-D requires up to 2.5W (500mA @ 5V)")
    print("   - Jetson USB ports may be power-limited")
    print("   - External powered hub can provide stable power")
    print()
    
    print("HARDWARE SOLUTIONS:")
    print("• Use OAK-D-PoE for network-powered operation")
    print("• Add external USB 3.0 PCIe card (if expansion available)")
    print("• Use USB-C PD power injector cable")
    print()
    
    print("SOFTWARE SOLUTIONS:")
    print("• Update to latest JetPack/L4T")
    print("• Check for DepthAI compatibility with your JetPack version")
    print("• Try DepthAI SDK with older/newer versions")
    print()


def main() -> None:
    """Main diagnostic function."""
    print("🚨 JETSON + OAK-D HARDWARE DIAGNOSTICS")
    print("=" * 39)
    print("Direct connection still failing - investigating hardware issues")
    print()
    
    # System diagnostics
    check_jetson_info()
    check_usb_system()
    check_power_supply()
    check_depthai_installation()
    
    # Recovery attempts
    recovery_success = test_device_recovery()
    print()
    
    # Recommendations
    provide_jetson_recommendations()
    
    print("🔍 DIAGNOSIS SUMMARY")
    print("=" * 19)
    if recovery_success:
        print("✅ Device recovery partially successful")
        print("   Issue likely related to power or USB stability")
        print("   Try powered USB hub or different ports")
    else:
        print("❌ Device recovery failed")
        print("   Possible hardware failure or incompatibility")
        print("   Contact Luxonis support with these diagnostic results")


if __name__ == "__main__":
    main()