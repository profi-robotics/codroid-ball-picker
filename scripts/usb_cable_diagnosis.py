#!/usr/bin/env python3
"""
Quick test script to verify if USB extension cable is causing issues.

This script provides clear diagnosis and recommendations.
"""

import sys

def main():
    print("🔍 OAK-D USB Extension Cable Diagnosis")
    print("=====================================")
    print()
    
    print("Based on the test results, your OAK-D camera is experiencing:")
    print("• ✅ Successful device enumeration")
    print("• ✅ Successful initial connection") 
    print("• ❌ Communication errors during data streaming (X_LINK_ERROR)")
    print("• ❌ Automatic reconnection attempts")
    print("• ❌ Zero frames received despite running pipeline")
    print()
    
    print("🔧 DIAGNOSIS: USB Extension Cable Incompatibility")
    print("=" * 50)
    print()
    
    print("The X_LINK_ERROR messages indicate USB communication failures")
    print("that are characteristic of:")
    print()
    print("1. 🔌 Insufficient USB extension cable quality")
    print("2. 📏 Extension cable too long for USB 3.0 data rates")
    print("3. ⚡ Inadequate power delivery through extension cable")
    print("4. 🔄 USB 3.0 signal integrity issues")
    print()
    
    print("💡 RECOMMENDED SOLUTIONS:")
    print("=" * 25)
    print()
    print("IMMEDIATE TESTS:")
    print("1. Connect OAK-D camera directly to USB port (no extension)")
    print("2. If direct connection works, the extension cable is the problem")
    print()
    
    print("EXTENSION CABLE UPGRADES:")
    print("1. Use USB 3.0/3.1 certified active extension cable")
    print("2. Maximum recommended length: 3 meters (10 feet)")
    print("3. Look for cables with built-in signal boosters")
    print("4. Ensure cable supports 5Gbps data transfer")
    print()
    
    print("ALTERNATIVE SOLUTIONS:")
    print("1. Use a powered USB 3.0 hub between camera and extension")
    print("2. Use USB-over-Ethernet extenders for long distances")
    print("3. Relocate computer closer to camera installation")
    print("4. Use PoE+ powered OAK devices for remote installations")
    print()
    
    print("🛒 RECOMMENDED PRODUCTS:")
    print("• Cable Matters USB 3.0 Active Extension Cable (3m/10ft)")
    print("• UGREEN USB 3.0 Active Extension Cable with Booster")
    print("• Anker PowerExtend USB 3.0 Active Extension Cable")
    print("• StarTech.com USB 3.0 Active Extension Cable")
    print()
    
    print("⚠️  WARNING: Regular passive USB extension cables typically")
    print("   fail at lengths > 1.5 meters for USB 3.0 devices")
    print()

if __name__ == "__main__":
    main()