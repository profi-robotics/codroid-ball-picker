#!/usr/bin/env python3
"""Integration test for OnRobot API refactoring."""

import sys
import ast

print("=" * 70)
print("INTEGRATION TEST: OnRobot API Refactoring")
print("=" * 70)

# Test 1: Verify onrobot package imports
print("\n1. Testing onrobot package imports...")
try:
    from onrobot.device import Device
    from onrobot.twofg import TWOFG
    print("   ✓ Device class imported")
    print("   ✓ TWOFG class imported")
except ImportError as e:
    print(f"   ✗ Import failed: {e}")
    sys.exit(1)

# Test 2: Verify main.py compiles
print("\n2. Testing main.py syntax...")
try:
    with open('app/main.py', 'r') as f:
        code = f.read()
        ast.parse(code)
    print("   ✓ main.py syntax is valid")
except SyntaxError as e:
    print(f"   ✗ Syntax error in main.py: {e}")
    sys.exit(1)

# Test 3: Verify onrobot package structure
print("\n3. Testing onrobot package structure...")
try:
    import onrobot
    print(f"   ✓ onrobot version: {onrobot.__version__}")
    print(f"   ✓ Available exports: {', '.join(onrobot.__all__)}")
except Exception as e:
    print(f"   ✗ Package structure error: {e}")
    sys.exit(1)

# Test 4: Verify TWOFG methods
print("\n4. Testing TWOFG class methods...")
twofg_methods = [m for m in dir(TWOFG) if not m.startswith('_')]
required_methods = ['isConnected', 'isGripped',
                    'get_ext_width', 'get_force', 'move', 'grip']
for method in required_methods:
    if method in twofg_methods:
        print(f"   ✓ {method}() available")
    else:
        print(f"   ✗ {method}() not found")
        sys.exit(1)

# Test 5: Verify Device class has getCB method
print("\n5. Testing Device class methods...")
device_methods = [m for m in dir(Device) if not m.startswith('_')]
if 'getCB' in device_methods:
    print("   ✓ getCB() method available")
else:
    print("   ✗ getCB() method not found")
    sys.exit(1)

# Test 6: Verify gripper imports in main.py
print("\n6. Testing gripper imports in main.py...")
try:
    # Check that the file contains the correct imports
    with open('app/main.py', 'r') as f:
        content = f.read()
        if 'from onrobot.device import Device as GripperDevice' in content:
            print("   ✓ GripperDevice import found")
        else:
            print("   ✗ GripperDevice import not found")
            sys.exit(1)

        if 'from onrobot.twofg import TWOFG as TwoFG7Gripper' in content:
            print("   ✓ TwoFG7Gripper import found")
        else:
            print("   ✗ TwoFG7Gripper import not found")
            sys.exit(1)
except Exception as e:
    print(f"   ✗ Error checking imports: {e}")
    sys.exit(1)

print("\n" + "=" * 70)
print("ALL TESTS PASSED ✓")
print("=" * 70)
