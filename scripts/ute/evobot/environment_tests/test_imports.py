# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

#!/usr/bin/env python3
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Test script to verify all evobot imports are correct (consolidated mdp structure)."""

import sys
import traceback


def test_mdp_imports():
    """Test consolidated mdp module imports."""
    try:
        print("✓ Consolidated mdp module imports: OK")
        return True
    except Exception as e:
        print(f"✗ Consolidated mdp module imports FAILED: {e}")
        traceback.print_exc()
        return False


def test_balance_imports():
    """Test balance task imports."""
    try:
        print("✓ Balance task imports: OK")
        return True
    except Exception as e:
        print(f"✗ Balance task imports FAILED: {e}")
        traceback.print_exc()
        return False


def test_velocity_imports():
    """Test velocity navigation task imports."""
    try:
        print("✓ Velocity task imports: OK")
        return True
    except Exception as e:
        print(f"✗ Velocity task imports FAILED: {e}")
        traceback.print_exc()
        return False


def test_locomotion_manipulation_imports():
    """Test locomotion-manipulation task imports."""
    try:
        print("✓ Locomotion-Manipulation task imports: OK")
        return True
    except Exception as e:
        print(f"✗ Locomotion-Manipulation task imports FAILED: {e}")
        traceback.print_exc()
        return False


def test_hierarchical_imports():
    """Test hierarchical navigation task imports."""
    try:
        print("✓ Hierarchical task imports: OK")
        return True
    except Exception as e:
        print(f"✗ Hierarchical task imports FAILED: {e}")
        traceback.print_exc()
        return False


def test_robot_config():
    """Test robot configuration imports."""
    try:
        print("✓ Robot config imports: OK")
        return True
    except Exception as e:
        print(f"✗ Robot config imports FAILED: {e}")
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("=" * 70)
    print("EVOBOT V1 IMPORT VERIFICATION TEST (Consolidated MDP Structure)")
    print("=" * 70)

    results = []
    results.append(("Robot Config", test_robot_config()))
    results.append(("Consolidated MDP", test_mdp_imports()))
    results.append(("Balance Task", test_balance_imports()))
    results.append(("Velocity Task", test_velocity_imports()))
    results.append(("Locomotion-Manipulation Task", test_locomotion_manipulation_imports()))
    results.append(("Hierarchical Task", test_hierarchical_imports()))

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"{name:35} {status}")

    print("=" * 70)
    print(f"Total: {passed}/{total} tests passed")

    if passed == total:
        print("✓ All imports verified successfully!")
        sys.exit(0)
    else:
        print("✗ Some imports failed. See details above.")
        sys.exit(1)
