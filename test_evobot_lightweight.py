#!/usr/bin/env python3
"""Lightweight test for evobot_v1 structure - no Isaac Sim dependencies."""

import sys
import ast
from pathlib import Path

def check_file_syntax(file_path):
    """Check if Python file has valid syntax."""
    try:
        with open(file_path, 'r') as f:
            ast.parse(f.read())
        return True
    except SyntaxError as e:
        print(f"  Syntax error in {file_path}: {e}")
        return False

def check_imports_in_file(file_path):
    """Check that imports in a file point to correct modules."""
    try:
        with open(file_path, 'r') as f:
            tree = ast.parse(f.read())

        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.append(node.module)

        return True, imports
    except Exception as e:
        return False, str(e)

def test_balance_cfg():
    """Test balance config specifically."""
    print("\nTesting balance config USD path logic:")
    print("-" * 70)

    cfg_file = Path("source/isaaclab_assets/isaaclab_assets/evobot_v1/navigation/balance/evobot_v1_balance_cfg.py")

    # Read the file
    with open(cfg_file, 'r') as f:
        content = f.read()

    # Check that it has parent.parent.parent (3 levels)
    if "parent.parent.parent" in content:
        print("✓ Config uses .parent.parent.parent (3 levels)")
    else:
        print("✗ Config doesn't use correct path level")
        return False

    # Check USD file name
    if "evoBOT_cfg.usd" in content:
        print("✓ Config references correct USD file (evoBOT_cfg.usd)")
    else:
        print("✗ Config references wrong USD file")
        return False

    # Verify USD file exists
    usd_file = Path("source/isaaclab_assets/isaaclab_assets/evobot_v1/usd_file/evoBOT_cfg.usd")
    if usd_file.exists():
        print(f"✓ USD file exists: {usd_file}")
    else:
        print(f"✗ USD file not found: {usd_file}")
        return False

    return True

def test_init_files():
    """Test __init__.py files have correct imports."""
    print("\nTesting __init__.py files:")
    print("-" * 70)

    tests = [
        ("source/isaaclab_assets/isaaclab_assets/evobot_v1/__init__.py",
         "should import navigation",
         "from . import navigation"),

        ("source/isaaclab_assets/isaaclab_assets/evobot_v1/navigation/__init__.py",
         "should import balance, velocity, manipulation, locomotion",
         ["balance", "velocity", "manipulation", "locomotion"]),

        ("source/isaaclab_assets/isaaclab_assets/evobot_v1/navigation/velocity/__init__.py",
         "should NOT import hierarchical_vel_env_cfg",
         None),
    ]

    all_pass = True
    for file_path, description, expected in tests:
        print(f"\n{file_path}")
        print(f"  {description}")

        with open(file_path, 'r') as f:
            content = f.read()

        if expected is None:
            # Should NOT contain this
            if "hierarchical_vel_env_cfg" not in content:
                print("  ✓ Does not import hierarchical_vel_env_cfg")
            else:
                print("  ✗ Still imports hierarchical_vel_env_cfg")
                all_pass = False
        elif isinstance(expected, list):
            # Should contain all these
            for item in expected:
                if item in content:
                    print(f"  ✓ Imports {item}")
                else:
                    print(f"  ✗ Missing import: {item}")
                    all_pass = False
        else:
            # Should contain this exact string
            if expected in content:
                print(f"  ✓ Imports {expected}")
            else:
                print(f"  ✗ Missing: {expected}")
                all_pass = False

    return all_pass

def test_gym_registration():
    """Test gym.register calls."""
    print("\nTesting gym registrations:")
    print("-" * 70)

    init_file = Path("source/isaaclab_assets/isaaclab_assets/evobot_v1/__init__.py")

    with open(init_file, 'r') as f:
        content = f.read()

    expected_registrations = [
        'id="Isaac-Evobot-V1-Balance"',
        'id="Isaac-Evobot-V1-Velocity"',
        'id="Isaac-Evobot-V1-Locomotion-Manipulation"',
        'id="Isaac-Evobot-V1-Locomotion"',
        'id="Isaac-Evobot-V1-Locomotion-Play"',
    ]

    print("Expected registrations:")
    all_found = True
    for env_id in expected_registrations:
        if env_id in content:
            print(f"  ✓ {env_id}")
        else:
            print(f"  ✗ {env_id} not found")
            all_found = False

    # Check removed registrations
    removed_registrations = [
        'id="Isaac-Evobot-V1-Velocity-Pretrained"',
        'id="Isaac-Evobot-V1-Navigation-Hierarchical"',
        'id="Isaac-Evobot-V1-Velocity-PID"',
    ]

    print("\nRemoved registrations (should not exist):")
    for env_id in removed_registrations:
        env_name = env_id.replace('id="', '').replace('"', '')
        if env_id not in content:
            print(f"  ✓ {env_name} removed")
        else:
            print(f"  ✗ {env_name} still present")
            all_found = False

    return all_found

def test_syntax():
    """Test all Python files have valid syntax."""
    print("\nTesting Python syntax:")
    print("-" * 70)

    base_dir = Path("source/isaaclab_assets/isaaclab_assets/evobot_v1")
    py_files = list(base_dir.rglob("*.py"))

    print(f"Checking {len(py_files)} Python files...")
    all_valid = True

    for py_file in py_files:
        if not check_file_syntax(py_file):
            all_valid = False
            print(f"  ✗ {py_file.relative_to(base_dir)}")

    if all_valid:
        print(f"  ✓ All {len(py_files)} files have valid syntax")

    return all_valid

def main():
    """Run all tests."""
    print("=" * 70)
    print("EVOBOT_V1 LIGHTWEIGHT STRUCTURE TEST")
    print("=" * 70)

    test1 = test_balance_cfg()
    test2 = test_init_files()
    test3 = test_gym_registration()
    test4 = test_syntax()

    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    print(f"{'✓' if test1 else '✗'} Balance USD path:  {'PASS' if test1 else 'FAIL'}")
    print(f"{'✓' if test2 else '✗'} Init imports:     {'PASS' if test2 else 'FAIL'}")
    print(f"{'✓' if test3 else '✗'} Gym registration: {'PASS' if test3 else 'FAIL'}")
    print(f"{'✓' if test4 else '✗'} Python syntax:    {'PASS' if test4 else 'FAIL'}")
    print("=" * 70)

    if test1 and test2 and test3 and test4:
        print("\n✅ ALL TESTS PASSED! Structure is correct.")
        return 0
    else:
        print("\n❌ Some tests failed!")
        return 1

if __name__ == "__main__":
    sys.exit(main())
