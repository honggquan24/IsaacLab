#!/usr/bin/env python3
"""Test script to verify evobot_v1 structure after reorganization."""

import sys
from pathlib import Path

def test_paths():
    """Test that all required files exist."""
    print("=" * 70)
    print("TESTING EVOBOT_V1 STRUCTURE")
    print("=" * 70)

    base_dir = Path(__file__).parent / "source/isaaclab_assets/isaaclab_assets/evobot_v1"

    # Test key files exist
    test_files = [
        "evobot_v1_cfg.py",
        "__init__.py",
        "navigation/__init__.py",
        "navigation/balance/__init__.py",
        "navigation/balance/evobot_v1_balance_cfg.py",
        "navigation/balance/evobot_v1_balance_env_cfg.py",
        "navigation/velocity/__init__.py",
        "navigation/velocity/velocity_env_cfg.py",
        "navigation/manipulation/__init__.py",
        "navigation/manipulation/manipulation_env_cfg.py",
        "navigation/locomotion/__init__.py",
        "navigation/locomotion/locomotion_env_cfg.py",
        "usd_file/evoBOT_cfg.usd",
        "usd_file/evoBOT_v2_cfg.usd",
    ]

    print("\n1. Checking file existence:")
    print("-" * 70)
    all_exist = True
    for file_path in test_files:
        full_path = base_dir / file_path
        exists = full_path.exists()
        status = "✓" if exists else "✗"
        print(f"{status} {file_path}")
        if not exists:
            all_exist = False
            print(f"  Expected: {full_path}")

    return all_exist

def test_balance_usd_path():
    """Test that balance config can find USD file."""
    print("\n2. Testing balance USD path resolution:")
    print("-" * 70)

    cfg_file = Path(__file__).parent / "source/isaaclab_assets/isaaclab_assets/evobot_v1/navigation/balance/evobot_v1_balance_cfg.py"

    # Simulate what the config file does
    CURRENT_DIR = cfg_file.resolve().parent.parent.parent
    usd_file_path = CURRENT_DIR / "usd_file" / "evoBOT_cfg.usd"
    EVOBOT_USD_PATH = usd_file_path.resolve()

    print(f"Config file: {cfg_file}")
    print(f"Going up 3 levels: {CURRENT_DIR}")
    print(f"USD path: {EVOBOT_USD_PATH}")
    print(f"USD exists: {EVOBOT_USD_PATH.exists()}")

    if EVOBOT_USD_PATH.exists():
        print("✓ USD path is correct!")
        return True
    else:
        print("✗ USD path is incorrect!")
        return False

def test_imports():
    """Test Python imports."""
    print("\n3. Testing Python imports:")
    print("-" * 70)

    # Add source paths
    sys.path.insert(0, str(Path(__file__).parent / "source/isaaclab"))
    sys.path.insert(0, str(Path(__file__).parent / "source/isaaclab_assets"))

    try:
        print("Importing isaaclab_assets.evobot_v1...")
        from isaaclab_assets import evobot_v1
        print("✓ evobot_v1 module imported successfully")

        print("Importing navigation submodule...")
        from isaaclab_assets.evobot_v1 import navigation
        print("✓ navigation submodule imported")

        print("Importing balance submodule...")
        from isaaclab_assets.evobot_v1.navigation import balance
        print("✓ balance submodule imported")

        print("Importing velocity submodule...")
        from isaaclab_assets.evobot_v1.navigation import velocity
        print("✓ velocity submodule imported")

        print("Importing manipulation submodule...")
        from isaaclab_assets.evobot_v1.navigation import manipulation
        print("✓ manipulation submodule imported")

        print("Importing locomotion submodule...")
        from isaaclab_assets.evobot_v1.navigation import locomotion
        print("✓ locomotion submodule imported")

        return True

    except Exception as e:
        print(f"✗ Import failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_gym_registration():
    """Test gym environment registrations."""
    print("\n4. Checking gym registrations:")
    print("-" * 70)

    try:
        import gymnasium as gym

        # Need to import to trigger registration
        sys.path.insert(0, str(Path(__file__).parent / "source/isaaclab_assets"))
        from isaaclab_assets import evobot_v1

        evobot_envs = [env_id for env_id in gym.envs.registry.keys() if 'Evobot' in env_id]

        print(f"Found {len(evobot_envs)} Evobot environments registered:")
        for env_id in sorted(evobot_envs):
            print(f"  • {env_id}")

        if evobot_envs:
            print("✓ Gym registrations successful")
            return True
        else:
            print("⚠ No Evobot environments registered (might need Isaac Sim loaded)")
            return None

    except Exception as e:
        print(f"⚠ Could not check gym registrations: {e}")
        return None

def main():
    """Run all tests."""
    test1 = test_paths()
    test2 = test_balance_usd_path()
    test3 = test_imports()
    test4 = test_gym_registration()

    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    print(f"✓ File structure: {'PASS' if test1 else 'FAIL'}")
    print(f"✓ USD path resolution: {'PASS' if test2 else 'FAIL'}")
    print(f"✓ Python imports: {'PASS' if test3 else 'FAIL'}")
    if test4 is not None:
        print(f"✓ Gym registration: {'PASS' if test4 else 'FAIL'}")
    print("=" * 70)

    if test1 and test2 and test3:
        print("\n✓ All critical tests passed!")
        return 0
    else:
        print("\n✗ Some tests failed!")
        return 1

if __name__ == "__main__":
    sys.exit(main())
