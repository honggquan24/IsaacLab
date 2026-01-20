#!/usr/bin/env python3
"""Test all relative imports in evobot_v1 configs."""

from pathlib import Path
import re

def check_config_imports():
    """Check all config files for correct relative imports."""
    print("=" * 70)
    print("CHECKING RELATIVE IMPORTS IN EVOBOT_V1 CONFIGS")
    print("=" * 70)

    base_dir = Path("source/isaaclab_assets/isaaclab_assets/evobot_v1")

    test_cases = [
        {
            "file": "navigation/balance/evobot_v1_balance_env_cfg.py",
            "rules": [
                ("from ...mdp import", "Should use 3 dots to reach evobot_v1/mdp"),
                ("from ... import mdp", "Should use 3 dots to reach evobot_v1/mdp"),
            ]
        },
        {
            "file": "navigation/velocity/velocity_env_cfg.py",
            "rules": [
                ("from ...evobot_v1_cfg import", "Should use 3 dots to reach evobot_v1/evobot_v1_cfg.py"),
            ]
        },
        {
            "file": "navigation/manipulation/manipulation_env_cfg.py",
            "rules": [
                ("from ...evobot_v1_cfg import", "Should use 3 dots to reach evobot_v1/evobot_v1_cfg.py"),
            ]
        },
        {
            "file": "navigation/locomotion/locomotion_env_cfg.py",
            "rules": [
                ("from ...mdp import", "Should use 3 dots to reach evobot_v1/mdp"),
            ]
        },
    ]

    print()
    all_pass = True

    for test in test_cases:
        file_path = base_dir / test["file"]
        print(f"\nChecking: {test['file']}")
        print("-" * 70)

        with open(file_path, 'r') as f:
            content = f.read()

        file_pass = True
        for import_pattern, description in test["rules"]:
            # Check if this pattern exists
            found = import_pattern in content
            status = "✓" if found else "✗"
            print(f"  {status} {description}")
            print(f"      Pattern: '{import_pattern}'")

            if not found:
                file_pass = False
                # Try to find similar imports
                similar = re.findall(r'from \.*\w+ import', content)
                if similar:
                    print(f"      Found instead: {similar[:3]}")

        if not file_pass:
            all_pass = False

    print()
    print("=" * 70)
    if all_pass:
        print("✅ ALL IMPORT PATHS CORRECT!")
        return 0
    else:
        print("❌ SOME IMPORT PATHS INCORRECT!")
        return 1

if __name__ == "__main__":
    import sys
    sys.exit(check_config_imports())
