#!/usr/bin/env python3
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Test script to verify all evobot_v1 imports are correct (consolidated mdp structure)."""

import sys
import traceback

def test_mdp_imports():
    """Test consolidated mdp module imports."""
    try:
        from isaaclab_assets.evobot_v1.mdp import (
            # Balance rewards
            rpy_alignment_imu,
            height_reward,
            angular_velocity_reward,
            linear_velocity_reward,
            feet_contact_force_symmetry,
            joint_pos_target_l2,
            joint_force_balance,
            # Balance observations
            obs_body_roll,
            obs_body_pitch,
            obs_body_yaw,
            lin_vel_b,
            angl_vel_b,
            obs_pos_world,
            # Balance terminations
            reset_when_fall,
            # Navigation rewards
            position_command_error_tanh,
            heading_command_error_abs,
            position_reached_bonus,
            navigation_velocity_reward,
            forward_velocity_tracking,
            lateral_velocity_penalty,
            velocity_goal_alignment,
            goal_progress_reward,
            velocity_towards_goal,
            heading_alignment_reward,
            yaw_rate_penalty,
            joint_velocity_penalty,
            upright_reward,
            tilt_penalty,
            # Hierarchical actions
            PreTrainedBalancePolicyActionCfg,
            # Manipulation utilities
            reward_wheel_speed,
            action_rate_l2,
            joint_acc_l2,
            undesired_contacts,
            reward_man,
        )
        print("✓ Consolidated mdp module imports: OK")
        return True
    except Exception as e:
        print(f"✗ Consolidated mdp module imports FAILED: {e}")
        traceback.print_exc()
        return False

def test_balance_imports():
    """Test balance task imports."""
    try:
        from isaaclab_assets.evobot_v1 import balance
        from isaaclab_assets.evobot_v1.balance import evobot_v1_balance_env_cfg
        from isaaclab_assets.evobot_v1.mdp import (
            rpy_alignment_imu,
            reset_when_fall,
        )
        print("✓ Balance task imports: OK")
        return True
    except Exception as e:
        print(f"✗ Balance task imports FAILED: {e}")
        traceback.print_exc()
        return False

def test_velocity_imports():
    """Test velocity navigation task imports."""
    try:
        from isaaclab_assets.evobot_v1.navigation import velocity
        from isaaclab_assets.evobot_v1.navigation.velocity import velocity_env_cfg
        from isaaclab_assets.evobot_v1.mdp import reward_wheel_speed
        print("✓ Velocity task imports: OK")
        return True
    except Exception as e:
        print(f"✗ Velocity task imports FAILED: {e}")
        traceback.print_exc()
        return False

def test_locomotion_manipulation_imports():
    """Test locomotion-manipulation task imports."""
    try:
        from isaaclab_assets.evobot_v1.navigation import manipulation
        from isaaclab_assets.evobot_v1.navigation.manipulation import manipulation_env_cfg
        from isaaclab_assets.evobot_v1.mdp import (
            action_rate_l2,
            joint_acc_l2,
            undesired_contacts,
            reward_man,
        )
        print("✓ Locomotion-Manipulation task imports: OK")
        return True
    except Exception as e:
        print(f"✗ Locomotion-Manipulation task imports FAILED: {e}")
        traceback.print_exc()
        return False

def test_hierarchical_imports():
    """Test hierarchical navigation task imports."""
    try:
        from isaaclab_assets.evobot_v1.navigation import navigation
        from isaaclab_assets.evobot_v1.navigation.navigation import navigation_env_cfg
        from isaaclab_assets.evobot_v1.mdp import (
            PreTrainedBalancePolicyActionCfg,
            goal_progress_reward,
            velocity_towards_goal,
            heading_alignment_reward,
            lateral_velocity_penalty,
            yaw_rate_penalty,
            joint_velocity_penalty,
            upright_reward,
            tilt_penalty,
            position_command_error_tanh,
            heading_command_error_abs,
            position_reached_bonus,
        )
        print("✓ Hierarchical task imports: OK")
        return True
    except Exception as e:
        print(f"✗ Hierarchical task imports FAILED: {e}")
        traceback.print_exc()
        return False

def test_robot_config():
    """Test robot configuration imports."""
    try:
        from isaaclab_assets.evobot_v1.evobot_v1_cfg import EVOBOT_V1_CFG
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
