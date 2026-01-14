# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Consolidated MDP components for Evobot V1.

This module contains all observation, reward, termination, and action functions
for the Evobot V1 robot across all tasks (balance, velocity, manipulation, hierarchical).

Instead of scattering MDP components across multiple directories, everything is
consolidated here with clear file naming:
- rewards_balance.py: Balance-specific rewards
- observations_balance.py: Balance observations
- terminations_balance.py: Balance terminations
- rewards_navigation.py: Shared navigation rewards
- rewards_hierarchical.py: Hierarchical navigation rewards
- actions_hierarchical.py: Pre-trained policy action term
- rewards_manipulation.py: Velocity and manipulation utilities
"""

# Balance rewards
from .rewards_balance import (
    rpy_alignment_imu,
    height_reward,
    angular_velocity_reward,
    linear_velocity_reward,
    feet_contact_force_symmetry,
    joint_pos_target_l2,
    joint_force_balance,
)

# Balance observations
from .observations_balance import (
    obs_body_roll,
    obs_body_pitch,
    obs_body_yaw,
    lin_vel_b,
    angl_vel_b,
    obs_pos_world,
)

# Balance terminations
from .terminations_balance import reset_when_fall

# Navigation rewards
from .rewards_navigation import (
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
)

# Hierarchical rewards (initially same as navigation)
from .rewards_hierarchical import (
    position_command_error_tanh as position_command_error_tanh_hierarchical,
    heading_command_error_abs as heading_command_error_abs_hierarchical,
    position_reached_bonus as position_reached_bonus_hierarchical,
    navigation_velocity_reward as navigation_velocity_reward_hierarchical,
    forward_velocity_tracking as forward_velocity_tracking_hierarchical,
    lateral_velocity_penalty as lateral_velocity_penalty_hierarchical,
    velocity_goal_alignment as velocity_goal_alignment_hierarchical,
    goal_progress_reward as goal_progress_reward_hierarchical,
    velocity_towards_goal as velocity_towards_goal_hierarchical,
    heading_alignment_reward as heading_alignment_reward_hierarchical,
    yaw_rate_penalty as yaw_rate_penalty_hierarchical,
    joint_velocity_penalty as joint_velocity_penalty_hierarchical,
    upright_reward as upright_reward_hierarchical,
    tilt_penalty as tilt_penalty_hierarchical,
)

# Hierarchical actions
from .actions_hierarchical import (
    PreTrainedBalancePolicyAction,
    PreTrainedBalancePolicyActionCfg,
)

# Manipulation utilities
from .rewards_manipulation import (
    reward_wheel_speed,
    action_rate_l2,
    joint_acc_l2,
    undesired_contacts,
    reward_man,
)

__all__ = [
    # Balance rewards
    "rpy_alignment_imu",
    "height_reward",
    "angular_velocity_reward",
    "linear_velocity_reward",
    "feet_contact_force_symmetry",
    "joint_pos_target_l2",
    "joint_force_balance",
    # Balance observations
    "obs_body_roll",
    "obs_body_pitch",
    "obs_body_yaw",
    "lin_vel_b",
    "angl_vel_b",
    "obs_pos_world",
    # Balance terminations
    "reset_when_fall",
    # Navigation rewards
    "position_command_error_tanh",
    "heading_command_error_abs",
    "position_reached_bonus",
    "navigation_velocity_reward",
    "forward_velocity_tracking",
    "lateral_velocity_penalty",
    "velocity_goal_alignment",
    "goal_progress_reward",
    "velocity_towards_goal",
    "heading_alignment_reward",
    "yaw_rate_penalty",
    "joint_velocity_penalty",
    "upright_reward",
    "tilt_penalty",
    # Hierarchical rewards (aliased)
    "position_command_error_tanh_hierarchical",
    "heading_command_error_abs_hierarchical",
    "position_reached_bonus_hierarchical",
    "navigation_velocity_reward_hierarchical",
    "forward_velocity_tracking_hierarchical",
    "lateral_velocity_penalty_hierarchical",
    "velocity_goal_alignment_hierarchical",
    "goal_progress_reward_hierarchical",
    "velocity_towards_goal_hierarchical",
    "heading_alignment_reward_hierarchical",
    "yaw_rate_penalty_hierarchical",
    "joint_velocity_penalty_hierarchical",
    "upright_reward_hierarchical",
    "tilt_penalty_hierarchical",
    # Hierarchical actions
    "PreTrainedBalancePolicyAction",
    "PreTrainedBalancePolicyActionCfg",
    # Manipulation utilities
    "reward_wheel_speed",
    "action_rate_l2",
    "joint_acc_l2",
    "undesired_contacts",
    "reward_man",
]
