# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""MDP components for Rotary Pendulum V2 (Furuta Pendulum).

This module contains observation, reward, and termination functions
for the rotary pendulum swing-up and balance task.
"""

# Observations
from .observations import (
    obs_joint_pos,
    obs_joint_vel,
    obs_joint_pos_sin,
    obs_joint_pos_cos,
)

# Rewards
from .rewards import (
    pendulum_upright_reward,
    pendulum_angular_velocity_penalty,
    pivot_velocity_penalty,
    energy_penalty,
    balance_reward,
    joint_pos_target_l2,
    pivot_heading_tracking_reward,
    action_rate_l2_pendulum,
)
from isaaclab.envs.mdp import action_rate_l2

# Terminations
from .terminations import reset_when_pivot_exceeds_limit

__all__ = [
    # Observations
    "obs_joint_pos",
    "obs_joint_vel",
    "obs_joint_pos_sin",
    "obs_joint_pos_cos",
    # Rewards
    "action_rate_l2",
    "action_rate_l2_pendulum",
    "pendulum_upright_reward",
    "pendulum_angular_velocity_penalty",
    "pivot_velocity_penalty",
    "energy_penalty",
    "balance_reward",
    "joint_pos_target_l2",
    "pivot_heading_tracking_reward",
    # Terminations
    "reset_when_pivot_exceeds_limit",
]
