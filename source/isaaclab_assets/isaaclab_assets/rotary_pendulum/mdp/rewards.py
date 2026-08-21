# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reward functions for Rotary Pendulum V2 (Furuta Pendulum) balance/swing-up task.

The goal is to swing the pendulum up and balance it in the upright position.
Key reward signals:
- Pendulum upright: cos(theta2) close to 1 (upright) or -1 (hanging)
- Low angular velocity when balanced
- Energy-efficient control (minimize torque)
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import wrap_to_pi

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def joint_pos_target_l2(env: ManagerBasedRLEnv, target: float, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize joint position deviation from a target value.

    Wraps the angle difference (not the angle itself) to avoid
    discontinuity at the +-pi boundary.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
    angle_error = wrap_to_pi(joint_pos - target)
    return torch.sum(torch.square(angle_error), dim=1)


def pendulum_upright_reward(
    env: ManagerBasedRLEnv,
    pendulum_joint_idx: int = 1,
    scale: float = 7.2,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward for pendulum being upright (cos(theta2) close to 1).

    Returns value in [0, 1] where 1 = perfectly upright (theta2 = pi).
    For Furuta pendulum, upright means pendulum angle = pi (inverted position).
    """
    robot: Articulation = env.scene[asset_cfg.name]
    theta2 = robot.data.joint_pos[:, pendulum_joint_idx]
    # cos(theta2) = -1 when upright (theta2 = pi), = 1 when hanging (theta2 = 0)
    # Reward: (1 - cos(theta2)) / 2 maps to [0, 1] where 1 = upright
    reward = torch.exp(abs(theta2) / scale) - 1
    return reward


def pendulum_angular_velocity_penalty(
    env: ManagerBasedRLEnv,
    pendulum_joint_idx: int = 1,
    swing_scale: float = 1.0,
    balance_scale: float = -1.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalty for high pendulum angular velocity.

    Conditional: penalize when upright (encourage stability),
    reward velocity when not upright (encourage swing-up).

    Args:
        swing_scale: Scale for reward when NOT upright (positive = encourage velocity).
        balance_scale: Scale for reward when upright (negative = penalize velocity).
    """
    robot: Articulation = env.scene[asset_cfg.name]
    dtheta2 = robot.data.joint_vel[:, asset_cfg.joint_ids].squeeze(-1)
    theta2 = robot.data.joint_pos[:, pendulum_joint_idx]
    is_upright = torch.cos(theta2) < -0.99
    vel_sq = torch.square(dtheta2)
    return torch.where(~is_upright, swing_scale * vel_sq * torch.exp(-theta2), balance_scale * vel_sq)


def pivot_velocity_penalty(
    env: ManagerBasedRLEnv,
    pendulum_joint_idx: int = 1,
    swing_scale: float = 1.0,
    balance_scale: float = -1.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalty for high pivot angular velocity.

    Conditional: penalize when upright (energy efficiency),
    reward velocity when not upright (encourage swing-up).

    Args:
        swing_scale: Scale for reward when NOT upright (positive = encourage velocity).
        balance_scale: Scale for reward when upright (negative = penalize velocity).
    """
    robot: Articulation = env.scene[asset_cfg.name]
    dtheta1 = robot.data.joint_vel[:, asset_cfg.joint_ids].squeeze(-1)
    theta2 = robot.data.joint_pos[:, pendulum_joint_idx]
    is_upright = torch.cos(theta2) < -0.99
    vel_sq = torch.square(dtheta1)
    return torch.where(~is_upright, swing_scale * vel_sq, balance_scale * vel_sq)


def energy_penalty(
    env: ManagerBasedRLEnv,
    pendulum_joint_idx: int = 1,
    swing_scale: float = 1.0,
    balance_scale: float = -1.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalty for applied torque (energy efficiency).

    Conditional: penalize when upright, reward when not upright.

    Args:
        swing_scale: Scale for reward when NOT upright (positive = encourage torque).
        balance_scale: Scale for reward when upright (negative = penalize torque).
    """
    robot: Articulation = env.scene[asset_cfg.name]
    theta2 = robot.data.joint_pos[:, pendulum_joint_idx]
    is_upright = torch.cos(theta2) < -0.99
    torques = robot.data.applied_torque
    torque_sq = torch.sum(torch.square(torques), dim=-1)
    return torch.where(~is_upright, swing_scale * torque_sq, balance_scale * torque_sq)


def balance_reward(
    env: ManagerBasedRLEnv,
    vel_threshold: float = 1.0,
    angle_range: float = 10.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Scaled reward when pendulum is near upright AND stable.

    Reward scales linearly from 0 (at 170 deg) to 1 (at 180 deg).
    Only active when angular velocity is below vel_threshold.

    Args:
        vel_threshold: Max angular velocity to be considered stable.
        angle_range: Degrees from upright where reward starts scaling (default 10 = 170-180 deg).
    """
    robot: Articulation = env.scene[asset_cfg.name]
    theta2 = robot.data.joint_pos[:, asset_cfg.joint_ids].squeeze(-1)
    dtheta2 = robot.data.joint_vel[:, asset_cfg.joint_ids].squeeze(-1)

    # Angle deviation from upright (pi)
    angle_dev = torch.abs(wrap_to_pi(theta2 - math.pi))
    max_dev = math.radians(angle_range)

    # Linear scale: 1.0 at 180 deg, 0.0 at (180 - angle_range) deg
    scale = torch.clamp(1.0 - angle_dev / max_dev, min=0.0, max=1.0)

    # Only reward when stable
    is_stable = torch.abs(dtheta2) < vel_threshold
    return torch.where(is_stable, scale, torch.zeros_like(scale))


def pivot_heading_tracking_reward(
    env: ManagerBasedRLEnv,
    command_name: str = "pose_cmd",
    pendulum_joint_idx: int = 1,
    vel_threshold: float = 0.5,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward for tracking the commanded heading angle with the pivot joint.

    Only active when pendulum is upright AND stable (balanced).
    Returns 0 during swing-up phase so it doesn't interfere.

    Computes L2 error between pivot joint angle (Revolute_1) and the
    commanded heading from UniformPose2dCommand.
    Command tensor shape: (num_envs, 3) where [:, 2] = heading.
    """
    robot: Articulation = env.scene[asset_cfg.name]
    pivot_angle = robot.data.joint_pos[:, asset_cfg.joint_ids].squeeze(-1)
    theta2 = robot.data.joint_pos[:, pendulum_joint_idx]
    dtheta2 = robot.data.joint_vel[:, pendulum_joint_idx]
    target_heading = env.command_manager.get_command(command_name)[:, 2]

    is_upright = torch.cos(theta2) < -0.99
    is_stable = torch.abs(dtheta2) < vel_threshold
    is_balanced = is_upright & is_stable

    angle_error = wrap_to_pi(pivot_angle - target_heading)
    return torch.where(is_balanced, torch.square(angle_error), torch.zeros_like(angle_error))


def action_rate_l2_pendulum(
    env: ManagerBasedRLEnv,
    pendulum_joint_idx: int = 1,
    swing_scale: float = 1.0,
    balance_scale: float = -1.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Action rate penalty, conditional on pendulum state.

    Penalize when upright (smooth control), reward when not upright (encourage dynamic actions).

    Args:
        swing_scale: Scale for reward when NOT upright (positive = encourage action change).
        balance_scale: Scale for reward when upright (negative = penalize action change).
    """
    robot: Articulation = env.scene[asset_cfg.name]
    theta2 = robot.data.joint_pos[:, pendulum_joint_idx]
    rate_sq = torch.sum(torch.square(env.action_manager.action - env.action_manager.prev_action), dim=1)
    is_upright = torch.cos(theta2) < -0.99
    return torch.where(~is_upright, swing_scale * rate_sq, balance_scale * rate_sq)
