# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Manipulation and velocity-specific reward functions for Evobot V1."""

from __future__ import annotations

import torch
import math
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg
from isaaclab.assets import Articulation

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# =========================================================
# VELOCITY-SPECIFIC REWARDS
# =========================================================


def reward_wheel_speed(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Reward for controlled wheel speed (velocity tracking)."""
    commands = env.command_manager.get_command("base_velocity")
    target_lin_vel = commands[:, :2]
    robot = env.scene["robot"]
    actual_lin_vel = robot.data.root_lin_vel_w[:, :2]
    vel_error = torch.norm(actual_lin_vel - target_lin_vel, dim=-1)
    reward = torch.exp(-5.0 * vel_error)
    return reward


# =========================================================
# LOCOMOTION-MANIPULATION UTILITIES
# =========================================================


def action_rate_l2(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize action changes for smooth control."""
    return torch.sum(torch.square(env.action_manager.action - env.action_manager.prev_action), dim=1)


def joint_acc_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize joint accelerations."""
    asset: Articulation = env.scene[asset_cfg.name]
    joint_vel = asset.data.joint_vel

    if not hasattr(asset.data, "prev_joint_vel"):
        asset.data.prev_joint_vel = joint_vel.clone()
        return torch.zeros(env.num_envs, device=env.device)

    joint_acc = (joint_vel - asset.data.prev_joint_vel) / env.physics_dt
    asset.data.prev_joint_vel = joint_vel.clone()

    return torch.sum(torch.square(joint_acc), dim=1)


def undesired_contacts(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    threshold: float = 1.0,
) -> torch.Tensor:
    """Penalize undesired contacts (e.g., arm touching ground)."""
    contact_sensor = env.scene.sensors[sensor_cfg.name]
    net_contact_force = torch.norm(contact_sensor.data.net_forces_w, dim=-1)
    contact_penalty = torch.where(
        net_contact_force > threshold,
        net_contact_force - threshold,
        torch.zeros_like(net_contact_force),
    )
    return contact_penalty


def reward_man(
    env: ManagerBasedRLEnv,
    command_name: str,
    std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Tanh-based position tracking reward for manipulation targets.

    Rewards the agent for moving the end-effector towards the commanded position.
    """
    command = env.command_manager.get_command(command_name)
    target_pos = command[:, :3]

    asset: Articulation = env.scene[asset_cfg.name]
    ee_pos = asset.data.body_pos_w[:, asset_cfg.body_ids[0], :]

    pos_error = torch.norm(ee_pos - target_pos, dim=-1)
    reward = torch.tanh(pos_error / std).unsqueeze(-1).squeeze(-1)
    reward = 1.0 - reward

    return reward

def joint_pos_target_l2(env: ManagerBasedRLEnv, target: float, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize joint position deviation from a target value."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # wrap the joint positions to (-pi, pi)
    joint_pos = wrap_to_pi(
        asset.data.joint_pos[:, asset_cfg.joint_ids]
    )
    # compute the reward
    return torch.sum(torch.square(joint_pos - target), dim=1)
