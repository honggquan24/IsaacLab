"""Navigation reward functions for Legged Robot V5 hierarchical policy.

All functions are called by NavRewardCfg in legged_v3_wheel_navigation_env_cfg.py.
The high-level policy receives these shaped rewards while the low-level locomotion
policy runs inside PreTrainedPolicyAction at a faster decimation rate.
"""
from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def progress_toward_goal(env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Reward for moving closer to the goal each step (delta distance)."""
    asset = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)  # (N, 3) [x, y, yaw] in world frame

    pos_w = asset.data.root_pos_w[:, :2]        # (N, 2)
    goal_w = command[:, :2]                      # (N, 2)

    dist = torch.linalg.norm(goal_w - pos_w, dim=-1)  # (N,)

    if not hasattr(env, "_nav_prev_dist"):
        env._nav_prev_dist = dist.clone()

    delta = env._nav_prev_dist - dist            # positive = moved closer
    env._nav_prev_dist = dist.clone()
    return delta


def goal_reached(env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg,
                 threshold: float = 0.3) -> torch.Tensor:
    """Bonus reward when robot reaches within threshold of goal."""
    asset = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)

    pos_w = asset.data.root_pos_w[:, :2]
    goal_w = command[:, :2]
    dist = torch.linalg.norm(goal_w - pos_w, dim=-1)
    return (dist < threshold).float()


def goal_distance_exp(env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg,
                      std: float = 1.0) -> torch.Tensor:
    """Exponential reward based on distance to goal (peaks at goal)."""
    asset = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)

    pos_w = asset.data.root_pos_w[:, :2]
    goal_w = command[:, :2]
    dist = torch.linalg.norm(goal_w - pos_w, dim=-1)
    return torch.exp(-dist / std)


def heading_toward_goal(env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Reward for robot heading pointing toward goal direction."""
    asset = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)

    pos_w = asset.data.root_pos_w[:, :2]
    goal_w = command[:, :2]
    direction = goal_w - pos_w                   # (N, 2)

    dist = torch.linalg.norm(direction, dim=-1, keepdim=True).clamp(min=1e-3)
    direction_norm = direction / dist            # unit vector to goal

    # Robot heading from quaternion (yaw component)
    quat = asset.data.root_quat_w               # (N, 4) w,x,y,z
    w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    heading = torch.stack([torch.cos(yaw), torch.sin(yaw)], dim=-1)  # (N, 2)

    alignment = (heading * direction_norm).sum(dim=-1)  # cosine similarity
    return alignment.clamp(-1.0, 1.0)
