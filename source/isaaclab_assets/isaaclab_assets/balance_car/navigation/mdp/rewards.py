# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Custom reward functions for balance car navigation."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.utils.math import euler_xyz_from_quat

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# =========================================================
# BASIC POSITION / HEADING REWARDS (GIỮ TƯƠNG THÍCH CŨ)
# =========================================================


def position_command_error_tanh(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str,
) -> torch.Tensor:
    command = env.command_manager.get_command(command_name)
    des_pos_b = command[:, :2]
    distance = torch.norm(des_pos_b, dim=1)
    return 1.0 - torch.tanh(distance / std)


def heading_command_error_abs(
    env: ManagerBasedRLEnv,
    command_name: str,
) -> torch.Tensor:
    command = env.command_manager.get_command(command_name)
    heading = command[:, 3]
    return torch.abs(heading)


def position_reached_bonus(
    env: ManagerBasedRLEnv,
    threshold: float,
    command_name: str,
) -> torch.Tensor:
    command = env.command_manager.get_command(command_name)
    des_pos = command[:, :2]
    distance = torch.norm(des_pos, dim=1)
    return (distance < threshold).float()


# =========================================================
# VELOCITY / ALIGNMENT (WORLD + BODY FRAME CHUẨN)
# =========================================================


def navigation_velocity_reward(
    env: ManagerBasedRLEnv,
    command_name: str,
    scale: float = 1.0,
) -> torch.Tensor:
    command = env.command_manager.get_command(command_name)
    des_pos_b = command[:, :2]

    vel_b = env.scene["robot"].data.root_lin_vel_b[:, :2]
    direction = torch.nn.functional.normalize(des_pos_b, dim=1, eps=1e-6)

    vel_towards_target = torch.sum(vel_b * direction, dim=1)
    return scale * torch.clamp(vel_towards_target, min=0.0)


def forward_velocity_tracking(
    env: ManagerBasedRLEnv,
) -> torch.Tensor:
    return env.scene["robot"].data.root_lin_vel_b[:, 0]


def lateral_velocity_penalty(
    env: ManagerBasedRLEnv,
) -> torch.Tensor:
    return -torch.abs(env.scene["robot"].data.root_lin_vel_b[:, 1])


def velocity_goal_alignment(
    env: ManagerBasedRLEnv,
    command_name: str = "pose_command",
) -> torch.Tensor:
    command = env.command_manager.get_command(command_name)
    goal_pos = command[:, :2]

    robot_pos = env.scene["robot"].data.root_pos_w[:, :2]
    to_goal = torch.nn.functional.normalize(goal_pos - robot_pos, dim=1, eps=1e-6)

    vel_w = env.scene["robot"].data.root_lin_vel_w[:, :2]
    return torch.sum(vel_w * to_goal, dim=1)


# =========================================================
# NAVIGATION CORE REWARD (RESET-SAFE)
# =========================================================


def goal_progress_reward(
    env: ManagerBasedRLEnv,
    command_name: str,
) -> torch.Tensor:
    command = env.command_manager.get_command(command_name)
    goal = command[:, :2]
    pos = env.scene["robot"].data.root_pos_w[:, :2]

    dist = torch.norm(goal - pos, dim=1)

    if "prev_dist" not in env.extras:
        env.extras["prev_dist"] = dist.clone()
        return torch.zeros_like(dist)

    progress = env.extras["prev_dist"] - dist
    env.extras["prev_dist"] = dist.clone()
    return progress


def velocity_towards_goal(
    env: ManagerBasedRLEnv,
    command_name: str,
    stop_radius: float = 0.4,
) -> torch.Tensor:
    command = env.command_manager.get_command(command_name)
    goal = command[:, :2]
    pos = env.scene["robot"].data.root_pos_w[:, :2]

    dist = torch.norm(goal - pos, dim=1)

    vel = env.scene["robot"].data.root_lin_vel_w[:, :2]
    to_goal = torch.nn.functional.normalize(goal - pos, dim=1, eps=1e-6)
    vel_proj = torch.sum(vel * to_goal, dim=1)

    return torch.where(dist > stop_radius, vel_proj, torch.zeros_like(vel_proj))


# =========================================================
# HEADING / STABILITY (ISAAC LAB SAFE)
# =========================================================


def heading_alignment_reward(
    env: ManagerBasedRLEnv,
    command_name: str,
) -> torch.Tensor:
    command = env.command_manager.get_command(command_name)
    desired_yaw = command[:, 3]

    quat = env.scene["robot"].data.root_quat_w
    _, _, robot_yaw = euler_xyz_from_quat(quat)

    err = torch.atan2(
        torch.sin(desired_yaw - robot_yaw),
        torch.cos(desired_yaw - robot_yaw),
    )
    return -torch.abs(err)


def yaw_rate_penalty(
    env: ManagerBasedRLEnv,
) -> torch.Tensor:
    return -torch.abs(env.scene["robot"].data.root_ang_vel_b[:, 2])


def joint_velocity_penalty(
    env: ManagerBasedRLEnv,
    scale: float = 0.01,
) -> torch.Tensor:
    v = env.scene["robot"].data.joint_vel
    return -scale * torch.sum(v**2, dim=1)


def upright_reward(env: ManagerBasedRLEnv) -> torch.Tensor:
    quat = env.scene["robot"].data.root_quat_w
    # body z-axis trong world frame
    z_axis = torch.stack(
        [
            2 * (quat[:, 1] * quat[:, 3] - quat[:, 0] * quat[:, 2]),
            2 * (quat[:, 2] * quat[:, 3] + quat[:, 0] * quat[:, 1]),
            1 - 2 * (quat[:, 1] ** 2 + quat[:, 2] ** 2),
        ],
        dim=1,
    )

    # dot với world up (0,0,1)
    upright = z_axis[:, 2]
    return torch.clamp(upright, min=0.0)


def tilt_penalty(env: ManagerBasedRLEnv) -> torch.Tensor:
    quat = env.scene["robot"].data.root_quat_w
    pitch = torch.asin(2 * (quat[:, 0] * quat[:, 2] - quat[:, 3] * quat[:, 1]))
    roll = torch.atan2(
        2 * (quat[:, 0] * quat[:, 1] + quat[:, 2] * quat[:, 3]),
        1 - 2 * (quat[:, 1] ** 2 + quat[:, 2] ** 2),
    )
    return -(pitch.abs() + roll.abs())
