# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from isaaclab.utils.math import euler_xyz_from_quat

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def reward_angle_r(
    env: ManagerBasedRLEnv,
    target: float = 90 * math.pi / 180,
    scale: float = 2.0,
):
    imu = env.scene["imu"]
    quat = imu.data.quat_w
    roll, _, _ = euler_xyz_from_quat(quat)

    err = torch.abs(roll) - target
    reward = scale * (-0.9 + torch.cos(err))
    return reward


def reward_angle_y(
    env: ManagerBasedRLEnv,
    target: float = 90 * math.pi / 180,
    scale: float = 2.0,
):
    imu = env.scene["imu"]
    quat = imu.data.quat_w
    _, _, yaw = euler_xyz_from_quat(quat)

    err = yaw - target
    reward = scale * (-0.9 + torch.cos(err))
    return reward


def reward_vel(
    env: ManagerBasedRLEnv,
    target: float = 0.0,
    scale: float = 0.005,
):
    robot = env.scene["robot"]
    joint_vel = robot.data.joint_vel

    err1 = torch.abs(joint_vel[:, 0]) - target
    err2 = torch.abs(joint_vel[:, 1]) - target

    reward = torch.exp(-scale * (err1**2 + err2**2))

    return reward


def bonus_reward(
    env: ManagerBasedRLEnv,
    target: float = 0.0,
    bonus: float = 0.5,
):
    imu = env.scene["imu"]
    quat = imu.data.quat_w
    roll_, _, _ = euler_xyz_from_quat(quat)
    roll = torch.abs(roll_)

    cond = (roll > (88 * math.pi / 180)) & (roll < (92 * math.pi / 180))

    reward = torch.where(cond, torch.full_like(roll, bonus), 0.0)
    return reward


def penalty_when_center_of_env_l2(
    env: ManagerBasedRLEnv, target_x: float = 0.0, target_y: float = 0.0, scale: float = 0.1
):
    robot = env.scene["robot"]
    pos = robot.data.root_pos_w - env.scene.env_origins

    err_x = pos[:, 0] - target_x
    err_y = pos[:, 1] - target_y

    penalty = -scale * (err_x**2 + err_y**2)
    return penalty


def reward_li_vel(
    env: ManagerBasedRLEnv,
    target: float = 0.0,
    scale: float = 1.0,
):
    imu = env.scene["imu"]
    l_vel = imu.data.lin_vel_b
    y_vel = l_vel[:, 1]
    err = y_vel - target

    reward = torch.exp(-scale * err**2)

    return reward


def reward_roll_rate(
    env: ManagerBasedRLEnv,
    target: float = 0.0,
    scale: float = 2.0,
):
    imu = env.scene["imu"]
    roll_rate = imu.data.ang_vel_b[:, 0]

    err = roll_rate - target

    reward = torch.exp(-scale * err**2)
    return reward


"""
Bám lệnh vận tốc — reward của TẦNG THẤP sau khi đổi từ "đứng yên giữ thăng bằng"
sang "vừa giữ thăng bằng vừa chạy theo lệnh".
"""


def track_lin_vel_exp(
    env: ManagerBasedRLEnv,
    command_name: str = "base_velocity",
    std: float = 0.25,
    forward_sign: float = 1.0,
):
    """Thưởng khi vận tốc tiến của xe bám lệnh ``vx``.

    Lấy ``imu.data.lin_vel_b[:, 1]`` làm vận tốc tiến — đúng thành phần mà quan sát
    :func:`~..observations.lin_vel_b` đang đưa vào mạng, nên reward và quan sát nói cùng một
    thứ. Thân xe bị xoay 90° quanh trục roll (tư thế đứng là ``|roll| = 90°``) nên KHÔNG dùng
    được ``root_lin_vel_b[:, 0]`` như các task locomotion của Isaac Lab.

    .. important::
        ``forward_sign`` phải kiểm bằng mắt một lần. Nếu +y của IMU chỉ về phía sau xe thì
        lệnh tiến sẽ làm xe lùi, mà reward vẫn báo bám tốt — sai kiểu này không lộ ra trong
        log, chỉ thấy khi nhìn robot chạy. Thấy ngược thì đặt -1.0.
    """
    imu = env.scene["imu"]
    command = env.command_manager.get_command(command_name)
    lin_vel = forward_sign * imu.data.lin_vel_b[:, 1]
    return torch.exp(-torch.square(command[:, 0] - lin_vel) / std**2)


def track_ang_vel_exp(
    env: ManagerBasedRLEnv,
    command_name: str = "base_velocity",
    std: float = 0.5,
    turn_sign: float = 1.0,
):
    """Thưởng khi tốc độ quay của xe bám lệnh ``wz``.

    Đo trong hệ WORLD (``root_ang_vel_w[:, 2]``) chứ không phải hệ thân: thân xe xoay 90° nên
    trục nào của nó là trục quay đứng còn tuỳ tư thế, trong khi z của world thì luôn là trục
    quay của việc rẽ.
    """
    robot = env.scene["robot"]
    command = env.command_manager.get_command(command_name)
    ang_vel = turn_sign * robot.data.root_ang_vel_w[:, 2]
    return torch.exp(-torch.square(command[:, 2] - ang_vel) / std**2)
