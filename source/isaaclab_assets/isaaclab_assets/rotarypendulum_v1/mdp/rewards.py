from __future__ import annotations
import torch
from typing import TYPE_CHECKING
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import wrap_to_pi ,  euler_xyz_from_quat
import math


if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def reward_angle_pen (
        env: ManagerBasedRLEnv,
):
    
    robot = env.scene["robot"]
    joint_pos = robot.data.joint_pos[:,1]
    joint_pos = wrap_to_pi (joint_pos)

    penangle = torch.abs(joint_pos)
    reward = -torch.cos(penangle)

    return reward

def penalty_vel_pen (
    env : ManagerBasedRLEnv,
    target: float = 0.0,
    scale : float = 1.0,  
):
    robot = env.scene['robot']
    joint_pos = robot.data.joint_pos
    joint_vel = robot.data.joint_vel

    penangle_ = joint_pos[:, 1]

    penangle = wrap_to_pi(penangle_)

    vel1 = joint_vel[:, 1]

    th = 160.0 * math.pi / 180.0
    near = (torch.abs(penangle) > th)

    k_balance = 0.01 
    k_swingup = 0.001

    penalty = torch.where(
        near,
        -k_balance * (vel1**2),
        -k_swingup * (vel1**2),
    )
    return penalty


def penalty_vel_Rotary (
    env : ManagerBasedRLEnv,
    target: float = 0.0,
    scale : float = 1.0,  
):
    robot = env.scene['robot']
    joint_pos = robot.data.joint_pos
    joint_vel = robot.data.joint_vel

    penangle_ = joint_pos[:, 0]

    penangle = wrap_to_pi(penangle_)

    vel1 = joint_vel[:, 0]

    th = 160.0 * math.pi / 180.0
    near = (torch.abs(penangle) > th)

    k_balance = 0.01 
    k_swingup = 0.001

    penalty = torch.where(
        near,
        -k_balance * (vel1**2),
        -k_swingup * (vel1**2),
    )
    return penalty

def Reward_bonus_near(
    env: ManagerBasedRLEnv,
    th_deg: float = 175,
    bonus: float = 1.0,
):
    robot = env.scene["robot"]
    joint_pos = robot.data.joint_pos

    theta1_ = joint_pos[:, 1]
    theta1 = wrap_to_pi(theta1_)

    th = th_deg * math.pi / 180.0
    near = (theta1.abs() > th)

    return torch.where(near, torch.full_like(theta1, bonus), torch.zeros_like(theta1))

def penalty_when_center (
        env: ManagerBasedRLEnv,
):
    robot = env.scene["robot"]
    joint_pos = robot.data.joint_pos
    joint_pos = wrap_to_pi (joint_pos[:,0])

    penalty = torch.cos (joint_pos)
    return penalty 







    