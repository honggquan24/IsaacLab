from __future__ import annotations
import torch
from typing import TYPE_CHECKING
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import wrap_to_pi
import math

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv



def Reward_Swing_up_rv1 (
        env : ManagerBasedRLEnv,
        target: float = 0.0,
        scale : float = 1.0,
):
    robot = env.scene['robot']
    joint_pos = robot.data.joint_pos

    theta1_ = joint_pos [:,1]
    theta1  = wrap_to_pi(theta1_)

    err = theta1 - target 

    reward = scale * torch.cos(err)
    return reward

def Reward_Swing_up_rv2 (
        env : ManagerBasedRLEnv,
        target: float = 0.0,
        scale : float = 1.0,
):
    robot = env.scene['robot']
    joint_pos = robot.data.joint_pos

    theta1_ = joint_pos [:,1]
    theta2_ = joint_pos [:,2]

    theta1  =wrap_to_pi (theta1_)
    theta2  = wrap_to_pi (theta1 + theta2_)

    err = theta2 - target 

    reward = scale * torch.cos(err)
    return reward

def Penalty_vel (
    env : ManagerBasedRLEnv,
    target: float = 0.0,
    scale : float = 1.0,  
):
    robot = env.scene['robot']
    joint_pos = robot.data.joint_pos
    joint_vel = robot.data.joint_vel

    theta1_ = joint_pos[:, 1]
    theta2_ = joint_pos[:, 2]

    theta1 = wrap_to_pi(theta1_)
    theta2 = wrap_to_pi(theta1 + theta2_)

    w1 = joint_vel[:, 1]
    w2_rel = joint_vel[:, 2]
    w2_abs = w1 + w2_rel

    th = 15.0 * math.pi / 180.0
    near = (theta1.abs() < th) & (theta2.abs() < th)

    k_balance = 0.01 
    k_swingup = 1.0e-4

    penalty = torch.where(
        near,
        -k_balance * (w1**2 + w2_abs**2),
        -k_swingup * (w1**2 + w2_abs**2),
    )
    return penalty

def action_penalty(
    env: ManagerBasedRLEnv,
    scale: float = 1.0e-3,
) -> torch.Tensor:
    u = env.action_manager.action
    return -scale * torch.sum(u**2, dim=-1)

def cart_not_center_penalty(
    env: ManagerBasedRLEnv,
    target: float = 0.6,
    scale: float = 5,
):
    robot =env.scene["robot"]
    joint_pos = robot.data.joint_pos
    err_pos = scale * (torch.abs(joint_pos[:, 0]) - target)
    err_pos = torch.clamp (err_pos, min = 0)
    
    penalty = -(1 - torch.exp(-(err_pos ** 2)))
    return penalty

def Reward_balance_rv2(
    env: ManagerBasedRLEnv,
    k: float = 20.0,
    scale: float = 1.0,
):
    robot = env.scene["robot"]
    joint_pos = robot.data.joint_pos

    theta1_ = joint_pos[:, 1]
    theta2_ = joint_pos[:, 2]

    theta1 = wrap_to_pi(theta1_)
    theta2_abs = wrap_to_pi(theta1 + theta2_)

    err2 = wrap_to_pi(theta2_abs - 0.0)
    reward = scale * torch.exp(-k * (err2**2))
    return reward

def Reward_bonus_near(
    env: ManagerBasedRLEnv,
    th_deg: float = 15.0,
    bonus: float = 1.0,
):
    robot = env.scene["robot"]
    joint_pos = robot.data.joint_pos

    theta1_ = joint_pos[:, 1]
    theta2_ = joint_pos[:, 2]

    theta1 = wrap_to_pi(theta1_)
    theta2_abs = wrap_to_pi(theta1 + theta2_)

    th = th_deg * math.pi / 180.0
    near = (theta1.abs() < th) & (theta2_abs.abs() < th)

    return torch.where(near, torch.full_like(theta1, bonus), torch.zeros_like(theta1))


    

