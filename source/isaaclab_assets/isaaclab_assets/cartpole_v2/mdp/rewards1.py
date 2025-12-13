from __future__ import annotations
import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def cartpole_reward_joint_pos_rv1(
    env: ManagerBasedRLEnv,
    scale_pos: float = 5.0,
)-> torch.Tensor:
    robot = env.scene['robot']
    joint_pos = robot.data.joint_pos 
    theta1 = joint_pos[:, 1]
    reward = scale_pos * torch.cos(theta1)
    return reward 

def cartpole_reward_joint_pos_rv2(
    env: ManagerBasedRLEnv,
    scale_pos: float = 5.0,
)-> torch.Tensor:
    robot = env.scene['robot']
    joint_pos = robot.data.joint_pos 
    theta2 = joint_pos[:, 2]
    reward = scale_pos * torch.cos(theta2)
    return reward

def cartpole_penalty_extreme_angle(
    env: ManagerBasedRLEnv,
    threshold_fail: float = 1.57,
    scale: float = 10.0,
)-> torch.Tensor:
    robot = env.scene["robot"]
    joint_pos = robot.data.joint_pos 

    theta1 = torch.abs(joint_pos[:, 1])
    theta2 = torch.abs(joint_pos[:, 2])
    

    err1 = torch.clamp(theta1 - threshold_fail, min=0.0) 
    err2 = torch.clamp(theta2 - threshold_fail, min=0.0) 
    
    penalty = -scale * (err1**2 + err2**2)
    return penalty



def cartpole_penalty_joint_vel(
    env: ManagerBasedRLEnv,
    scale: float = 0.1
)-> torch.Tensor:
    robot = env.scene['robot']
    joint_vel = robot.data.joint_vel  
    penalty = -scale * (joint_vel[:, 0]**2 + joint_vel[:, 1]**2 + joint_vel[:, 2]**2)
    return penalty

def cartpole_penalty_action_effort(
    env: ManagerBasedRLEnv,
    scale: float = 0.001,
) -> torch.Tensor:
    actions = env.action_manager.action
    penalty = -scale * torch.sum(actions**2, dim=-1)
    return penalty



def cart_center_reward(
    env: ManagerBasedRLEnv,
    target: float = 0.6,
    scale: float = 1.5,
    x_limit: float = 0.8,
) -> torch.Tensor:
    robot = env.scene["robot"]
    joint_pos = robot.data.joint_pos
    device = joint_pos.device

    x = joint_pos[:, 0] - target
    reward = torch.exp(-((scale * x) ** 2))
    reward = torch.where(
        torch.abs(x) < x_limit,
        reward,
        torch.zeros_like(reward)
    )
    return reward



def cart_not_center_penalty(
    env: ManagerBasedRLEnv,
    target: float = 0.6,
    tolerance: float = 0.05,
    scale: float = 2.0,
) -> torch.Tensor:
    robot = env.scene["robot"]
    joint_pos = robot.data.joint_pos

    x = torch.abs(joint_pos[:, 0] - target)
    err = torch.clamp(x - tolerance, min=0.0)

    penalty = -(1.0 - torch.exp(-scale * err**2))
    return penalty
