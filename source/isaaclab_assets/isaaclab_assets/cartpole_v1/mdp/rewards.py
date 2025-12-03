from __future__ import annotations
import torch
from typing import TYPE_CHECKING
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import wrap_to_pi, euler_xyz_from_quat
import math

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# TARGET JOINT
TARGET_JOINT = torch.tensor([
    # index: joint_name       
    0.0, 2.0, 2.0 ,0.0 ,          
])

def cartpole_reward_joint_pos(
    env: ManagerBasedRLEnv,
    target: torch.Tensor = TARGET_JOINT,
    scale_pos: float = 0.6,
)-> torch.Tensor:
    robot = env.scene['robot']
    joint_pos = robot.data.joint_pos      
 
    robot = env.scene['robot']
    joint_pos = robot.data.joint_pos
 
    # Move tensors to correct device
    device = joint_pos.device
    if target.device != device: 
        target = target.to(device)

    err = (joint_pos[:, 1] - target[0])
    reward = torch.exp(-scale_pos*(err ** 2))
    return reward


def cartpole_reward_joint_vel(
    env: ManagerBasedRLEnv,
    target: torch.Tensor = TARGET_JOINT,
    scale: float = 0.8
)-> torch.Tensor:
    robot = env.scene['robot']
    joint_vel = robot.data.joint_vel  
    
    device = joint_vel.device
    if target.device != device:
        target = target.to(device)


    err_cart = torch.abs(joint_vel[:, 0]) - target[1]
    err_cart = torch.clamp (err_cart , min= 0.0 )

    err_pendulum = torch.abs(joint_vel[:, 1]) - target[2]
    err_pendulum = torch.clamp (err_pendulum , min= 0.0)


    reward=1.0 - torch.exp(
        -scale * (err_pendulum**2 + err_cart**2)
        )

    return reward

def cartpole_reward_fall(
    env: ManagerBasedRLEnv,
    threshold_fall: float = 0.3, 
    scale: float = 0.8,
)-> torch.Tensor:
    robot = env.scene["robot"]
    joint_pos = robot.data.joint_pos 

    theta = joint_pos[:, 1]

    err = torch.abs(theta) - threshold_fall
    err = torch.clamp(err, min=0.0)


    reward = 1.0 - torch.exp(-scale * err**2)
    return reward 

