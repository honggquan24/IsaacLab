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
    # index: joint_name                # comment
    3.14, 3.14, 0.0, 1.0 ,1.0, 1.0 , 0.0,        
])

def cartpole_reward_joint_pos_rv1(
    env: ManagerBasedRLEnv,
    target: torch.Tensor = TARGET_JOINT,
    scale_pos: float = 0.6,
    scale_vel: float = 0.3
)-> torch.Tensor:
    robot = env.scene['robot']
    joint_pos = robot.data.joint_pos 
    joint_vel = robot.data.joint_vel


    # Move tensors to correct device
    device = joint_pos.device
    if target.device != device: 
        target = target.to(device)

        
    err_pos = scale_pos * (torch.abs(joint_pos[:, 1]) - target[0])
    err_vel = scale_vel * (torch.abs(joint_vel[:, 1]) - target[2])

    reward = torch.exp(-(err_pos ** 2 + err_vel**2))

    return reward

def cartpole_reward_joint_pos_rv2(
    env: ManagerBasedRLEnv,
    target: torch.Tensor = TARGET_JOINT,
    scale_pos: float = 0.6,
    scale_vel: float = 0.3
)-> torch.Tensor:
    robot = env.scene['robot']
    joint_pos = robot.data.joint_pos
    joint_vel = robot.data.joint_vel

    
    # Move tensors to correct device
    device = joint_pos.device

    if target.device != device: 
        target = target.to(device)
       
    err_pos = scale_pos * (torch.abs(joint_pos[:, 2]) - target[1])
    err_vel = scale_vel * (torch.abs(joint_vel[:, 2]) - target[2])
    reward = torch.exp(-(err_pos ** 2 + err_vel ** 2 ))

    return reward



def cartpole_reward_joint_vel(
    env: ManagerBasedRLEnv,
    target: torch.Tensor = TARGET_JOINT,
    scale: float = 0.002
):
    robot = env.scene['robot']
    joint_vel = robot.data.joint_vel  
    
    device = joint_vel.device
    if target.device != device:
        target = target.to(device)


    err_cart = torch.abs(joint_vel[:, 0]) - target[3]
    err_cart = torch.clamp (err_cart , min= 0.0 )

    err_pendulum1 = torch.abs(joint_vel[:, 1]) - target[4]
    err_pendulum1 = torch.clamp (err_pendulum1 , min= 0.0)
    
    err_pendulum2 = torch.abs(joint_vel[:, 2]) - target[5]
    err_pendulum2 = torch.clamp (err_pendulum2 , min= 0.0)

    reward = 1.0- torch.exp(-scale * (err_cart**2 + err_pendulum1**2 + err_pendulum2**2))
    return reward


def cartpole_reward_fall(
    env: ManagerBasedRLEnv,
    threshold_fall: float = 3.11, 
    scale: float = 0.35,
):
    robot = env.scene["robot"]
    joint_pos = robot.data.joint_pos 

    theta1 = torch.abs(joint_pos[:, 1]) 
    theta2 = torch.abs(joint_pos[:, 2])
    
    err1 =  threshold_fall - torch.abs(theta1)
    err1 = torch.clamp (err1 , min = 0.0)

    err2 =  threshold_fall - torch.abs(theta2)
    err2 =  torch.clamp (err2, min= 0.0)


    reward = 1.0 - torch.exp(-scale * (err1**2 + err2**2))
    return reward 

def cart_center_reward (
    env: ManagerBasedRLEnv,
    target: torch.Tensor = TARGET_JOINT,
    scale: float = 8,
):
    robot =env.scene["robot"]
    joint_pos = robot.data.joint_pos
    joint_vel = robot.data.joint_pos

    device = joint_vel.device
    if target.device != device:
        target = target.to(device)
 

    err_pos = scale * (torch.abs(joint_pos[:, 0]) - target[6])
    err_vel = scale * (torch.abs(joint_vel[:, 0]) - target[6])
    
    reward = torch.exp(-(err_pos ** 2 + err_vel ** 2 ))
    return reward


