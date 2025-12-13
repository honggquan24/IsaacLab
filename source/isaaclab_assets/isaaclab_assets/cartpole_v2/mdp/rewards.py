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

    0.0,  # cartpole_reward_joint_pos_rv1
    0.0,
    0.0,  # cartpole_reward_joint_pos_rv2
    2.0,  # cartpole_reward_joint_vel 0
    2.0,  # cartpole_reward_joint_vel 1
    0.3,  # cartpole_reward_joint_vel 2
    0.0,  # cart_center_reward
    0.6,  # cart_not_center_penalty   
])

def cartpole_reward_joint_pos_rv1(
    env: ManagerBasedRLEnv,
    target: torch.Tensor = TARGET_JOINT,
    scale_pos: float = 0.6,
)-> torch.Tensor:
    robot = env.scene['robot']
    joint_pos = robot.data.joint_pos 

    device = joint_pos.device
    if target.device != device: 
        target = target.to(device)

        
    err_pos = scale_pos * (torch.abs(joint_pos[:, 1]) - target[0])
    reward = torch.exp(-(err_pos ** 2 ))
    return reward 

def cartpole_reward_joint_pos_rv2(
    env: ManagerBasedRLEnv,
    target: torch.Tensor = TARGET_JOINT,
    # scale_pos: float = 0.6,
)-> torch.Tensor:
    robot = env.scene['robot']
    joint_pos = robot.data.joint_pos 

    device = joint_pos.device
    if target.device != device: 
        target = target.to(device)

    
    err_pos1 = torch.abs(joint_pos[:, 2] - target[1])
    reward = torch.where(
    torch.abs((torch.cos(joint_pos[:, 1])) > 0.85 ),
    -0.3 + torch.cos(err_pos1),
    torch.tensor(0.0, device=joint_pos.device)
    )
    return reward

def cartpole_reward_joint_pos_rv2_rv1(
    env: ManagerBasedRLEnv,
    scale_pos: float = 0.6,
)-> torch.Tensor:
    robot = env.scene['robot']
    joint_pos = robot.data.joint_pos

    theta1 = torch.abs(joint_pos[:,1])
    theta2 = torch.abs(joint_pos[:,2])

    reward = torch.cos(theta1) + torch.cos (theta2)
    return reward

def cartpole_penalty_joint_vel(
    env: ManagerBasedRLEnv,
    target: torch.Tensor = TARGET_JOINT,
    scale: float = 0.05
)-> torch.Tensor:
    robot = env.scene['robot']
    joint_vel = robot.data.joint_vel  
    
    device = joint_vel.device
    if target.device != device:
        target = target.to(device)


    err_cart = torch.abs(joint_vel[:, 0]) - target[3]
    err_cart = torch.clamp (err_cart , min= 0.0 )
    penalty = 1.0- torch.exp(-scale * (err_cart**2 ))
    return penalty

def cartpole_penalty_joint_vel_pe1(
    env: ManagerBasedRLEnv,
    target: torch.Tensor = TARGET_JOINT,
    scale: float = 0.05
)-> torch.Tensor:
    robot = env.scene['robot']
    joint_vel = robot.data.joint_vel  
    
    device = joint_vel.device
    if target.device != device:
        target = target.to(device)

    err_pendulum1 = torch.abs(joint_vel[:, 1]) - target[4]
    err_pendulum1 = torch.clamp (err_pendulum1 , min= 0.0) 
    penalty = 1.0- torch.exp(-scale * (err_pendulum1 **2))
    return penalty

def cartpole_penalty_joint_vel_pe2(
    env: ManagerBasedRLEnv,
    target: torch.Tensor = TARGET_JOINT,
    scale: float = 0.05
)-> torch.Tensor:
    robot = env.scene['robot']
    joint_vel = robot.data.joint_vel  
    
    device = joint_vel.device
    if target.device != device:
        target = target.to(device)

    err_pendulum2 = torch.abs(joint_vel[:, 2]) - target[5]
    err_pendulum2 = torch.clamp (err_pendulum2 , min= 0.0)
    penalty = 1.0- torch.exp(-scale * (err_pendulum2 **2))
    return penalty



def cartpole_penalty_fall_p1(
    env: ManagerBasedRLEnv,
    threshold_fall: float = 0.2, 
    scale: float = 1.0,
)-> torch.Tensor:
    robot = env.scene["robot"]
    joint_pos = robot.data.joint_pos 

    theta = joint_pos[:, 1]

    err = torch.abs(theta) - threshold_fall
    err = torch.clamp(err, min=0.0)


    penalty = 1.0 - torch.exp(-scale * err**2)
    return penalty 
 

def cartpole_penalty_fall_p2(
    env: ManagerBasedRLEnv,
    threshold_fall: float = 0.2,
    scale: float = 1.0,
)-> torch.Tensor:
    robot = env.scene["robot"]
    joint_pos = robot.data.joint_pos 

    theta1 = torch.abs(joint_pos[:, 1])
    theta2 = torch.abs(joint_pos[:, 2])
    err = torch.clamp(theta2 - threshold_fall, min=0.0) 
    
    penalty = torch.where (theta1 < 0.2, 1.0 - torch.exp(-scale * err**2),0.0)
    return penalty


def cart_center_reward (
    env: ManagerBasedRLEnv,
    target: torch.Tensor = TARGET_JOINT,
    scale: float = 2.5,
)-> torch.Tensor:
    robot =env.scene["robot"]
    joint_pos = robot.data.joint_pos

    device = joint_pos.device
    if target.device != device:
        target = target.to(device)
 
    err_pos = scale * (torch.abs(joint_pos[:, 0]) - target[6])
    cond = torch.abs(joint_pos[:, 0]) < 0.6
    reward = torch.where(cond, torch.exp(-(err_pos ** 2)), 0.0)

    return reward

def cart_not_center_penalty(
    env: ManagerBasedRLEnv,
    target: torch.Tensor = TARGET_JOINT,
    scale: float = 5,
)-> torch.Tensor:
    robot =env.scene["robot"]
    joint_pos = robot.data.joint_pos

    device = joint_pos.device
    if target.device != device:
        target = target.to(device)
 
    err_pos = scale * (torch.abs(joint_pos[:, 0]) - target[7])
    err_pos = torch.clamp (err_pos, min = 0)
    
    penalty = 1 - torch.exp(-(err_pos ** 2))
    return penalty


