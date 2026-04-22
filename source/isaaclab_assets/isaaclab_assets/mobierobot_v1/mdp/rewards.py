from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import wrap_to_pi



if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def reward_when_at_goal(env: ManagerBasedRLEnv):
    robot = env.scene["robot"]

    pos = robot.data.root_pos_w[:, 0:2] 

    command = env.command_manager.get_command("goal_pos")

    dx = command[:, 0] - pos[:, 0]
    dy = command[:, 1] - pos[:, 1]

    dis = torch.sqrt(dx**2 + dy**2 + 1e-6)

    reward = torch.exp(-0.6 * (dis - 0.2)**2)
    return reward


def reward_linear_velocity(
    env: ManagerBasedRLEnv,
):
    imu = env.scene["imu"]
    vel = imu.data.lin_vel_b[:, 1]

    target = 1.0
    err = vel - target
    abs_err = torch.abs(err)

    reward = torch.exp(-0.5 * err**2)
    penalty = torch.clamp(abs_err - 0.5, min=0.0)
    total = reward - penalty

    return total

def reward_when_reward_forward(
        env: ManagerBasedRLEnv,
):
    robot = env.scene["robot"]
    revolute1_vel = robot.data.joint_vel[:, 0]
    revolute2_vel = robot.data.joint_vel[:, 1]

    error = torch.abs(revolute1_vel - revolute2_vel)
    reward = torch.exp(-0.2 * error**2)
    return reward

def reward_avoid_obstacle(
    env: ManagerBasedRLEnv,
):
    raycaster = env.scene["raycaster"]

    ray_hits = raycaster.data.ray_hits_w 
    ray_pos = raycaster.data.pos_w          

    diff = ray_hits - ray_pos.unsqueeze(1)
    dist = torch.norm(diff, dim=-1)  

    dist = torch.nan_to_num(dist, nan=10.0, posinf=10.0)

    inv_dist = 1.0 / (dist + 0.05)
    inv_dist = torch.clamp(inv_dist, 0.0, 10.0)
    inv_dist = inv_dist / 10.0 

    raycast1 = inv_dist[:, 1]
    raycast5 = inv_dist[:, 5]
    raycast2 = inv_dist[:, 2]
    raycast4 = inv_dist[:, 4]

    penalty1 = 1 - torch.exp(-0.5 * raycast1**2)
    penalty5 = 1 - torch.exp(-0.5 * raycast5**2)
    penalty2 = 1 - torch.exp(-0.5 * raycast2**2)
    penalty4 = 1 - torch.exp(-0.5 * raycast4**2)

    return -(penalty1 + penalty5 + penalty2 + penalty4)/4



def reward_distance_from_start(
    env: ManagerBasedRLEnv,
):
    robot: Articulation = env.scene["robot"]

    pos = robot.data.root_pos_w[:, 0:2] 
    origin = env.scene.env_origins[:, 0:2]
    dist = torch.norm(pos - origin, dim=1)

    return dist



def penalty_spin_when_stuck(env:ManagerBasedRLEnv):
    imu = env.scene["imu"]
    robot: Articulation = env.scene["robot"]

    yaw_rate = torch.abs(imu.data.ang_vel_b[:, 2])
    speed = torch.norm(robot.data.root_lin_vel_w[:, 0:2], dim=1)

    penalty = torch.where(
        speed < 0.2,
        yaw_rate**2 ,  
        yaw_rate**2     
    )
    return -penalty


def raycast0_forward(env: ManagerBasedRLEnv):
    raycaster = env.scene["raycaster"]

    ray_hits = raycaster.data.ray_hits_w 
    ray_pos = raycaster.data.pos_w          

    diff = ray_hits - ray_pos.unsqueeze(1)
    dist = torch.norm(diff, dim=-1)  

    dist = torch.nan_to_num(dist, nan=10.0, posinf=10.0)

    inv_dist = 1.0 / (dist + 0.05)
    inv_dist = torch.clamp(inv_dist, 0.0, 10.0)
    inv_dist = inv_dist / 10.0 

    raycast0 = inv_dist[:, 0]

    threshold = 0.5
    is_far = raycast0 < threshold

    reward_far = 1 - torch.exp(-20 * raycast0**2)

    penalty_near = -(1-torch.exp(-20 * (raycast0**2)))

    Reward = torch.where(is_far, reward_far, penalty_near)

    return Reward

def raycast3_forward(env: ManagerBasedRLEnv):
    raycaster = env.scene["raycaster"]

    ray_hits = raycaster.data.ray_hits_w 
    ray_pos = raycaster.data.pos_w          

    diff = ray_hits - ray_pos.unsqueeze(1)
    dist = torch.norm(diff, dim=-1)  

    dist = torch.nan_to_num(dist, nan=10.0, posinf=10.0)

    inv_dist = 1.0 / (dist + 0.05)
    inv_dist = torch.clamp(inv_dist, 0.0, 10.0)
    inv_dist = inv_dist / 10.0 

    raycast3 = inv_dist[:, 3]

    reward_far = -(1-torch.exp(-20 * raycast3**2))

    return reward_far




def reward_move_to_goal(env: ManagerBasedRLEnv):
    robot = env.scene["robot"]

    vel = robot.data.root_lin_vel_w[:, :2]
    pos = robot.data.root_pos_w[:, :2]

    command = env.command_manager.get_command("goal_pos")
    goal = command[:, :2]

    direction = goal - pos
    direction = direction / (torch.norm(direction, dim=1, keepdim=True) + 1e-6)

    vel_proj = torch.sum(vel * direction, dim=1)

    return vel_proj




