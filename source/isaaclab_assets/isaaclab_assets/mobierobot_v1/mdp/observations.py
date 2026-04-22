from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import wrap_to_pi, euler_xyz_from_quat

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

def command_distance(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
):
    robot = env.scene["robot"]

    root_pos = robot.data.root_pos_w[:, 0:2]
    command = env.command_manager.get_command("goal_pos") 

    dx = command[:, 0] - root_pos[:, 0]
    dy = command[:, 1] - root_pos[:, 1]

    dist = torch.sqrt(dx**2 + dy**2 + 1e-6)
    return dist.unsqueeze(-1)


def imu_linear_vel(
    env: ManagerBasedRLEnv, 
    asset_cfg: SceneEntityCfg
):
    imu = env.scene["imu"]

    vel = imu.data.lin_vel_b[:, 0:2]

    return vel

def imu_ang_vel_z(
    env: ManagerBasedRLEnv, 
    asset_cfg: SceneEntityCfg
):
    imu = env.scene["imu"]

    ang_vel_z = imu.data.ang_vel_b[:, 2:3]

    return ang_vel_z




def raycaster_observation(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
):
    raycaster = env.scene["raycaster"]

    ray_hits = raycaster.data.ray_hits_w 
    ray_pos = raycaster.data.pos_w          

    diff = ray_hits - ray_pos.unsqueeze(1)
    dist = torch.norm(diff, dim=-1)  

    dist = torch.nan_to_num(dist, posinf=10.0)

    inv_dist = 1.0 / (dist + 0.05)

    inv_dist = torch.clamp(inv_dist, 0.0, 10.0)

    inv_dist = inv_dist / 10.0

    return inv_dist



def velocity_linear_driver(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg):

    robot = env.scene["robot"]

    wheel_radius = 0.05

    right_w = robot.data.joint_vel[:, 0]
    left_w = robot.data.joint_vel[:, 1]

    v_r = right_w * wheel_radius
    v_l = left_w * wheel_radius

    lin_vel = (v_r + v_l) / 2.0

    return lin_vel.unsqueeze(-1)


def velocity_angular_driver(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg):

    robot = env.scene["robot"]

    wheel_radius = 0.05
    L = 0.157

    right_w = robot.data.joint_vel[:, 0]
    left_w = robot.data.joint_vel[:, 1]

    v_r = right_w * wheel_radius
    v_l = left_w * wheel_radius

    ang_vel = (v_r - v_l) / L

    return ang_vel.unsqueeze(-1)


