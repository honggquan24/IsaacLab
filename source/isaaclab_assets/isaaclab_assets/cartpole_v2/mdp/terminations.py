from __future__ import annotations
import torch
from typing import TYPE_CHECKING
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import wrap_to_pi, euler_xyz_from_quat
import math

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def Cart_pole_angle_reset (
    env: ManagerBasedRLEnv,    # cond3 = vel3 > limit1
    limit1: float = math.pi * 90/180,
    limit2: float = math.pi
):
    robot = env.scene['robot']
    joint_pos = robot.data.joint_pos 

    theta1 = torch.abs(joint_pos [:,1])
    theta2 = torch.abs(joint_pos [:,2])

    cond1 = theta1 > limit1
    cond2 = theta2 > limit2

    terminated = cond1 | cond2

    return terminated

def Cart_pole_vel_reset(
    env: ManagerBasedRLEnv,
    limit1: float = 60,
    limit2: float = 20,
):
    robot = env.scene['robot']
    jont_vel = robot.data.joint_vel

    vel1 = torch.abs(jont_vel [:,0])
    vel2 = torch.abs(jont_vel [:,1])
    vel3 = torch.abs(jont_vel [:,2])

    # cond1 = vel1 > limit2
    cond2 = vel2 > limit1
    # cond3 = vel3 > limit1
    erminated = cond2 

    return erminated 


     
