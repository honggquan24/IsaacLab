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
    limit2: float = math.pi * 90/180,
):
    robot = env.scene['robot']
    joint_pos = robot.data.joint_pos 

    theta1 = torch.abs(joint_pos [:,1])
    theta2 = torch.abs(joint_pos [:,2])

    cond1 = theta1 > limit1
    cond2 = theta2 > limit2

    terminated = cond1 | cond2

    return terminated



def Cart_pole_angle_reset_1(
    env: ManagerBasedRLEnv,    # cond3 = vel3 > limit1
    limit1: float = math.pi * 340/180,
):
    robot = env.scene['robot']
    joint_pos = robot.data.joint_pos 

    theta1 = torch.abs(joint_pos [:,1] - math.pi)

    cond1 = theta1 > limit1
    terminated = cond1 

    return terminated

def Cart_pole_pos_reset(
    env: ManagerBasedRLEnv,
    limit1: float = 0.99,
):
    robot = env.scene['robot']
    jont_pos = robot.data.joint_pos

    vel1 = torch.abs(jont_pos [:,0])

    cond1 = vel1 > limit1
    erminated = cond1

    return erminated 

