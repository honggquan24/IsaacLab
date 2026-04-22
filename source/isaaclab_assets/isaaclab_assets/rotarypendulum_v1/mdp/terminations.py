from __future__ import annotations
import torch
from typing import TYPE_CHECKING
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import wrap_to_pi ,  euler_xyz_from_quat
import math



if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

def reset_joint_limit(env: ManagerBasedRLEnv):

    robot = env.scene["robot"]
    joint_pos = robot.data.joint_pos

    revolute1 =joint_pos[:,0]

    cond1 = torch.abs(revolute1) > (134*math.pi/180)


    return cond1



