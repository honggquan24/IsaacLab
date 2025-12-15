from __future__ import annotations
import torch
from typing import TYPE_CHECKING
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import wrap_to_pi ,  euler_xyz_from_quat
import math

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def reward_angle (
        env: ManagerBasedRLEnv,
        tagret: float =  90 * math.pi/180,
        scale : float = 2,
):
    robot = env.scene["robot"]
    quat = robot.data.root_quat_w
    roll,_,_ = euler_xyz_from_quat(quat)

    err = roll - tagret
    reward = scale* (-0.9+ torch.cos(err))
    return reward




