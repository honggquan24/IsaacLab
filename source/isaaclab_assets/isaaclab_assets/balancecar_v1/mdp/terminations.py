from __future__ import annotations
import torch
from typing import TYPE_CHECKING
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import wrap_to_pi, euler_xyz_from_quat
import math

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

def reset_when_fall(env: ManagerBasedRLEnv):
    robot = env.scene["robot"]
    quat = robot.data.root_quat_w

    roll, _, _ = euler_xyz_from_quat(quat)

    upright = math.pi / 2  
    threshold = math.pi/180 * 50

    terminate = torch.abs(roll - upright) > threshold
    return terminate


    


