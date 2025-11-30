from __future__ import annotations
import torch
from typing import TYPE_CHECKING
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import wrap_to_pi, euler_xyz_from_quat
import math

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

# ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py --task=Isaac-Legged-Robot-V2-Pose --num_envs 4096 --resume --load_run=pose_1 --checkpoint=model_150.pt --video


# TARGET JOINT
TARGET_JOINT_POS = torch.tensor([
    # index: joint_name                # comment
    0.0,                  
])


def cartpole_reward(
    env: ManagerBasedRLEnv,
    target: torch.Tensor = TARGET_JOINT_POS,
    scale: float = 5.0
):
    robot = env.scene['robot']
    joint_pos = robot.data.joint_pos
    
    # Move tensors to correct device
    device = joint_pos.device
    if target.device != device:
        target = target.to(device)
    
    err = joint_pos[0][1].item() - TARGET_JOINT_POS.item()
    
    reward = torch.exp(
        torch.tensor(-scale*(err)**2)
    )
    
    return reward