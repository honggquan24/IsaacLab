from __future__ import annotations
import torch
from typing import TYPE_CHECKING
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import wrap_to_pi , euler_xyz_from_quat
import math

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

def obs_body_roll(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
):
    robot = env.scene[asset_cfg.name]
    quat = robot.data.root_quat_w
    r, p, y = euler_xyz_from_quat(quat)
    return r.unsqueeze(-1)


def obs_body_pitch(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
):
    robot = env.scene[asset_cfg.name]
    quat = robot.data.root_quat_w
    r, p, y = euler_xyz_from_quat(quat)
    return p.unsqueeze(-1)

def obs_body_yaw(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
):
    robot = env.scene[asset_cfg.name]
    quat = robot.data.root_quat_w
    r, p, y = euler_xyz_from_quat(quat)
    return y.unsqueeze(-1)

