from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import wrap_to_pi



if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

def termination_on_collision(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
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

    # detect collision
    threshold = 0.85 
    cond1 = inv_dist[:,0] > threshold
    cond2 = inv_dist[:,1] > threshold
    cond3 = inv_dist[:,2] > threshold
    cond4 = inv_dist[:,3] > threshold
    cond5 = inv_dist[:,4] > threshold
    cond6 = inv_dist[:,5] > threshold
    # print(f"inv_dist: {inv_dist}")
    return cond1 | cond2 | cond3 | cond4 | cond5 | cond6