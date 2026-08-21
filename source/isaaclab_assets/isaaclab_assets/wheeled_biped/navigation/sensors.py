# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""LiDAR (RayCaster) observation, reward và termination cho navigation tránh vật cản.

Cảm biến LiDAR 2D được dựng bằng MultiMeshRayCasterCfg + LidarPatternCfg
(360°, 1 kênh), raycast vào mesh kho (warehouse) đã merge. Các hàm dưới đây
biến dữ liệu tia thành:
  - lidar_ranges       : observation (khoảng cách chuẩn hoá [0,1] mỗi tia)
  - obstacle_proximity : reward phạt khi lại gần vật cản
  - too_close_to_obstacle : termination khi sắp đâm vật cản
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _ray_distances(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, max_distance: float) -> torch.Tensor:
    """Khoảng cách từ gốc cảm biến tới điểm va chạm mỗi tia — (N, num_rays).

    Tia không trúng gì (hit ở vô cực) được clamp về max_distance.
    """
    sensor = env.scene.sensors[sensor_cfg.name]
    hits = sensor.data.ray_hits_w  # (N, R, 3) toạ độ va chạm (world)
    origin = sensor.data.pos_w.unsqueeze(1)  # (N, 1, 3) gốc cảm biến (world)

    dist = torch.norm(hits - origin, dim=-1)  # (N, R)
    # hit vô cực -> nan/inf; thay bằng max_distance
    dist = torch.nan_to_num(dist, nan=max_distance, posinf=max_distance, neginf=max_distance)
    return dist.clamp(max=max_distance)


def lidar_ranges(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("lidar"),
    max_distance: float = 10.0,
) -> torch.Tensor:
    """Observation: khoảng cách LiDAR chuẩn hoá về [0, 1] (1 = không thấy vật cản).

    Trả về (N, num_rays). num_rays = (horizontal_fov / horizontal_res).
    """
    dist = _ray_distances(env, sensor_cfg, max_distance)
    return dist / max_distance


def obstacle_proximity_penalty(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("lidar"),
    safe_distance: float = 0.6,
    max_distance: float = 10.0,
) -> torch.Tensor:
    """Reward (dùng weight ÂM): phạt khi tia gần nhất < safe_distance.

    Penalty = (safe_distance - d_min) / safe_distance, kẹp về [0, 1].
    = 0 khi vật cản còn xa hơn safe_distance, tăng dần tới 1 khi sát vật cản.
    """
    dist = _ray_distances(env, sensor_cfg, max_distance)
    d_min = dist.min(dim=-1).values  # (N,)
    penalty = (safe_distance - d_min).clamp(min=0.0) / safe_distance
    return penalty


def too_close_to_obstacle(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("lidar"),
    min_distance: float = 0.25,
    max_distance: float = 10.0,
) -> torch.Tensor:
    """Termination: True khi tia gần nhất < min_distance (coi như đâm vật cản)."""
    dist = _ray_distances(env, sensor_cfg, max_distance)
    d_min = dist.min(dim=-1).values
    return d_min < min_distance
