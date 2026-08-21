# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Đo projected_gravity_b khi nghiêng robot quanh world-X vs world-Y.

Mục tiêu: xác định thành phần nào của projected_gravity_b là NGHIÊNG NGANG
(lateral, đổ sang bên) vs NGHIÊNG DỌC (fore-aft, lean để chạy theo Y).
Robot drive theo world-Y, init xoay 90° quanh X.

    ./isaaclab.sh -p source/isaaclab_assets/isaaclab_assets/legged_v3/probe_gravity_v5.py --headless
"""

import argparse
import math

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args, _ = parser.parse_known_args()
app = AppLauncher(args).app


import torch

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.utils.math import quat_mul

from isaaclab_assets.wheeled_biped.locomotion.wheel_env_cfg import WheeledBipedWheelEnvCfg

_bp = print


def print(*a, **k):  # noqa
    s = " ".join(str(x) for x in a)
    _bp(s, flush=True)


cfg = WheeledBipedWheelEnvCfg()
cfg.scene.num_envs = 1
cfg.sim.device = "cuda:0"
env = ManagerBasedRLEnv(cfg=cfg)
env.reset()
robot = env.scene["robot"]
DEV = env.device

q_def = robot.data.default_root_state[0, 3:7].clone()  # w,x,y,z
pos = robot.data.default_root_state[0, :3].clone()


def quat_axis(angle_deg, axis):
    a = math.radians(angle_deg) / 2
    w = math.cos(a)
    s = math.sin(a)
    v = [s * axis[0], s * axis[1], s * axis[2]]
    return torch.tensor([[w, v[0], v[1], v[2]]], device=DEV)


def set_and_read(label, dq_world):
    # q_cur = dq_world ⊗ q_def  (xoay trong world frame quanh trục cho trước)
    q_cur = quat_mul(dq_world, q_def.unsqueeze(0))
    pose = torch.cat([pos.unsqueeze(0), q_cur], dim=-1)
    robot.write_root_pose_to_sim(pose)
    robot.write_root_velocity_to_sim(torch.zeros(1, 6, device=DEV))
    env.sim.step(render=False)
    robot.update(cfg.sim.dt)
    g = robot.data.projected_gravity_b[0]
    print(f"  {label:<34} proj_grav_b = ({g[0]:+.3f}, {g[1]:+.3f}, {g[2]:+.3f})")


print("\n==== PROBE projected_gravity_b ====")
set_and_read("upright (default)", quat_axis(0, [1, 0, 0]))
set_and_read("nghiêng +15° quanh world-X", quat_axis(15, [1, 0, 0]))
set_and_read("nghiêng +15° quanh world-Y (drive dir)", quat_axis(15, [0, 1, 0]))
print("\n→ Thành phần thay đổi khi xoay quanh world-Y (drive dir) = NGHIÊNG DỌC (cho lean)")
print("→ Thành phần thay đổi khi xoay quanh world-X = NGHIÊNG NGANG (phạt mạnh)")
app.close()
