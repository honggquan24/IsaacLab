# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Probe: set 2 bánh CÙNG vận tốc target, xem robot TIẾN hay XOAY (trục bánh mirror?).

Nếu wheel & wheel_01 quay NGƯỢC nhau trong world → joint trái bị mirror → set cùng
dấu joint-vel sẽ làm robot xoay/triệt tiêu, KHÔNG tiến. Khi đó PI-ANN cần wheel_dir=[1,-1].

Run:
    ./isaaclab.sh -p source/isaaclab_assets/isaaclab_assets/legged_v3/tools/probe_wheel_sign.py --headless
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args, _ = parser.parse_known_args()
simulation_app = AppLauncher(args).app

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.sim import SimulationContext

from isaaclab_assets.wheeled_biped.wheeled_biped_cfg import WHEELED_BIPED_CFG


def main():
    sim = SimulationContext(sim_utils.SimulationCfg(dt=1 / 200.0, device="cuda:0"))
    sim_utils.GroundPlaneCfg().func("/World/ground", sim_utils.GroundPlaneCfg())
    robot = Articulation(WHEELED_BIPED_CFG.replace(prim_path="/World/Robot"))
    sim.reset()

    jn = robot.data.joint_names
    bn = robot.data.body_names
    wheel_ids = [jn.index(n) for n in ["right_wheel_joint", "left_wheel_joint"]]
    wbody_ids = [bn.index(n) for n in ["wheel", "wheel_01"]]
    print(f"\nwheel joints idx={wheel_ids}  wheel bodies idx={wbody_ids} ({['wheel', 'wheel_01']})\n")

    # CÙNG vận tốc target cho cả 2 bánh
    vtarget = torch.zeros(1, robot.num_joints, device="cuda:0")
    vtarget[0, wheel_ids[0]] = 8.0
    vtarget[0, wheel_ids[1]] = 8.0

    for step in range(60):
        robot.set_joint_velocity_target(vtarget[:, wheel_ids], joint_ids=wheel_ids)
        sim.step()
        robot.update(sim.cfg.dt)
        if step % 15 == 14:
            wjv = robot.data.joint_vel[0, wheel_ids].tolist()  # joint vel (sim)
            waw = robot.data.body_ang_vel_w[0, wbody_ids]  # world ang vel mỗi bánh
            rlv = robot.data.root_lin_vel_w[0].tolist()
            rav = robot.data.root_ang_vel_w[0].tolist()
            print(
                f"[step {step + 1:3d}] joint_vel(R,L)=[{wjv[0]:+.2f},{wjv[1]:+.2f}]  "
                f"wheelR_angvel_w={[f'{v:+.2f}' for v in waw[0].tolist()]}  "
                f"wheelL_angvel_w={[f'{v:+.2f}' for v in waw[1].tolist()]}"
            )
            print(
                f"            root_lin_vel_w={[f'{v:+.3f}' for v in rlv]}  root_ang_vel_w={[f'{v:+.3f}' for v in rav]}"
            )

    print(
        "\n>>> KẾT LUẬN: nếu wheelR & wheelL world-angvel NGƯỢC dấu trên cùng trục → "
        "MIRROR → PI-ANN cần wheel_dir=[1,-1]. Nếu CÙNG dấu → set cùng joint-vel là tiến (code hiện đúng).\n"
    )
    simulation_app.close()


if __name__ == "__main__":
    main()
