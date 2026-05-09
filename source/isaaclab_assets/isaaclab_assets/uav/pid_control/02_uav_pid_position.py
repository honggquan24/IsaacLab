# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""
Mô phỏng Position Control cho quadcopter dùng PID.

Điều khiển vị trí [x, y, z] trực tiếp từ position error.
Single-loop: position error → attitude + thrust (không qua velocity loop).
PID output thẳng vào lực/moment vật lý (N, Nm) — không dùng feedforward.

3 vòng PID song song:
    pos_z error  →  thrust
    pos_x error  →  desired_pitch
    pos_y error  →  desired_roll
    att error    →  moments

Chạy:
    ./isaaclab.sh -p source/isaaclab_assets/isaaclab_assets/uav/pid_control/02_uav_pid_position.py
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import math

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Position PID control cho Crazyflie.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import collections
import torch
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.sim import SimulationContext
from isaaclab.utils.math import euler_xyz_from_quat

from isaaclab_assets.uav.uav_cfg import UAV_CFG
from pid_controller import QuadcopterPID, PIDController, make_alloc_inv


WINDOW_S   = 20.0
PLOT_EVERY = 5

MOTOR_THRUST_STD = 1e-4   # noise tuyệt đối [N]
MOTOR_MOMENT_STD = 1e-5   # noise tuyệt đối [Nm]


def make_plot():
    plt.ion()
    fig, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
    axes[0].set_title("UAV Position Response — X/Y/Z(t)")
    for ax, lbl in zip(axes, ["X [m]", "Y [m]", "Z [m]"]):
        ax.set_ylabel(lbl)
        ax.grid(True)
    axes[2].set_xlabel("Time [s]")
    lines_pos, lines_tgt = [], []
    for ax in axes:
        lp, = ax.plot([], [], "b-",  lw=2,   label="pos")
        lt, = ax.plot([], [], "r--", lw=1.5, label="target")
        ax.legend(loc="upper right", fontsize=8)
        lines_pos.append(lp)
        lines_tgt.append(lt)
    fig.tight_layout()
    return fig, axes, lines_pos, lines_tgt


def update_plot(fig, axes, lines_pos, lines_tgt, times, pos_data, tgt_data):
    t = list(times)
    if not t:
        return
    for i, (lp, lt) in enumerate(zip(lines_pos, lines_tgt)):
        p  = list(pos_data[i])
        tv = list(tgt_data[i])
        lp.set_data(t, p)
        lt.set_data(t, tv)
        all_v = p + tv
        margin = 0.1
        axes[i].set_xlim(max(0.0, t[-1] - WINDOW_S), t[-1] + 0.5)
        axes[i].set_ylim(min(all_v) - margin, max(all_v) + margin)
    fig.canvas.draw()
    fig.canvas.flush_events()


def main():
    # ── Setup ──────────────────────────────────────────────────────────────────
    sim_cfg = sim_utils.SimulationCfg(dt=0.005, device=args_cli.device)
    sim = SimulationContext(sim_cfg)
    sim.set_camera_view(eye=[1.5, 1.5, 2.0], target=[0.0, 0.0, 1.0])

    cfg = sim_utils.GroundPlaneCfg()
    cfg.func("/World/defaultGroundPlane", cfg)
    cfg = sim_utils.DistantLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    cfg.func("/World/Light", cfg)

    robot_cfg = UAV_CFG.replace(prim_path="/World/Crazyflie")
    robot_cfg = robot_cfg.replace(init_state=robot_cfg.init_state.replace(pos=(0.0, 0.0, 0.05)))
    robot_cfg.spawn.func("/World/Crazyflie", robot_cfg.spawn, translation=robot_cfg.init_state.pos)
    robot = Articulation(robot_cfg)

    sim.reset()

    prop_body_ids = robot.find_bodies("m.*_prop")[0]
    A_inv = make_alloc_inv(device=sim.device)

    TARGET_POS = torch.tensor([2.0, 2.0, 1.0], device=sim.device)
    print(f"[INFO] target = {TARGET_POS.tolist()}")

    # ── PID Controllers ────────────────────────────────────────────────────────
    pid = QuadcopterPID()
    pid_roll  = PIDController(kp=0.01, ki=0.001, kd=0.002, integral_limit=0.01)
    pid_pitch = PIDController(kp=0.01, ki=0.001, kd=0.002, integral_limit=0.01)
    pid_yaw   = PIDController(kp=0.03, ki=0.0,   kd=0.001, integral_limit=0.01)

    # ── Live plot ─────────────────────────────────────────────────────────────
    sim_dt   = sim.get_physics_dt()
    maxlen   = int(WINDOW_S / sim_dt) + 50
    times    = collections.deque(maxlen=maxlen)
    pos_data = [collections.deque(maxlen=maxlen) for _ in range(3)]
    tgt_data = [collections.deque(maxlen=maxlen) for _ in range(3)]
    fig, axes, lines_pos, lines_tgt = make_plot()

    sim_time = 0.0
    count    = 0

    while simulation_app.is_running():
        if count % 2000 == 0 and count > 0:
            sim_time = 0.0
            count = 0
            joint_pos, joint_vel = robot.data.default_joint_pos, robot.data.default_joint_vel
            robot.write_joint_state_to_sim(joint_pos, joint_vel)
            robot.write_root_pose_to_sim(robot.data.default_root_state[:, :7])
            robot.write_root_velocity_to_sim(robot.data.default_root_state[:, 7:])
            robot.reset()
            pid.reset()
            pid_roll.reset(); pid_pitch.reset(); pid_yaw.reset()
            print(f">>>>>>>> Reset!")

        # ── Đọc trạng thái ─────────────────────────────────────────────────
        pos  = robot.data.root_pos_w[0]
        vel  = robot.data.root_lin_vel_w[0]
        quat = robot.data.root_quat_w[0]
        roll, pitch, yaw = [x[0].item() for x in euler_xyz_from_quat(quat.unsqueeze(0))]

        # ── Outer PID: position → thrust + desired attitude ────────────────
        thrust_N, desired_roll, desired_pitch = pid.compute(pos=pos, vel=vel, target_pos=TARGET_POS, dt=sim_dt)

        # ── Inner attitude PID ─────────────────────────────────────────────
        yaw_err = (0.0 - yaw + math.pi) % (2 * math.pi) - math.pi
        m_roll  = pid_roll.update(desired_roll  - roll,  sim_dt)
        m_pitch = pid_pitch.update(desired_pitch - pitch, sim_dt)
        m_yaw   = pid_yaw.update(yaw_err,                 sim_dt)

        # ── Allocation matrix: [Fz,Tx,Ty,Tz] → [F1,F2,F3,F4] ───────────
        wrench  = torch.tensor([thrust_N, m_roll, m_pitch, m_yaw], device=sim.device)
        F_props = (A_inv @ wrench).clamp(min=0.0)
        # F_props += torch.randn(4, device=sim.device) * MOTOR_THRUST_STD

        forces_prop = torch.zeros(robot.num_instances, 4, 3, device=sim.device)
        forces_prop[0, :, 2] = F_props
        robot.permanent_wrench_composer.set_forces_and_torques(
            forces=forces_prop,
            torques=torch.zeros_like(forces_prop),
            body_ids=prop_body_ids,
        )

        robot.write_data_to_sim()
        sim.step()
        sim_time += sim_dt
        count    += 1
        robot.update(sim_dt)

        if count % int(1.0 / sim_dt) == 0:
            err = (TARGET_POS - pos).norm().item()
            print(f"t={sim_time:5.1f}s | pos=({pos[0].item():+.2f},{pos[1].item():+.2f},{pos[2].item():+.2f}) | err={err:.3f}m | thrust={thrust_N:.4f}N")

        times.append(sim_time)
        for i in range(3):
            pos_data[i].append(pos[i].item())
            tgt_data[i].append(TARGET_POS[i].item())
        if count % PLOT_EVERY == 0:
            update_plot(fig, axes, lines_pos, lines_tgt, times, pos_data, tgt_data)

        p = robot.data.root_pos_w[0].cpu().numpy()
        sim.set_camera_view(
            eye=[p[0] - 0.8, p[1] - 0.8, p[2] + 0.8],
            target=[p[0], p[1], p[2]],
        )


if __name__ == "__main__":
    main()
    simulation_app.close()
