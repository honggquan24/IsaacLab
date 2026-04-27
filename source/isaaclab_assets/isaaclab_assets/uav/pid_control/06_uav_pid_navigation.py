# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""
Hierarchical Navigation — điều hướng waypoint ngẫu nhiên.

File này là bước tiếp theo của file 05 (hierarchical):
    - Dùng cùng kiến trúc 3-tầng (pos → vel → att) tần số 10/25/200 Hz
    - Thêm logic cấp cao: tự động chuyển waypoint khi đến gần mục tiêu

Sơ đồ hệ thống:
    ┌─────────────────────────────────────┐
    │  Mission Logic (mỗi step)           │
    │  - Sample waypoint ngẫu nhiên       │
    │  - Switch nếu err < 0.15m hoặc      │
    │    timeout > WP_HOLD_S              │
    └──────────────────┬──────────────────┘
                       │ target_pos
                       ▼
    ┌─────────────────────────────────────┐
    │  Tầng 1 — Position PID (10 Hz)      │
    │  pos_err → desired_velocity         │
    └──────────────────┬──────────────────┘
                       │ des_vel [vx,vy,vz]
                       ▼
    ┌─────────────────────────────────────┐
    │  Tầng 2 — Velocity PID (25 Hz)      │
    │  vel_err → thrust + des_angles      │
    └──────────────────┬──────────────────┘
                       │ thrust (N), des_roll/pitch (rad)
                       ▼
    ┌─────────────────────────────────────┐
    │  Tầng 3 — Attitude PID (200 Hz)     │
    │  att_err → moments [mx, my, mz]     │
    └──────────────────┬──────────────────┘
                       │ forces + torques (N, Nm)
                       ▼
              [UAV Physics Engine]

Chạy:
    ./isaaclab.sh -p source/isaaclab_assets/isaaclab_assets/uav/pid_control/06_uav_pid_navigation.py
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import math
import random

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Hierarchical navigation với random waypoints.")
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
from pid_controller import PIDController, make_alloc_inv


WINDOW_S   = 30.0
PLOT_EVERY = 5

MOTOR_THRUST_STD = 1e-4   # noise tuyệt đối [N]
MOTOR_MOMENT_STD = 1e-5   # noise tuyệt đối [Nm]

# ── Mission parameters ─────────────────────────────────────────────────────────
WP_X_RANGE  = (-1.5, 1.5)
WP_Y_RANGE  = (-1.5, 1.5)
WP_Z_RANGE  = (0.5,  2.0)
WP_HOLD_S   = 8.0     # timeout chuyển waypoint [s]
ERR_THRESH  = 0.12    # chuyển waypoint khi err < ngưỡng [m]

# ── Multi-rate parameters ──────────────────────────────────────────────────────
SIM_HZ    = 200
POS_HZ    = 10
VEL_HZ    = 25
SIM_DT    = 1.0 / SIM_HZ
POS_EVERY = SIM_HZ // POS_HZ    # mỗi 20 step
VEL_EVERY = SIM_HZ // VEL_HZ    # mỗi 8 step

MAX_VEL_XY = 1.2
MAX_VEL_Z  = 0.8
MAX_TILT   = math.radians(20.0)


def sample_waypoint(device):
    x = random.uniform(*WP_X_RANGE)
    y = random.uniform(*WP_Y_RANGE)
    z = random.uniform(*WP_Z_RANGE)
    return torch.tensor([x, y, z], device=device)


def make_plot():
    plt.ion()
    fig, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
    axes[0].set_title("UAV Hierarchical Navigation — X/Y/Z(t)")
    for ax, lbl in zip(axes, ["X [m]", "Y [m]", "Z [m]"]):
        ax.set_ylabel(lbl)
        ax.grid(True)
    axes[2].set_xlabel("Time [s]")
    lines_pos, lines_tgt = [], []
    for ax in axes:
        lp, = ax.plot([], [], "b-",  lw=2,   label="pos")
        lt, = ax.plot([], [], "r--", lw=1.5, label="waypoint")
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
        margin = 0.2
        axes[i].set_xlim(max(0.0, t[-1] - WINDOW_S), t[-1] + 0.5)
        axes[i].set_ylim(min(all_v) - margin, max(all_v) + margin)
    fig.canvas.draw()
    fig.canvas.flush_events()


def main():
    # ── Setup ──────────────────────────────────────────────────────────────────
    sim_cfg = sim_utils.SimulationCfg(dt=SIM_DT, device=args_cli.device)
    sim = SimulationContext(sim_cfg)
    sim.set_camera_view(eye=[2.0, 2.0, 2.5], target=[0.0, 0.0, 1.0])

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

    print(f"[INFO] POS@{POS_HZ}Hz  VEL@{VEL_HZ}Hz  ATT@{SIM_HZ}Hz")
    print(f"[INFO] err_thresh={ERR_THRESH}m  wp_timeout={WP_HOLD_S}s")

    # ── PID Controllers (3 tầng, giống file 05) ────────────────────────────────
    # Tầng 1 — Position (10 Hz): pos_err → desired velocity
    pid_px = PIDController(kp=0.8,  ki=0.05, kd=0.2,  integral_limit=0.5)
    pid_py = PIDController(kp=0.8,  ki=0.05, kd=0.2,  integral_limit=0.5)
    pid_pz = PIDController(kp=1.0,  ki=0.1,  kd=0.3,  integral_limit=0.5)

    # Tầng 2 — Velocity (25 Hz): vel_err → thrust + desired angles
    pid_vx = PIDController(kp=0.4,  ki=0.02, kd=0.15, integral_limit=0.5)
    pid_vy = PIDController(kp=0.4,  ki=0.02, kd=0.15, integral_limit=0.5)
    pid_vz = PIDController(kp=0.5,  ki=0.1,  kd=0.2,  integral_limit=1.0)

    # Tầng 3 — Attitude (200 Hz): att_err → moments
    pid_roll  = PIDController(kp=0.006, ki=0.001, kd=0.002, integral_limit=0.01)
    pid_pitch = PIDController(kp=0.006, ki=0.001, kd=0.002, integral_limit=0.01)
    pid_yaw   = PIDController(kp=0.003, ki=0.0,   kd=0.001, integral_limit=0.01)

    def reset_all_pids():
        for c in (pid_px, pid_py, pid_pz, pid_vx, pid_vy, pid_vz,
                  pid_roll, pid_pitch, pid_yaw):
            c.reset()

    # ── ZOH state variables ────────────────────────────────────────────────────
    des_vx = des_vy = des_vz = 0.0
    thrust = 0.0
    des_roll = des_pitch = 0.0

    # ── Mission state ──────────────────────────────────────────────────────────
    target   = sample_waypoint(sim.device)
    wp_time  = 0.0
    wp_count = 0
    print(f"[WP {wp_count}] → {[round(v, 2) for v in target.tolist()]}")

    # ── Live plot ─────────────────────────────────────────────────────────────
    maxlen   = int(WINDOW_S / SIM_DT) + 50
    times    = collections.deque(maxlen=maxlen)
    pos_data = [collections.deque(maxlen=maxlen) for _ in range(3)]
    tgt_data = [collections.deque(maxlen=maxlen) for _ in range(3)]
    fig, axes, lines_pos, lines_tgt = make_plot()

    sim_time = 0.0
    count    = 0

    while simulation_app.is_running():
        # ── Đọc trạng thái ─────────────────────────────────────────────────
        pos  = robot.data.root_pos_w[0]
        vel  = robot.data.root_lin_vel_w[0]
        quat = robot.data.root_quat_w[0]
        roll, pitch, yaw = [x[0].item() for x in euler_xyz_from_quat(quat.unsqueeze(0))]

        # ── Mission logic: chuyển waypoint khi đạt hoặc timeout ───────────
        err     = (target - pos).norm().item()
        wp_time += SIM_DT
        if err < ERR_THRESH or wp_time > WP_HOLD_S:
            target   = sample_waypoint(sim.device)
            wp_time  = 0.0
            wp_count += 1
            reset_all_pids()
            des_vx = des_vy = des_vz = 0.0
            thrust = 0.0
            des_roll = des_pitch = 0.0
            print(f"[WP {wp_count}] → {[round(v,2) for v in target.tolist()]}  (prev_err={err:.3f}m)")

        # ══════════════════════════════════════════════════════════════════
        # TẦNG 1 — Position PID (10 Hz)
        # ══════════════════════════════════════════════════════════════════
        if count % POS_EVERY == 0:
            dt1 = SIM_DT * POS_EVERY   # 0.1 s
            des_vx = max(-MAX_VEL_XY, min(MAX_VEL_XY, pid_px.update((target[0] - pos[0]).item(), dt1)))
            des_vy = max(-MAX_VEL_XY, min(MAX_VEL_XY, pid_py.update((target[1] - pos[1]).item(), dt1)))
            des_vz = max(-MAX_VEL_Z,  min(MAX_VEL_Z,  pid_pz.update((target[2] - pos[2]).item(), dt1)))

        # ══════════════════════════════════════════════════════════════════
        # TẦNG 2 — Velocity PID (25 Hz)
        # ══════════════════════════════════════════════════════════════════
        if count % VEL_EVERY == 0:
            dt2 = SIM_DT * VEL_EVERY   # 0.04 s
            thrust    = max(0.0, pid_vz.update(des_vz - vel[2].item(), dt2))
            des_pitch = max(-MAX_TILT, min(MAX_TILT, -pid_vx.update(des_vx - vel[0].item(), dt2)))
            des_roll  = max(-MAX_TILT, min(MAX_TILT,  pid_vy.update(des_vy - vel[1].item(), dt2)))

        # ══════════════════════════════════════════════════════════════════
        # TẦNG 3 — Attitude PID (200 Hz, mỗi step)
        # ══════════════════════════════════════════════════════════════════
        yaw_err = (0.0 - yaw + math.pi) % (2 * math.pi) - math.pi
        m_roll  = pid_roll.update(des_roll  - roll,  SIM_DT)
        m_pitch = pid_pitch.update(des_pitch - pitch, SIM_DT)
        m_yaw   = pid_yaw.update(yaw_err,             SIM_DT)

        # ── Allocation matrix: [Fz,Tx,Ty,Tz] → [F1,F2,F3,F4] ───────────
        wrench  = torch.tensor([thrust, m_roll, m_pitch, m_yaw], device=sim.device)
        F_props = (A_inv @ wrench).clamp(min=0.0)
        F_props += torch.randn(4, device=sim.device) * MOTOR_THRUST_STD

        forces_prop = torch.zeros(robot.num_instances, 4, 3, device=sim.device)
        forces_prop[0, :, 2] = F_props
        robot.permanent_wrench_composer.set_forces_and_torques(
            forces=forces_prop,
            torques=torch.zeros_like(forces_prop),
            body_ids=prop_body_ids,
        )

        robot.write_data_to_sim()
        sim.step()
        sim_time += SIM_DT
        count    += 1
        robot.update(SIM_DT)

        if count % SIM_HZ == 0:
            print(f"t={sim_time:5.1f}s | pos=({pos[0].item():+.2f},{pos[1].item():+.2f},{pos[2].item():+.2f}) | err={err:.3f}m | wp={wp_count}")

        times.append(sim_time)
        for i in range(3):
            pos_data[i].append(pos[i].item())
            tgt_data[i].append(target[i].item())
        if count % PLOT_EVERY == 0:
            update_plot(fig, axes, lines_pos, lines_tgt, times, pos_data, tgt_data)

        p = robot.data.root_pos_w[0].cpu().numpy()
        sim.set_camera_view(
            eye=[p[0] - 1.0, p[1] - 1.0, p[2] + 1.0],
            target=[p[0], p[1], p[2]],
        )


if __name__ == "__main__":
    main()
    simulation_app.close()
