# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""
Mô phỏng Hierarchical (Cascaded) PID Control cho quadcopter.

PID output thẳng vào lực/moment vật lý (N, Nm) — không dùng feedforward.

Kiến trúc 3 tầng tần số khác nhau:
    ┌─────────────────────────────────────────────────────────┐
    │  Tầng 1 – Position (10 Hz)                              │
    │    pos_error [x,y,z]  →  desired_velocity [vx,vy,vz]   │
    ├─────────────────────────────────────────────────────────┤
    │  Tầng 2 – Velocity  (25 Hz)                             │
    │    vel_error [vx,vy]  →  desired_pitch / desired_roll   │
    │    vel_error [vz]     →  thrust (N)                     │
    ├─────────────────────────────────────────────────────────┤
    │  Tầng 3 – Attitude  (50 Hz / every step)               │
    │    att_error [r,p,y]  →  moments [mx, my, mz] (Nm)     │
    └─────────────────────────────────────────────────────────┘

Chạy:
    ./isaaclab.sh -p source/isaaclab_assets/isaaclab_assets/uav/pid_control/05_uav_pid_hierarchical.py
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import math

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Hierarchical PID control cho Crazyflie.")
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


WINDOW_S   = 20.0
PLOT_EVERY = 5

MOTOR_THRUST_STD = 1e-4   # noise tuyệt đối [N]
MOTOR_MOMENT_STD = 1e-5   # noise tuyệt đối [Nm]


def make_plot():
    plt.ion()
    fig, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
    axes[0].set_title("UAV Hierarchical PID — X/Y/Z(t)")
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

    print("[INFO]: Setup complete — Hierarchical PID control")

    # ── Waypoints ─────────────────────────────────────────────────────────────
    WAYPOINTS = [
        torch.tensor([0.0,  0.0,  1.0], device=sim.device),
        torch.tensor([1.0,  0.0,  1.0], device=sim.device),
        torch.tensor([1.0,  1.0,  1.5], device=sim.device),
        torch.tensor([0.0,  0.0,  1.0], device=sim.device),
    ]
    WAYPOINT_HOLD_S = 5.0
    wp_idx = 0

    # ── Tần số mỗi tầng ───────────────────────────────────────────────────────
    SIM_HZ    = int(1.0 / 0.005)   # 200 Hz
    POS_HZ    = 10
    VEL_HZ    = 25
    POS_EVERY = SIM_HZ // POS_HZ   # mỗi 20 bước
    VEL_EVERY = SIM_HZ // VEL_HZ   # mỗi 8 bước

    MAX_VEL_XY = 1.0
    MAX_VEL_Z  = 0.8
    MAX_TILT   = math.radians(20.0)

    # ── PID Controllers ────────────────────────────────────────────────────────
    # Tầng 1 — Position → desired velocity
    pid_px = PIDController(kp=0.8,  ki=0.05, kd=0.2,  integral_limit=0.5)
    pid_py = PIDController(kp=0.8,  ki=0.05, kd=0.2,  integral_limit=0.5)
    pid_pz = PIDController(kp=1.0,  ki=0.1,  kd=0.3,  integral_limit=0.5)

    # Tầng 2 — Velocity → desired angle + thrust (không nhân hover_thrust)
    pid_vx = PIDController(kp=0.4,  ki=0.02, kd=0.15, integral_limit=0.5)
    pid_vy = PIDController(kp=0.4,  ki=0.02, kd=0.15, integral_limit=0.5)
    pid_vz = PIDController(kp=0.5,  ki=0.1,  kd=0.2,  integral_limit=1.0)

    # Tầng 3 — Attitude → moments
    pid_roll  = PIDController(kp=0.006, ki=0.001, kd=0.002, integral_limit=0.01)
    pid_pitch = PIDController(kp=0.006, ki=0.001, kd=0.002, integral_limit=0.01)
    pid_yaw   = PIDController(kp=0.003, ki=0.0,   kd=0.001, integral_limit=0.01)

    des_vx = des_vy = des_vz = 0.0
    thrust = 0.0
    des_roll = des_pitch = 0.0

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
        if count % (SIM_HZ * 25) == 0 and count > 0:
            sim_time = 0.0
            count = 0
            wp_idx = 0
            joint_pos, joint_vel = robot.data.default_joint_pos, robot.data.default_joint_vel
            robot.write_joint_state_to_sim(joint_pos, joint_vel)
            robot.write_root_pose_to_sim(robot.data.default_root_state[:, :7])
            robot.write_root_velocity_to_sim(robot.data.default_root_state[:, 7:])
            robot.reset()
            pid_px.reset(); pid_py.reset(); pid_pz.reset()
            pid_vx.reset(); pid_vy.reset(); pid_vz.reset()
            pid_roll.reset(); pid_pitch.reset(); pid_yaw.reset()
            des_vx = des_vy = des_vz = 0.0
            thrust = 0.0
            des_roll = des_pitch = 0.0
            print(">>>>>>>> Reset!")

        wp_idx = min(int(sim_time / WAYPOINT_HOLD_S), len(WAYPOINTS) - 1)
        target = WAYPOINTS[wp_idx]

        # ── Đọc trạng thái ─────────────────────────────────────────────────
        pos  = robot.data.root_pos_w[0]
        vel  = robot.data.root_lin_vel_w[0]
        quat = robot.data.root_quat_w[0]
        roll, pitch, yaw = [x[0].item() for x in euler_xyz_from_quat(quat.unsqueeze(0))]

        # ══════════════════════════════════════════════════════════════════════
        # TẦNG 1 — Position PID (10 Hz)
        # ══════════════════════════════════════════════════════════════════════
        if count % POS_EVERY == 0:
            dt1 = sim_dt * POS_EVERY
            des_vx = max(-MAX_VEL_XY, min(MAX_VEL_XY, pid_px.update((target[0] - pos[0]).item(), dt1)))
            des_vy = max(-MAX_VEL_XY, min(MAX_VEL_XY, pid_py.update((target[1] - pos[1]).item(), dt1)))
            des_vz = max(-MAX_VEL_Z,  min(MAX_VEL_Z,  pid_pz.update((target[2] - pos[2]).item(), dt1)))

        # ══════════════════════════════════════════════════════════════════════
        # TẦNG 2 — Velocity PID (25 Hz)
        # ══════════════════════════════════════════════════════════════════════
        if count % VEL_EVERY == 0:
            dt2 = sim_dt * VEL_EVERY
            thrust    = max(0.0, pid_vz.update(des_vz - vel[2].item(), dt2))
            des_pitch = max(-MAX_TILT, min(MAX_TILT, -pid_vx.update(des_vx - vel[0].item(), dt2)))
            des_roll  = max(-MAX_TILT, min(MAX_TILT,  pid_vy.update(des_vy - vel[1].item(), dt2)))

        # ══════════════════════════════════════════════════════════════════════
        # TẦNG 3 — Attitude PID (200 Hz)
        # ══════════════════════════════════════════════════════════════════════
        yaw_err = (0.0 - yaw + math.pi) % (2 * math.pi) - math.pi
        m_roll  = pid_roll.update(des_roll  - roll,  sim_dt)
        m_pitch = pid_pitch.update(des_pitch - pitch, sim_dt)
        m_yaw   = pid_yaw.update(yaw_err,             sim_dt)

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
        sim_time += sim_dt
        count    += 1
        robot.update(sim_dt)

        if count % SIM_HZ == 0:
            err = (target - pos).norm().item()
            print(
                f"t={sim_time:5.1f}s | wp={wp_idx} target={target.tolist()} | "
                f"pos=({pos[0].item():+.2f},{pos[1].item():+.2f},{pos[2].item():+.2f}) | "
                f"err={err:.3f}m"
            )

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
