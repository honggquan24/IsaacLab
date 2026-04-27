# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""
PID Position Control với waypoints ngẫu nhiên.

PID output thẳng vào lực/moment vật lý (N, Nm) — không dùng feedforward.
Waypoint được sample ngẫu nhiên sau mỗi lần đạt mục tiêu hoặc timeout.

Luồng:
    SimulationContext + UAV_CFG
        ↓
    QuadcopterPID tính (thrust_N, desired_roll, desired_pitch)
        ↓
    PIDController attitude tính (m_roll, m_pitch, m_yaw)
        ↓
    permanent_wrench_composer apply lực vật lý

Chạy:
    ./isaaclab.sh -p source/isaaclab_assets/isaaclab_assets/uav/pid_control/06_uav_pid_env.py
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import math
import random

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="PID control với random waypoints.")
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
from pid_controller import QuadcopterPID, PIDController


WINDOW_S   = 20.0
PLOT_EVERY = 5

MOTOR_THRUST_STD = 1e-4   # noise tuyệt đối [N]
MOTOR_MOMENT_STD = 1e-5   # noise tuyệt đối [Nm]

# Phạm vi sample waypoint ngẫu nhiên
WP_X_RANGE = (-1.5, 1.5)
WP_Y_RANGE = (-1.5, 1.5)
WP_Z_RANGE = (0.5,  2.0)
WP_HOLD_S  = 8.0    # giữ waypoint tối đa [s]
ERR_THRESH = 0.15   # chuyển waypoint khi error < ngưỡng này [m]


def sample_waypoint(device):
    x = random.uniform(*WP_X_RANGE)
    y = random.uniform(*WP_Y_RANGE)
    z = random.uniform(*WP_Z_RANGE)
    return torch.tensor([x, y, z], device=device)


def make_plot():
    plt.ion()
    fig, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
    axes[0].set_title("UAV PID — X/Y/Z(t) với Random Waypoints")
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
    sim_cfg = sim_utils.SimulationCfg(dt=0.005, device=args_cli.device)
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
    body_id       = robot.find_bodies("body")[0]

    print("[INFO] Setup complete — PID với random waypoints")

    # ── PID Controllers ────────────────────────────────────────────────────────
    pid = QuadcopterPID()
    pid_roll  = PIDController(kp=0.006, ki=0.001, kd=0.002, integral_limit=0.01)
    pid_pitch = PIDController(kp=0.006, ki=0.001, kd=0.002, integral_limit=0.01)
    pid_yaw   = PIDController(kp=0.003, ki=0.0,   kd=0.001, integral_limit=0.01)

    # ── State ─────────────────────────────────────────────────────────────────
    sim_dt      = sim.get_physics_dt()
    target      = sample_waypoint(sim.device)
    wp_time     = 0.0
    wp_count    = 0
    print(f"[WP {wp_count}] target = {target.tolist()}")

    maxlen   = int(WINDOW_S / sim_dt) + 50
    times    = collections.deque(maxlen=maxlen)
    pos_data = [collections.deque(maxlen=maxlen) for _ in range(3)]
    tgt_data = [collections.deque(maxlen=maxlen) for _ in range(3)]
    fig, axes, lines_pos, lines_tgt = make_plot()

    sim_time = 0.0
    count    = 0

    while simulation_app.is_running():
        pos  = robot.data.root_pos_w[0]
        vel  = robot.data.root_lin_vel_w[0]
        quat = robot.data.root_quat_w[0]
        roll, pitch, yaw = [x[0].item() for x in euler_xyz_from_quat(quat.unsqueeze(0))]

        # ── Chuyển waypoint khi đạt mục tiêu hoặc timeout ─────────────────
        err = (target - pos).norm().item()
        wp_time += sim_dt
        if err < ERR_THRESH or wp_time > WP_HOLD_S:
            target   = sample_waypoint(sim.device)
            wp_time  = 0.0
            wp_count += 1
            pid.reset()
            pid_roll.reset(); pid_pitch.reset(); pid_yaw.reset()
            print(f"[WP {wp_count}] target = {[round(v, 2) for v in target.tolist()]}  (prev_err={err:.3f}m)")

        # ── PID ───────────────────────────────────────────────────────────
        thrust_N, desired_roll, desired_pitch = pid.compute(pos=pos, vel=vel, target_pos=target, dt=sim_dt)

        yaw_err = (0.0 - yaw + math.pi) % (2 * math.pi) - math.pi
        m_roll  = pid_roll.update(desired_roll  - roll,  sim_dt)
        m_pitch = pid_pitch.update(desired_pitch - pitch, sim_dt)
        m_yaw   = pid_yaw.update(yaw_err,                 sim_dt)

        # ── Apply lực ─────────────────────────────────────────────────────
        forces_prop  = torch.zeros(robot.num_instances, 4, 3, device=sim.device)
        torques_prop = torch.zeros_like(forces_prop)
        forces_prop[..., 2] = (thrust_N + random.gauss(0.0, MOTOR_THRUST_STD)) / 4.0
        robot.permanent_wrench_composer.set_forces_and_torques(
            forces=forces_prop, torques=torques_prop, body_ids=prop_body_ids
        )

        forces_body  = torch.zeros(robot.num_instances, 1, 3, device=sim.device)
        torques_body = torch.zeros_like(forces_body)
        torques_body[0, 0, 0] = m_roll  + random.gauss(0.0, MOTOR_MOMENT_STD)
        torques_body[0, 0, 1] = m_pitch + random.gauss(0.0, MOTOR_MOMENT_STD)
        torques_body[0, 0, 2] = m_yaw   + random.gauss(0.0, MOTOR_MOMENT_STD)
        robot.permanent_wrench_composer.set_forces_and_torques(
            forces=forces_body, torques=torques_body, body_ids=body_id
        )

        robot.write_data_to_sim()
        sim.step()
        sim_time += sim_dt
        count    += 1
        robot.update(sim_dt)

        if count % int(1.0 / sim_dt) == 0:
            print(f"t={sim_time:5.1f}s | pos=({pos[0].item():+.2f},{pos[1].item():+.2f},{pos[2].item():+.2f}) | err={err:.3f}m | thrust={thrust_N:.4f}N")

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
