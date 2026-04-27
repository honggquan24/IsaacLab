# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""
Altitude PID Control — chỉ điều khiển độ cao Z.
PID output thẳng vào lực/moment vật lý (N, Nm).

Chạy:
    ./isaaclab.sh -p source/isaaclab_assets/isaaclab_assets/uav/pid_control/01_uav_pid_altitude.py
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import math

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Altitude PID control — direct force.")
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

TARGET_Z   = 1.0   # độ cao mục tiêu [m]

MOTOR_THRUST_STD = 1e-4    # noise tuyệt đối trên thrust [N]
MOTOR_MOMENT_STD = 1e-5    # noise tuyệt đối [Nm]


def make_plot():
    plt.ion()
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.set_title("UAV Altitude Response — Z(t)")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Height [m]")
    ax.grid(True)
    line_pos,    = ax.plot([], [], "b-",  lw=2,   label="pos_z")
    line_target, = ax.plot([], [], "r--", lw=1.5, label="target_z")
    line_err,    = ax.plot([], [], "g:",  lw=1,   label="error_z")
    ax.legend(loc="upper right")
    fig.tight_layout()
    return fig, ax, line_pos, line_target, line_err


def update_plot(fig, ax, line_pos, line_target, line_err, times, pos_zs, target_zs, err_zs):
    t  = list(times)
    pz = list(pos_zs)
    tz = list(target_zs)
    ez = list(err_zs)

    line_pos.set_data(t, pz)
    line_target.set_data(t, tz)
    line_err.set_data(t, ez)

    if t:
        ax.set_xlim(max(0.0, t[-1] - WINDOW_S), t[-1] + 0.5)
        all_vals = pz + tz + ez
        margin = 0.3
        ax.set_ylim(min(all_vals) - margin, max(all_vals) + margin)

    fig.canvas.draw()
    fig.canvas.flush_events()


def main():
    # ── Setup ──────────────────────────────────────────────────────────────────
    sim_cfg = sim_utils.SimulationCfg(dt=0.005, device=args_cli.device)
    sim = SimulationContext(sim_cfg)
    sim.set_camera_view(eye=[1.5, 0.5, 1.5], target=[0.0, 0.0, 1.0])

    cfg = sim_utils.GroundPlaneCfg()
    cfg.func("/World/defaultGroundPlane", cfg)
    cfg = sim_utils.DistantLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    cfg.func("/World/Light", cfg)

    robot_cfg = UAV_CFG.replace(prim_path="/World/Crazyflie")
    robot_cfg = robot_cfg.replace(init_state=robot_cfg.init_state.replace(pos=(0.0, 0.0, 0.05)))
    robot_cfg.spawn.func("/World/Crazyflie", robot_cfg.spawn, translation=robot_cfg.init_state.pos)
    robot = Articulation(robot_cfg)

    sim.reset()

    prop_body_ids = robot.find_bodies("m.*_prop")[0]   # [m1, m2, m3, m4]
    A_inv = make_alloc_inv(device=sim.device)          # (4×4) ma trận phân bổ nghịch đảo

    print(f"[INFO] target_z = {TARGET_Z} m")

    # ── PID Controllers ────────────────────────────────────────────────────────
    # Vòng ngoài: altitude Z → thrust (N) — không dùng feedforward, integral tự bù trọng lực
    pid_z     = PIDController(kp=1.0, ki=0.0, kd=0.0, integral_limit=1.0)
    # Vòng trong: attitude → moments
    pid_roll  = PIDController(kp=0.001, ki=0.001, kd=0.002, integral_limit=0.01)
    pid_pitch = PIDController(kp=0.001, ki=0.001, kd=0.002, integral_limit=0.01)
    pid_yaw   = PIDController(kp=0.001, ki=0.0,   kd=0.001, integral_limit=0.01)

    # ── Live plot ─────────────────────────────────────────────────────────────
    sim_dt = sim.get_physics_dt()
    maxlen = int(WINDOW_S / sim_dt) + 50
    times    = collections.deque(maxlen=maxlen)
    pos_zs   = collections.deque(maxlen=maxlen)
    target_zs= collections.deque(maxlen=maxlen)
    err_zs   = collections.deque(maxlen=maxlen)
    fig, ax, line_pos, line_target, line_err = make_plot()

    sim_time = 0.0
    count    = 0

    # ── Simulation loop ───────────────────────────────────────────────────────
    while simulation_app.is_running():
        pos  = robot.data.root_pos_w[0]
        vel  = robot.data.root_lin_vel_w[0]
        quat = robot.data.root_quat_w[0]
        roll, pitch, yaw = [x[0].item() for x in euler_xyz_from_quat(quat.unsqueeze(0))]

        # ── Altitude PID: Z error → thrust ────────────────────────────────
        err_z  = TARGET_Z - pos[2].item()
        thrust = max(0.0, pid_z.update(err_z, sim_dt))

        # ── Attitude PID: giữ drone level ─────────────────────────────────
        yaw_err = (0.0 - yaw + math.pi) % (2 * math.pi) - math.pi
        m_roll  = pid_roll.update(0.0 - roll,  sim_dt)
        m_pitch = pid_pitch.update(0.0 - pitch, sim_dt)
        m_yaw   = pid_yaw.update(yaw_err,       sim_dt)

        # ── Allocation matrix: [Fz, Tx, Ty, Tz] → [F1, F2, F3, F4] ──────
        # Mỗi prop nhận lực riêng — roll/pitch/yaw tạo ra từ hiệu lực giữa các prop
        # Noise độc lập mỗi motor mô phỏng chênh lệch coreless motor
        wrench  = torch.tensor([thrust, m_roll, m_pitch, m_yaw], device=sim.device)
        F_props = (A_inv @ wrench).clamp(min=0.0)                # (4,) — không âm
        F_props += torch.randn(4, device=sim.device) * MOTOR_THRUST_STD  # noise/motor

        forces_prop = torch.zeros(robot.num_instances, 4, 3, device=sim.device)
        forces_prop[0, :, 2] = F_props                           # chỉ lực Z (nâng lên)
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

        # log mỗi 1 giây
        if count % int(1.0 / sim_dt) == 0:
            print(f"t={sim_time:5.1f}s | z={pos[2].item():+.3f}m | err_z={err_z:+.3f}m | thrust={thrust:.4f}N")

        # Thu thập data
        pos_z    = pos[2].item()
        times.append(sim_time)
        pos_zs.append(pos_z)
        target_zs.append(TARGET_Z)
        err_zs.append(TARGET_Z - pos_z)

        if count % PLOT_EVERY == 0:
            update_plot(fig, ax, line_pos, line_target, line_err,
                        times, pos_zs, target_zs, err_zs)

        p = robot.data.root_pos_w[0].cpu().numpy()
        sim.set_camera_view(
            eye=[p[0] - 1.0, p[1] - 1.0, p[2] + 1.0],
            target=[p[0], p[1], p[2]],
        )


if __name__ == "__main__":
    main()
    simulation_app.close()
