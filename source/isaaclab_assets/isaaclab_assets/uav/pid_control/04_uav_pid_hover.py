# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""
Multi-rate Position Control cho quadcopter.

Điểm khác so với file 02 (position): PID KHÔNG chạy mỗi sim step.
Thay vào đó:
    - Physics (sim): 200 Hz  (dt = 0.005 s)
    - Control PID:    50 Hz  (dt = 0.020 s, mỗi DECIMATION = 4 bước)

Điều này phản ánh thực tế phần cứng: vi xử lý điều khiển (STM32, ESP32...)
thường chạy 50–200 Hz, trong khi physics engine mô phỏng ở 200–1000 Hz.

Kỹ thuật: Zero-Order Hold (ZOH) — giữ nguyên lệnh điều khiển cuối cùng
cho đến khi vòng lặp PID chạy lần tiếp theo.

Sơ đồ thời gian:
    sim:  |--|--|--|--|--|--|--|--|  200 Hz
    PID:  |--------|--------|----   50 Hz  (tính lại mỗi 4 step)
           ← ZOH →  ← ZOH →

Chạy:
    ./isaaclab.sh -p source/isaaclab_assets/isaaclab_assets/uav/pid_control/04_uav_pid_hover.py
    ./isaaclab.sh -p source/isaaclab_assets/isaaclab_assets/uav/pid_control/04_uav_pid_hover.py --headless
"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Multi-rate PID cho Crazyflie — PID_freq != sim_freq.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import collections
import math
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

TARGET_POS = torch.tensor([0.0, 0.0, 1.0])

# ── Multi-rate parameters ───────────────────────────────────────────────────────
SIM_HZ     = 200    # physics timestep
CONTROL_HZ = 50     # PID update rate (phản ánh phần cứng thực)
DECIMATION = SIM_HZ // CONTROL_HZ   # = 4 — PID tính lại mỗi 4 sim step
SIM_DT     = 1.0 / SIM_HZ          # = 0.005 s
CONTROL_DT = 1.0 / CONTROL_HZ      # = 0.020 s — dt đúng cho PID


def make_plot():
    plt.ion()
    fig, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
    axes[0].set_title(f"UAV Multi-rate PID — X/Y/Z(t)  [PID@{CONTROL_HZ}Hz / sim@{SIM_HZ}Hz]")
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
    # ── Setup ─────────────────────────────────────────────────────────────────
    sim_cfg = sim_utils.SimulationCfg(dt=SIM_DT, device=args_cli.device)
    sim = SimulationContext(sim_cfg)
    sim.set_camera_view(eye=[1.5, 1.5, 1.5], target=[0.0, 0.0, 1.0])

    sim_utils.GroundPlaneCfg().func("/World/defaultGroundPlane", sim_utils.GroundPlaneCfg())
    sim_utils.DistantLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75)).func(
        "/World/Light", sim_utils.DistantLightCfg(intensity=3000.0)
    )

    robot_cfg = UAV_CFG.replace(prim_path="/World/Crazyflie")
    robot_cfg = robot_cfg.replace(init_state=robot_cfg.init_state.replace(pos=(0.0, 0.0, 0.05)))
    robot_cfg.spawn.func("/World/Crazyflie", robot_cfg.spawn, translation=robot_cfg.init_state.pos)
    robot = Articulation(robot_cfg)

    sim.reset()

    prop_body_ids = robot.find_bodies("m.*_prop")[0]
    A_inv  = make_alloc_inv(device=sim.device)
    target = TARGET_POS.to(sim.device)

    print(f"[INFO] SIM={SIM_HZ}Hz  PID={CONTROL_HZ}Hz  DECIMATION={DECIMATION}x")
    print(f"[INFO] target = {target.tolist()}")

    # ── PID Controllers ────────────────────────────────────────────────────────
    pid       = QuadcopterPID()
    pid_roll  = PIDController(kp=0.006, ki=0.001, kd=0.002, integral_limit=0.01)
    pid_pitch = PIDController(kp=0.006, ki=0.001, kd=0.002, integral_limit=0.01)
    pid_yaw   = PIDController(kp=0.003, ki=0.0,   kd=0.001, integral_limit=0.01)

    # ZOH buffer — giữ lực prop giữa 2 lần PID cập nhật
    F_props_hold = torch.zeros(4, device=sim.device)

    # ── Live plot ─────────────────────────────────────────────────────────────
    maxlen   = int(WINDOW_S / SIM_DT) + 50
    times    = collections.deque(maxlen=maxlen)
    pos_data = [collections.deque(maxlen=maxlen) for _ in range(3)]
    tgt_data = [collections.deque(maxlen=maxlen) for _ in range(3)]
    fig, axes, lines_pos, lines_tgt = make_plot()

    sim_time  = 0.0
    step      = 0
    log_every = int(1.0 / SIM_DT)

    while simulation_app.is_running():
        if step % int(20.0 / SIM_DT) == 0 and step > 0:
            sim_time = 0.0
            joint_pos, joint_vel = robot.data.default_joint_pos, robot.data.default_joint_vel
            robot.write_joint_state_to_sim(joint_pos, joint_vel)
            robot.write_root_pose_to_sim(robot.data.default_root_state[:, :7])
            robot.write_root_velocity_to_sim(robot.data.default_root_state[:, 7:])
            robot.reset()
            pid.reset()
            pid_roll.reset(); pid_pitch.reset(); pid_yaw.reset()
            F_props_hold.zero_()
            print("[RESET]")

        pos  = robot.data.root_pos_w[0]
        vel  = robot.data.root_lin_vel_w[0]
        quat = robot.data.root_quat_w[0]
        roll, pitch, yaw = euler_xyz_from_quat(quat.unsqueeze(0))
        roll, pitch, yaw = roll[0].item(), pitch[0].item(), yaw[0].item()

        # ── PID chỉ tính lại mỗi DECIMATION bước (ZOH) ───────────────────
        if step % DECIMATION == 0:
            thrust_N, desired_roll, desired_pitch = pid.compute(
                pos=pos, vel=vel, target_pos=target, dt=CONTROL_DT
            )
            yaw_err = (0.0 - yaw + math.pi) % (2 * math.pi) - math.pi
            m_roll  = pid_roll.update(desired_roll  - roll,  CONTROL_DT)
            m_pitch = pid_pitch.update(desired_pitch - pitch, CONTROL_DT)
            m_yaw   = pid_yaw.update(yaw_err,                 CONTROL_DT)

            # Allocation matrix: [Fz,Tx,Ty,Tz] → [F1,F2,F3,F4] rồi lưu ZOH
            wrench       = torch.tensor([thrust_N, m_roll, m_pitch, m_yaw], device=sim.device)
            F_props_hold = (A_inv @ wrench).clamp(min=0.0)
            F_props_hold = F_props_hold + torch.randn(4, device=sim.device) * MOTOR_THRUST_STD

        # ── Apply lực MỖI sim step — ZOH giữ F_props_hold từ lần PID trước ──
        forces_prop = torch.zeros(robot.num_instances, 4, 3, device=sim.device)
        forces_prop[0, :, 2] = F_props_hold   # chỉ lực Z (nâng) cho 4 prop
        robot.permanent_wrench_composer.set_forces_and_torques(
            forces=forces_prop,
            torques=torch.zeros_like(forces_prop),
            body_ids=prop_body_ids,
        )
        robot.write_data_to_sim()
        sim.step()
        sim_time += SIM_DT
        step     += 1
        robot.update(SIM_DT)

        if step % log_every == 0:
            err = (target - pos).norm().item()
            print(
                f"t={sim_time:6.1f}s | pos=({pos[0].item():+.2f},{pos[1].item():+.2f},{pos[2].item():+.2f}) | "
                f"err={err:.3f}m | Fz={F_props_hold.sum().item():.4f}N | "
                f"rpy=({math.degrees(roll):+.1f}°,{math.degrees(pitch):+.1f}°,{math.degrees(yaw):+.1f}°)"
            )

        times.append(sim_time)
        for i in range(3):
            pos_data[i].append(pos[i].item())
            tgt_data[i].append(target[i].item())
        if step % PLOT_EVERY == 0:
            update_plot(fig, axes, lines_pos, lines_tgt, times, pos_data, tgt_data)

        p = pos.cpu().numpy()
        sim.set_camera_view(
            eye=[p[0] - 1.0, p[1] - 1.0, p[2] + 0.8],
            target=[p[0], p[1], p[2]],
        )


if __name__ == "__main__":
    main()
    simulation_app.close()
