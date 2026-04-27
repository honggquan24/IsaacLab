# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""
Mô phỏng Velocity Control cho quadcopter dùng PID — có điều khiển bàn phím.

PID output thẳng vào lực/moment vật lý (N, Nm) — không dùng feedforward.

2 vòng PID:
    Vòng ngoài  : vel_z error    →  thrust (N)
    Vòng ngoài  : vel_x/y error  →  desired_pitch / desired_roll
    Vòng trong  : att error      →  moments (Nm)

Bàn phím (cửa sổ Isaac Sim phải được focus):
    W / S   →  vx  +/- (tiến / lùi)
    A / D   →  vy  +/- (trái / phải)
    Q / E   →  vz  +/- (lên / xuống)
    R       →  reset drone về vị trí ban đầu

Chạy:
    ./isaaclab.sh -p source/isaaclab_assets/isaaclab_assets/uav/pid_control/03_uav_pid_velocity.py
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import math

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Velocity PID control cho Crazyflie.")
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
import carb
import omni.appwindow

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
    axes[0].set_title("UAV Velocity Response — Vx/Vy/Vz(t)")
    for ax, lbl in zip(axes, ["Vx [m/s]", "Vy [m/s]", "Vz [m/s]"]):
        ax.set_ylabel(lbl)
        ax.grid(True)
    axes[2].set_xlabel("Time [s]")
    lines_vel, lines_cmd = [], []
    for ax in axes:
        lv, = ax.plot([], [], "b-",  lw=2,   label="vel")
        lc, = ax.plot([], [], "r--", lw=1.5, label="cmd")
        ax.legend(loc="upper right", fontsize=8)
        lines_vel.append(lv)
        lines_cmd.append(lc)
    fig.tight_layout()
    return fig, axes, lines_vel, lines_cmd


def update_plot(fig, axes, lines_vel, lines_cmd, times, vel_data, cmd_data):
    t = list(times)
    if not t:
        return
    for i, (lv, lc) in enumerate(zip(lines_vel, lines_cmd)):
        v  = list(vel_data[i])
        c  = list(cmd_data[i])
        lv.set_data(t, v)
        lc.set_data(t, c)
        all_v = v + c
        margin = 0.1
        axes[i].set_xlim(max(0.0, t[-1] - WINDOW_S), t[-1] + 0.5)
        axes[i].set_ylim(min(all_v) - margin, max(all_v) + margin)
    fig.canvas.draw()
    fig.canvas.flush_events()


def _setup_keyboard():
    app_window  = omni.appwindow.get_default_app_window()
    input_iface = carb.input.acquire_input_interface()
    keyboard    = app_window.get_keyboard()
    return input_iface, keyboard


def _read_velocity_cmd(input_iface, keyboard, vel_step: float = 0.5) -> tuple[float, float, float, bool]:
    key = carb.input.KeyboardInput

    def held(k):
        return input_iface.get_keyboard_value(keyboard, k) > 0.0

    dvx = vel_step if held(key.W) else (-vel_step if held(key.S) else 0.0)
    dvy = vel_step if held(key.A) else (-vel_step if held(key.D) else 0.0)
    dvz = vel_step if held(key.E) else (-vel_step if held(key.Q) else 0.0)
    do_reset = held(key.R)

    return dvx, dvy, dvz, do_reset


def main():
    # ── Setup ──────────────────────────────────────────────────────────────────
    sim_cfg = sim_utils.SimulationCfg(dt=0.005, device=args_cli.device)
    sim = SimulationContext(sim_cfg)

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

    print("[INFO]: Setup complete — Velocity PID control (keyboard interactive)")
    print("[INFO]: W/S=vx  A/D=vy  Q/E=vz  R=reset  (focus Isaac Sim window)")

    # ── Bàn phím ──────────────────────────────────────────────────────────────
    input_iface, keyboard = _setup_keyboard()
    VEL_CMD = 0.5
    MAX_CMD = 2.0

    TARGET_VEL = torch.zeros(3, device=sim.device)
    MAX_TILT   = math.radians(20.0)

    # ── PID Controllers ────────────────────────────────────────────────────────
    # Vòng ngoài velocity — output thẳng N / rad (không cần hover_thrust)
    pid_vz    = PIDController(kp=0.5, ki=0.1, kd=0.2, integral_limit=1.0)
    pid_vx    = PIDController(kp=0.5, ki=0.05, kd=0.2, integral_limit=0.5)
    pid_vy    = PIDController(kp=0.5, ki=0.05, kd=0.2, integral_limit=0.5)
    # Vòng trong attitude
    pid_roll  = PIDController(kp=0.006, ki=0.001, kd=0.002, integral_limit=0.01)
    pid_pitch = PIDController(kp=0.006, ki=0.001, kd=0.002, integral_limit=0.01)
    pid_yaw   = PIDController(kp=0.003, ki=0.0,   kd=0.001, integral_limit=0.01)

    # ── Live plot ─────────────────────────────────────────────────────────────
    sim_dt   = sim.get_physics_dt()
    maxlen   = int(WINDOW_S / sim_dt) + 50
    times    = collections.deque(maxlen=maxlen)
    vel_data = [collections.deque(maxlen=maxlen) for _ in range(3)]
    cmd_data = [collections.deque(maxlen=maxlen) for _ in range(3)]
    fig, axes, lines_vel, lines_cmd = make_plot()

    sim_time = 0.0
    count    = 0
    _prev_reset_held = False

    while simulation_app.is_running():
        # ── Đọc bàn phím ──────────────────────────────────────────────────
        dvx, dvy, dvz, do_reset = _read_velocity_cmd(input_iface, keyboard, VEL_CMD)

        TARGET_VEL[0] = max(-MAX_CMD, min(MAX_CMD, dvx))
        TARGET_VEL[1] = max(-MAX_CMD, min(MAX_CMD, dvy))
        TARGET_VEL[2] = max(-MAX_CMD, min(MAX_CMD, dvz))

        if do_reset and not _prev_reset_held:
            sim_time = 0.0
            count = 0
            joint_pos, joint_vel = robot.data.default_joint_pos, robot.data.default_joint_vel
            robot.write_joint_state_to_sim(joint_pos, joint_vel)
            robot.write_root_pose_to_sim(robot.data.default_root_state[:, :7])
            robot.write_root_velocity_to_sim(robot.data.default_root_state[:, 7:])
            robot.reset()
            pid_vz.reset(); pid_vx.reset(); pid_vy.reset()
            pid_roll.reset(); pid_pitch.reset(); pid_yaw.reset()
            TARGET_VEL[:] = 0.0
            print(">>>>>>>> Reset!")
        _prev_reset_held = do_reset

        # ── Đọc trạng thái ─────────────────────────────────────────────────
        vel_w  = robot.data.root_lin_vel_w[0]
        quat   = robot.data.root_quat_w[0]
        roll, pitch, yaw = [x[0].item() for x in euler_xyz_from_quat(quat.unsqueeze(0))]
        vx, vy, vz = vel_w[0].item(), vel_w[1].item(), vel_w[2].item()

        # ── Vòng ngoài: velocity → thrust + desired angles ─────────────────
        thrust = max(0.0, pid_vz.update(TARGET_VEL[2].item() - vz, sim_dt))

        des_pitch = -pid_vx.update(TARGET_VEL[0].item() - vx, sim_dt)
        des_roll  =  pid_vy.update(TARGET_VEL[1].item() - vy, sim_dt)
        des_pitch = max(-MAX_TILT, min(MAX_TILT, des_pitch))
        des_roll  = max(-MAX_TILT, min(MAX_TILT, des_roll))

        # ── Vòng trong: attitude error → moments ──────────────────────────
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

        if count % int(1.0 / sim_dt) == 0:
            pos = robot.data.root_pos_w[0]
            print(
                f"t={sim_time:5.1f}s | "
                f"cmd=({TARGET_VEL[0].item():+.1f},{TARGET_VEL[1].item():+.1f},{TARGET_VEL[2].item():+.1f})m/s | "
                f"vel=({vx:+.2f},{vy:+.2f},{vz:+.2f})m/s | "
                f"pos=({pos[0].item():+.2f},{pos[1].item():+.2f},{pos[2].item():+.2f})m"
            )

        times.append(sim_time)
        for i, v in enumerate([vx, vy, vz]):
            vel_data[i].append(v)
            cmd_data[i].append(TARGET_VEL[i].item())
        if count % PLOT_EVERY == 0:
            update_plot(fig, axes, lines_vel, lines_cmd, times, vel_data, cmd_data)

        p = robot.data.root_pos_w[0].cpu().numpy()
        sim.set_camera_view(
            eye=[p[0] - 0.5, p[1] - 0.5, p[2] + 0.5],
            target=[p[0], p[1], p[2]],
        )


if __name__ == "__main__":
    main()
    simulation_app.close()
