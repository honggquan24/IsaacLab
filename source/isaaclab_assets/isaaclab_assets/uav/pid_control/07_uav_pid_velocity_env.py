# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""
Velocity PID Control — dùng UAVHoverEnvCfg thay vì SimulationContext thủ công.

Bàn phím (focus cửa sổ Isaac Sim):
    W / S   →  vx  +/- (tiến / lùi)
    A / D   →  vy  +/- (trái / phải)
    Q / E   →  vz  +/- (lên / xuống)
    R       →  reset episode

Chạy:
    ./isaaclab.sh -p source/isaaclab_assets/isaaclab_assets/uav/pid_control/uav_pid_velocity_env.py
"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Velocity PID + UAVHoverEnvCfg.")
parser.add_argument("--num_envs", type=int, default=1)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import math
import torch
import carb
import omni.appwindow

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.utils.math import euler_xyz_from_quat

from isaaclab_assets.uav.uav_hover_env_cfg import UAVHoverEnvCfg


# ── Thông số UAVThrustAction (phải khớp với ActionCfg) ────────────────────────
THRUST_TO_WEIGHT = 1.9
MOMENT_SCALE     = 0.01


def pid_to_action(thrust_N, m_roll, m_pitch, m_yaw, robot_weight):
    """Chuyển PID output (N, Nm) → action [-1,1] cho UAVThrustAction."""
    a0 = max(-1.0, min(1.0, thrust_N * 2.0 / (THRUST_TO_WEIGHT * robot_weight) - 1.0))
    a1 = max(-1.0, min(1.0, m_roll  / MOMENT_SCALE))
    a2 = max(-1.0, min(1.0, m_pitch / MOMENT_SCALE))
    a3 = max(-1.0, min(1.0, m_yaw   / MOMENT_SCALE))
    return torch.tensor([[a0, a1, a2, a3]], dtype=torch.float32)


def main():
    # ── Env từ config sẵn có ───────────────────────────────────────────────────
    env_cfg = UAVHoverEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env = ManagerBasedRLEnv(cfg=env_cfg)

    robot        = env.scene["robot"]
    robot_mass   = robot.root_physx_view.get_masses()[0].sum().item()
    gravity_mag  = torch.tensor(env.sim.cfg.gravity, device=env.device).norm().item()
    robot_weight = robot_mass * gravity_mag
    hover_thrust = robot_weight

    control_dt = env.physics_dt * env_cfg.decimation

    print(f"[INFO] robot_mass={robot_mass:.4f}kg  hover_thrust={hover_thrust:.4f}N")
    print("[INFO] W/S=vx  A/D=vy  Q/E=vz  R=reset")

    # ── Bàn phím ──────────────────────────────────────────────────────────────
    app_window  = omni.appwindow.get_default_app_window()
    input_iface = carb.input.acquire_input_interface()
    keyboard    = app_window.get_keyboard()

    def held(k):
        return input_iface.get_keyboard_value(keyboard, k) > 0.0

    KEY = carb.input.KeyboardInput
    VEL_CMD = 0.5   # m/s khi nhấn phím
    MAX_CMD = 2.0   # giới hạn lệnh

    # ── Thông số PID ───────────────────────────────────────────────────────────
    kp_vz, ki_vz, kd_vz = 1.5 * hover_thrust, 0.3 * hover_thrust, 0.5 * hover_thrust
    kp_vh, ki_vh, kd_vh = 0.5, 0.05, 0.2
    kp_att, ki_att, kd_att = 0.006, 0.001, 0.002
    MAX_TILT  = math.radians(20.0)
    INTEG_LIM = 0.5

    def reset_pid_state():
        return [0.0] * 9  # int_vz, int_vx, int_vy, int_ro, int_pi, int_ya, prev_vz, prev_vx, prev_vy

    def reset_att_state():
        return [0.0] * 6  # prev_ro, prev_pi, prev_ya, int_ro_att, int_pi_att, int_ya_att

    # ── Reset env ─────────────────────────────────────────────────────────────
    obs, _ = env.reset()
    int_vz = int_vx = int_vy = 0.0
    int_ro = int_pi = int_ya = 0.0
    prev_vz = prev_vx = prev_vy = 0.0
    prev_ro = prev_pi = prev_ya = 0.0

    TARGET_VEL = torch.zeros(3, device=env.device)
    _prev_reset = False

    sim_time = 0.0
    step     = 0
    log_every = max(1, int(1.0 / control_dt))

    # ── Simulation loop ────────────────────────────────────────────────────────
    while simulation_app.is_running():
        # ── Bàn phím → vận tốc mục tiêu ───────────────────────────────────
        dvx = VEL_CMD if held(KEY.W) else (-VEL_CMD if held(KEY.S) else 0.0)
        dvy = VEL_CMD if held(KEY.A) else (-VEL_CMD if held(KEY.D) else 0.0)
        dvz = VEL_CMD if held(KEY.E) else (-VEL_CMD if held(KEY.Q) else 0.0)
        do_reset = held(KEY.R)

        TARGET_VEL[0] = max(-MAX_CMD, min(MAX_CMD, dvx))
        TARGET_VEL[1] = max(-MAX_CMD, min(MAX_CMD, dvy))
        TARGET_VEL[2] = max(-MAX_CMD, min(MAX_CMD, dvz))

        if do_reset and not _prev_reset:
            obs, _ = env.reset()
            int_vz = int_vx = int_vy = 0.0
            int_ro = int_pi = int_ya = 0.0
            prev_vz = prev_vx = prev_vy = 0.0
            prev_ro = prev_pi = prev_ya = 0.0
            TARGET_VEL[:] = 0.0
            sim_time = 0.0
            print(">>>>>>>> Reset!")
        _prev_reset = do_reset

        # ── Đọc trạng thái ─────────────────────────────────────────────────
        pos  = robot.data.root_pos_w[0]
        vel  = robot.data.root_lin_vel_w[0]
        quat = robot.data.root_quat_w[0]
        roll, pitch, yaw = [x[0].item() for x in euler_xyz_from_quat(quat.unsqueeze(0))]
        vx, vy, vz = vel[0].item(), vel[1].item(), vel[2].item()

        # ── Vòng vận tốc → thrust + desired attitude ───────────────────────
        err_vz = TARGET_VEL[2].item() - vz
        int_vz = max(-INTEG_LIM, min(INTEG_LIM, int_vz + err_vz * control_dt))
        dz     = (err_vz - prev_vz) / control_dt;  prev_vz = err_vz
        thrust = hover_thrust + kp_vz * err_vz + ki_vz * int_vz + kd_vz * dz
        thrust = max(0.0, min(2.0 * hover_thrust, thrust))

        err_vx    = TARGET_VEL[0].item() - vx
        int_vx    = max(-INTEG_LIM, min(INTEG_LIM, int_vx + err_vx * control_dt))
        dx        = (err_vx - prev_vx) / control_dt;  prev_vx = err_vx
        des_pitch = max(-MAX_TILT, min(MAX_TILT, -(kp_vh * err_vx + ki_vh * int_vx + kd_vh * dx)))

        err_vy   = TARGET_VEL[1].item() - vy
        int_vy   = max(-INTEG_LIM, min(INTEG_LIM, int_vy + err_vy * control_dt))
        dy       = (err_vy - prev_vy) / control_dt;  prev_vy = err_vy
        des_roll = max(-MAX_TILT, min(MAX_TILT,  (kp_vh * err_vy + ki_vh * int_vy + kd_vh * dy)))

        # ── Vòng attitude → moments ────────────────────────────────────────
        err_ro = des_roll  - roll
        err_pi = des_pitch - pitch
        err_ya = (0.0 - yaw + math.pi) % (2 * math.pi) - math.pi

        int_ro = max(-0.01, min(0.01, int_ro + err_ro * control_dt))
        int_pi = max(-0.01, min(0.01, int_pi + err_pi * control_dt))
        int_ya = max(-0.01, min(0.01, int_ya + err_ya * control_dt))

        dro = (err_ro - prev_ro) / control_dt;  prev_ro = err_ro
        dpi = (err_pi - prev_pi) / control_dt;  prev_pi = err_pi
        dya = (err_ya - prev_ya) / control_dt;  prev_ya = err_ya

        m_roll  = kp_att * err_ro + ki_att * int_ro + kd_att * dro
        m_pitch = kp_att * err_pi + ki_att * int_pi + kd_att * dpi
        m_yaw   = kp_att * err_ya + ki_att * int_ya + kd_att * dya

        # ── PID → action → env.step ────────────────────────────────────────
        action = pid_to_action(thrust, m_roll, m_pitch, m_yaw, robot_weight)
        action = action.to(env.device).expand(env.num_envs, -1)

        obs, reward, terminated, truncated, _ = env.step(action)
        sim_time += control_dt
        step     += 1

        if (terminated | truncated).any():
            obs, _ = env.reset()
            int_vz = int_vx = int_vy = 0.0
            int_ro = int_pi = int_ya = 0.0
            prev_vz = prev_vx = prev_vy = 0.0
            prev_ro = prev_pi = prev_ya = 0.0
            print("[AUTO RESET] episode ended")

        # ── Log ───────────────────────────────────────────────────────────
        if step % log_every == 0:
            print(
                f"t={sim_time:6.1f}s | "
                f"cmd=({TARGET_VEL[0].item():+.1f},{TARGET_VEL[1].item():+.1f},{TARGET_VEL[2].item():+.1f})m/s | "
                f"vel=({vx:+.2f},{vy:+.2f},{vz:+.2f})m/s | "
                f"pos=({pos[0].item():+.2f},{pos[1].item():+.2f},{pos[2].item():+.2f})m | "
                f"reward={reward[0].item():+.3f}"
            )

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
