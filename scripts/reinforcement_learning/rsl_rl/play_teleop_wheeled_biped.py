# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Lái robot bipedal wheel wheel bằng bàn phím (pygame GUI) — ghi đè velocity_command lên policy đã train.

Điều khiển (focus vào cửa sổ pygame "Wheeled Biped Teleop"):
    W / S : tiến / lùi   (lin_vel_y, forward = body +Y)
    A / D : xoay trái / phải (ang_vel_z)
    SPACE : dừng (zero lệnh)
    R     : reset môi trường
    ESC   : thoát

Ví dụ:
    python scripts/reinforcement_learning/rsl_rl/play_teleop_v5.py \
        --task Isaac-Wheeled-Biped-Wheel --num_envs 1 \
        --checkpoint logs/rsl_rl/<exp>/<run>/model_xxx.pt
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Teleop a trained V5 wheel RSL-RL agent with a pygame GUI.")
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="Isaac-Wheeled-Biped-Wheel", help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--vy_max", type=float, default=0.5, help="Max forward/back command (m/s).")
parser.add_argument("--wz_max", type=float, default=0.3, help="Max yaw-rate command (rad/s).")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import os
import time

import gymnasium as gym
import pygame
import torch
from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.math import euler_xyz_from_quat, quat_conjugate, quat_mul, wrap_to_pi

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import isaaclab_assets  # noqa: F401  ← register custom tasks (wheeled_biped, ...)


# ---------------------------------------------------------------------------- #
# pygame teleop panel
# ---------------------------------------------------------------------------- #
class TeleopPanel:
    """Cửa sổ pygame nhỏ: đọc phím WASD và hiển thị lệnh hiện tại."""

    def __init__(self, vy_max: float, wz_max: float):
        pygame.init()
        self.vy_max = vy_max
        self.wz_max = wz_max
        self.screen = pygame.display.set_mode((460, 430))
        pygame.display.set_caption("Wheeled Biped Teleop + Debug")
        self.font = pygame.font.SysFont("monospace", 17)
        self.big = pygame.font.SysFont("monospace", 21, bold=True)

    def poll(self) -> tuple[float, float, bool, bool]:
        """Trả về (vy, wz, reset, quit)."""
        reset = quit_flag = False
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                quit_flag = True
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    quit_flag = True
                elif event.key == pygame.K_r:
                    reset = True

        keys = pygame.key.get_pressed()
        vy = (self.vy_max if keys[pygame.K_w] else 0.0) - (self.vy_max if keys[pygame.K_s] else 0.0)
        wz = (self.wz_max if keys[pygame.K_a] else 0.0) - (self.wz_max if keys[pygame.K_d] else 0.0)
        if keys[pygame.K_SPACE]:
            vy = wz = 0.0
        return vy, wz, reset, quit_flag

    def draw(self, vy: float, wz: float, dbg: dict | None = None):
        self.screen.fill((20, 20, 28))
        WHITE, GREY, GREEN, BLUE = (220, 220, 220), (150, 150, 150), (120, 220, 120), (120, 180, 255)

        def tilt_color(deg, warn, bad):
            a = abs(deg)
            return (235, 80, 80) if a > bad else (235, 200, 80) if a > warn else WHITE

        lines = [
            (self.big, f"CMD vy: {vy:+.2f}  wz: {wz:+.2f}", BLUE),
        ]
        if dbg is not None:
            lines += [
                (self.font, "--- THUC TE (lech so voi spawn) ---", GREY),
                # roll = lat nghieng NGANG -> thu pham lat khi xoay
                (self.font, f"roll  (ngang): {dbg['roll']:+6.1f} deg", tilt_color(dbg["roll"], 25, 50)),
                (self.font, f"pitch (doc) : {dbg['pitch']:+6.1f} deg", tilt_color(dbg["pitch"], 25, 50)),
                (self.font, f"yaw   (huong): {dbg['yaw']:+6.1f} deg", WHITE),
                (self.font, "--- van toc THUC vs LENH ---", GREY),
                (self.font, f"vy thuc: {dbg['vy']:+.2f}  (lenh {vy:+.2f})", GREEN),
                (self.font, f"wz thuc: {dbg['wz']:+.2f}  (lenh {wz:+.2f})", GREEN),
                (self.font, "--- banh xe (rad/s) ---", GREY),
                (self.font, f"L: {dbg['wl']:+6.1f}   R: {dbg['wr']:+6.1f}", WHITE),
                (self.font, f"|L|-|R| dif: {abs(dbg['wl']) - abs(dbg['wr']):+6.1f}", WHITE),
                (self.font, f"height: {dbg['h']:.3f} m", WHITE),
            ]
        lines += [
            (self.font, "W/S=tien/lui  A/D=xoay  SPACE=dung", GREY),
            (self.font, "R=reset  ESC=thoat (focus o day)", GREY),
        ]
        y = 16
        for font, text, color in lines:
            self.screen.blit(font.render(text, True, color), (16, y))
            y += 30 if font is self.big else 26
        pygame.display.flip()

    def close(self):
        pygame.quit()


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Teleop a trained RSL-RL agent."""
    # override configurations with non-hydra CLI arguments
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # tat resample ngau nhien + standing env => giu nguyen lenh tay lai cho den khi nguoi dung doi
    env_cfg.commands.velocity_command.resampling_time_range = (1.0e9, 1.0e9)
    env_cfg.commands.velocity_command.rel_standing_envs = 0.0
    env_cfg.commands.velocity_command.heading_command = False

    # resolve checkpoint
    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    log_dir = os.path.dirname(resume_path)
    env_cfg.log_dir = log_dir

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    dt = env.unwrapped.step_dt

    # grab the velocity command term so we can override it every step
    cmd_term = env.unwrapped.command_manager.get_term("velocity_command")
    device = env.unwrapped.device

    # robot handle + wheel joint ids cho debug (env 0)
    robot = env.unwrapped.scene["robot"]
    wheel_ids = robot.find_joints(["left_wheel_joint", "right_wheel_joint"])[0]

    def gather_debug() -> dict:
        """Lệch orientation so với tư thế spawn + vận tốc thực + tốc 2 bánh (env 0)."""
        q_cur = robot.data.root_quat_w[0:1]  # (1,4)
        q_ref = robot.data.default_root_state[0:1, 3:7]  # (1,4) quat spawn
        q_rel = quat_mul(quat_conjugate(q_ref), q_cur)  # lệch so với spawn
        r, p, yw = euler_xyz_from_quat(q_rel)
        rad2deg = 180.0 / 3.14159265
        wl, wr = robot.data.joint_vel[0, wheel_ids[0]], robot.data.joint_vel[0, wheel_ids[1]]
        return {
            "roll": wrap_to_pi(r)[0].item() * rad2deg,  # nghiêng NGANG (lateral) — thủ phạm lật khi xoay
            "pitch": wrap_to_pi(p)[0].item() * rad2deg,  # nghiêng DỌC (fore-aft)
            "yaw": wrap_to_pi(yw)[0].item() * rad2deg,  # đổi hướng
            "vy": robot.data.root_lin_vel_b[0, 1].item(),  # tiến thực (body +Y)
            "wz": robot.data.root_ang_vel_b[0, 2].item(),  # yaw rate thực
            "wl": wl.item(),
            "wr": wr.item(),
            "h": robot.data.root_pos_w[0, 2].item(),
        }

    # pygame control panel
    panel = TeleopPanel(args_cli.vy_max, args_cli.wz_max)

    obs = env.get_observations()
    print_every = max(1, int(0.25 / dt))  # in console ~4 Hz
    step_i = 0
    try:
        while simulation_app.is_running():
            start_time = time.time()

            vy, wz, do_reset, do_quit = panel.poll()
            if do_quit:
                break
            if do_reset:
                obs = env.get_observations()
                env.unwrapped.reset()
                obs = env.get_observations()

            with torch.inference_mode():
                actions = policy(obs)
                obs, _, _, _ = env.step(actions)
                # ghi đè lệnh sau khi step (manager đã resample) → giữ giá trị tay lái
                cmd_term.vel_command_b[:, 0] = 0.0  # lin_vel_x: khoa (mimic khong di ngang)
                cmd_term.vel_command_b[:, 1] = torch.tensor(vy, device=device)  # tien/lui
                cmd_term.vel_command_b[:, 2] = torch.tensor(wz, device=device)  # xoay

            dbg = gather_debug()
            panel.draw(vy, wz, dbg)

            step_i += 1
            if step_i % print_every == 0:
                flag = "  <<< LAT NGANG" if abs(dbg["roll"]) > 35 else ""
                print(
                    f"[DBG] cmd(vy={vy:+.2f} wz={wz:+.2f}) | "
                    f"thuc(vy={dbg['vy']:+.2f} wz={dbg['wz']:+.2f}) | "
                    f"roll={dbg['roll']:+6.1f} pitch={dbg['pitch']:+6.1f} yaw={dbg['yaw']:+6.1f} | "
                    f"wheelL={dbg['wl']:+6.1f} wheelR={dbg['wr']:+6.1f} | h={dbg['h']:.3f}{flag}"
                )

            sleep_time = dt - (time.time() - start_time)
            if sleep_time > 0:
                time.sleep(sleep_time)
    finally:
        panel.close()
        env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
