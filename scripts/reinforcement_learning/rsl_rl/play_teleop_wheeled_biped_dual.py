# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Teleop V5 wheel — 2 MODEL (fwd + turn) + camera gắn robot + HUD.

Camera được inject vào scene config lúc runtime (không sửa task file):
    env_cfg.scene.follow_cam = CameraCfg(prim_path="{ENV_REGEX_NS}/Robot/base/follow_cam", ...)
Frame lấy qua: env.unwrapped.scene["follow_cam"].data.output["rgb"][0]

Layout pygame:
    [ CAMERA  cam_width × cam_height ] [ HUD 560px ]

Điều khiển (focus cửa sổ pygame):
    W/S  tiến/lùi    A/D  xoay    SPACE  dừng    R  reset    ESC  thoát

Chạy:
    python scripts/reinforcement_learning/rsl_rl/play_teleop_v5_dual.py \
        --task Isaac-Wheeled-Biped-Wheel --num_envs 1 \
        [--base_link base]   # tên link gắn camera, default = "base"
"""

# ── Launch app trước mọi thứ ────────────────────────────────────────────────
import argparse
import sys

from isaaclab.app import AppLauncher

import cli_args  # isort: skip

DEFAULT_FWD = "logs/rsl_rl/wheeled_biped_wheel_mimic/2026-06-18_02-19-07/model_3500.pt"
DEFAULT_TURN = "logs/rsl_rl/wheeled_biped_wheel_mimic/2026-06-18_00-59-33/model_600.pt"

parser = argparse.ArgumentParser(description="Dual-policy teleop V5 wheel — camera + HUD.")
parser.add_argument("--disable_fabric", action="store_true", default=False)
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--task", type=str, default="Isaac-Wheeled-Biped-Wheel")
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point")
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--checkpoint_fwd", type=str, default=DEFAULT_FWD)
parser.add_argument("--checkpoint_turn", type=str, default=DEFAULT_TURN)
parser.add_argument("--vy_max", type=float, default=0.5)
parser.add_argument("--wz_max", type=float, default=0.3)
parser.add_argument("--switch_thr", type=float, default=0.02, help="|wz| > thr → dùng TURN policy.")
parser.add_argument("--cam_width", type=int, default=640)
parser.add_argument("--cam_height", type=int, default=480)
parser.add_argument("--base_link", type=str, default="base", help="Tên link gắn camera (xem robot.data.body_names).")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ── Imports sau khi app đã launch ───────────────────────────────────────────
import math
import time

import gymnasium as gym
import numpy as np
import pygame
import torch
from rsl_rl.runners import OnPolicyRunner

import isaaclab.sim as sim_utils
from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.sensors import CameraCfg
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.math import euler_xyz_from_quat, quat_conjugate, quat_mul, wrap_to_pi

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config

import isaaclab_assets  # noqa: F401

# ─────────────────────────────────────────────────────────────────────────────
# Màu sắc
# ─────────────────────────────────────────────────────────────────────────────
C_BG = (10, 13, 20)
C_BORDER = (35, 45, 70)
C_TEXT = (200, 210, 225)
C_DIM = (90, 100, 120)
C_FWD = (70, 160, 255)  # xanh lam — FWD policy
C_TURN = (255, 200, 60)  # vàng     — TURN policy
C_GREEN = (80, 220, 130)
C_RED = (235, 75, 75)
C_ORANGE = (235, 145, 50)
C_WHITE = (230, 235, 245)
C_BARBG = (30, 36, 54)


# ─────────────────────────────────────────────────────────────────────────────
# Draw helpers
# ─────────────────────────────────────────────────────────────────────────────
def rfill(surf, color, rect, r=4):
    pygame.draw.rect(surf, color, rect, border_radius=r)


def rborder(surf, color, rect, r=4, w=1):
    pygame.draw.rect(surf, color, rect, w, border_radius=r)


def lerp_c(a, b, t):
    t = max(0.0, min(1.0, t))
    return tuple(int(a[i] + (b[i] - a[i]) * t) for i in range(3))


def tilt_color(deg):
    a = abs(deg)
    return C_RED if a > 50 else C_ORANGE if a > 25 else C_TEXT


def speed_color(val, mx):
    t = abs(val) / max(mx, 1e-6)
    return lerp_c(C_GREEN, C_ORANGE, t / 0.6) if t < 0.6 else lerp_c(C_ORANGE, C_RED, (t - 0.6) / 0.4)


# ─────────────────────────────────────────────────────────────────────────────
# HUD
# ─────────────────────────────────────────────────────────────────────────────
class HUDPanel:
    HUD_W = 560

    def __init__(self, cam_w, cam_h, vy_max, wz_max):
        pygame.init()
        self.CW, self.CH = cam_w, cam_h
        self.WH = max(cam_h, 600)
        self.WW = cam_w + self.HUD_W
        self.vy_max, self.wz_max = vy_max, wz_max

        self.screen = pygame.display.set_mode((self.WW, self.WH))
        pygame.display.set_caption("Wheeled Biped Teleop — Dual Policy")

        self.fT = pygame.font.SysFont("monospace", 13)  # tiny
        self.fB = pygame.font.SysFont("monospace", 15)  # body
        self.fL = pygame.font.SysFont("monospace", 15, bold=True)  # label
        self.fV = pygame.font.SysFont("monospace", 22, bold=True)  # value
        self.fH = pygame.font.SysFont("monospace", 19, bold=True)  # header

        self._no_cam = self._make_nocam()

    def _make_nocam(self):
        s = pygame.Surface((self.CW, self.CH))
        s.fill((8, 10, 18))
        t = self.fB.render("[ NO CAMERA ]", True, C_DIM)
        s.blit(t, (self.CW // 2 - t.get_width() // 2, self.CH // 2 - t.get_height() // 2))
        for y in range(0, self.CH, 4):
            pygame.draw.line(s, (0, 0, 0), (0, y), (self.CW, y), 1)
        return s

    # ── input ────────────────────────────────────────────────────────────────
    def poll(self):
        reset = quit_f = False
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                quit_f = True
            elif ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_ESCAPE:
                    quit_f = True
                if ev.key == pygame.K_r:
                    reset = True
        k = pygame.key.get_pressed()
        vy = (self.vy_max if k[pygame.K_w] else 0.0) - (self.vy_max if k[pygame.K_s] else 0.0)
        wz = (self.wz_max if k[pygame.K_a] else 0.0) - (self.wz_max if k[pygame.K_d] else 0.0)
        if k[pygame.K_SPACE]:
            vy = wz = 0.0
        return vy, wz, reset, quit_f

    # ── sub-widgets ──────────────────────────────────────────────────────────
    def _gauge(self, x, y, w, h, val, mx, label, accent):
        rfill(self.screen, C_BARBG, (x, y, w, h), r=3)
        cx = x + w // 2
        bw = int(abs(val / mx) * w / 2)
        bx = cx - bw if val < 0 else cx
        if bw > 0:
            rfill(self.screen, speed_color(val, mx), (bx, y + 1, bw, h - 2), r=2)
        pygame.draw.line(self.screen, C_BORDER, (cx, y), (cx, y + h), 1)
        rborder(self.screen, C_BORDER, (x, y, w, h), r=3)
        self.screen.blit(self.fT.render(label, True, C_DIM), (x, y - 16))
        vs = self.fB.render(f"{val:+.3f}", True, accent)
        self.screen.blit(vs, (x + w - vs.get_width(), y - 16))

    def _row(self, x, y, label, val_str, color=None, unit=""):
        color = color or C_TEXT
        self.screen.blit(self.fB.render(label, True, C_DIM), (x, y))
        vs = self.fV.render(val_str, True, color)
        self.screen.blit(vs, (x + 170, y - 3))
        self.screen.blit(self.fT.render(unit, True, C_DIM), (x + 170 + vs.get_width() + 4, y + 4))
        return y + 32

    def _badge(self, x, y, w, name, is_turn):
        acc = C_TURN if is_turn else C_FWD
        gl = pygame.Surface((w, 44), pygame.SRCALPHA)
        gl.fill((*acc, 22))
        self.screen.blit(gl, (x, y))
        rborder(self.screen, acc, (x, y, w, 44), r=6, w=2)
        icon = "↻ TURN" if is_turn else "→ FWD"
        self.screen.blit(self.fH.render(f"POLICY  {icon}", True, acc), (x + 12, y + 6))
        self.screen.blit(self.fT.render(name, True, C_DIM), (x + 12, y + 27))

    def _joystick(self, cx, cy, r, vy, wz):
        pygame.draw.circle(self.screen, C_BARBG, (cx, cy), r)
        pygame.draw.circle(self.screen, C_BORDER, (cx, cy), r, 1)
        pygame.draw.line(self.screen, C_BORDER, (cx - r, cy), (cx + r, cy), 1)
        pygame.draw.line(self.screen, C_BORDER, (cx, cy - r), (cx, cy + r), 1)
        dx = int(wz / self.wz_max * r * 0.88)
        dy = int(-vy / self.vy_max * r * 0.88)
        dot = C_TURN if abs(wz) > 0.01 else C_FWD
        pygame.draw.circle(self.screen, dot, (cx + dx, cy + dy), 8)
        pygame.draw.circle(self.screen, C_WHITE, (cx + dx, cy + dy), 8, 2)

    # ── main draw ────────────────────────────────────────────────────────────
    def draw(self, frame, vy, wz, name, is_turn, dbg, fps):
        self.screen.fill(C_BG)

        # camera
        if frame is not None:
            surf = pygame.surfarray.make_surface(np.transpose(frame, (1, 0, 2)))
            surf = pygame.transform.scale(surf, (self.CW, self.CH))
        else:
            surf = self._no_cam
        cy0 = (self.WH - self.CH) // 2
        self.screen.blit(surf, (0, cy0))
        self.screen.blit(self.fT.render(f"{fps:.0f} fps", True, C_DIM), (8, cy0 + self.CH - 20))
        rborder(self.screen, C_BORDER, (0, cy0, self.CW, self.CH), r=0)

        # HUD origin
        hx, hw = self.CW + 14, self.HUD_W - 20
        y = 0

        # title
        self.screen.blit(self.fH.render("V5  WHEEL  TELEOP", True, C_TEXT), (hx, y + 12))
        pygame.draw.line(self.screen, C_BORDER, (hx, y + 36), (hx + hw, y + 36), 1)
        y += 48

        # model badge
        self._badge(hx, y, hw, name, is_turn)
        y += 58

        # joystick + gauges
        jr = 52
        jcx, jcy = hx + jr + 6, y + jr + 10
        self._joystick(jcx, jcy, jr, vy, wz)
        gx = hx + jr * 2 + 24
        gw = hw - (jr * 2 + 30)
        self._gauge(gx, y + 20, gw, 14, vy, self.vy_max, "VY CMD", C_FWD)
        self._gauge(gx, y + 68, gw, 14, wz, self.wz_max, "WZ CMD", C_TURN)
        y = jcy + jr + 20

        pygame.draw.line(self.screen, C_BORDER, (hx, y), (hx + hw, y), 1)
        y += 12

        # telemetry
        if dbg:
            self.screen.blit(self.fL.render("TELEMETRY", True, C_DIM), (hx, y))
            y += 22
            y = self._row(hx, y, "ROLL  (ngang)", f"{dbg['roll']:+6.1f}", tilt_color(dbg["roll"]), "deg")
            y = self._row(hx, y, "PITCH (dọc)  ", f"{dbg['pitch']:+6.1f}", tilt_color(dbg["pitch"]), "deg")
            y = self._row(hx, y, "YAW          ", f"{dbg['yaw']:+6.1f}", C_TEXT, "deg")
            pygame.draw.line(self.screen, C_BORDER, (hx, y), (hx + hw, y), 1)
            y += 10
            y = self._row(hx, y, "WHEEL L      ", f"{dbg['wl']:+.1f}", C_TEXT, "rad/s")
            y = self._row(hx, y, "WHEEL R      ", f"{dbg['wr']:+.1f}", C_TEXT, "rad/s")
            y = self._row(hx, y, "HEIGHT       ", f"{dbg['h']:.4f}", C_GREEN, "m")
            if abs(dbg["roll"]) > 35:
                self.screen.blit(self.fL.render("⚠  NGUY CƠ LẬT NGANG", True, C_RED), (hx, y + 4))
        else:
            self.screen.blit(self.fB.render("waiting for sim...", True, C_DIM), (hx, y + 10))

        # legend
        pygame.draw.line(self.screen, C_BORDER, (hx, self.WH - 50), (hx + hw, self.WH - 50), 1)
        for i, ln in enumerate(
            [
                "W/S  tiến/lùi      A/D  xoay",
                "SPACE  dừng    R  reset    ESC  thoát",
            ]
        ):
            self.screen.blit(self.fT.render(ln, True, C_DIM), (hx, self.WH - 44 + i * 16))

        pygame.display.flip()

    def close(self):
        pygame.quit()


# ─────────────────────────────────────────────────────────────────────────────
# Helper: lấy frame từ CameraCfg sensor
# ─────────────────────────────────────────────────────────────────────────────
def get_cam_frame(cam_sensor) -> np.ndarray | None:
    """
    cam_sensor.data.output["rgb"] → Tensor (num_envs, H, W, 4) RGBA uint8.
    Trả về numpy (H, W, 3) RGB của env_0, hoặc None.
    """
    try:
        rgb = cam_sensor.data.output["rgb"]  # Tensor
        if rgb is None or rgb.shape[0] == 0:
            return None
        return rgb[0, :, :, :3].cpu().numpy().astype(np.uint8)
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # ── Đổi terrain sang Simple Warehouse ──────────────────────────────────
    # Isaac Lab built-in asset: isaaclab.sim.spawners.from_files
    # USD path chuẩn trong Isaac Sim: omniverse://localhost/NVIDIA/Assets/...
    # Dùng path nội bộ Isaac Lab luôn có sẵn, không cần OV Nucleus
    from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

    env_cfg.scene.terrain.terrain_type = "usd"
    env_cfg.scene.terrain.usd_path = f"{ISAAC_NUCLEUS_DIR}/Environments/Simple_Warehouse/warehouse.usd"
    env_cfg.scene.terrain.max_init_terrain_level = None

    # giữ nguyên lệnh tay lái
    env_cfg.commands.velocity_command.resampling_time_range = (1e9, 1e9)
    env_cfg.commands.velocity_command.rel_standing_envs = 0.0
    env_cfg.commands.velocity_command.heading_command = False

    # ── Inject CameraCfg vào scene runtime ──────────────────────────────────
    # Cấu trúc USD thực tế (xem log):
    #   /World/envs/env_0/Robot/Robot/Robot/base/base_B  ...
    # Vậy parent của "base" link là: {ENV_REGEX_NS}/Robot/Robot/Robot/<base_link>
    base_link = args_cli.base_link  # default "base"
    cam_prim_path = "{ENV_REGEX_NS}/Robot/Robot/Robot/" + base_link + "/follow_cam"

    env_cfg.scene.follow_cam = CameraCfg(
        prim_path=cam_prim_path,
        update_period=0.0,  # update mỗi sim step
        height=args_cli.cam_height,
        width=args_cli.cam_width,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=12.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.05, 200.0),
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(0.0, 0.2, 0.25),  # 0.35 m trước, 0.25 m trên link
            # Camera đang nhìn lên trần → cần pitch 180° để lật lại + pitch -20° nhìn xuống nhẹ
            # compose(pitch=180°, pitch=-20°) = (w=0.1736, x=0.9848, y=0, z=0)  [wxyz]
            rot=(0.1736, 0.9848, 0.0, 0.0),
            convention="ros",
        ),
    )

    # tạo env
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # load 2 policy
    def make_policy(ckpt_path):
        path = retrieve_file_path(ckpt_path)
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        runner.load(path)
        print(f"[INFO] loaded: {path}")
        return runner.get_inference_policy(device=env.unwrapped.device)

    policy_fwd = make_policy(args_cli.checkpoint_fwd)
    policy_turn = make_policy(args_cli.checkpoint_turn)

    dt = env.unwrapped.step_dt
    cmd_term = env.unwrapped.command_manager.get_term("velocity_command")
    device = env.unwrapped.device
    robot = env.unwrapped.scene["robot"]
    cam_sensor = env.unwrapped.scene["follow_cam"]
    wheel_ids = robot.find_joints(["left_wheel_joint", "right_wheel_joint"])[0]

    print(f"[INFO] Camera prim path: {cam_prim_path}")
    print(f"[INFO] Robot body names: {robot.data.body_names}")

    def gather_debug():
        q_cur = robot.data.root_quat_w[0:1]
        q_ref = robot.data.default_root_state[0:1, 3:7]
        q_rel = quat_mul(quat_conjugate(q_ref), q_cur)
        r, p, yw = euler_xyz_from_quat(q_rel)
        k = 180.0 / math.pi
        return {
            "roll": wrap_to_pi(r)[0].item() * k,
            "pitch": wrap_to_pi(p)[0].item() * k,
            "yaw": wrap_to_pi(yw)[0].item() * k,
            "vy": robot.data.root_lin_vel_b[0, 1].item(),
            "wz": robot.data.root_ang_vel_b[0, 2].item(),
            "wl": robot.data.joint_vel[0, wheel_ids[0]].item(),
            "wr": robot.data.joint_vel[0, wheel_ids[1]].item(),
            "h": robot.data.root_pos_w[0, 2].item(),
        }

    panel = HUDPanel(args_cli.cam_width, args_cli.cam_height, args_cli.vy_max, args_cli.wz_max)
    obs = env.get_observations()
    print_every = max(1, int(0.25 / dt))
    step_i = fps = fps_count = 0
    t_fps = time.time()

    try:
        while simulation_app.is_running():
            t0 = time.time()

            vy, wz, do_reset, do_quit = panel.poll()
            if do_quit:
                break
            if do_reset:
                env.unwrapped.reset()
                obs = env.get_observations()

            is_turn = abs(wz) > args_cli.switch_thr
            policy, active_name = (policy_turn, "TURN  model_600") if is_turn else (policy_fwd, "FWD   model_3500")

            with torch.inference_mode():
                actions = policy(obs)
                obs, _, _, _ = env.step(actions)
                cmd_term.vel_command_b[:, 0] = 0.0
                cmd_term.vel_command_b[:, 1] = torch.tensor(vy, device=device)
                cmd_term.vel_command_b[:, 2] = torch.tensor(wz, device=device)

            frame = get_cam_frame(cam_sensor)
            dbg = gather_debug()
            panel.draw(frame, vy, wz, active_name, is_turn, dbg, fps)

            # fps
            fps_count += 1
            if time.time() - t_fps >= 0.5:
                fps = fps_count / (time.time() - t_fps)
                fps_count = 0
                t_fps = time.time()

            step_i += 1
            if step_i % print_every == 0:
                flag = "  <<< LAT NGANG" if abs(dbg["roll"]) > 35 else ""
                print(
                    f"[{active_name:18s}] "
                    f"cmd(vy={vy:+.2f} wz={wz:+.2f}) | "
                    f"thực(vy={dbg['vy']:+.2f} wz={dbg['wz']:+.2f}) | "
                    f"roll={dbg['roll']:+6.1f} pitch={dbg['pitch']:+6.1f} | "
                    f"L={dbg['wl']:+6.1f} R={dbg['wr']:+6.1f} | "
                    f"h={dbg['h']:.3f}{flag}"
                )

            sleep_time = dt - (time.time() - t0)
            if sleep_time > 0:
                time.sleep(sleep_time)

    finally:
        panel.close()
        env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
