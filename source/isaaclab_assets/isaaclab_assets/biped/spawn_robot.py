"""Spawn robot với zero action — quan sát robot rớt theo trục nào.

Không apply lực, chỉ ghi roll/pitch/yaw và vẽ đồ thị.

Run:
    ./isaaclab.sh -p source/isaaclab_assets/isaaclab_assets/biped/spawn_robot.py --headless
    ./isaaclab.sh -p source/isaaclab_assets/isaaclab_assets/biped/spawn_robot.py
"""

from isaaclab.app import AppLauncher

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--max_steps", type=int, default=400,
                    help="Số bước tối đa mỗi episode")
parser.add_argument("--num_episodes", type=int, default=3,
                    help="Số episode để quan sát")
AppLauncher.add_app_launcher_args(parser)
args, _ = parser.parse_known_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import torch
import numpy as np
from isaaclab.utils.math import euler_xyz_from_quat

from isaaclab_assets.biped.rl_control.inner_tilt_env_cfg import BipedInnerTiltEnvCfg
from isaaclab.envs import ManagerBasedRLEnv

# ── Build env ─────────────────────────────────────────────────────────────────
env_cfg = BipedInnerTiltEnvCfg()
env_cfg.scene.num_envs = 1
env_cfg.sim.device = "cuda:0"

env = ManagerBasedRLEnv(cfg=env_cfg)
obs, _ = env.reset()

robot = env.scene["robot"]
action_dim = env.action_manager.total_action_dim
zero_action = torch.zeros(1, action_dim, device="cuda:0")

print(f"\nAction dim: {action_dim}")
print(f"Joints: {robot.data.joint_names}\n")

# ── Collect data ───────────────────────────────────────────────────────────────
all_episodes = []

for ep in range(args.num_episodes):
    rolls, pitches, yaws, heights, times = [], [], [], [], []
    step = 0

    obs, _ = env.reset()

    while step < args.max_steps:
        obs, reward, terminated, truncated, info = env.step(zero_action)
        step += 1

        quat  = robot.data.root_quat_w[0:1]           # (1, 4)
        roll, pitch, yaw = euler_xyz_from_quat(quat)   # each (1,)
        height = robot.data.root_pos_w[0, 2].item()

        rolls.append(roll[0].item() * 57.296)          # → degrees
        pitches.append(pitch[0].item() * 57.296)
        yaws.append(yaw[0].item() * 57.296)
        heights.append(height)
        times.append(step / 200.0)                     # 200 Hz → seconds

        done = terminated[0] or truncated[0]
        if done:
            term_info = ""
            for name in env.termination_manager.active_terms:
                if env.termination_manager.get_term(name)[0]:
                    term_info += name + " "
            print(f"[ep {ep}] terminated at step {step} ({step/200:.3f}s): {term_info}")
            print(f"  roll={rolls[-1]:.1f}°  pitch={pitches[-1]:.1f}°  yaw={yaws[-1]:.1f}°  h={height:.3f}m")
            break

    all_episodes.append({
        "t": times, "roll": rolls, "pitch": pitches, "yaw": yaws, "h": heights,
        "end_step": step,
    })
    print(f"[ep {ep}] {step} steps, "
          f"max |roll|={max(abs(r) for r in rolls):.1f}°  "
          f"max |pitch|={max(abs(p) for p in pitches):.1f}°")

env.close()

# ── Plot ───────────────────────────────────────────────────────────────────────
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=False)
colors = ["tab:blue", "tab:orange", "tab:green"]

for i, ep_data in enumerate(all_episodes):
    t = ep_data["t"]
    ax = axes[0]
    ax.plot(t, ep_data["roll"],  color=colors[i % 3], linestyle="-",  label=f"ep{i} roll")
    ax.plot(t, ep_data["pitch"], color=colors[i % 3], linestyle="--", label=f"ep{i} pitch")
    ax.plot(t, ep_data["yaw"],   color=colors[i % 3], linestyle=":",  label=f"ep{i} yaw")

ax = axes[0]
ax.axhline(y=40.0,  color="red",  linestyle="--", alpha=0.5, label="term +40°")
ax.axhline(y=-40.0, color="red",  linestyle="--", alpha=0.5)
ax.set_ylabel("Góc (độ)")
ax.set_title("Roll / Pitch / Yaw theo thời gian (zero action)")
ax.legend(fontsize=7, ncol=4)
ax.grid(True, alpha=0.3)

ax2 = axes[1]
for i, ep_data in enumerate(all_episodes):
    ax2.plot(ep_data["t"], ep_data["h"], color=colors[i % 3], label=f"ep{i} height")
ax2.axhline(y=0.10, color="red", linestyle="--", alpha=0.5, label="min height 0.10m")
ax2.set_ylabel("Height (m)")
ax2.set_xlabel("Thời gian (s)")
ax2.legend(fontsize=8)
ax2.grid(True, alpha=0.3)

out_path = "/tmp/robot_angles.png"
plt.tight_layout()
plt.savefig(out_path, dpi=120)
print(f"\nĐã lưu đồ thị: {out_path}")

simulation_app.close()
