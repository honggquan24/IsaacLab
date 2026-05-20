"""Test Legged V3 environment — spawn robot and verify termination conditions.

Run with:
    ./isaaclab.sh -p source/isaaclab_assets/isaaclab_assets/legged_v3/spawn_robot.py
    ./isaaclab.sh -p source/isaaclab_assets/isaaclab_assets/legged_v3/spawn_robot.py --headless
"""

from isaaclab.app import AppLauncher

import argparse
parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args, _ = parser.parse_known_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import torch
from isaaclab_assets.legged_v3.locomotion.legged_v3_wheel_env_cfg import LeggedV3WheelEnvCfg
from isaaclab.envs import ManagerBasedRLEnv

# ── Build env directly (no gym registry needed) ───────────────────────────────
env_cfg = LeggedV3WheelEnvCfg()
env_cfg.scene.num_envs = 1
env_cfg.sim.device = "cuda:0"

env = ManagerBasedRLEnv(cfg=env_cfg)
obs, _ = env.reset()

print("\n" + "=" * 60)
print(f"Joints  ({env.scene['robot'].num_joints}): {env.scene['robot'].data.joint_names}")
print(f"Bodies  ({env.scene['robot'].num_bodies}): {env.scene['robot'].data.body_names}")
print("=" * 60 + "\n")

# ── Zero-action loop ──────────────────────────────────────────────────────────
zero_action = torch.zeros(1, env.action_manager.total_action_dim, device="cuda:0")

episode      = 0
step_in_ep   = 0
total_steps  = 0

while simulation_app.is_running():
    obs, reward, terminated, truncated, info = env.step(zero_action)
    step_in_ep  += 1
    total_steps += 1

    # Camera follows robot
    robot = env.scene["robot"]
    pos   = robot.data.root_pos_w[0].cpu()           # (3,)
    eye   = (pos[0].item() + 2.0, pos[1].item() + 2.0, pos[2].item() + 1.5)
    target = (pos[0].item(), pos[1].item(), pos[2].item())
    env.sim.set_camera_view(eye=eye, target=target)

    # Print state every 30 env steps
    if step_in_ep % 30 == 0:
        height   = robot.data.root_pos_w[0, 2].item()
        w, x, y, z = robot.data.root_quat_w[0]
        up_z     = (1.0 - 2.0 * (x*x + y*y)).item()
        tilt_deg = torch.acos(torch.tensor(up_z).clamp(-1, 1)).item() * 57.296
        max_jvel = robot.data.joint_vel[0].abs().max().item()
        print(f"[ep {episode:3d} step {step_in_ep:4d}] "
              f"height={height:.3f}m  tilt={tilt_deg:.1f}°  "
              f"jvel_max={max_jvel:.1f}  reward={reward[0]:.3f}")

    # Termination
    done = terminated[0] or truncated[0]
    if done:
        # Print which condition fired
        term_mgr = env.termination_manager
        print(f"\n[EPISODE {episode} END at step {step_in_ep}]")
        for name in term_mgr.active_terms:
            if term_mgr.get_term(name)[0]:
                print(f"  terminated: {name}")

        obs, _ = env.reset()
        episode    += 1
        step_in_ep  = 0

env.close()
simulation_app.close()
