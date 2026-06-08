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

robot = env.scene["robot"]
joint_names = robot.data.joint_names
print("\n" + "=" * 60)
print(f"Joints  ({robot.num_joints}): {joint_names}")
print(f"Bodies  ({robot.num_bodies}): {robot.data.body_names}")
print(f"Action  dim = {env.action_manager.total_action_dim}")
print("=" * 60)

# Find wheel joint indices for targeted monitoring
wheel_ids = [joint_names.index(n) for n in ["left_wheel_joint", "right_wheel_joint"] if n in joint_names]
hip_ids   = [joint_names.index(n) for n in ["left_hip_joint_A1", "right_hip_joint_A1"] if n in joint_names]
print(f"wheel_ids={wheel_ids}  hip_ids={hip_ids}\n")

episode    = 0
step_in_ep = 0

while simulation_app.is_running():
    test_action = torch.randn(1, env.action_manager.total_action_dim, device="cuda:0")

    obs, reward, terminated, truncated, info = env.step(test_action)
    step_in_ep += 1

    # Print state every 10 env steps
    if step_in_ep % 10 == 0:
        height   = robot.data.root_pos_w[0, 2].item()
        w, x, y, z = robot.data.root_quat_w[0]
        up_z     = (1.0 - 2.0 * (x*x + y*y)).item()
        tilt_deg = torch.acos(torch.tensor(up_z).clamp(-1, 1)).item() * 57.296

        wvel = robot.data.joint_vel[0, wheel_ids].tolist() if wheel_ids else []
        hvel = robot.data.joint_vel[0, hip_ids].tolist()   if hip_ids   else []
        hpos = robot.data.joint_pos[0, hip_ids].tolist()   if hip_ids   else []
        weff = robot.data.applied_torque[0, wheel_ids].tolist() if wheel_ids else []
        heff = robot.data.applied_torque[0, hip_ids].tolist()   if hip_ids   else []

        print(f"[ep {episode:3d} step {step_in_ep:4d}] "
              f"h={height:.3f}m tilt={tilt_deg:.1f}°  "
              f"wheel_vel={[f'{v:.2f}' for v in wvel]}  "
              f"hip_pos={[f'{p:.3f}' for p in hpos]}  "
              f"hip_vel={[f'{v:.3f}' for v in hvel]}  "
              f"hip_torque={[f'{t:.2f}' for t in heff]}  "
              f"reward={reward[0]:.3f}")

    done = terminated[0] or truncated[0]
    if done:
        term_mgr = env.termination_manager
        print(f"\n[EPISODE {episode} END at step {step_in_ep}]")
        for name in term_mgr.active_terms:
            if term_mgr.get_term(name)[0]:
                print(f"  terminated: {name}")
        obs, _ = env.reset()
        episode   += 1
        step_in_ep = 0

env.close()
simulation_app.close()
