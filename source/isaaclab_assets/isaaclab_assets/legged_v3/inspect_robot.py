"""Utility script to inspect Legged V3 robot joint/body names at runtime.

Run ONCE after exporting the USD from Onshape to verify that the joint names
in legged_v3_cfg.py match the actual USD prim names:

    ./isaaclab.sh -p source/isaaclab_assets/isaaclab_assets/legged_v3/inspect_robot.py

Then update legged_v3_cfg.py and legged_v3_velocity_env_cfg.py accordingly.
"""

from isaaclab.app import AppLauncher

app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

# Isaac Sim đã boot — giờ mới import các module khác
import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab_assets.legged_v3.legged_v3_cfg import LEGGED_ROBOT_V3_CFG

sim_cfg = sim_utils.SimulationCfg(dt=1 / 60.0, device="cpu")
sim = sim_utils.SimulationContext(sim_cfg)

# Spawn ground
sim_utils.GroundPlaneCfg().func("/World/ground", sim_utils.GroundPlaneCfg())

# Spawn robot
cfg = LEGGED_ROBOT_V3_CFG.replace(prim_path="/World/Robot")
robot = Articulation(cfg)
sim.reset()

print("\n" + "=" * 60)
print("LEGGED V3 — ROBOT INSPECTION")
print("=" * 60)
print(f"\nJoint names ({robot.num_joints}):")
for i, name in enumerate(robot.data.joint_names):
    print(f"  [{i:2d}] {name}")

print(f"\nBody names ({robot.num_bodies}):")
for i, name in enumerate(robot.data.body_names):
    print(f"  [{i:2d}] {name}")

# ── Joint positions sau khi reset ────────────────────────────────────────────
print("\n" + "─" * 60)
print("JOINT POSITIONS after sim.reset():")
print(f"  {'Joint':<25} {'default (cfg)':>14}  {'actual (sim)':>12}")
print("  " + "-" * 55)
default_pos = robot.data.default_joint_pos[0]   # shape: (num_joints,)
actual_pos  = robot.data.joint_pos[0]            # shape: (num_joints,)
for i, name in enumerate(robot.data.joint_names):
    print(f"  {name:<25} {default_pos[i].item():>14.4f}  {actual_pos[i].item():>12.4f}")

# ── Body positions (world frame) ─────────────────────────────────────────────
print("\n" + "─" * 60)
print("BODY POSITIONS in world frame after reset:")
print(f"  {'Body':<30} {'x':>8}  {'y':>8}  {'z':>8}")
print("  " + "-" * 58)
body_pos = robot.data.body_pos_w[0]              # shape: (num_bodies, 3)
for i, name in enumerate(robot.data.body_names):
    x, y, z = body_pos[i].tolist()
    print(f"  {name:<30} {x:>8.4f}  {y:>8.4f}  {z:>8.4f}")

print("\n" + "=" * 60)
print("IMU should be placed on the BASE body above.")
print("Use the body name to set imu.prim_path in the env cfg.")
print("Pattern: {ENV_REGEX_NS}/Robot/<body_name>")
print("=" * 60 + "\n")

simulation_app.close()
