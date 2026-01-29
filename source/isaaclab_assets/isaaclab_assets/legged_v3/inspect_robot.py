"""Utility script to inspect Legged V3 robot joint/body names at runtime.

Run ONCE after exporting the USD from Onshape to verify that the joint names
in legged_v3_cfg.py match the actual USD prim names:

    ./isaaclab.sh -p source/isaaclab_assets/isaaclab_assets/legged_v3/inspect_robot.py

Then update legged_v3_cfg.py and legged_v3_velocity_env_cfg.py accordingly.
"""

import isaaclab.sim as sim_utils
from isaaclab.app import AppLauncher

app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

import torch
from isaaclab.assets import Articulation
from isaaclab_assets.legged_v3.legged_v3_cfg import LEGGED_ROBOT_V3_CFG

sim_cfg = sim_utils.SimulationCfg(dt=1 / 60.0, device="cpu")
sim = sim_utils.SimulationContext(sim_cfg)
sim_utils.set_camera_view([2.0, 2.0, 2.0], [0.0, 0.0, 0.5])

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

print("\n" + "=" * 60)
print("IMU should be placed on the BASE body above.")
print("Use the body name to set imu.prim_path in the env cfg.")
print("Pattern: {ENV_REGEX_NS}/Robot/<body_name>")
print("=" * 60 + "\n")

simulation_app.close()
