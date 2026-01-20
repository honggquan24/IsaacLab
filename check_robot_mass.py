#!/usr/bin/env python3
"""Check mass properties of Evobot V1 robot bodies."""

import argparse
from isaaclab.app import AppLauncher

# Parse arguments
parser = argparse.ArgumentParser(description="Check robot mass properties")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Launch Isaac Sim
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.sim import SimulationContext

from isaaclab_assets.evobot_v1.evobot_v1_cfg import EVOBOT_V1_CFG

def main():
    # Create simulation context
    sim_cfg = sim_utils.SimulationCfg(dt=0.01)
    sim = SimulationContext(sim_cfg)
    sim.set_camera_view(eye=[2.5, 2.5, 2.5], target=[0.0, 0.0, 0.0])

    # Spawn robot
    robot_cfg = EVOBOT_V1_CFG.replace(prim_path="/World/Robot")
    robot = Articulation(robot_cfg)

    # Design scene
    sim.reset()

    # Step simulation to initialize physics
    for _ in range(10):
        robot.update(dt=sim.get_physics_dt())
        sim.step()

    # Print mass properties
    print("\n" + "="*60)
    print("EVOBOT V1 MASS PROPERTIES")
    print("="*60)

    # Get body names and masses
    body_names = robot.body_names

    # Get masses from physics view (masses are stored per body)
    # Note: Isaac Lab stores masses differently, we'll use get_masses()
    try:
        # Try to get masses from root_physx_view
        body_masses = robot.root_physx_view.get_masses()
        print(f"\nTotal bodies: {len(body_names)}")
        print(f"Body masses shape: {body_masses.shape}")

        # Print each body
        for i, name in enumerate(body_names):
            mass = body_masses[0, i].item()
            print(f"\n{name}:")
            print(f"  Mass: {mass:.4f} kg")

            # Highlight arm and gripper
            if "arm" in name.lower() or "gripper" in name.lower():
                print(f"  ⚠️  IMPORTANT FOR FORCE/TORQUE CALC")

        # Calculate total arm + gripper mass
        arm_gripper_mass = 0.0
        for i, name in enumerate(body_names):
            if "arm" in name.lower() or "gripper" in name.lower():
                arm_gripper_mass += body_masses[0, i].item()

    except AttributeError:
        # Fallback: use data buffer if available
        print(f"\nTotal bodies: {len(body_names)}")
        print("\nBody names:")
        for i, name in enumerate(body_names):
            print(f"  {i}: {name}")

        # Use estimated masses based on typical robot proportions
        print("\n⚠️  Could not get masses from physics view")
        print("Using estimated values based on robot size:")

        # Estimate based on typical self-balancing robot
        arm_gripper_mass = 0.5  # Estimate 500g for arm + grippers

    print("\n" + "="*60)
    print(f"TOTAL ARM + GRIPPER MASS: {arm_gripper_mass:.4f} kg")
    print("="*60)

    # Calculate recommended force/torque ranges
    print("\n" + "="*60)
    print("RECOMMENDED FORCE/TORQUE RANGES")
    print("="*60)

    m = arm_gripper_mass
    L = 0.2  # Estimate arm length (you can measure from USD)
    g = 9.81

    print(f"\nAssuming arm length L ≈ {L} m")
    print(f"\n1. LIGHT disturbance (0.5g):")
    print(f"   force_range: ({-0.5*m*g:.1f}, {0.5*m*g:.1f}) N")
    print(f"   torque_range: ({-0.1*m*L*g:.1f}, {0.1*m*L*g:.1f}) Nm")

    print(f"\n2. MEDIUM disturbance (1.0g):")
    print(f"   force_range: ({-1.0*m*g:.1f}, {1.0*m*g:.1f}) N")
    print(f"   torque_range: ({-0.2*m*L*g:.1f}, {0.2*m*L*g:.1f}) Nm")

    print(f"\n3. STRONG disturbance (2.0g):")
    print(f"   force_range: ({-2.0*m*g:.1f}, {2.0*m*g:.1f}) N")
    print(f"   torque_range: ({-0.5*m*L*g:.1f}, {0.5*m*L*g:.1f}) Nm")

    print("\n" + "="*60)
    print("YOUR TRAINED CONFIG:")
    print("  force_range: (-5.0, 5.0)")
    print("  torque_range: (-10.0, 10.0)")
    print("="*60)

if __name__ == "__main__":
    main()
    simulation_app.close()
