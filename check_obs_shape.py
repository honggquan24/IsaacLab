#!/usr/bin/env python3
"""Script to calculate exact observation dimensions for evobot velocity policy."""

# Robot has 5 DOF total:
# - 2 wheels: left_wheel_joint, right_wheel_joint
# - 1 arm: arm_joint
# - 2 grippers: left_gripper_joint, right_gripper_joint

print("=" * 80)
print("EVOBOT VELOCITY POLICY - OBSERVATION SHAPE CALCULATION")
print("=" * 80)

# Observation terms from velocity_env_cfg.py PolicyCfg
obs_dims = {}

# 1. base_lin_vel (from mdp_v.base_lin_vel)
obs_dims["base_lin_vel"] = 3  # [vx, vy, vz]

# 2. IMU sensors
obs_dims["imu_lin_acc"] = 3     # [ax, ay, az]
obs_dims["imu_ang_vel"] = 3     # [wx, wy, wz]
obs_dims["imu_orientation"] = 4 # [qw, qx, qy, qz]

# 3. Joint states (ALL 5 joints)
obs_dims["joint_pos"] = 5       # All joints: [left_wheel, right_wheel, arm, left_grip, right_grip]
obs_dims["joint_vel"] = 5       # All joints velocities
obs_dims["joint_effort"] = 5    # All joints efforts

# 4. Previous actions (for smoothness)
# Action space = 5D (wheel_left, wheel_right, arm, grip_left, grip_right)
obs_dims["last_action"] = 5

# 5. Commands
# base_velocity_cmd: UniformVelocityCommandCfg with heading_command=False
# Output shape: [lin_vel_x, lin_vel_y, ang_vel_z] = 3D
obs_dims["base_velocity_cmd"] = 3

# arm_ee_pose_cmd: UniformPoseCommandCfg
# Output shape: [pos_x, pos_y, pos_z, quat_w, quat_x, quat_y, quat_z] = 7D
obs_dims["arm_ee_pose_cmd"] = 7

# grip_ee_pose_left_cmd: UniformPoseCommandCfg
obs_dims["grip_ee_pose_left_cmd"] = 7

# grip_ee_pose_right_cmd: UniformPoseCommandCfg
obs_dims["grip_ee_pose_right_cmd"] = 7

print("\nObservation Term Dimensions:")
print("-" * 80)
total = 0
for name, dim in obs_dims.items():
    print(f"{name:30s} : {dim:3d} dims")
    total += dim

print("-" * 80)
print(f"{'TOTAL OBSERVATION DIMENSION':30s} : {total:3d} dims")
print("=" * 80)

print("\n" + "=" * 80)
print("ACTION SPACE CALCULATION")
print("=" * 80)
print("\nAction dimensions:")
print("-" * 80)
print(f"{'wheel_effort (2 wheels)':30s} : 2 dims")
print(f"{'arm_effort (1 arm)':30s} : 1 dim")
print(f"{'grabber_effort (2 grippers)':30s} : 2 dims")
print("-" * 80)
print(f"{'TOTAL ACTION DIMENSION':30s} : 5 dims")
print("=" * 80)

print("\n" + "=" * 80)
print("NETWORK ARCHITECTURE (from PPO config)")
print("=" * 80)
print("\nTypical RSL-RL PPO network:")
print("-" * 80)
print(f"Input layer:  {total} dims")
print(f"Hidden layers: [128, 256, 128] (typical)")
print(f"Output layer: 5 dims (actions)")
print("=" * 80)

print("\n" + "!" * 80)
print("KEY INSIGHT: LOW-LEVEL POLICY EXPECTS EXACT INPUT SHAPE")
print("!" * 80)
print(f"\nThe pre-trained velocity policy was trained with input shape: {total}D")
print("\nThis includes:")
print("  - Robot state (3 + 3 + 4 + 5 + 5 + 5 = 25D)")
print("  - Previous actions (5D)")
print("  - Base velocity command (3D)")
print("  - Arm EE pose command (7D)")
print("  - Left gripper EE pose command (7D)")
print("  - Right gripper EE pose command (7D)")
print("\nFor hierarchical control, you MUST provide ALL these observations!")
print("=" * 80)
