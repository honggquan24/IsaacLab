# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the Evobot V1 Robot imported from Onshape.

USD Structure (see EVOBOT.md for details):
- 6 khớp: arm_joint, base_joint, left_wheel_joint, right_wheel_joint, left_gripper_joint, right_gripper_joint
- All joints at root level: /evobot/evobot/<joint_name>
- 7 link: top_link (root), leg_link, arm_link, wheel, wheel_01, gripper, gripper_01
"""

from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import DCMotorCfg
from isaaclab.assets import ArticulationCfg

# PATH CONFIG
CURRENT_DIR = Path(__file__).resolve().parent

usd_file_path = CURRENT_DIR / "usd" / "evoBOT_v2_cfg.usd"
EVOBOT_USD_PATH = usd_file_path.resolve()

if not EVOBOT_USD_PATH.exists():
    raise FileNotFoundError(f"USD file not found at: {EVOBOT_USD_PATH}\nExpected relative to: {__file__}")

# nho check articulation_root_prim_path

# ROBOT CONFIG
EVOBOT_CFG = ArticulationCfg(
    articulation_root_prim_path="/evobot/evobot/top_link",
    spawn=sim_utils.UsdFileCfg(
        usd_path=str(EVOBOT_USD_PATH),
        # mass_props=sim_utils.schemas.MassPropertiesCfg(
        #     mass=10,
        #     density=100,
        # ),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            max_linear_velocity=500.0,
            max_angular_velocity=500.0,
            # linear_damping=0.002,
            # angular_damping=0.005,
            # max_depenetration_velocity=1.0,
            enable_gyroscopic_forces=True,
            # max_contact_impulse=5000,
            retain_accelerations=True,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            #     solver_position_iteration_count=20,
            #     solver_velocity_iteration_count=1,
        ),
        # collision_props=sim_utils.schemas.CollisionPropertiesCfg(
        #     collision_enabled=True,
        # ),
        activate_contact_sensors=True,
    ),
    # INITIAL STATE - Must match reset_position in env config
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.75),  # Starting height above ground (upright position)
        joint_pos={
            # --- Wheels (Revolute joints) ---
            "left_wheel_joint": 0.0,
            "right_wheel_joint": 0.0,
            # --- Arm (Revolute joint) ---
            "arm_joint": 0.0,
            # --- Grabbing mechanism (Prismatic joints) ---
            "left_gripper_joint": 0.0,
            "right_gripper_joint": 0.0,
        },
    ),
    actuators={
        # ===== Wheels =====
        "wheels": DCMotorCfg(
            joint_names_expr=[".*_wheel_joint"],
            saturation_effort=500.0,
            effort_limit=400.0,
            velocity_limit=400.0,
            stiffness=0.0,
            damping=1.0,
            armature=0.01,
            friction=0.01,
            dynamic_friction=0.01,
            viscous_friction=0.01,
        ),
        # ===== Arm =====
        "arm": DCMotorCfg(
            joint_names_expr=["arm_joint"],
            saturation_effort=400.0,  # ↓ Giảm (từ 500)
            effort_limit=380.0,
            velocity_limit=2.0,  # ↓ Giảm (từ 100)
            stiffness=0.0,
            damping=1.0,  # ↓ Giảm (từ 100) - QUAN TRỌNG
            armature=0.01,
            friction=0.05,
            dynamic_friction=0.01,
            viscous_friction=0.05,
        ),
        # ===== Grabbers =====
        "grabbers": DCMotorCfg(
            joint_names_expr=[".*_gripper_joint"],
            saturation_effort=100.0,  # Matched with trained config
            effort_limit=80.0,  # Matched with trained config
            velocity_limit=30.0,
            stiffness=0.0,
            damping=1.0,
            armature=0.001,
            friction=0.01,
            dynamic_friction=0.01,
            viscous_friction=0.01,
        ),
    },
)
