# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the Evobot V1 Robot for BALANCE task.

This config matches the actuator parameters used during training.
Critical difference from root evobot_cfg.py:
- Wheels: stiffness=1.0, damping=0.0, velocity_limit=20.0
- Training used these specific parameters for balance task
"""

from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import DCMotorCfg
from isaaclab.assets import ArticulationCfg

# PATH CONFIG - Use evobot_v1 root directory's USD file
# File is at: evobot_v1/navigation/balance/balance_cfg.py
# USD is at:  evobot_v1/usd_file/evoBOT_cfg.usd
# Need to go up 2 levels: balance/ -> navigation/ -> evobot_v1/
CURRENT_DIR = Path(__file__).resolve().parent.parent
usd_file_path = CURRENT_DIR / "usd" / "evoBOT_cfg.usd"
EVOBOT_USD_PATH = usd_file_path.resolve()

if not EVOBOT_USD_PATH.exists():
    raise FileNotFoundError(f"USD file not found at: {EVOBOT_USD_PATH}\nExpected relative to: {__file__}")

# ROBOT CONFIG FOR BALANCE TASK
EVOBOT_BALANCE_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=str(EVOBOT_USD_PATH),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            max_linear_velocity=500.0,
            max_angular_velocity=500.0,
            enable_gyroscopic_forces=True,
            retain_accelerations=True,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
        ),
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
            "left_grabbing_joint": 0.0,
            "right_grabbing_joint": 0.0,
        },
    ),
    actuators={
        # ===== Wheels - TRAINING CONFIG =====
        "wheels": DCMotorCfg(
            joint_names_expr=[".*_wheel_joint"],
            effort_limit=2000.0,  # Match training
            velocity_limit=200.0,  # Match training (NOT 100.0!)
            stiffness=0.0,  # Match training (NOT 0.0!)
            damping=4.0,  # Match training (NOT 1.0!)
            armature=0.01,  # Match training (NOT 0.001!)
            friction=0.1,  # Match training (NOT 0.01!)
            dynamic_friction=0.05,  # Match training
            viscous_friction=0.02,  # Match training
            saturation_effort=500.0,  # Match training
        ),
        # ===== Arm - TRAINING CONFIG =====
        "arm": DCMotorCfg(
            joint_names_expr=["arm_joint"],
            effort_limit=2000.0,  # Match training (NOT 500!)
            velocity_limit=10.0,  # Match training (NOT 100!)
            stiffness=0.0,  # Match training (NOT 0.0!)
            damping=1.0,  # Match training (NOT 2.0!)
            armature=0.02,  # Match training
            friction=0.2,  # Match training
            dynamic_friction=0.1,  # Match training
            viscous_friction=0.05,  # Match training
            saturation_effort=500.0,  # Match training
        ),
        # ===== Grabbers - TRAINING CONFIG =====
        "grabbers": DCMotorCfg(
            joint_names_expr=[".*_grabbing_joint"],
            effort_limit=2000.0,  # Match training (NOT 200!)
            velocity_limit=1.0,  # Match training (NOT 100!)
            stiffness=1.0,  # Match training (NOT 0.0!)
            damping=0.1,  # Match training (NOT 1.0!)
            armature=0.01,  # Match training
            friction=0.2,  # Match training
            dynamic_friction=0.1,  # Match training
            viscous_friction=0.05,  # Match training
            saturation_effort=50.0,  # Match training (NOT 200!)
        ),
    },
)
