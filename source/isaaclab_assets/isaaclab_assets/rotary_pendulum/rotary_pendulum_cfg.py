# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the Rotary Pendulum V2 (Furuta Pendulum).

USD Structure:
- 2 DOF: Revolute_1 (pivot motor), Revolute_2 (pendulum)
- Prim hierarchy: /rotary_pendulum/rotary_pendulum/pivot/Revolute_1, Revolute_2
- Main bodies: base (fixed), pivot (rotating arm), pendulum (free-swinging)
"""

from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import DCMotorCfg
from isaaclab.assets import ArticulationCfg

# PATH CONFIG
CURRENT_DIR = Path(__file__).resolve().parent

usd_file_path = CURRENT_DIR / "usd" / "rotary_pendulum_v2_base.usd"
ROTARY_PENDULUM_USD_PATH = usd_file_path.resolve()

if not ROTARY_PENDULUM_USD_PATH.exists():
    raise FileNotFoundError(f"USD file not found at: {ROTARY_PENDULUM_USD_PATH}\nExpected relative to: {__file__}")

# ROBOT CONFIG
ROTARY_PENDULUM_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=str(ROTARY_PENDULUM_USD_PATH),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            # rigid_body_enabled=True,
            # max_linear_velocity=100.0,
            # max_angular_velocity=100.0,
            # enable_gyroscopic_forces=True,
            # retain_accelerations=True,
        ),
        # articulation_props=sim_utils.ArticulationRootPropertiesCfg(
        #     enabled_self_collisions=False,
        # ),
        # activate_contact_sensors=False,
    ),
    # INITIAL STATE - pendulum hanging down (theta2 = pi for swing-up task)
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.0),
        joint_pos={
            "Revolute_1": 0.0,  # Pivot angle (motor-driven)
            "Revolute_2": 0.0,  # Pendulum angle (0 = hanging down)
        },
    ),
    actuators={
        # ===== Pivot Motor (Revolute_1) =====
        # JointEffortActionCfg = torque control => stiffness=0, damping=0
        "pivot_motor": DCMotorCfg(
            joint_names_expr=["Revolute_1"],
            saturation_effort=100.0,
            effort_limit=100.0,
            velocity_limit=100.0,
            stiffness=0.01,  # No position control (pure torque)
            damping=0.0,  # No velocity damping (policy controls all)
            armature=0.01,
            friction=0.0,
        ),
    },
)
