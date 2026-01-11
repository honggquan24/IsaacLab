"""Configuration for the Evobot V1 Robot imported from Onshape.

USD Structure (see EVOBOT.md for details):
- 5 DOF: left_wheel_joint, right_wheel_joint, arm_joint, left_grabbing_joint, right_grabbing_joint
- All joints at root level: /evobot/evobot/<joint_name>
- Main bodies: base_link (root), head_link, arm_link, wheel, wheel_01, gripper, gripper_01
"""
import os
import math
from pathlib import Path
import isaaclab.sim as sim_utils
from isaaclab.actuators import DCMotorCfg
from isaaclab.assets import ArticulationCfg

# PATH CONFIG
CURRENT_DIR = Path(__file__).resolve().parents[2]

usd_file_path = CURRENT_DIR / "usd_file" / "evoBOT_cfg.usd"
EVOBOT_USD_PATH = usd_file_path.resolve()

if not EVOBOT_USD_PATH.exists():
    raise FileNotFoundError(
        f"USD file not found at: {EVOBOT_USD_PATH}\n"
        f"Expected relative to: {__file__}"
    )

# ROBOT CONFIG
EVOBOT_V1_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=str(EVOBOT_USD_PATH),
        # mass_props=sim_utils.schemas.MassPropertiesCfg(
        #     mass=10,
        #     density=100,
        # ),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            # max_linear_velocity=10.0,
            # max_angular_velocity=110.0,
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
            "left_grabbing_joint": 0.0,
            "right_grabbing_joint": 0.0,
        },
    ),

    actuators = {
        # ===== Wheels (Revolute joints) =====
        "wheels": DCMotorCfg(
            joint_names_expr=[".*_wheel_joint"],  # Matches: left_wheel_joint, right_wheel_joint
            saturation_effort=500.0,
            effort_limit=2000.0,
            velocity_limit=20.0,
            stiffness=1.0,
            damping=0.0,
            armature=0.01,
            friction=0.1,
            dynamic_friction=0.05,
            viscous_friction=0.02,
        ),

        # ===== Arm (Revolute joint) =====
        "arm": DCMotorCfg(
            joint_names_expr=["arm_joint"],
            saturation_effort=500.0,
            effort_limit=2000.0,
            velocity_limit=10.0,
            stiffness=1.0,
            damping=0.1,
            armature=0.02,
            friction=0.2,
            dynamic_friction=0.1,
            viscous_friction=0.05,
        ),

        # ===== Grabbing mechanism (Prismatic joints) =====
        "grabbers": DCMotorCfg(
            joint_names_expr=[".*_grabbing_joint"],  # Matches: left_grabbing_joint, right_grabbing_joint
            saturation_effort=50.0,
            effort_limit=2000.0,
            velocity_limit=1.0,
            stiffness=1.0,
            damping=0.1,
            armature=0.01,
            friction=0.2,
            dynamic_friction=0.1,
            viscous_friction=0.05,
        ),
    }
)
