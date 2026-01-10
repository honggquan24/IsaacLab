"""Configuration for the Evobot V1 Robot imported from Onshape."""
import os
import math
from pathlib import Path
import isaaclab.sim as sim_utils
from isaaclab.actuators import DCMotorCfg
from isaaclab.assets import ArticulationCfg

# PATH CONFIG
CURRENT_DIR = Path(__file__).resolve().parents[2]

usd_file_path = CURRENT_DIR / "usd_file" / "evobot_v1.usd"
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
            # max_linear_velocity=25.0,
            # max_angular_velocity=50.0,
            # linear_damping=0.002,
            # angular_damping=0.005,
            # max_depenetration_velocity=1.0,
            enable_gyroscopic_forces=True,
            # max_contact_impulse=5000,
            retain_accelerations=True,
        ),
        # articulation_props=sim_utils.ArticulationRootPropertiesCfg(
        #     enabled_self_collisions=True,
        #     solver_position_iteration_count=20,
        #     solver_velocity_iteration_count=1,
        # ),
        # collision_props=sim_utils.schemas.CollisionPropertiesCfg(
        #     collision_enabled=True,
        # ),
        activate_contact_sensors=True,
    ),

    # INITIAL STATE - Must match reset_position in env config
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.1),  # Starting height above ground
        joint_pos={
            # --- Wheels ---
            "left_wheel_joint": 0.0,
            "right_wheel_joint": 0.0,

            # --- Arm and grabbing mechanism ---
            "arm_joint": 0.0,
            "base_joint": 0.0,
            "left_grabbing_joint": 0.0,
            "right_grabbing_joint": 0.0,

            # --- Levers and turntables (passive) ---
            "left_levers_joint_1": 0.0,
            "left_levers_2_joint": 0.0,
            "left_turntable_joint": 0.0,
            "right_levers_joint_1": 0.0,
            "right_levers_2_joint": 0.0,
            "right_turntable_joint": 0.0,
        },
    ),

    actuators = {
        # ===== Wheels (main drive) =====
        "wheels": DCMotorCfg(
            joint_names_expr=[".*_wheel_joint"],
            saturation_effort=400.0,
            effort_limit=500.0,
            velocity_limit=20.0,
            stiffness=0.0,
            damping=0.0,
            armature=0.01,
            friction=0.1,
            dynamic_friction=0.05,
            viscous_friction=0.02,
        ),

        # ===== Arm =====
        "arm": DCMotorCfg(
            joint_names_expr=["arm_joint"],
            saturation_effort=100.0,
            effort_limit=150.0,
            velocity_limit=10.0,
            stiffness=1.0,
            damping=0.1,
            armature=0.02,
            friction=0.2,
            dynamic_friction=0.1,
            viscous_friction=0.05,
        ),

        # ===== Base joint =====
        "base": DCMotorCfg(
            joint_names_expr=["base_joint"],
            saturation_effort=100.0,
            effort_limit=150.0,
            velocity_limit=10.0,
            stiffness=1.0,
            damping=0.1,
            armature=0.02,
            friction=0.2,
            dynamic_friction=0.1,
            viscous_friction=0.05,
        ),

        # ===== Grabbing mechanism =====
        "grabbers": DCMotorCfg(
            joint_names_expr=[".*_grabbing_joint"],
            saturation_effort=50.0,
            effort_limit=100.0,
            velocity_limit=5.0,
            stiffness=1.0,
            damping=0.1,
            armature=0.01,
            friction=0.2,
            dynamic_friction=0.1,
            viscous_friction=0.05,
        ),

        # ===== Passive joints (levers, turntables) =====
        "passive": DCMotorCfg(
            joint_names_expr=[".*_levers.*|.*_turntable.*"],
            saturation_effort=0.0,
            effort_limit=10.0,
            velocity_limit=10.0,
            stiffness=0.0,
            damping=0.1,
            armature=0.01,
            friction=0.1,
            viscous_friction=0.05,
        ),
    }
)
