"""Configuration for the 2-Legged Robot imported from Onshape."""
import os
import math
from pathlib import Path
import isaaclab.sim as sim_utils
from isaaclab.actuators import DCMotorCfg
from isaaclab.assets import ArticulationCfg

# PATH CONFIG
CURRENT_DIR = Path(__file__).resolve().parents[2] 

usd_file_path = CURRENT_DIR / "usd_file" / "robot_v2_cfg.usd"
LEGGED_ROBOT_USD_PATH = usd_file_path.resolve()

if not LEGGED_ROBOT_USD_PATH.exists():
    raise FileNotFoundError(
        f"USD file not found at: {LEGGED_ROBOT_USD_PATH}\n"
        f"Expected relative to: {__file__}"
    )

# ROBOT CONFIG
LEGGED_ROBOT_V2_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=str(LEGGED_ROBOT_USD_PATH),
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
        pos=(0.0, 0.0, 0.5),  # FIX: Changed from 0.2 to 0.27 (match env reset)
        joint_pos={
            # --- Left leg joints ---
            "Left_Revolute_01": 0.0,      # Hip
            "Left_Revolute_02": 0.34,     # Knee
            "Left_Revolute_03": 0.08,     # Ankle
            "Left_Revolute_04": 0.0,      # Wheel 

            # --- Right leg joints ---
            "Right_Revolute_01": 0.0,     # Hip
            "Right_Revolute_02": -0.34,   # Knee
            "Right_Revolute_03": -0.08,   # Ankle
            "Right_Revolute_04": 0.0,     # Wheel 

        },
    ),
    
    actuators = {
        # ===== Right Leg =====
        "hip_right": DCMotorCfg(
            joint_names_expr=["Right_Revolute_01"],
            saturation_effort=170.0,
            effort_limit=250.0,
            velocity_limit=250.0,
            stiffness=1.0,
            damping=0.0,
            armature=0.02,
            friction=0.2,
            dynamic_friction=0.1,
            viscous_friction=0.05,
        ),
        "knee_right": DCMotorCfg(
            joint_names_expr=["Right_Revolute_02"],
            saturation_effort=170.0,
            effort_limit=250.0,
            velocity_limit=250.0,
            stiffness=1.0,
            damping=0.0,
            armature=0.02,
            friction=0.2,
            dynamic_friction=0.1,
            viscous_friction=0.05,
        ),
        "ankle_right": DCMotorCfg(
            joint_names_expr=["Right_Revolute_03"],
            saturation_effort=170.0,
            effort_limit=250.0,
            velocity_limit=250.0,
            stiffness=1.0,
            damping=0.0,
            armature=0.02,
            friction=0.2,
            dynamic_friction=0.1,
            viscous_friction=0.05,
        ),
        
        "hip_left": DCMotorCfg(
            joint_names_expr=["Left_Revolute_01"],
            saturation_effort=170.0,
            effort_limit=250.0,
            velocity_limit=250.0,
            stiffness=1.0,
            damping=0.0,
            armature=0.02,
            friction=0.2,
            dynamic_friction=0.1,
            viscous_friction=0.05,
        ),
        "knee_left": DCMotorCfg(
            joint_names_expr=["Left_Revolute_02"],
            saturation_effort=170.0,
            effort_limit=250.0,
            velocity_limit=250.0,
            stiffness=1.0,
            damping=0.0,
            armature=0.02,
            friction=0.2,
            dynamic_friction=0.1,
            viscous_friction=0.05,
        ),
        "ankle_left": DCMotorCfg(
            joint_names_expr=["Left_Revolute_03"],
            saturation_effort=170.0,
            effort_limit=250.0,
            velocity_limit=250.0,
            stiffness=1.0,
            damping=0.0,
            armature=0.02,
            friction=0.2,
            dynamic_friction=0.1,
            viscous_friction=0.05,
        ),

        "passive": DCMotorCfg(
            joint_names_expr=[".*_Revolute_0[5-7]"],
            saturation_effort=0.0,
            effort_limit=250.0,
            velocity_limit=250.0,
            stiffness=1.0,
            damping=0.0,
            armature=0.2,
            friction=0.1,
            viscous_friction=0.05,
        ),
    }
)