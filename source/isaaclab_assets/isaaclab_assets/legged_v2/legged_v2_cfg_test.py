"""Configuration for the 2-Legged Robot imported from Onshape."""
import os
import math
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

# PATH CONFIG
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
LEGGED_ROBOT_USD_PATH = os.path.join(
    CURRENT_DIR, "usd_file", "robot_v2_cfg.usd"
)

# ROBOT CONFIG
LEGGED_ROBOT_V2_CFG_TEST = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=LEGGED_ROBOT_USD_PATH,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            max_linear_velocity=25.0,
            max_angular_velocity=50.0,
            linear_damping=0.002,
            angular_damping=0.005,
            max_depenetration_velocity=1.0,
            enable_gyroscopic_forces=True,
            max_contact_impulse=500,
            retain_accelerations=True,
            
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=20,
            solver_velocity_iteration_count=1,
        ),
        collision_props=sim_utils.schemas.CollisionPropertiesCfg(
            collision_enabled=True,
            contact_offset=0.02,
            rest_offset=0.1,
            torsional_patch_radius=0.001,
            min_torsional_patch_radius = 0.001,
        ),
    ),
    # INITIAL STATE - Đồng bộ với reset_position
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.2),  # Tăng chiều cao spawn để phù hợp với reset range
        joint_pos={
            "Left_Revolute_01": 0.0,
            "Left_Revolute_02": 0.0,
            "Left_Revolute_03": 0.263,
            "Left_Revolute_04": 0.0,
            "Left_Revolute_05": 0.0,
            "Left_Revolute_06": 0.0,
            "Right_Revolute_01": 0.0,
            "Right_Revolute_02": 0.0,
            "Right_Revolute_03": -0.263,
            "Right_Revolute_04": 0.0,
            "Right_Revolute_05": 0.0,
            "Right_Revolute_06": 0.0,
        },
        # Đảm bảo velocity ban đầu = 0
        joint_vel={
            "Left_Revolute_01": 0.0,
            "Left_Revolute_02": 0.0,
            "Left_Revolute_03": 0.0,
            "Left_Revolute_04": 0.0,
            "Left_Revolute_05": 0.0,
            "Left_Revolute_06": 0.0,
            "Right_Revolute_01": 0.0,
            "Right_Revolute_02": 0.0,
            "Right_Revolute_03": 0.0,
            "Right_Revolute_04": 0.0,
            "Right_Revolute_05": 0.0,
            "Right_Revolute_06": 0.0,
        },
    ),
    # ACTUATORS
    actuators={
        "hip_right": ImplicitActuatorCfg(
            joint_names_expr=["Right_Revolute_01"],
            effort_limit_sim=1000,
            stiffness=0,
            damping=0,
            velocity_limit=250.0,
        ),
        "knee_right": ImplicitActuatorCfg(
            joint_names_expr=["Right_Revolute_02"],
            effort_limit_sim=1000,
            stiffness=0,
            damping=0,
            velocity_limit=50.0,
        ),
        "ankle_right": ImplicitActuatorCfg(
            joint_names_expr=["Right_Revolute_03"],
            effort_limit_sim=1000,
            stiffness=0,
            damping=0,
            velocity_limit=50.0,
        ),
        "wheel_right": ImplicitActuatorCfg(
            joint_names_expr=["Right_Revolute_04"],
            effort_limit_sim=1000,
            stiffness=0,
            damping=0,
            velocity_limit=50.0,
        ),
        "hip_left": ImplicitActuatorCfg(
            joint_names_expr=["Left_Revolute_01"],
            effort_limit_sim=1000,
            stiffness=0,
            damping=0,
            velocity_limit=50.0,
        ),
        "knee_left": ImplicitActuatorCfg(
            joint_names_expr=["Left_Revolute_02"],
            effort_limit_sim=1000,
            stiffness=0,
            damping=0,
            velocity_limit=50.0,
        ),
        "ankle_left": ImplicitActuatorCfg(
            joint_names_expr=["Left_Revolute_03"],
            effort_limit_sim=1000,
            stiffness=0,
            damping=0,
            velocity_limit=50.0,
        ),
        "wheel_left": ImplicitActuatorCfg(
            joint_names_expr=["Left_Revolute_04"],
            effort_limit_sim=1000,
            stiffness=0,
            damping=0,
            velocity_limit=250.0,
        ),
    },
)