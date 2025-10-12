"""Configuration for the 2-Legged Robot imported from Onshape."""
import os
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

# PATH CONFIG
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
LEGGED_ROBOT_USD_PATH = os.path.join(
    CURRENT_DIR, "usd_file", "legged_robot_v1", "Robot_2_leg_cfg.usd"
)

# ROBOT CONFIG
LEGGED_ROBOT_V1_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=LEGGED_ROBOT_USD_PATH,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            max_linear_velocity=100.0,
            max_angular_velocity=100.0,
            max_depenetration_velocity=100.0,
            enable_gyroscopic_forces=True,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
        ),
    ),

    # INITIAL STATE
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.5),  # spawn height
        joint_pos={
            "Revolute_1": 0.0,
            "Revolute_2": 0.0,
            "Revolute_3": 0.0,
            "Revolute_4": 0.0,
            "Revolute_5": 0.0,
            "Revolute_6": 0.0,
        },
    ),

    # ACTUATORS
    actuators={
        "hip_right": ImplicitActuatorCfg(
            joint_names_expr=["Revolute_2"],
            effort_limit_sim=400.0,
            stiffness=10.0,
            damping=5.0,
        ),
        "knee_right": ImplicitActuatorCfg(
            joint_names_expr=["Revolute_4"],
            effort_limit_sim=400.0,
            stiffness=10.0,
            damping=5.0,
        ),
        "ankle_right": ImplicitActuatorCfg(
            joint_names_expr=["Revolute_6"],
            effort_limit_sim=400.0,
            stiffness=10.0,
            damping=5.0,
        ),
        "hip_left": ImplicitActuatorCfg(
            joint_names_expr=["Revolute_1"],
            effort_limit_sim=400.0,
            stiffness=10.0,
            damping=5.0,
        ),
        "knee_left": ImplicitActuatorCfg(
            joint_names_expr=["Revolute_3"],
            effort_limit_sim=400.0,
            stiffness=10.0,
            damping=5.0,
        ),
        "ankle_left": ImplicitActuatorCfg(
            joint_names_expr=["Revolute_5"],
            effort_limit_sim=400.0,
            stiffness=10.0,
            damping=5.0,
        ),
    },
)

