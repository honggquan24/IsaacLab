"""Configuration for the 2-Legged Robot imported from Onshape."""
import os
import math
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg

# PATH CONFIG
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

ROTARY_ROBOT_USD_PATH = os.path.join(
    CURRENT_DIR, "usd_file", "rotary_pendulumv2_cfg.usd"
)

# ROBOT CONFIG
ROTARY_ROBOT_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=ROTARY_ROBOT_USD_PATH,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            # rigid_body_enabled=True,
            # max_linear_velocity=500.0,
            # max_angular_velocity=500.0,
            # linear_damping=0.002,
            # angular_damping=0.005,
            # max_depenetration_velocity=1.0,
            # enable_gyroscopic_forces=True,
            # max_contact_impulse=500,
            # retain_accelerations=True,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            # enabled_self_collisions=True,
            # solver_position_iteration_count=20,
            # solver_velocity_iteration_count=1,
        ),
    ),

    actuators={
        "revolute1": ImplicitActuatorCfg( 
            joint_names_expr=["Revolute_1"],
            effort_limit_sim=100.0,      
            stiffness=1.0,
            damping=0.2,                 
            velocity_limit_sim=25.0,    

        ),
        "revolute2": ImplicitActuatorCfg( 
            joint_names_expr=["Revolute_2"],
            effort_limit_sim=100.0,      
            stiffness=1e-7,
            damping=1e-7,                 
            velocity_limit_sim=50.0,     
        ),
    },
    init_state= ArticulationCfg.InitialStateCfg(
        pos= (0.0, 0.0 , 0.1 ) ,
        joint_vel = {"Revolute_1": 0.1},
    ),
)
