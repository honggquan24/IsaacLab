"""Configuration for the 2-Legged Robot imported from Onshape."""
import os
import math
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR

# PATH CONFIG
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

CAR_ROBOT_USD_PATH = os.path.join(
    CURRENT_DIR, "usd_file", "balance_car_v1_cfg.usd"
)

CART_BALANCE_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=CAR_ROBOT_USD_PATH,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=100.0,
            enable_gyroscopic_forces=True,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
            sleep_threshold=0.005,
            stabilization_threshold=0.001,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.5),
    ),
    actuators={
        "wheel_left": ImplicitActuatorCfg(
            joint_names_expr=["Revolute_1"], 
            effort_limit_sim=400.0, 
            stiffness=0.0, 
            damping=0.0,
            velocity_limit_sim=20.0,
        ),
        "wheel_right": ImplicitActuatorCfg(
            joint_names_expr=["Revolute_2"], 
            effort_limit_sim=400.0, 
            stiffness=0.0, 
            damping=0.0,
            velocity_limit_sim=20.0,
        ),
    },
)
"""Configuration for a simple inverted Double Pendulum on a Cart robot."""