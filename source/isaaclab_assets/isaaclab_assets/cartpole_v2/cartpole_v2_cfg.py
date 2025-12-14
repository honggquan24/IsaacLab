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

CARTPOLE_ROBOT_USD_PATH = os.path.join(
    CURRENT_DIR, "usd_file", "cartpole_v2_cfg_up.usd"
)

# # ROBOT CONFIG
# CARTPOLE_V2_CFG = ArticulationCfg(
#     spawn=sim_utils.UsdFileCfg(
#         usd_path=CARTPOLE_ROBOT_USD_PATH,
#         rigid_props=sim_utils.RigidBodyPropertiesCfg(
#             rigid_body_enabled=True,
#             max_linear_velocity=100.0,
#             max_angular_velocity=100.0,
#             linear_damping=0.002,
#             angular_damping=0.005,
#             max_depenetration_velocity=100.0,
#             enable_gyroscopic_forces=True,
#             max_contact_impulse=500,
#             retain_accelerations=True,
#         ),
#         articulation_props=sim_utils.ArticulationRootPropertiesCfg(
#             enabled_self_collisions=False,
#             solver_position_iteration_count=4,
#             solver_velocity_iteration_count=0,
#             sleep_threshold=0.005,
#             stabilization_threshold=0.001,
#         ),
#     ),

#     actuators={
#         "slider": ImplicitActuatorCfg( 
#             joint_names_expr=["Slider_1"],
#             effort_limit_sim=400.0,      
#             stiffness=0.0,
#             damping=10.0,                 
#             velocity_limit_sim=10.0,     
#         ),
#         "passive1": ImplicitActuatorCfg( 
#             joint_names_expr=["Revolute_1"],
#             effort_limit_sim=400.0,      
#             stiffness=0.001,
#             damping=0.01,                 
#             velocity_limit_sim=20.0,     
#         ),
#         "passive2": ImplicitActuatorCfg(
#             joint_names_expr=["Revolute_2"],
#             effort_limit_sim=400.0,      
#             stiffness=0.0,
#             damping=0.0,                 
#             velocity_limit_sim=25.0, 
#         ),
#     },
#     init_state= ArticulationCfg.InitialStateCfg(
#         pos= (0.0, 0.0 , 1.5 ) ,
#         joint_vel = {"Slider_1": 0.1},
#         joint_pos = {"Revolute_1": math.pi},
#     ),
# )


CART_DOUBLE_PENDULUM_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Robots/Classic/CartDoublePendulum/cart_double_pendulum.usd",
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
        pos=(0.0, 0.0, 2.0), joint_pos={"slider_to_cart": 0.0, "cart_to_pole": math.pi, "pole_to_pendulum": 0.0}
    ),
    actuators={
        "cart_actuator": ImplicitActuatorCfg(
            joint_names_expr=["slider_to_cart"],
            effort_limit_sim=400.0,
            stiffness=0.0,
            damping=10.0,
        ),
        "pole_actuator": ImplicitActuatorCfg(
            joint_names_expr=["cart_to_pole"], effort_limit_sim=400.0, stiffness=0.0, damping=0.0
        ),
        "pendulum_actuator": ImplicitActuatorCfg(
            joint_names_expr=["pole_to_pendulum"], effort_limit_sim=400.0, stiffness=0.0, damping=0.0
        ),
    },
)
"""Configuration for a simple inverted Double Pendulum on a Cart robot."""