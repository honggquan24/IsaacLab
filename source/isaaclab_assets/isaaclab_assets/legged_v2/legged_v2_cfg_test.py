"""Configuration for the 2-Legged Robot imported from Onshape."""
import os
import math
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg

# PATH CONFIG
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

LEGGED_ROBOT_USD_PATH = os.path.join(
    CURRENT_DIR, "usd_file", "robot_v2_cfg_copy_p.usd"
)

# ROBOT CONFIG
LEGGED_ROBOT_V2_CFG_TEST = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=LEGGED_ROBOT_USD_PATH,
        mass_props=sim_utils.schemas.MassPropertiesCfg(
            mass=10,
            density=100,
        ),
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
        ),
    ),
    
    # INITIAL STATE - Must match reset_position in env config
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.27),  # FIX: Changed from 0.2 to 0.27 (match env reset)
        joint_pos={
            # --- Left leg joints ---
            "Left_Revolute_01": 0.0,                    # Hip
            "Left_Revolute_02": 0.7,     # Knee
            "Left_Revolute_03": 0.27,     # Ankle
            "Left_Revolute_04": 0.0,                    # Wheel - FIX: Uncommented
            # "Left_Revolute_05": math.radians(-13.0),    # Passive - FIX: Fixed angle
            # "Left_Revolute_06": math.radians(-13.0),    # Passive - FIX: Fixed angle

            # --- Right leg joints ---
            "Right_Revolute_01": 0.0,                   # Hip
            "Right_Revolute_02": -0.7,   # Knee
            "Right_Revolute_03": -0.27,   # Ankle
            "Right_Revolute_04": 0.0,                   # Wheel - FIX: Uncommented
            # "Right_Revolute_05": math.radians(12.6),    # Passive
            # "Right_Revolute_06": math.radians(12.6),    # Passive
        },
        joint_vel={
            # --- All joints start with zero velocity ---
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
        # --- RIGHT LEG ---
        "hip_right": ImplicitActuatorCfg( 
            joint_names_expr=["Right_Revolute_01"],
            effort_limit_sim=100.0,      # FIX: Reduced from 1000 (too high)
            stiffness=0.0,
            damping=0.5,                 # FIX: Added damping for stability
            velocity_limit_sim=50.0,     # FIX: Reduced from 250 (unrealistic)
        ),
        "knee_right": ImplicitActuatorCfg(
            joint_names_expr=["Right_Revolute_02"],
            effort_limit_sim=100.0,      # FIX: Reduced from 1000
            stiffness=0.0,
            damping=0.5,                 # FIX: Added damping
            velocity_limit_sim=50.0,
        ),
        "ankle_right": ImplicitActuatorCfg(
            joint_names_expr=["Right_Revolute_03"],
            effort_limit_sim=100.0,      # FIX: Reduced from 1000
            stiffness=10.0,
            damping=0.5,                 # FIX: Added damping
            velocity_limit_sim=50.0,
        ),
        "wheel_right": ImplicitActuatorCfg(
            joint_names_expr=["Right_Revolute_04"],
            effort_limit_sim=50.0,       # FIX: Reduced from 1000 (wheels need less)
            stiffness=0.0,
            damping=0.1,                 # FIX: Lower damping for wheels
            velocity_limit_sim=100.0,    # FIX: Increased from 50 (wheels can spin fast)
        ),
        
        # --- LEFT LEG ---
        "hip_left": ImplicitActuatorCfg(
            joint_names_expr=["Left_Revolute_01"],
            effort_limit_sim=100.0,      # FIX: Reduced from 1000
            stiffness=0.0,
            damping=0.5,                 # FIX: Added damping
            velocity_limit_sim=50.0,
        ),
        "knee_left": ImplicitActuatorCfg(
            joint_names_expr=["Left_Revolute_02"],
            effort_limit_sim=100.0,      # FIX: Reduced from 1000
            stiffness=0.0,
            damping=0.5,                 # FIX: Added damping
            velocity_limit_sim=50.0,
        ),
        "ankle_left": ImplicitActuatorCfg(
            joint_names_expr=["Left_Revolute_03"],
            effort_limit_sim=100.0,      # FIX: Reduced from 1000
            stiffness=10.0,
            damping=0.5,                 # FIX: Added damping
            velocity_limit_sim=50.0,
        ),
        "wheel_left": ImplicitActuatorCfg(
            joint_names_expr=["Left_Revolute_04"],
            effort_limit_sim=50.0,       # FIX: Reduced from 1000
            stiffness=0.0,
            damping=0.1,                 # FIX: Lower damping for wheels
            velocity_limit_sim=100.0,    # FIX: Increased from 50
        ),
        
        # --- PASSIVE JOINTS ---
        "passive": ImplicitActuatorCfg(
            joint_names_expr=[".*_Revolute_0[5-6]"],
            effort_limit_sim=0.0,        # No actuation
            stiffness=0.0,
            damping=0.0,
        ),
    },
)