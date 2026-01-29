"""Configuration for the Legged Robot V3 (2-legged wheeled robot from Onshape robot_legged_v2).

Robot structure (from Onshape screenshots):
- base (chassis)
- left_leg:  left_hip, thigh, left_calf_motor (calf + motor), left_wheel
  Joints (from Onshape Mate Features):
    root level : left_hip_joint
    in left_leg: left_thigh_joint, left_knee_joint, left_wheel_joint
- right_leg: symmetric
  Joints:
    root level : right_hip_joint
    in right_leg: right_thigh_joint, right_knee_joint, right_wheel_joint

Total: 8 DOF (4 per leg)

NOTE: Joint names MUST match the USD prim names (Onshape mate feature names).
      If training crashes with joint-not-found errors, check actual names with:
          print(env.scene["robot"].data.joint_names)
      and update joint_pos, joint_vel, and actuators accordingly.

      IMU prim path also needs to be verified after USD export:
          print(env.scene["robot"].data.body_names)
"""
import os
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

# PATH CONFIG
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
LEGGED_ROBOT_V3_USD_PATH = os.path.join(
    CURRENT_DIR, "usd_file", "robot_legged_v3_cfg.usd"   # actual file name
)

# ROBOT CONFIG
LEGGED_ROBOT_V3_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=LEGGED_ROBOT_V3_USD_PATH,
        activate_contact_sensors=True,
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

    # INITIAL STATE — standing pose
    # Signs: left and right joints are mirrored (opposite sign for symmetric joints).
    # Adjust if the robot spawns in an incorrect pose.
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.30),
        joint_pos={
            # --- Left leg (Onshape mate feature names) ---
            "left_hip_joint":   0.0,   # hip abduction/adduction
            "left_thigh_joint": 0.3,   # hip pitch (forward lean)
            "left_knee_joint": -0.6,   # knee bend
            "left_wheel_joint": 0.0,   # wheel (free spinning)

            # --- Right leg ---
            "right_hip_joint":   0.0,
            "right_thigh_joint": -0.3,  # mirrored sign
            "right_knee_joint":   0.6,  # mirrored sign
            "right_wheel_joint":  0.0,
        },
        joint_vel={
            "left_hip_joint":    0.0,
            "left_thigh_joint":  0.0,
            "left_knee_joint":   0.0,
            "left_wheel_joint":  0.0,
            "right_hip_joint":   0.0,
            "right_thigh_joint": 0.0,
            "right_knee_joint":  0.0,
            "right_wheel_joint": 0.0,
        },
    ),

    # ACTUATORS
    actuators={
        # --- LEFT LEG ---
        "hip_left": ImplicitActuatorCfg(
            joint_names_expr=["left_hip_joint"],
            effort_limit_sim=100.0,
            stiffness=0.0,
            damping=0.5,
            velocity_limit_sim=50.0,
        ),
        "thigh_left": ImplicitActuatorCfg(
            joint_names_expr=["left_thigh_joint"],
            effort_limit_sim=100.0,
            stiffness=0.0,
            damping=0.5,
            velocity_limit_sim=50.0,
        ),
        "knee_left": ImplicitActuatorCfg(
            joint_names_expr=["left_knee_joint"],
            effort_limit_sim=100.0,
            stiffness=0.0,
            damping=0.5,
            velocity_limit_sim=50.0,
        ),
        "wheel_left": ImplicitActuatorCfg(
            joint_names_expr=["left_wheel_joint"],
            effort_limit_sim=50.0,
            stiffness=0.0,
            damping=0.1,
            velocity_limit_sim=100.0,
        ),

        # --- RIGHT LEG ---
        "hip_right": ImplicitActuatorCfg(
            joint_names_expr=["right_hip_joint"],
            effort_limit_sim=100.0,
            stiffness=0.0,
            damping=0.5,
            velocity_limit_sim=50.0,
        ),
        "thigh_right": ImplicitActuatorCfg(
            joint_names_expr=["right_thigh_joint"],
            effort_limit_sim=100.0,
            stiffness=0.0,
            damping=0.5,
            velocity_limit_sim=50.0,
        ),
        "knee_right": ImplicitActuatorCfg(
            joint_names_expr=["right_knee_joint"],
            effort_limit_sim=100.0,
            stiffness=0.0,
            damping=0.5,
            velocity_limit_sim=50.0,
        ),
        "wheel_right": ImplicitActuatorCfg(
            joint_names_expr=["right_wheel_joint"],
            effort_limit_sim=50.0,
            stiffness=0.0,
            damping=0.1,
            velocity_limit_sim=100.0,
        ),
    },
)
