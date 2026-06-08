"""Robot config for Biped Inverted-Pendulum PID-RL — Legged V3 URDF.

Control architecture (cascade):
  Vòng ngoài (outer, 50 Hz) : body pitch/roll angle  → desired wheel angular velocity
  Vòng trong (inner, 200 Hz): wheel angular velocity error → wheel torque

Actuator modes:
  Wheels  : effort mode (stiffness=damping=0) — action term applies torques directly.
  Hip A1  : PD mode (stiffness=30, damping=1) — held at default stance, not RL-controlled.
  Hip A2  : mimic (PhysxMimicJointAPI).
  Knees   : passive (loop-closure constraint).
"""
from isaaclab.actuators import ImplicitActuatorCfg

from isaaclab_assets.legged_v3.legged_v3_cfg import LEGGED_ROBOT_V3_CFG

# Joints driven by cascade PID
WHEEL_JOINT_NAMES   = ["left_wheel_joint", "right_wheel_joint"]
HIP_ALL_JOINT_NAMES = ["left_hip_joint_A1", "right_hip_joint_A1",
                        "left_hip_joint_A2", "right_hip_joint_A2"]
# Default hip positions (all 0.0 from URDF init_state)
HIP_DEFAULT_Q = [0.0, 0.0, 0.0, 0.0]

# Legged V3 with wheels switched to effort mode for cascade PID control
BIPED_CFG = LEGGED_ROBOT_V3_CFG.replace(
    actuators={
        # Hip A1: pure effort mode — custom PID in TiltPIDAction has full authority
        # stiffness=0 prevents implicit PD from fighting the custom PID torque
        "hip_active": ImplicitActuatorCfg(
            joint_names_expr=["left_hip_joint_A1", "right_hip_joint_A1"],
            effort_limit_sim=200.0,
            stiffness=0.0,
            damping=0.5,
            velocity_limit_sim=50.0,
        ),
        # Hip A2: mimic — PhysxMimicJointAPI tracks A1, zero drive
        "hip_mimic": ImplicitActuatorCfg(
            joint_names_expr=["left_hip_joint_A2", "right_hip_joint_A2"],
            effort_limit_sim=200.0,
            stiffness=0.0,
            damping=0.0,
            velocity_limit_sim=50.0,
        ),
        # Passive knees — loop-closure D6 spring provides geometric constraint
        "knee_b1": ImplicitActuatorCfg(
            joint_names_expr=["left_knee_joint_B1", "right_knee_joint_B1"],
            effort_limit_sim=40.0,
            stiffness=0.0,
            damping=0.5,
            velocity_limit_sim=50.0,
        ),
        "knee_b2": ImplicitActuatorCfg(
            joint_names_expr=["left_knee_joint_B2", "right_knee_joint_B2"],
            effort_limit_sim=40.0,
            stiffness=0.0,
            damping=0.5,
            velocity_limit_sim=50.0,
        ),
        # Wheels: effort mode + back-EMF damping to prevent runaway spin
        "wheel": ImplicitActuatorCfg(
            joint_names_expr=["left_wheel_joint", "right_wheel_joint"],
            effort_limit_sim=200.0,
            stiffness=0.0,
            damping=1.0,
            velocity_limit_sim=30.0,
        ),
    }
)
