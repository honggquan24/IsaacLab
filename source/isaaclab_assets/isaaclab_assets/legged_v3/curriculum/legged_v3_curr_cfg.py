"""Robot configuration for Legged V3 auto-curriculum training.

Extends the base LEGGED_ROBOT_V3_CFG with curriculum-specific actuator setup:
  - Leg joints frozen initially via ImplicitActuatorCfg(stiffness=5000).
  - Wheels always active via velocity control (stiffness=0, damping=5).
  - LEGGED_CURRICULUM_ROBOT_CFG is used directly in CurriculumSceneCfg.

Frozen joint parameters:
  FROZEN_STIFFNESS   = 5000.0   : >> gravity, holds joint in place
  FROZEN_DAMPING     = 500.0    : prevents oscillation at spawn impact
  FROZEN_EFFORT_LIM  = 10000.0  : must not clamp (stiffness × max_dev=2rad)

Normal joint parameters (restored at unlock):
  NORMAL_STIFFNESS   = 20.0     : matches original DelayedPDActuatorCfg
  NORMAL_DAMPING     = 0.0
"""
import os
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg, DelayedPDActuatorCfg
from isaaclab.assets import ArticulationCfg

# ──────────────────────────────────────────────────────────────────────────────
# PATH CONFIG
# ──────────────────────────────────────────────────────────────────────────────

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
LEGGED_ROBOT_V3_USD_PATH = os.path.join(
    CURRENT_DIR, "usd_file", "robot_legged_v3_cfg.usd"
)

# ──────────────────────────────────────────────────────────────────────────────
# Frozen / normal joint parameters — edit here to tune curriculum dynamics
# ──────────────────────────────────────────────────────────────────────────────

FROZEN_STIFFNESS  = 5000.0    # >> gravity → joint doesn't move
FROZEN_DAMPING    = 500.0     # high damping prevents oscillation at spawn impact
FROZEN_EFFORT_LIM = 10000.0   # must not clamp: stiffness × max_dev=2rad → 10000 Nm

NORMAL_STIFFNESS  = 20.0      # matches original DelayedPDActuatorCfg
NORMAL_DAMPING    = 0.0

# ──────────────────────────────────────────────────────────────────────────────
# Base robot config (same USD, same init pose)
# ──────────────────────────────────────────────────────────────────────────────

LEGGED_ROBOT_V3_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=LEGGED_ROBOT_V3_USD_PATH,
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            enable_gyroscopic_forces=True,
            retain_accelerations=True,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
        ),
    ),

    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.50),
        joint_pos={
            "left_hip_joint":     0.0,
            "left_thigh_joint":  -0.3,
            "left_knee_joint":    0.0,
            "left_wheel_joint":   0.0,
            "right_hip_joint":    0.0,
            "right_thigh_joint":  0.3,
            "right_knee_joint":   0.0,
            "right_wheel_joint":  0.0,
        },
        joint_vel={k: 0.0 for k in [
            "left_hip_joint", "left_thigh_joint", "left_knee_joint", "left_wheel_joint",
            "right_hip_joint", "right_thigh_joint", "right_knee_joint", "right_wheel_joint",
        ]},
    ),

    actuators={
        "hip_left":    DelayedPDActuatorCfg(joint_names_expr=["left_hip_joint"],    effort_limit_sim=100.0, stiffness=NORMAL_STIFFNESS, damping=NORMAL_DAMPING, velocity_limit_sim=50.0),
        "thigh_left":  DelayedPDActuatorCfg(joint_names_expr=["left_thigh_joint"],  effort_limit_sim=100.0, stiffness=NORMAL_STIFFNESS, damping=NORMAL_DAMPING, velocity_limit_sim=50.0),
        "knee_left":   DelayedPDActuatorCfg(joint_names_expr=["left_knee_joint"],   effort_limit_sim=100.0, stiffness=NORMAL_STIFFNESS, damping=NORMAL_DAMPING, velocity_limit_sim=50.0),
        "wheel_left":  DelayedPDActuatorCfg(joint_names_expr=["left_wheel_joint"],  effort_limit_sim=100.0, stiffness=0.0,              damping=5.0,            velocity_limit_sim=100.0),
        "hip_right":   DelayedPDActuatorCfg(joint_names_expr=["right_hip_joint"],   effort_limit_sim=100.0, stiffness=NORMAL_STIFFNESS, damping=NORMAL_DAMPING, velocity_limit_sim=50.0),
        "thigh_right": DelayedPDActuatorCfg(joint_names_expr=["right_thigh_joint"], effort_limit_sim=100.0, stiffness=NORMAL_STIFFNESS, damping=NORMAL_DAMPING, velocity_limit_sim=50.0),
        "knee_right":  DelayedPDActuatorCfg(joint_names_expr=["right_knee_joint"],  effort_limit_sim=100.0, stiffness=NORMAL_STIFFNESS, damping=NORMAL_DAMPING, velocity_limit_sim=50.0),
        "wheel_right": DelayedPDActuatorCfg(joint_names_expr=["right_wheel_joint"], effort_limit_sim=100.0, stiffness=0.0,              damping=5.0,            velocity_limit_sim=100.0),
    },
)

# ──────────────────────────────────────────────────────────────────────────────
# Curriculum robot config — leg joints frozen, wheels always active
# ──────────────────────────────────────────────────────────────────────────────

def _frozen_implicit(joint_names_expr: list[str]) -> ImplicitActuatorCfg:
    """Leg joint held frozen by high stiffness. Restored at unlock time."""
    return ImplicitActuatorCfg(
        joint_names_expr=joint_names_expr,
        effort_limit_sim=FROZEN_EFFORT_LIM,
        stiffness=FROZEN_STIFFNESS,
        damping=FROZEN_DAMPING,
        velocity_limit_sim=50.0,
    )


def _wheel_implicit(joint_names_expr: list[str]) -> ImplicitActuatorCfg:
    """Wheel joint: velocity-controlled via damping (stiffness=0)."""
    return ImplicitActuatorCfg(
        joint_names_expr=joint_names_expr,
        effort_limit_sim=100.0,
        stiffness=0.0,
        damping=5.0,
        velocity_limit_sim=100.0,
    )


LEGGED_CURRICULUM_ROBOT_CFG = LEGGED_ROBOT_V3_CFG.replace(
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.4),   # lower spawn to reduce drop on reset
        joint_pos=LEGGED_ROBOT_V3_CFG.init_state.joint_pos,
        joint_vel=LEGGED_ROBOT_V3_CFG.init_state.joint_vel,
    ),
    actuators={
        "hip_left":    _frozen_implicit(["left_hip_joint"]),
        "thigh_left":  _frozen_implicit(["left_thigh_joint"]),
        "knee_left":   _frozen_implicit(["left_knee_joint"]),
        "hip_right":   _frozen_implicit(["right_hip_joint"]),
        "thigh_right": _frozen_implicit(["right_thigh_joint"]),
        "knee_right":  _frozen_implicit(["right_knee_joint"]),
        "wheel_left":  _wheel_implicit(["left_wheel_joint"]),
        "wheel_right": _wheel_implicit(["right_wheel_joint"]),
    },
)
