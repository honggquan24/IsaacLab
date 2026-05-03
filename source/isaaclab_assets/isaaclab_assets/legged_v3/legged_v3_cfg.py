"""Configuration for the Legged Robot V3 — loaded from URDF.

Robot structure (5-bar parallel linkage, 2 legs + wheels):
  base_link
  ├── pad_link_right (revolute Z) → hip_frame_link_right (fixed)
  │     ├── thigh_right_1 → calf_right_link_1 → wheel_link_right  [ACTIVE chain]
  │     └── thigh_right_2 → calf_right_link_2                     [PASSIVE]
  └── pad_link_left (revolute Z) → hip_frame_link_left (fixed)
        ├── thigh_left_1 → calf_left_link_1 → wheel_link_left     [ACTIVE chain]
        └── thigh_left_2 → calf_left_link_2                       [PASSIVE]

Closed-loop: calf_*_2 tip → wheel_* added via revolute joint in URDF.
             close_loop_* are excluded from articulation at spawn time via
             _spawn_urdf_with_loop_joints(), which runs for every env.
"""
import os
from collections.abc import Callable

import isaaclab.sim as sim_utils
from isaaclab.actuators import DelayedPDActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.sim.spawners.from_files import UrdfFileCfg
from isaaclab.sim.spawners.from_files.from_files import spawn_from_urdf
from isaaclab.utils import configclass

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
LEGGED_ROBOT_V3_URDF_PATH = os.path.join(
    CURRENT_DIR, "cad", "robot.SLDASM", "urdf", "robot.SLDASM.urdf"
)

_LOOP_JOINT_NAMES = {"close_loop_right", "close_loop_left"}


def _spawn_urdf_with_loop_joints(
    prim_path: str,
    cfg: "UrdfFileCfgWithLoops",
    translation=None,
    orientation=None,
    **kwargs,
):
    """Spawn URDF and immediately set close_loop_* joints to excludeFromArticulation=True."""
    import omni.usd
    from pxr import Usd, UsdPhysics

    prim = spawn_from_urdf(prim_path, cfg, translation, orientation, **kwargs)

    stage = omni.usd.get_context().get_stage()
    actual_path = str(prim.GetPath())
    print(f"[legged_v3_cfg] spawn actual_path = {actual_path}")

    found = []
    for joint_name in _LOOP_JOINT_NAMES:
        candidates = [
            f"{actual_path}/joints/{joint_name}",
            f"{actual_path}/{joint_name}",
        ]
        joint_prim = None
        for candidate in candidates:
            p = stage.GetPrimAtPath(candidate)
            if p.IsValid():
                joint_prim = p
                break

        if joint_prim is None:
            print(f"[legged_v3_cfg] WARNING: prim not found for {joint_name}, tried: {candidates}")
            continue

        # Write into session layer (highest priority, always writable) so the
        # override is not shadowed by the referenced cached USD layer.
        with Usd.EditContext(stage, stage.GetSessionLayer()):
            joint_api = UsdPhysics.Joint(joint_prim)
            joint_api.GetExcludeFromArticulationAttr().Set(True)
            drive = UsdPhysics.DriveAPI.Get(joint_prim, "angular")
            if drive:
                drive.GetStiffnessAttr().Set(0.0)
                drive.GetDampingAttr().Set(0.0)

        # Verify the value was written
        val = UsdPhysics.Joint(joint_prim).GetExcludeFromArticulationAttr().Get()
        print(f"[legged_v3_cfg] {joint_prim.GetPath()} excludeFromArticulation={val}")
        found.append(joint_name)

    if len(found) != 2:
        print(f"[legged_v3_cfg] WARNING: expected 2 loop joints, found {len(found)}: {found}")

    return prim


@configclass
class UrdfFileCfgWithLoops(UrdfFileCfg):
    """UrdfFileCfg that sets excludeFromArticulation on close_loop_* after spawn."""

    func: Callable = _spawn_urdf_with_loop_joints


LEGGED_ROBOT_V3_CFG = ArticulationCfg(
    spawn=UrdfFileCfgWithLoops(
        asset_path=LEGGED_ROBOT_V3_URDF_PATH,
        fix_base=False,
        merge_fixed_joints=True,
        self_collision=False,
        activate_contact_sensors=True,
        joint_drive=None,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            enable_gyroscopic_forces=True,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_depenetration_velocity=1.0,
        ),
        collision_props=sim_utils.CollisionPropertiesCfg(
            contact_offset=0.001,
            rest_offset=0.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
        ),
    ),

    soft_joint_pos_limit_factor=0.95,

    # INITIAL STATE
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.30),
        joint_pos={
            "pad_joint_right":      0.0,
            "pad_joint_left":       0.0,
            "thigh_joint_right_1":  0.0,
            "calf_joint_right_1":   0.0,
            "wheel_joint_right":    0.0,
            "thigh_joint_left_1":   0.0,
            "calf_joint_left_1":    0.0,
            "wheel_joint_left":     0.0,
            # thigh_*_2 omitted: passive, PhysX may auto-exclude if close_loop fix fails
            "calf_joint_right_2":   0.0,
            "calf_joint_left_2":    0.0,
            # close_loop_* excluded from articulation → not listed here
        },
        joint_vel={".*": 0.0},
    ),

    # ACTUATORS
    actuators={
        # pad: 100 N·m motor, ±20° range → Kp=100 gives 100 N·m at full range
        "pad": DelayedPDActuatorCfg(
            joint_names_expr=["pad_joint_right", "pad_joint_left"],
            effort_limit_sim=100.0,
            stiffness=100.0,
            damping=4.0,
            velocity_limit_sim=10.0,
        ),
        # thigh/calf: 20 N·m motor, Kp=80 → 24 N·m at 0.3 rad error (clipped to limit)
        # Kd = 2·√(80·0.01) ≈ 1.8 → use 2.0
        "thigh_active": DelayedPDActuatorCfg(
            joint_names_expr=["thigh_joint_right_1", "thigh_joint_left_1"],
            effort_limit_sim=20.0,
            stiffness=80.0,
            damping=2.0,
            velocity_limit_sim=50.0,
        ),
        "calf_active": DelayedPDActuatorCfg(
            joint_names_expr=["calf_joint_right_1", "calf_joint_left_1"],
            effort_limit_sim=20.0,
            stiffness=80.0,
            damping=2.0,
            velocity_limit_sim=50.0,
        ),
        # Wheels: velocity control → stiffness=0, damping = drive gain (N·m·s/rad)
        "wheel": DelayedPDActuatorCfg(
            joint_names_expr=["wheel_joint_right", "wheel_joint_left"],
            effort_limit_sim=20.0,
            stiffness=0.0,
            damping=5.0,
            velocity_limit_sim=100.0,
        ),
        # Passive chain: zero torque, tiny damping to prevent numerical jitter
        "calf_passive": DelayedPDActuatorCfg(
            joint_names_expr=["calf_joint_right_2", "calf_joint_left_2"],
            effort_limit_sim=0.0,
            stiffness=0.0,
            damping=0.05,
            velocity_limit_sim=50.0,
        ),
    },
)
