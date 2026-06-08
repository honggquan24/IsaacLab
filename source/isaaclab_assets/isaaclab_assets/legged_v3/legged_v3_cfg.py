"""Configuration for the Legged Robot V3 — loaded from URDF.

Robot structure (5-bar parallel linkage, 2 legs + wheels):
  base_link
  ├── left_thigh_link_A1 → left_shin_link_B1 → left_foot_link   [ACTIVE chain]
  ├── left_thigh_link_A2 → left_shin_link_B2                    [PASSIVE / mimic]
  ├── right_thigh_link_A1 → right_shin_link_B1 → right_foot_link [ACTIVE chain]
  └── right_thigh_link_A2 → right_shin_link_B2                   [PASSIVE / mimic]

Actuation:
  - hip_A1   : position-controlled by policy
  - hip_A2   : mimic (PhysxMimicJointAPI, tracks hip_A1 1:1)
  - knee_B1/B2: passive (stiffness=0, damped)
  - wheel    : velocity-controlled by policy

Loop closure:
  D6 joint (position spring drive) created programmatically after URDF spawn.
  Using spring drives (stiffness=100000 N/m) instead of rigid SphericalJoint so
  any small geometry gap at q=0 is handled gracefully without impulse explosion.
  ExcludeFromArticulation=True keeps the articulation tree intact.

  Wheel geometry (URDF):
    left_foot_link  cylinder: origin=(0, -0.0225, 0), length=0.045
      → B1 face at Y=0  (foot_link origin = wheel_joint location)
      → B2 face at Y=-0.045  ← loop-closure pin
    right_foot_link cylinder: origin=(0, +0.0225, 0), length=0.045
      → B1 face at Y=0
      → B2 face at Y=+0.045  ← loop-closure pin
"""
import os
from collections.abc import Callable

import isaaclab.sim as sim_utils
from isaaclab.actuators import IdealPDActuatorCfg, IdealPDActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.sim.spawners.from_files import UrdfFileCfg
from isaaclab.sim.spawners.from_files.from_files import spawn_from_urdf
from isaaclab.utils import configclass

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
LEGGED_ROBOT_V3_URDF_PATH = os.path.join(
    CURRENT_DIR, "cad", "robot_urdf", "urdf", "robot.SLDASM.urdf"
)

# (secondary, primary) — secondary mimics primary at gear 1:1
_MIMIC_PAIRS = [
    ("right_hip_joint_A2", "right_hip_joint_A1"),
    ("left_hip_joint_A2",  "left_hip_joint_A1"),
]

# Loop-closing D6 spring joints: shin_B2 tip ↔ B2-side axle face of foot_link.
# b2_axle_offset: position of the B2-side pin in foot_link's local frame.
#   Left : wheel cylinder origin=(0,-0.0225,0), B2 face at Y=-0.045
#   Right: wheel cylinder origin=(0,+0.0225,0), B2 face at Y=+0.045
_LOOP_JOINT_PAIRS = [
    ("close_loop_left",  "left_shin_link_B2",  "left_foot_link",  (0.0, -0.045, 0.0)),
    ("close_loop_right", "right_shin_link_B2", "right_foot_link", (0.0,  0.045, 0.0)),
]

# Spring parameters for compliant loop closure
# High enough to maintain constraint under gravity loads, no hard impulse at init
_LOOP_SPRING_STIFFNESS = 100_000.0   # N/m — 0.5mm error under 50N load
_LOOP_SPRING_DAMPING   =   2_000.0   # N·s/m — critically damp ~5Hz oscillation


def _find_prim_by_name(stage, root_path, name):
    from pxr import Usd
    p = stage.GetPrimAtPath(f"{root_path}/{name}")
    if p.IsValid():
        return p
    root = stage.GetPrimAtPath(root_path)
    if not root.IsValid():
        return None
    for prim in Usd.PrimRange(root):
        if prim.GetName() == name:
            return prim
    return None


def _spawn_urdf_with_mimic_and_loops(
    prim_path: str,
    cfg: "UrdfFileCfgExt",
    translation=None,
    orientation=None,
    **kwargs,
):
    """Spawn URDF, apply mimic on hip_A2, create compliant loop-closing joints."""
    import omni.usd
    from pxr import Usd, UsdGeom, UsdPhysics, PhysxSchema, Gf, Sdf

    prim = spawn_from_urdf(prim_path, cfg, translation, orientation, **kwargs)

    stage = omni.usd.get_context().get_stage()
    actual_path = str(prim.GetPath())
    print(f"[legged_v3_cfg] spawn actual_path = {actual_path}")

    # ── 1. PhysxMimicJointAPI: hip_A2 tracks hip_A1 1:1 ─────────────────────
    with Usd.EditContext(stage, stage.GetSessionLayer()):
        for secondary_name, primary_name in _MIMIC_PAIRS:
            pri = _find_prim_by_name(stage, f"{actual_path}/joints", primary_name) \
               or _find_prim_by_name(stage, actual_path, primary_name)
            sec = _find_prim_by_name(stage, f"{actual_path}/joints", secondary_name) \
               or _find_prim_by_name(stage, actual_path, secondary_name)
            if pri is None or sec is None:
                print(f"[legged_v3_cfg] WARNING: mimic pair not found: {secondary_name} → {primary_name}")
                continue
            mimic = PhysxSchema.PhysxMimicJointAPI.Apply(sec, "rotX")
            mimic.GetGearingAttr().Set(1.0)
            mimic.GetOffsetAttr().Set(0.0)
            mimic.GetReferenceJointRel().AddTarget(pri.GetPath())
            print(f"[legged_v3_cfg] mimic: {secondary_name} → {primary_name}")

    # ── 2. Compliant loop-closing joints (D6 + spring drive) ─────────────────
    # Uses spring drives instead of rigid constraints to tolerate geometry gap.
    # At init, spring force = stiffness × gap_distance (no impulse).
    with Usd.EditContext(stage, stage.GetSessionLayer()):
        for joint_name, body0_name, body1_name, b2_axle_offset in _LOOP_JOINT_PAIRS:
            body0_prim = _find_prim_by_name(stage, actual_path, body0_name)
            body1_prim = _find_prim_by_name(stage, actual_path, body1_name)

            if body0_prim is None or body1_prim is None:
                print(f"[legged_v3_cfg] WARNING: bodies not found for {joint_name} "
                      f"({body0_name}, {body1_name})")
                continue

            # World transforms at default time (initial pose, q=0)
            T0_world = UsdGeom.Xformable(body0_prim).ComputeLocalToWorldTransform(
                Usd.TimeCode.Default()
            )
            T1_world = UsdGeom.Xformable(body1_prim).ComputeLocalToWorldTransform(
                Usd.TimeCode.Default()
            )

            # Pin world position = B2-side axle face on foot_link (A-chain side)
            pin_world = T1_world.Transform(Gf.Vec3d(*b2_axle_offset))

            # localPos0: where pin_world lands in shin_B2's local frame (B-chain tip)
            pos0 = T0_world.GetInverse().Transform(pin_world)

            # localPos1: B2-axle face in foot_link local frame
            pos1 = Gf.Vec3d(*b2_axle_offset)

            print(f"[legged_v3_cfg] {joint_name}: "
                  f"localPos0=({pos0[0]:.4f},{pos0[1]:.4f},{pos0[2]:.4f})  "
                  f"localPos1=({pos1[0]:.4f},{pos1[1]:.4f},{pos1[2]:.4f})")

            joint_path = f"{actual_path}/{joint_name}"
            if stage.GetPrimAtPath(joint_path).IsValid():
                stage.RemovePrim(joint_path)

            # D6 joint: all rotational DOF free, translational DOF spring-driven
            # to target position = 0 (enforce coincidence of localPos0 and localPos1)
            j = UsdPhysics.Joint.Define(stage, joint_path)

            j.CreateBody0Rel().SetTargets([Sdf.Path(str(body0_prim.GetPath()))])
            j.CreateBody1Rel().SetTargets([Sdf.Path(str(body1_prim.GetPath()))])

            j.CreateLocalPos0Attr(Gf.Vec3f(float(pos0[0]), float(pos0[1]), float(pos0[2])))
            j.CreateLocalRot0Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
            j.CreateLocalPos1Attr(Gf.Vec3f(float(pos1[0]), float(pos1[1]), float(pos1[2])))
            j.CreateLocalRot1Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))

            # Spring drives on all 3 translational axes — target=0 means enforce coincidence
            for dof in ("transX", "transY", "transZ"):
                drive = UsdPhysics.DriveAPI.Apply(j.GetPrim(), dof)
                drive.CreateTypeAttr("force")
                drive.CreateTargetPositionAttr(0.0)
                drive.CreateStiffnessAttr(_LOOP_SPRING_STIFFNESS)
                drive.CreateDampingAttr(_LOOP_SPRING_DAMPING)

            # Exclude from articulation tree; PhysX enforces as a separate constraint
            UsdPhysics.Joint(j.GetPrim()).GetExcludeFromArticulationAttr().Set(True)
            print(f"[legged_v3_cfg] loop joint (D6 spring) created: {joint_path}")

    return prim


@configclass
class UrdfFileCfgExt(UrdfFileCfg):
    """UrdfFileCfg with post-spawn mimic + loop-joint processing."""

    func: Callable = _spawn_urdf_with_mimic_and_loops


LEGGED_ROBOT_V3_CFG = ArticulationCfg(
    spawn=UrdfFileCfgExt(
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
            max_depenetration_velocity=1.0,
        ),
        collision_props=sim_utils.CollisionPropertiesCfg(
            contact_offset=0.001,
            rest_offset=0.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=16,   # higher for loop closure stability
            solver_velocity_iteration_count=4,
        ),
    ),

    soft_joint_pos_limit_factor=0.95,

    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.40),
        joint_pos={
            "left_hip_joint_A1":   0.0,
            "left_knee_joint_B1":  0.0,
            "left_wheel_joint":    0.0,
            "right_hip_joint_A1":  0.0,
            "right_knee_joint_B1": 0.0,
            "right_wheel_joint":   0.0,
            "left_hip_joint_A2":   0.0,
            "left_knee_joint_B2":  0.0,
            "right_hip_joint_A2":  0.0,
            "right_knee_joint_B2": 0.0,
        },
        joint_vel={".*": 0.0},
    ),

    actuators={
        # ── Active: hip A1 — position control, stiffness drives stance maintenance
        "hip_active": IdealPDActuatorCfg(
            joint_names_expr=["left_hip_joint_A1", "right_hip_joint_A1"],
            effort_limit_sim=150.0,
            stiffness=20.0,
            damping=1.0,
            velocity_limit_sim=50.0,
        ),
        # ── Mimic: hip A2 tracks hip A1 via PhysxMimicJointAPI ───────────────
        "hip_mimic": IdealPDActuatorCfg(
            joint_names_expr=["left_hip_joint_A2", "right_hip_joint_A2"],
            effort_limit_sim=100.0,
            stiffness=0.0,
            damping=0.0,
            velocity_limit_sim=50.0,
        ),
        # ── Passive: knee B1 — loop closure provides geometric constraint ─────
        "knee": IdealPDActuatorCfg(
            joint_names_expr=["left_knee_joint_B1", "right_knee_joint_B1", "left_knee_joint_B2", "right_knee_joint_B2"],
            effort_limit_sim=200.0,
            stiffness=0.0,
            damping=0.5,
            velocity_limit_sim=50.0,
        ),
        # ── Active: wheels — IdealPD velocity control ─────────────────────────
        # stiffness=0: no position hold; damping=X: torque = X*(vel_target - vel_current)
        "wheel": IdealPDActuatorCfg(
            joint_names_expr=["left_wheel_joint", "right_wheel_joint"],
            effort_limit=12.0,
            stiffness=0.0,
            damping=1.0,
            velocity_limit=30.0,
        ),
    },
)
