"""Configuration for the Legged Robot V3 — loaded from URDF.

Robot structure (5-bar parallel linkage, 2 legs + wheels):
  base_link
  ├── left_thigh_link_A1 → left_shin_link_B1 → left_foot_link   [ACTIVE chain]
  ├── left_thigh_link_A2 → left_shin_link_B2                    [PASSIVE / mimic]
  ├── right_thigh_link_A1 → right_shin_link_B1 → right_foot_link [ACTIVE chain]
  └── right_thigh_link_A2 → right_shin_link_B2                   [PASSIVE / mimic]

Closed-loop: loop joints created programmatically after URDF spawn via
             _spawn_urdf_with_mimic_and_loops(). ExcludeFromArticulation=True
             so PhysX enforces them as separate constraints (not in the tree).
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
    CURRENT_DIR, "cad", "robot_urdf", "urdf", "robot.SLDASM.urdf"
)

# (secondary, primary) — secondary mimics primary at gear 1:1
_MIMIC_PAIRS = [
    ("right_hip_joint_A2", "right_hip_joint_A1"),
    ("left_hip_joint_A2",  "left_hip_joint_A1"),
]

# Loop-closing joints (created programmatically, excluded from articulation).
# pos0/rpy0: joint frame in body0 (shin_B2) local frame — from FK at q=0.
# localPos1/localRot1 = identity (joint anchored at foot_link origin).
_LOOP_JOINTS = [
    {
        "name":  "close_loop_left",
        "body0": "left_shin_link_B2",
        "body1": "left_foot_link",
        "pos0":  (0.08316, -0.17891, 0.05833),
        "rpy0":  (2.93061, -0.52798, -1.07098),
    },
    {
        "name":  "close_loop_right",
        "body0": "right_shin_link_B2",
        "body1": "right_foot_link",
        "pos0":  (0.08940, -0.12800, 0.01434),
        "rpy0":  (2.90234,  0.06987, -1.03781),
    },
]


def _rpy_to_quatf(roll: float, pitch: float, yaw: float):
    """URDF extrinsic-XYZ RPY → pxr.Gf.Quatf(w, x, y, z)."""
    import math
    from pxr import Gf
    cr, sr = math.cos(roll / 2),  math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cy, sy = math.cos(yaw / 2),   math.sin(yaw / 2)
    w =  cr * cp * cy + sr * sp * sy
    x =  sr * cp * cy - cr * sp * sy
    y =  cr * sp * cy + sr * cp * sy
    z =  cr * cp * sy - sr * sp * cy
    return Gf.Quatf(float(w), float(x), float(y), float(z))


def _spawn_urdf_with_mimic_and_loops(
    prim_path: str,
    cfg: "UrdfFileCfgExt",
    translation=None,
    orientation=None,
    **kwargs,
):
    """Spawn URDF, add mimic constraints on hip_A2, then create loop joints."""
    import omni.usd
    from pxr import Usd, UsdPhysics, PhysxSchema, Gf, Sdf

    prim = spawn_from_urdf(prim_path, cfg, translation, orientation, **kwargs)

    stage = omni.usd.get_context().get_stage()
    actual_path = str(prim.GetPath())
    print(f"[legged_v3_cfg] spawn actual_path = {actual_path}")

    def _find_prim(name: str):
        p = stage.GetPrimAtPath(f"{actual_path}/joints/{name}")
        if p.IsValid():
            return p
        robot_prim = stage.GetPrimAtPath(actual_path)
        for _p in Usd.PrimRange(robot_prim):
            if _p.GetName() == name:
                return _p
        return None

    # ── 1. PhysxMimicJointAPI: hip_A2 tracks hip_A1 ──────────────────────────
    with Usd.EditContext(stage, stage.GetSessionLayer()):
        for secondary_name, primary_name in _MIMIC_PAIRS:
            primary_prim   = _find_prim(primary_name)
            secondary_prim = _find_prim(secondary_name)
            if primary_prim is None or secondary_prim is None:
                print(f"[legged_v3_cfg] WARNING: mimic pair not found: {secondary_name} → {primary_name}")
                continue
            mimic = PhysxSchema.PhysxMimicJointAPI.Apply(secondary_prim, "rotX")
            mimic.GetGearingAttr().Set(1.0)
            mimic.GetOffsetAttr().Set(0.0)
            mimic.GetReferenceJointRel().AddTarget(primary_prim.GetPath())
            print(f"[legged_v3_cfg] mimic: {secondary_name} → {primary_name}")

    # ── 2. Create loop-closing revolute joints (excluded from articulation) ───
    with Usd.EditContext(stage, stage.GetSessionLayer()):
        for jdef in _LOOP_JOINTS:
            joint_path = f"{actual_path}/{jdef['name']}"
            # Remove stale prim from a prior spawn in the same stage session
            if stage.GetPrimAtPath(joint_path).IsValid():
                stage.RemovePrim(joint_path)

            j = UsdPhysics.RevoluteJoint.Define(stage, joint_path)
            j.CreateAxisAttr("Y")

            body0_path = Sdf.Path(f"{actual_path}/{jdef['body0']}")
            body1_path = Sdf.Path(f"{actual_path}/{jdef['body1']}")
            j.CreateBody0Rel().SetTargets([body0_path])
            j.CreateBody1Rel().SetTargets([body1_path])

            px, py, pz = jdef["pos0"]
            j.CreateLocalPos0Attr(Gf.Vec3f(px, py, pz))
            j.CreateLocalRot0Attr(_rpy_to_quatf(*jdef["rpy0"]))
            j.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
            j.CreateLocalRot1Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))

            # ±180° revolute limit (degrees in USD)
            j.CreateLowerLimitAttr(-180.0)
            j.CreateUpperLimitAttr(180.0)

            # Exclude from articulation tree → becomes a PhysX constraint
            UsdPhysics.Joint(j.GetPrim()).GetExcludeFromArticulationAttr().Set(True)
            print(f"[legged_v3_cfg] loop joint created: {joint_path} (excludeFromArticulation=True)")

    return prim


@configclass
class UrdfFileCfgExt(UrdfFileCfg):
    """UrdfFileCfg with mimic + loop-joint post-processing."""

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
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=1,
        ),
    ),

    soft_joint_pos_limit_factor=0.95,

    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.3),
        joint_pos={
            "left_hip_joint_A1":   0.0,
            "left_knee_joint_B1":  0.0,
            "left_wheel_joint":    0.0,
            "right_hip_joint_A1":  0.0,
            "right_knee_joint_B1": 0.0,
            "right_wheel_joint":   0.0,
            # A2/B2 passive — mimic + loop closure determines their angles
            "left_hip_joint_A2":   0.0,
            "left_knee_joint_B2":  0.0,
            "right_hip_joint_A2":  0.0,
            "right_knee_joint_B2": 0.0,
        },
        joint_vel={".*": 0.0},
    ),

    actuators={
        # ── Active: hip A1 (position-controlled by policy) ───────────────────
        "hip_active": DelayedPDActuatorCfg(
            joint_names_expr=["left_hip_joint_A1", "right_hip_joint_A1"],
            effort_limit_sim=20.0,
            stiffness=20.0,
            damping=1.0,
            velocity_limit_sim=50.0,
        ),
        # ── Mimic: hip A2 tracks hip A1 via PhysxMimicJointAPI ───────────────
        "hip_mimic": DelayedPDActuatorCfg(
            joint_names_expr=["left_hip_joint_A2", "right_hip_joint_A2"],
            effort_limit_sim=20.0,
            stiffness=0.0,
            damping=0.5,
            velocity_limit_sim=50.0,
        ),
        # ── Passive: knee B1 — constrained by loop closure geometry ──────────
        "knee_b1": DelayedPDActuatorCfg(
            joint_names_expr=["left_knee_joint_B1", "right_knee_joint_B1"],
            effort_limit_sim=20.0,
            stiffness=0.0,
            damping=2.0,
            velocity_limit_sim=50.0,
        ),
        # ── Passive: knee B2 — constrained by loop closure geometry ──────────
        "knee_b2": DelayedPDActuatorCfg(
            joint_names_expr=["left_knee_joint_B2", "right_knee_joint_B2"],
            effort_limit_sim=5.0,
            stiffness=0.0,
            damping=2.0,
            velocity_limit_sim=50.0,
        ),
        # ── Active: wheels (velocity-controlled by policy) ────────────────────
        "wheel": DelayedPDActuatorCfg(
            joint_names_expr=["left_wheel_joint", "right_wheel_joint"],
            effort_limit_sim=20.0,
            stiffness=0.0,
            damping=5.0,
            velocity_limit_sim=100.0,
        ),
    },
)
