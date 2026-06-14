"""Scan joint prims và fix PhysxMimicJointAPI gearing = -1.0 cho hip_mimic.

Mục đích:
  - PhysxMimicJointAPI (trong articulation) hoạt động đúng ở multi-env
  - excludeFromArticulation constraints KHÔNG được remap khi clone → dùng MimicJoint thay thế
  - gearing = -1.0 → hip_mimic quay ngược chiều Z so với hip chủ động

Run (dry-run, chỉ scan):
    ./isaaclab.sh -p .../fix_mimic_v5.py --headless

Run (apply fix vào USD):
    ./isaaclab.sh -p .../fix_mimic_v5.py --headless --apply
"""
import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--apply", action="store_true", help="Thực sự ghi PhysxMimicJointAPI vào USD")
AppLauncher.add_app_launcher_args(parser)
args, _ = parser.parse_known_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import omni.usd
from pxr import UsdPhysics, PhysxSchema, Usd

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.sim import SimulationContext
from isaaclab_assets.legged_v3.legged_v5_cfg import LEGGED_V5_CFG

# ── Khởi động sim ──────────────────────────────────────────────────────────
sim = SimulationContext(sim_utils.SimulationCfg(dt=1 / 200.0, device="cpu"))
sim_utils.GroundPlaneCfg().func("/World/ground", sim_utils.GroundPlaneCfg())

robot_cfg = LEGGED_V5_CFG.replace(prim_path="/World/Robot")
robot = Articulation(robot_cfg)
sim.reset()

# ── Scan stage ─────────────────────────────────────────────────────────────
stage = omni.usd.get_context().get_stage()
root = stage.GetPrimAtPath("/World/Robot")

SEP = "=" * 72
print(f"\n{SEP}")
print("JOINT SCAN — RevoluteJoint | PhysxMimicJointAPI | gearing")
print(SEP)

joint_paths = {}   # name → prim
for prim in Usd.PrimRange(root):
    ptype = prim.GetTypeName()
    if ptype not in ("PhysicsRevoluteJoint", "PhysicsPrismaticJoint",
                     "PhysicsFixedJoint", "PhysicsD6Joint", "PhysicsSphericalJoint"):
        continue
    path = str(prim.GetPath())
    has_mimic = prim.HasAPI(PhysxSchema.PhysxMimicJointAPI)
    name = prim.GetName()
    joint_paths[name] = prim

    print(f"\n  [{ptype}]  {path}")
    if has_mimic:
        for inst in PhysxSchema.PhysxMimicJointAPI.GetAll(prim):
            gear = inst.GetGearingAttr().Get()
            offset = inst.GetOffsetAttr().Get()
            refs = inst.GetReferenceJointRel().GetTargets()
            print(f"    MimicJointAPI gearing={gear}  offset={offset}")
            print(f"    referenceJoint → {refs}")
    else:
        print("    (no MimicJointAPI)")

print(f"\n{SEP}")

# ── Apply fix ──────────────────────────────────────────────────────────────
MIMIC_PAIRS = [
    # (slave_joint_name, master_joint_name, gearing)
    ("right_hip_joint_mimic", "right_hip_joint", -1.0),
    ("left_hip_joint_mimic",  "left_hip_joint",  -1.0),
]

if args.apply:
    print("\nAPPLYING PhysxMimicJointAPI ...")
    for slave_name, master_name, gearing in MIMIC_PAIRS:
        slave  = joint_paths.get(slave_name)
        master = joint_paths.get(master_name)
        if slave is None:
            print(f"  [WARN] joint '{slave_name}' not found in stage")
            continue
        if master is None:
            print(f"  [WARN] joint '{master_name}' not found in stage")
            continue

        # Apply (idempotent) với instance name "rotX" (revolute axis)
        api = PhysxSchema.PhysxMimicJointAPI.Apply(slave, "rotX")
        api.GetGearingAttr().Set(gearing)
        api.GetOffsetAttr().Set(0.0)
        api.GetReferenceJointRel().SetTargets([master.GetPath()])
        print(f"  OK  {slave_name}  gearing={gearing}  → {master_name}")

    # Lưu lại USD
    from isaaclab_assets.legged_v3.legged_v5_cfg import LEGGED_ROBOT_V5_USD_PATH
    stage.GetRootLayer().Export(LEGGED_ROBOT_V5_USD_PATH)
    print(f"\nSaved → {LEGGED_ROBOT_V5_USD_PATH}")
else:
    print("\n[DRY-RUN] Dùng --apply để ghi vào USD.")
    print("Pairs sẽ được fix:")
    for s, m, g in MIMIC_PAIRS:
        print(f"  {s}  gearing={g}  → {m}")

print(SEP + "\n")
simulation_app.close()
