"""Add UsdPhysics.RigidBodyAPI + PhysxContactReportAPI to hip/knee links in robot_v5.usd.

Run with:
    ./isaaclab.sh -p source/isaaclab_assets/isaaclab_assets/legged_v3/add_contact_api_v5.py
"""
import os, sys, ctypes, glob

# ── Bootstrap pxr + PhysxSchema from isaacsim extscache ──────────────────────
try:
    import isaacsim
    _isaac_path = os.environ.get("ISAAC_PATH", "")
except ImportError:
    _isaac_path = ""

if _isaac_path:
    _usd_libs = glob.glob(os.path.join(_isaac_path, "extscache", "omni.usd.libs-*"))
    _physx_schema = glob.glob(os.path.join(_isaac_path, "extscache", "omni.usd.schema.physx-*"))
    for _p in _usd_libs + _physx_schema:
        if _p not in sys.path:
            sys.path.insert(0, _p)
        # preload shared libs
        _bin = os.path.join(_p, "bin")
        if os.path.isdir(_bin):
            for _so in sorted(glob.glob(os.path.join(_bin, "*.so"))):
                try:
                    ctypes.CDLL(_so, ctypes.RTLD_GLOBAL)
                except OSError:
                    pass

from pxr import Usd, UsdPhysics, PhysxSchema

USD_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "usd_file", "robot_v5.usd")

# Body names from Stage panel (USD prim names, not joint names)
HIP_KNEE_NAMES = {
    "hip", "hip_01", "hip_02", "hip_03",
    "knee", "knee_01", "knee_02", "knee_03",
}

stage = Usd.Stage.Open(USD_PATH)

modified = []
skipped = []

for prim in stage.Traverse():
    if prim.GetName() not in HIP_KNEE_NAMES:
        continue

    path = str(prim.GetPath())
    has_rigid = prim.HasAPI(UsdPhysics.RigidBodyAPI)
    has_contact = prim.HasAPI(PhysxSchema.PhysxContactReportAPI)

    if not has_rigid:
        UsdPhysics.RigidBodyAPI.Apply(prim)
        # kinematic = True để không bị gravity kéo (link passive, chỉ detect contact)
        rigid_api = UsdPhysics.RigidBodyAPI(prim)
        rigid_api.GetKinematicEnabledAttr().Set(True)

    if not has_contact:
        PhysxSchema.PhysxContactReportAPI.Apply(prim)

    if not has_rigid or not has_contact:
        modified.append(path)
        print(f"  [ADD]  {path}  (rigid_added={not has_rigid}, contact_added={not has_contact})")
    else:
        skipped.append(path)
        print(f"  [OK]   {path}  (already has both APIs)")

if modified:
    stage.GetRootLayer().Save()
    print(f"\nSaved {USD_PATH}")
    print(f"Modified {len(modified)} prims, skipped {len(skipped)} prims.")
else:
    print("\nNo changes needed.")
    print(f"Skipped {len(skipped)} prims (already configured).")

# Verify
print("\n--- Verification ---")
stage2 = Usd.Stage.Open(USD_PATH)
for prim in stage2.Traverse():
    if prim.GetName() in HIP_KNEE_NAMES:
        apis = prim.GetAppliedSchemas()
        print(f"  {prim.GetPath()}  schemas={apis}")
