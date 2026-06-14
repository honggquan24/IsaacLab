"""Soi mass/inertia body bánh + drive/limit/armature khớp bánh, trên CẢ 2 usd."""
import os, sys
from isaacsim import SimulationApp
app = SimulationApp({"headless": True})
from pxr import Usd, UsdPhysics, PhysxSchema

HERE = os.path.dirname(os.path.dirname(__file__))
OUT = open(os.path.join(HERE, "mass_inspect.txt"), "w")
_p = print
def log(*a):
    s = " ".join(str(x) for x in a); _p(s, flush=True); OUT.write(s+"\n"); OUT.flush()

for fn in ["robot_v5.usd", "robot_v5_fixed.usd"]:
    path = os.path.join(HERE, "usd_file", fn)
    stage = Usd.Stage.Open(path)
    log(f"\n================ {fn} ================")
    for prim in stage.Traverse():
        name = prim.GetName()
        if name in ("wheel", "wheel_01", "knee_01", "knee"):
            mass = None; inertia = None
            if prim.HasAPI(UsdPhysics.MassAPI):
                m = UsdPhysics.MassAPI(prim)
                mass = m.GetMassAttr().Get()
                inertia = m.GetDiagonalInertiaAttr().Get()
            log(f"  BODY {name:<10} mass={mass}  diagInertia={inertia}")
        if "wheel_joint" in name:
            log(f"  JOINT {name}")
            for drv_axis in ("angular", "linear", "rotX", "transX"):
                if prim.HasAPI(UsdPhysics.DriveAPI, drv_axis):
                    d = UsdPhysics.DriveAPI(prim, drv_axis)
                    log(f"     DriveAPI[{drv_axis}] stiff={d.GetStiffnessAttr().Get()} "
                        f"damp={d.GetDampingAttr().Get()} targetVel={d.GetTargetVelocityAttr().Get()} "
                        f"maxForce={d.GetMaxForceAttr().Get()}")
            if prim.HasAPI(PhysxSchema.PhysxJointAPI):
                pj = PhysxSchema.PhysxJointAPI(prim)
                log(f"     PhysxJoint: maxJointVel={pj.GetMaxJointVelocityAttr().Get()} "
                    f"jointFriction={pj.GetJointFrictionAttr().Get()} "
                    f"armature={pj.GetArmatureAttr().Get()}")
            if prim.IsA(UsdPhysics.RevoluteJoint):
                rj = UsdPhysics.RevoluteJoint(prim)
                log(f"     Revolute limit lo/hi = {rj.GetLowerLimitAttr().Get()}/{rj.GetUpperLimitAttr().Get()}")

OUT.close(); app.close()
