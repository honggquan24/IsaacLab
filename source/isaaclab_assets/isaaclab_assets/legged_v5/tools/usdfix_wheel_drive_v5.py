"""Thêm UsdPhysics.DriveAPI (angular) vào 2 khớp bánh xe.

LỖI: USD export từ Onshape không tạo DriveAPI cho right/left_wheel_joint →
PhysX không có drive hoạt động → ImplicitActuator velocity-drive vô hiệu
(bánh không quay dù set target/damping). Đã verify: áp effort thuần thì bánh
quay tới velocity_limit (joint tự do), nhưng implicit damping không sinh lực.

FIX: apply DriveAPI("angular") với driveType=force, để PhysX có drive. Sau đó
ImplicitActuatorCfg(stiffness=0, damping=25) sẽ điều khiển vận tốc ổn định.

Chỉnh trực tiếp robot_v5_fixed.usd (đã gỡ bánh khỏi loop).

    ./isaaclab.sh -p source/isaaclab_assets/isaaclab_assets/legged_v3/add_wheel_drive_v5.py
"""
import os
from isaacsim import SimulationApp
app = SimulationApp({"headless": True})
from pxr import Usd, UsdPhysics

HERE = os.path.dirname(os.path.dirname(__file__))
USD = os.path.join(HERE, "usd_file", "robot_v5_fixed.usd")

OUT = open(os.path.join(HERE, "add_drive_log.txt"), "w")
_p = print
def log(*a):
    s = " ".join(str(x) for x in a); _p(s, flush=True); OUT.write(s+"\n"); OUT.flush()

stage = Usd.Stage.Open(USD)

WHEEL_JOINTS = ["right_wheel_joint", "left_wheel_joint"]

for prim in stage.Traverse():
    if prim.GetName() in WHEEL_JOINTS:
        existed = prim.HasAPI(UsdPhysics.DriveAPI, "angular")
        drive = UsdPhysics.DriveAPI.Apply(prim, "angular")
        drive.CreateTypeAttr().Set("force")
        drive.CreateDampingAttr().Set(25.0)       # velocity drive: chỉ damping
        drive.CreateStiffnessAttr().Set(0.0)
        drive.CreateMaxForceAttr().Set(300.0)
        drive.CreateTargetVelocityAttr().Set(0.0)
        log(f"[{prim.GetName()}] DriveAPI(angular) "
            f"{'đã có (ghi đè)' if existed else 'MỚI thêm'}: "
            f"type=force k=0 d=25 maxForce=300")

stage.GetRootLayer().Save()
log(f"\n✓ Đã lưu DriveAPI vào: {USD}")

OUT.close(); app.close()
