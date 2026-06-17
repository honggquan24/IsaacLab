"""Đặt giới hạn góc ±60° cho 4 khớp hip (active + mimic) trong robot_v5_fixed.usd.

CHỈ limit hip — KHÔNG đụng wheel_joint (phải -inf/inf để bánh quay tự do) và
close_loop_* (cơ cấu đóng vòng kín). Limit ở đơn vị ĐỘ (USD revolute dùng độ).

    ./isaaclab.sh -p source/isaaclab_assets/isaaclab_assets/legged_v3/add_hip_limits_v5.py
"""
import os
from isaacsim import SimulationApp
app = SimulationApp({"headless": True})
from pxr import Usd, UsdPhysics

HERE = os.path.dirname(os.path.dirname(__file__))
USD = os.path.join(HERE, "usd_file", "robot_v5_fixed.usd")

OUT = open(os.path.join(HERE, "add_hip_limits_log.txt"), "w")
_p = print
def log(*a):
    s = " ".join(str(x) for x in a); _p(s, flush=True); OUT.write(s+"\n"); OUT.flush()

HIP_JOINTS = ["right_hip_joint", "right_hip_joint_mimic",
              "left_hip_joint",  "left_hip_joint_mimic"]
LIMIT_DEG = 60.0

stage = Usd.Stage.Open(USD)
for prim in stage.Traverse():
    if prim.GetName() in HIP_JOINTS and prim.IsA(UsdPhysics.RevoluteJoint):
        rj = UsdPhysics.RevoluteJoint(prim)
        old = (rj.GetLowerLimitAttr().Get(), rj.GetUpperLimitAttr().Get())
        rj.CreateLowerLimitAttr().Set(-LIMIT_DEG)
        rj.CreateUpperLimitAttr().Set(+LIMIT_DEG)
        log(f"[{prim.GetName():<24}] limit {old} -> (-{LIMIT_DEG}, +{LIMIT_DEG}) deg")

stage.GetRootLayer().Save()
log(f"\n✓ Đã lưu limit hip vào: {USD}")
OUT.close(); app.close()
