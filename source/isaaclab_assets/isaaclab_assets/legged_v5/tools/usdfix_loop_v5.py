"""Phẫu thuật USD: gỡ bánh xe ra khỏi vòng kín 5-bar.

LỖI: `close_loop_linear` neo vào body BÁNH XE (wheel_01 / wheel) → khớp quay
bánh bị nhét vào vòng kín → bánh không quay tự do được.

FIX: chuyển body0 của close_loop_linear sang COUPLER (knee_01 / knee), tính lại
localPos0/localRot0 để GIỮ NGUYÊN điểm neo world (hình học không đổi). Sau đó
bánh xe trở thành revolute LÁ độc lập, drive vận tốc hoạt động bình thường.

Xuất: usd_file/robot_v5_fixed.usd  (KHÔNG ghi đè bản gốc)

Chạy:
    ./isaaclab.sh -p source/isaaclab_assets/isaaclab_assets/legged_v3/fix_loop_v5.py
"""
import os, shutil
from isaacsim import SimulationApp
app = SimulationApp({"headless": True})

from pxr import Usd, UsdGeom, UsdPhysics, Gf, Sdf

HERE = os.path.dirname(os.path.dirname(__file__))
SRC = os.path.join(HERE, "usd_file", "robot_v5.usd")
DST = os.path.join(HERE, "usd_file", "robot_v5_fixed.usd")

OUT = open(os.path.join(HERE, "fix_loop_log.txt"), "w")
_p = print
def log(*a):
    s = " ".join(str(x) for x in a)
    _p(s, flush=True); OUT.write(s + "\n"); OUT.flush()

# Bắt đầu từ bản copy để không đụng bản gốc
shutil.copy(SRC, DST)
stage = Usd.Stage.Open(DST)
cache = UsdGeom.XformCache(Usd.TimeCode.Default())

def find_prim(name):
    for p in stage.Traverse():
        if p.GetName() == name:
            return p
    return None

def mat_from_local(pos, quat):
    """Dựng ma trận joint-frame (row-vector): worldPt = jointPt * M."""
    m = Gf.Matrix4d()
    m.SetRotate(Gf.Quatd(quat.GetReal(), Gf.Vec3d(*quat.GetImaginary())))
    m.SetTranslateOnly(Gf.Vec3d(pos[0], pos[1], pos[2]))
    return m

# (joint, body bánh hiện tại, coupler đích)
FIXES = [
    ("right_close_loop_linear", "wheel_01", "knee_01"),
    ("left_close_loop_linear",  "wheel",    "knee"),
]

for jn, old_body, new_body in FIXES:
    jp = find_prim(jn)
    j = UsdPhysics.Joint(jp)

    old_prim = find_prim(old_body)
    new_prim = find_prim(new_body)
    W_old = cache.GetLocalToWorldTransform(old_prim)
    W_new = cache.GetLocalToWorldTransform(new_prim)

    lp0 = j.GetLocalPos0Attr().Get()
    lr0 = j.GetLocalRot0Attr().Get()

    # M_joint0 trong frame body cũ; world anchor = M_joint0 * W_old
    M_joint0 = mat_from_local(lp0, lr0)
    world_anchor_before = M_joint0 * W_old

    # frame mới relative coupler: M_new = M_joint0 * W_old * inverse(W_new)
    M_new = M_joint0 * W_old * W_new.GetInverse()
    new_pos = M_new.ExtractTranslation()
    new_quat = M_new.ExtractRotationQuat()  # GfQuatd

    # kiểm chứng: world anchor sau khi đổi
    world_anchor_after = M_new * W_new

    log(f"\n[{jn}]  body0: {old_body} -> {new_body}")
    log(f"  localPos0: {tuple(round(v,5) for v in lp0)} -> "
        f"({new_pos[0]:+.5f}, {new_pos[1]:+.5f}, {new_pos[2]:+.5f})")
    log(f"  anchor world TRƯỚC = {world_anchor_before.ExtractTranslation()}")
    log(f"  anchor world SAU   = {world_anchor_after.ExtractTranslation()}")

    # Ghi lại body0 + localPose mới
    j.GetBody0Rel().SetTargets([new_prim.GetPath()])
    j.GetLocalPos0Attr().Set(Gf.Vec3f(float(new_pos[0]), float(new_pos[1]), float(new_pos[2])))
    nq = Gf.Quatf(float(new_quat.GetReal()),
                  Gf.Vec3f(*[float(x) for x in new_quat.GetImaginary()]))
    j.GetLocalRot0Attr().Set(nq)

stage.GetRootLayer().Save()
log(f"\n✓ Đã xuất: {DST}")

OUT.close()
app.close()
