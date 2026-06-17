"""Xác định euler component nào của rpy_alignment = nghiêng DỌC (chạy) vs NGANG (ngã).

q_ref = init_state.rot = (0.7071,0.7071,0,0) (90° quanh X, robot Y-up).
Robot chạy dọc world X → lean tăng tốc = xoay quanh world Y (trục bánh).
Tip ngang (ngã) = xoay quanh world X.

Toán quaternion thuần (w,x,y,z), euler XYZ giống isaaclab.utils.math — KHÔNG import isaac.
"""
import math


def qmul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return (
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    )


def euler_xyz(q):
    w, x, y, z = q
    roll  = math.atan2(2*(w*x + y*z), 1 - 2*(x*x + y*y))
    pitch = math.asin(max(-1.0, min(1.0, 2*(w*y - z*x))))
    yaw   = math.atan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))
    return roll, pitch, yaw


def qaxis(axis, deg):
    a = math.radians(deg) / 2
    s, c = math.sin(a), math.cos(a)
    return {"x": (c, s, 0, 0), "y": (c, 0, s, 0), "z": (c, 0, 0, s)}[axis]


q_ref = (0.7071, 0.7071, 0.0, 0.0)
q_ref_inv = (q_ref[0], -q_ref[1], -q_ref[2], -q_ref[3])


def report(label, q_cur):
    q_rel = qmul(q_ref_inv, q_cur)
    r, p, yw = euler_xyz(q_rel)
    print(f"{label:<38} roll={math.degrees(r):+7.2f}  pitch={math.degrees(p):+7.2f}  yaw={math.degrees(yw):+7.2f}")


print()
report("upright (default)",                  q_ref)
report("LEAN FWD 20 (worldY) -> CHAY",       qmul(qaxis("y", 20), q_ref))
report("LEAN BACK 20 (worldY)",              qmul(qaxis("y", -20), q_ref))
report("TIP RIGHT 20 (worldX) -> NGA",       qmul(qaxis("x", 20), q_ref))
report("TIP LEFT 20 (worldX)",               qmul(qaxis("x", -20), q_ref))
print()
