# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Dump joint topology của robot_v5.usd để soi vòng kín 5-bar.

Mục tiêu: tìm xem `wheel_joint` có phải revolute lá độc lập không, và
`close_loop_linear` đóng vòng giữa body nào — để xác nhận bánh xe có bị
khớp cứng động học với hip qua loop closure hay không.

Chạy:
    ./isaaclab.sh -p source/isaaclab_assets/isaaclab_assets/legged_v3/check_usd_joints.py
"""

import os
import sys

from isaacsim import SimulationApp

app = SimulationApp({"headless": True})

from pxr import Usd, UsdPhysics

_arg = [a for a in sys.argv[1:] if a.endswith(".usd")]
USD_PATH = _arg[0] if _arg else os.path.join(os.path.dirname(os.path.dirname(__file__)), "usd", "robot_v5.usd")
_builtin_print = print


def emit(*a):
    line = " ".join(str(x) for x in a)
    _builtin_print(line, flush=True)


# alias để giữ nguyên phần dưới
print = emit  # noqa

stage = Usd.Stage.Open(USD_PATH)
if not stage:
    print(f"ERROR: cannot open {USD_PATH}")
    sys.exit(1)


def short(target_paths):
    """Lấy tên body cuối từ relationship target."""
    if not target_paths:
        return "<world/base>"
    return target_paths[0].name


def joint_type(prim):
    if prim.IsA(UsdPhysics.RevoluteJoint):
        return "REVOLUTE"
    if prim.IsA(UsdPhysics.PrismaticJoint):
        return "PRISMATIC"
    if prim.IsA(UsdPhysics.FixedJoint):
        return "FIXED"
    if prim.IsA(UsdPhysics.SphericalJoint):
        return "SPHERICAL"
    if prim.IsA(UsdPhysics.Joint):
        return "D6/GENERIC"
    return "?"


print(f"\nStage: {USD_PATH}\n")
print(f"{'joint':<26} {'type':<11} {'parent(body0)':<22} {'child(body1)':<22} {'axis':<5} {'limit(lo/hi)'}")
print("-" * 110)

joints = []
for prim in stage.Traverse():
    if not prim.IsA(UsdPhysics.Joint):
        continue
    j = UsdPhysics.Joint(prim)

    b0 = short(j.GetBody0Rel().GetTargets())
    b1 = short(j.GetBody1Rel().GetTargets())

    axis = ""
    lo = hi = ""
    if prim.IsA(UsdPhysics.RevoluteJoint):
        rj = UsdPhysics.RevoluteJoint(prim)
        axis = rj.GetAxisAttr().Get() or ""
        lo = rj.GetLowerLimitAttr().Get()
        hi = rj.GetUpperLimitAttr().Get()
    elif prim.IsA(UsdPhysics.PrismaticJoint):
        pj = UsdPhysics.PrismaticJoint(prim)
        axis = pj.GetAxisAttr().Get() or ""
        lo = pj.GetLowerLimitAttr().Get()
        hi = pj.GetUpperLimitAttr().Get()

    lim = f"{lo}/{hi}" if lo != "" else "—"
    name = prim.GetName()
    joints.append((name, b0, b1))
    print(f"{name:<26} {joint_type(prim):<11} {b0:<22} {b1:<22} {str(axis):<5} {lim}")

# ── Phân tích chuỗi: ai là parent của wheel, loop closure nối đâu ─────────────
print("\n" + "=" * 110)
print("PHÂN TÍCH CHUỖI ĐỘNG HỌC (parent -> child)")
print("=" * 110)

# build child->parent map theo body
edges = {}  # child_body -> (joint, parent_body)
for name, b0, b1 in joints:
    edges.setdefault(b1, []).append((name, b0))


def trace_up(body, depth=0):
    if depth > 12 or body not in edges:
        return
    for jname, parent in edges[body]:
        print("  " * depth + f"{body}  <--[{jname}]--  {parent}")
        trace_up(parent, depth + 1)


for leaf in ("wheel", "wheel_01"):
    print(f"\n▶ Chuỗi từ {leaf} ngược về gốc:")
    trace_up(leaf)

print("\n▶ Các joint có 'close_loop' (loop closure):")
for name, b0, b1 in joints:
    if "close_loop" in name.lower():
        print(f"  {name}: {b0}  <->  {b1}")

print("\n▶ Các joint có 'wheel':")
for name, b0, b1 in joints:
    if "wheel" in name.lower():
        print(f"  {name}: parent={b0}  child={b1}")

app.close()
