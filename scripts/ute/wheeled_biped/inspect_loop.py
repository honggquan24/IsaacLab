# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""In chi tiết anchor của close_loop joints + world transform các body liên quan.
Dùng để tính lại localPose khi reparent loop khỏi bánh xe.

Chạy:
    ./isaaclab.sh -p source/isaaclab_assets/isaaclab_assets/legged_v3/inspect_loop_v5.py
"""

import os

from isaacsim import SimulationApp

app = SimulationApp({"headless": True})

from pxr import Usd, UsdGeom, UsdPhysics

USD_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "usd", "robot_v5.usd")
_p = print


def log(*a):
    s = " ".join(str(x) for x in a)
    _p(s, flush=True)


stage = Usd.Stage.Open(USD_PATH)
cache = UsdGeom.XformCache(Usd.TimeCode.Default())


def find_prim(name):
    for prim in stage.Traverse():
        if prim.GetName() == name:
            return prim
    return None


def world_xf(body_name):
    p = find_prim(body_name)
    if p is None:
        return None
    return cache.GetLocalToWorldTransform(p)


JOINTS = ["right_close_loop_linear", "left_close_loop_linear", "right_wheel_joint", "left_wheel_joint"]

for jn in JOINTS:
    jp = find_prim(jn)
    if jp is None:
        log(f"\n[{jn}] NOT FOUND")
        continue
    j = UsdPhysics.Joint(jp)
    b0 = j.GetBody0Rel().GetTargets()
    b1 = j.GetBody1Rel().GetTargets()
    lp0 = j.GetLocalPos0Attr().Get()
    lr0 = j.GetLocalRot0Attr().Get()
    lp1 = j.GetLocalPos1Attr().Get()
    lr1 = j.GetLocalRot1Attr().Get()
    log(f"\n[{jn}]")
    log(f"  body0 = {[t.name for t in b0]}   body1 = {[t.name for t in b1]}")
    log(f"  localPos0 = {lp0}   localRot0 = {lr0}")
    log(f"  localPos1 = {lp1}   localRot1 = {lr1}")

log("\n── World transforms (translation) các body ──")
for bn in [
    "base",
    "knee_01",
    "wheel_01",
    "knee",
    "wheel",
    "right_close_loop_proxy",
    "left_close_loop_proxy",
    "knee_02",
    "knee_03",
]:
    m = world_xf(bn)
    if m is None:
        log(f"  {bn}: <none>")
        continue
    t = m.ExtractTranslation()
    log(f"  {bn:<24} pos=({t[0]:+.4f}, {t[1]:+.4f}, {t[2]:+.4f})")

app.close()
