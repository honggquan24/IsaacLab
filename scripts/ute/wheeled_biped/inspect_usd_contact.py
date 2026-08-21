# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Check USD prims for RigidBodyAPI and ContactReporterAPI.

Chạy bằng:
    ./isaaclab.sh -p source/isaaclab_assets/isaaclab_assets/legged_v3/check_usd_contact.py
"""

import os
import sys

from isaacsim import SimulationApp

app = SimulationApp({"headless": True})

from pxr import PhysxSchema, Usd, UsdPhysics

USD_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "usd", "robot_v5.usd")

stage = Usd.Stage.Open(USD_PATH)
if not stage:
    print(f"ERROR: cannot open {USD_PATH}")
    sys.exit(1)

print(f"\nStage: {USD_PATH}\n")
print(f"{'Prim path':<60} {'RigidBodyAPI':>14} {'ContactReporterAPI':>20}")
print("-" * 96)

for prim in stage.Traverse():
    has_rigid = prim.HasAPI(UsdPhysics.RigidBodyAPI)
    has_contact = prim.HasAPI(PhysxSchema.PhysxContactReportAPI)
    name = prim.GetName().lower()
    is_body = any(k in name for k in ("hip", "knee", "wheel", "base"))
    if has_rigid or has_contact or is_body:
        marker = "  <<<" if is_body else ""
        print(f"{str(prim.GetPath()):<60} {str(has_rigid):>14} {str(has_contact):>20}{marker}")

app.close()
