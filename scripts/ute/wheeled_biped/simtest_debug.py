# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Debug spawn V5: in toàn bộ thông tin USD stage, joints, bodies, mimic, orientation.

Run:
    ./isaaclab.sh -p source/isaaclab_assets/isaaclab_assets/legged_v3/debug_v5.py
    ./isaaclab.sh -p source/isaaclab_assets/isaaclab_assets/legged_v3/debug_v5.py --headless
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args, _ = parser.parse_known_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import omni.usd
from pxr import PhysxSchema, Usd, UsdPhysics

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.sim import SimulationContext

from isaaclab_assets.wheeled_biped.wheeled_biped_cfg import WHEELED_BIPED_CFG

SEP = "=" * 72


def quat_to_rpy_deg(w, x, y, z):
    """Quaternion (w,x,y,z) → roll/pitch/yaw degrees."""
    import math

    sinr = 2.0 * (w * x + y * z)
    cosr = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr, cosr)
    sinp = 2.0 * (w * y - z * x)
    sinp = max(-1.0, min(1.0, sinp))
    pitch = math.asin(sinp)
    siny = 2.0 * (w * z + x * y)
    cosy = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny, cosy)
    return roll * 57.296, pitch * 57.296, yaw * 57.296


def main():
    sim = SimulationContext(sim_utils.SimulationCfg(dt=1 / 200.0, device="cuda:0"))
    sim.set_camera_view(eye=(2.0, 2.0, 1.5), target=(0.0, 0.0, 0.3))

    sim_utils.GroundPlaneCfg().func("/World/ground", sim_utils.GroundPlaneCfg())

    robot_cfg = WHEELED_BIPED_CFG.replace(prim_path="/World/Robot")
    robot = Articulation(robot_cfg)

    sim.reset()

    stage = omni.usd.get_context().get_stage()
    root_prim = stage.GetPrimAtPath("/World/Robot")

    # ── 1. Cấu trúc prim ──────────────────────────────────────────────────────
    print(f"\n{SEP}")
    print("USD STAGE — tất cả prims dưới /World/Robot")
    print(SEP)
    for prim in Usd.PrimRange(root_prim):
        depth = len(str(prim.GetPath()).split("/")) - 3
        indent = "  " * depth
        apis = [a.split("(")[0] for a in [str(s) for s in prim.GetAppliedSchemas()]]
        apis_str = f"  [{', '.join(apis)}]" if apis else ""
        print(f"{indent}{prim.GetName()}  <{prim.GetTypeName()}>{apis_str}")

    # ── 2. Scan joints: RevoluteJoint + MimicJointAPI ─────────────────────────
    print(f"\n{SEP}")
    print("JOINTS — RevoluteJoint và MimicJointAPI")
    print(SEP)
    joint_types = {
        "PhysicsRevoluteJoint",
        "PhysicsPrismaticJoint",
        "PhysicsFixedJoint",
        "PhysicsD6Joint",
        "PhysicsSphericalJoint",
    }
    for prim in Usd.PrimRange(root_prim):
        if prim.GetTypeName() not in joint_types:
            continue
        path = str(prim.GetPath())
        has_mimic = prim.HasAPI(PhysxSchema.PhysxMimicJointAPI)
        axis_attr = prim.GetAttribute("physics:axis")
        axis = axis_attr.Get() if axis_attr.IsValid() else "?"
        print(f"\n  {prim.GetName()}  ({prim.GetTypeName()})  axis={axis}")
        print(f"    path: {path}")
        if has_mimic:
            for inst in PhysxSchema.PhysxMimicJointAPI.GetAll(prim):
                g = inst.GetGearingAttr().Get()
                o = inst.GetOffsetAttr().Get()
                refs = inst.GetReferenceJointRel().GetTargets()
                print(f"    MimicJointAPI  gearing={g}  offset={o}  ref={refs}")
        else:
            print("    (no MimicJointAPI)")

    # ── 3. Scan rigid bodies ───────────────────────────────────────────────────
    print(f"\n{SEP}")
    print("RIGID BODIES — prims có UsdPhysics.RigidBodyAPI")
    print(SEP)
    rb_count = 0
    for prim in Usd.PrimRange(root_prim):
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            print(f"  {prim.GetName()}  →  {prim.GetPath()}")
            rb_count += 1
    if rb_count == 0:
        print("  [WARN] Không tìm thấy rigid body nào! USD thiếu RigidBodyAPI.")

    # ── 4. Isaac Lab articulation info ────────────────────────────────────────
    print(f"\n{SEP}")
    print("ARTICULATION — joints và bodies từ Isaac Lab")
    print(SEP)
    print(f"  num_joints : {robot.num_joints}")
    print(f"  num_bodies : {robot.num_bodies}")
    print(f"  joint_names: {robot.data.joint_names}")
    print(f"  body_names : {robot.data.body_names}")

    # ── 5. Orientation ngay sau reset ─────────────────────────────────────────
    print(f"\n{SEP}")
    print("ROOT STATE — ngay sau sim.reset()")
    print(SEP)
    pos = robot.data.root_pos_w[0]
    quat = robot.data.root_quat_w[0]  # (w, x, y, z)
    w, x, y, z = quat[0].item(), quat[1].item(), quat[2].item(), quat[3].item()
    roll, pitch, yaw = quat_to_rpy_deg(w, x, y, z)
    print(f"  pos  = ({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})")
    print(f"  quat = (w={w:.4f}, x={x:.4f}, y={y:.4f}, z={z:.4f})")
    print(f"  RPY  = roll={roll:.1f}°  pitch={pitch:.1f}°  yaw={yaw:.1f}°")
    up_z = 1.0 - 2.0 * (x * x + y * y)
    print(f"  up_z = {up_z:.3f}  (1.0=thẳng đứng, 0.0=nằm ngang, -1.0=lộn ngược)")

    # ── 6. Joint positions sau reset ─────────────────────────────────────────
    print(f"\n{SEP}")
    print("JOINT POSITIONS — ngay sau sim.reset()")
    print(SEP)
    jpos = robot.data.joint_pos[0]
    for i, name in enumerate(robot.data.joint_names):
        print(f"  [{i:2d}] {name:35s} {jpos[i].item():+.4f} rad  ({jpos[i].item() * 57.296:+.1f}°)")

    print(f"\n{SEP}")
    print("GỢI Ý FIX ORIENTATION (nếu robot bị nghiêng):")
    print("  up_z ≈  0  và roll≈90°  → rot=(0.7071, 0.7071, 0, 0)   [X+90°]  ← đang dùng")
    print("  up_z ≈  0  và roll≈-90° → rot=(0.7071,-0.7071, 0, 0)   [X-90°]")
    print("  up_z ≈  0  và pitch≈90° → rot=(0.7071, 0, 0.7071, 0)   [Y+90°]")
    print("  up_z ≈  0  và pitch≈-90°→ rot=(0.7071, 0,-0.7071, 0)   [Y-90°]")
    print("  up_z ≈ -1  (lộn ngược)  → rot=(0,     1,     0,     0) [X180°]")
    print(SEP + "\n")

    # ── 7. In góc nghiêng liên tục ────────────────────────────────────────────
    import math

    print("\nChạy sim, in góc mỗi bước (Ctrl+C để dừng)...\n")
    print(f"{'step':>6}  {'h(m)':>6}  {'roll°':>7}  {'pitch°':>7}  {'yaw°':>7}  {'tilt°':>7}  {'up_z':>6}")
    print("-" * 60)

    step = 0
    while simulation_app.is_running():
        sim.step()
        robot.update(sim.cfg.dt)
        step += 1

        pos = robot.data.root_pos_w[0]
        quat = robot.data.root_quat_w[0]
        w, x, y, z = quat[0].item(), quat[1].item(), quat[2].item(), quat[3].item()
        up_z = max(-1.0, min(1.0, 1.0 - 2.0 * (x * x + y * y)))
        tilt = math.degrees(math.acos(up_z))
        roll, pitch, yaw = quat_to_rpy_deg(w, x, y, z)

        print(f"{step:>6}  {pos[2]:>6.3f}  {roll:>+7.1f}  {pitch:>+7.1f}  {yaw:>+7.1f}  {tilt:>7.2f}  {up_z:>+6.3f}")

    sim.clear_all_callbacks()
    simulation_app.close()


if __name__ == "__main__":
    main()
