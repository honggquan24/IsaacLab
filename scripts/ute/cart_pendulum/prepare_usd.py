# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Vá USD con lắc trên xe đẩy (export từ Onshape) thành bản chạy được với Isaac Lab.

Đọc ``usd/cart_pendulum_base.usd``, ghi ra ``usd/cart_pendulum_cfg.usd`` (không đụng bản gốc).
Bốn việc, tất cả đều idempotent — chạy lại nhiều lần cho cùng kết quả:

1. ``reverse_joint_parents`` — Onshape ghi ``body0``/``body1`` theo thứ tự mate người vẽ chọn,
   nên bản export có ``Slider_1`` với body0=cart body1=rack và ``Revolute_1`` với
   body0=pendulum body1=cart, tức là CON làm CHA. Đảo lại thành rack → cart → pendulum,
   đảo kèm ``localPos``/``localRot`` để khung khớp trong world không đổi (hình học giữ nguyên,
   chỉ dấu của bậc tự do đảo về chiều thuận).
2. ``anchor_root_to_world`` — bản export neo rack vào ``/World/Plane`` (một quad 10 mm không
   có API vật lý nào) bằng D6 khoá cả 6 trục. Thay bằng ``UsdPhysics.FixedJoint`` nối rack với
   world; đây đúng thứ mà :func:`find_global_fixed_joint_prim` của Isaac Lab tìm để nhận ra
   articulation nền cố định, và không cần lôi theo cái Plane vào mỗi env.
3. ``lift_above_ground`` — mô hình CAD có đáy con lắc ở z = -0.212 m, tức nằm dưới mặt sàn.
   Nâng cả cụm lên để đáy cách sàn ``--clearance``. Nâng luôn cả khung world của FixedJoint
   để hai bên không đá nhau.
4. ``add_joint_drives`` — Onshape không tạo ``UsdPhysics.DriveAPI``, thiếu nó thì
   ``ImplicitActuatorCfg`` không sinh được lực. Thêm drive lực (stiffness/damping = 0) cho cả
   hai khớp; Isaac Lab ghi đè hệ số lúc chạy.

Chạy:
    ./isaaclab.sh -p scripts/ute/cart_pendulum/prepare_usd.py
    ./isaaclab.sh -p scripts/ute/cart_pendulum/prepare_usd.py --verify
"""

import argparse
import os
import shutil

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--verify", action="store_true", help="In lại trạng thái USD sau khi vá.")
parser.add_argument("--clearance", type=float, default=0.04, help="Khoảng hở giữa đáy con lắc và sàn [m].")
parser.add_argument(
    "--density",
    type=float,
    default=None,
    help="Ghi physics:density [kg/m^3] cho các thân. Bỏ trống thì để PhysX tự tính (mặc định 1000).",
)
args = parser.parse_args()

from isaacsim import SimulationApp  # noqa: E402

simulation_app = SimulationApp({"headless": True})

from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics  # noqa: E402

PACKAGE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
    "source",
    "isaaclab_assets",
    "isaaclab_assets",
    "cart_pendulum",
)
SOURCE_USD = os.path.join(PACKAGE_DIR, "usd", "cart_pendulum_base.usd")
OUTPUT_USD = os.path.join(PACKAGE_DIR, "usd", "cart_pendulum_cfg.usd")

ROBOT_PATH = "/World/pendulum_1"
ARTICULATION_PATH = f"{ROBOT_PATH}/pendulum_1"
BODIES = ["rack", "cart", "pendulum"]
# (khớp, cha đúng, con đúng) — bản export ghi ngược cặp này
JOINT_PARENTS = [
    ("Slider_1", "rack", "cart"),
    ("Revolute_1", "cart", "pendulum"),
]
# (tên khớp, tên trục của DriveAPI)
JOINT_DRIVES = [("Slider_1", "linear"), ("Revolute_1", "angular")]
CLUTTER_PATHS = ["/World/Plane", "/Environment", "/Render"]
ANCHOR_JOINT = "FixedJoint"
LEGACY_ANCHOR_JOINT = "D6Joint"


def joint_path(name: str) -> str:
    return f"{ARTICULATION_PATH}/{name}"


def reverse_joint_parents(stage: Usd.Stage) -> None:
    """Đảo body0/body1 để cha là thân đứng gần gốc hơn."""
    for name, parent, child in JOINT_PARENTS:
        prim = stage.GetPrimAtPath(joint_path(name))
        joint = UsdPhysics.Joint(prim)
        body0 = [str(t) for t in joint.GetBody0Rel().GetTargets()]
        body1 = [str(t) for t in joint.GetBody1Rel().GetTargets()]
        want0 = f"{ARTICULATION_PATH}/{parent}"
        want1 = f"{ARTICULATION_PATH}/{child}"
        if body0 == [want0] and body1 == [want1]:
            print(f"  [bỏ qua] {name}: đã đúng chiều {parent} -> {child}")
            continue
        if body0 != [want1] or body1 != [want0]:
            raise RuntimeError(f"{name}: body0={body0} body1={body1}, không khớp cặp {parent}/{child} để đảo")

        # đảo cả khung cục bộ, nếu không khớp sẽ nhảy chỗ
        pos0 = prim.GetAttribute("physics:localPos0").Get()
        pos1 = prim.GetAttribute("physics:localPos1").Get()
        rot0 = prim.GetAttribute("physics:localRot0").Get()
        rot1 = prim.GetAttribute("physics:localRot1").Get()
        joint.GetBody0Rel().SetTargets([Sdf.Path(want0)])
        joint.GetBody1Rel().SetTargets([Sdf.Path(want1)])
        prim.GetAttribute("physics:localPos0").Set(pos1)
        prim.GetAttribute("physics:localPos1").Set(pos0)
        prim.GetAttribute("physics:localRot0").Set(rot1)
        prim.GetAttribute("physics:localRot1").Set(rot0)
        print(f"  [đảo]    {name}: {parent} -> {child}")


def anchor_root_to_world(stage: Usd.Stage) -> None:
    """Thay D6 neo vào Plane bằng FixedJoint nối thẳng rack với world."""
    if stage.GetPrimAtPath(joint_path(ANCHOR_JOINT)):
        print(f"  [bỏ qua] {ANCHOR_JOINT}: đã có")
        return

    legacy = stage.GetPrimAtPath(joint_path(LEGACY_ANCHOR_JOINT))
    if legacy:
        local_pos = legacy.GetAttribute("physics:localPos0").Get()
        local_rot = legacy.GetAttribute("physics:localRot0").Get()
        stage.RemovePrim(legacy.GetPath())
    else:
        # không có D6 thì lấy thẳng tư thế world của rack làm điểm neo
        rack = stage.GetPrimAtPath(f"{ARTICULATION_PATH}/rack")
        matrix = UsdGeom.XformCache().GetLocalToWorldTransform(rack)
        local_pos = Gf.Vec3f(matrix.ExtractTranslation())
        quat = matrix.ExtractRotationQuat()
        local_rot = Gf.Quatf(quat.GetReal(), Gf.Vec3f(quat.GetImaginary()))

    fixed = UsdPhysics.FixedJoint.Define(stage, joint_path(ANCHOR_JOINT))
    # body0 để trống = world; đây là dấu hiệu Isaac Lab dùng để nhận ra nền cố định
    fixed.GetBody1Rel().SetTargets([Sdf.Path(f"{ARTICULATION_PATH}/rack")])
    fixed.GetLocalPos0Attr().Set(local_pos)
    fixed.GetLocalRot0Attr().Set(local_rot)
    fixed.GetLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
    fixed.GetLocalRot1Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    print(f"  [thêm]   {ANCHOR_JOINT}: world -> rack (bỏ {LEGACY_ANCHOR_JOINT})")


def remove_clutter(stage: Usd.Stage) -> None:
    """Xoá Plane, đèn và scope Render mà phiên GUI để lại."""
    for path in CLUTTER_PATHS:
        if stage.GetPrimAtPath(path):
            stage.RemovePrim(path)
            print(f"  [xoá]    {path}")


def lift_above_ground(stage: Usd.Stage, clearance: float) -> float:
    """Nâng cụm robot để đáy con lắc cao hơn sàn ``clearance`` mét."""
    robot = stage.GetPrimAtPath(ROBOT_PATH)
    xform = UsdGeom.Xformable(robot)
    translate_op = None
    for op in xform.GetOrderedXformOps():
        if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
            translate_op = op
            break
    if translate_op is None:
        translate_op = xform.AddTranslateOp()
        translate_op.Set(Gf.Vec3d(0.0, 0.0, 0.0))

    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
    lowest = min(
        cache.ComputeWorldBound(stage.GetPrimAtPath(f"{ARTICULATION_PATH}/{body}"))
        .ComputeAlignedRange()
        .GetMin()[2]
        for body in BODIES
    )
    delta = clearance - lowest
    if abs(delta) < 1e-6:
        print(f"  [bỏ qua] chiều cao: đáy đã ở z={lowest:.4f} m")
        return 0.0

    current = translate_op.Get()
    translate_op.Set(Gf.Vec3d(current[0], current[1], current[2] + delta))
    # khung world của FixedJoint phải đi theo, nếu không nó kéo rack về chỗ cũ
    anchor = stage.GetPrimAtPath(joint_path(ANCHOR_JOINT))
    if anchor:
        pos0 = anchor.GetAttribute("physics:localPos0").Get()
        anchor.GetAttribute("physics:localPos0").Set(Gf.Vec3f(pos0[0], pos0[1], pos0[2] + delta))
    print(f"  [nâng]   {delta:+.4f} m — đáy từ z={lowest:.4f} lên z={clearance:.4f}")
    return delta


def add_joint_drives(stage: Usd.Stage) -> None:
    """Thêm DriveAPI dạng lực cho hai khớp."""
    for name, axis in JOINT_DRIVES:
        prim = stage.GetPrimAtPath(joint_path(name))
        if prim.HasAPI(UsdPhysics.DriveAPI, axis):
            print(f"  [bỏ qua] {name}: đã có DriveAPI:{axis}")
            continue
        drive = UsdPhysics.DriveAPI.Apply(prim, axis)
        drive.CreateTypeAttr().Set("force")
        # hệ số để 0: Isaac Lab ghi đè theo ImplicitActuatorCfg lúc khởi tạo
        drive.CreateStiffnessAttr().Set(0.0)
        drive.CreateDampingAttr().Set(0.0)
        drive.CreateMaxForceAttr().Set(1.0e6)
        drive.CreateTargetPositionAttr().Set(0.0)
        drive.CreateTargetVelocityAttr().Set(0.0)
        print(f"  [thêm]   {name}: DriveAPI:{axis}")


def set_density(stage: Usd.Stage, density: float) -> None:
    """Ghi khối lượng riêng cho các thân để PhysX tính khối lượng theo thể tích thật."""
    for body in BODIES:
        prim = stage.GetPrimAtPath(f"{ARTICULATION_PATH}/{body}")
        mass_api = UsdPhysics.MassAPI.Apply(prim)
        mass_api.CreateDensityAttr().Set(density)
        print(f"  [đặt]    {body}: density={density} kg/m^3")


def report(stage: Usd.Stage) -> None:
    print("\n--- trạng thái sau khi vá ---")
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
    for prim in stage.Traverse():
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            print(f"articulation root : {prim.GetPath()}")
    for name, _, _ in JOINT_PARENTS:
        prim = stage.GetPrimAtPath(joint_path(name))
        joint = UsdPhysics.Joint(prim)
        low = prim.GetAttribute("physics:lowerLimit")
        high = prim.GetAttribute("physics:upperLimit")
        drives = [s for s in prim.GetAppliedSchemas() if "DriveAPI" in s]
        print(f"\n{name} ({prim.GetTypeName()})")
        print(f"  body0 (cha) : {[str(t) for t in joint.GetBody0Rel().GetTargets()]}")
        print(f"  body1 (con) : {[str(t) for t in joint.GetBody1Rel().GetTargets()]}")
        print(f"  axis        : {prim.GetAttribute('physics:axis').Get()}")
        limits = "không giới hạn" if not low.HasAuthoredValue() else f"[{low.Get():.4f}, {high.Get():.4f}]"
        print(f"  limit       : {limits}")
        print(f"  drive       : {drives if drives else 'KHÔNG CÓ'}")
    anchor = stage.GetPrimAtPath(joint_path(ANCHOR_JOINT))
    if anchor:
        joint = UsdPhysics.Joint(anchor)
        print(f"\n{ANCHOR_JOINT} ({anchor.GetTypeName()})")
        print(f"  body0 (cha) : {[str(t) for t in joint.GetBody0Rel().GetTargets()] or 'world'}")
        print(f"  body1 (con) : {[str(t) for t in joint.GetBody1Rel().GetTargets()]}")
    print("\nbounding box world:")
    for body in BODIES:
        rng = cache.ComputeWorldBound(stage.GetPrimAtPath(f"{ARTICULATION_PATH}/{body}")).ComputeAlignedRange()
        print(
            f"  {body:9} min={tuple(round(v, 4) for v in rng.GetMin())}"
            f"  max={tuple(round(v, 4) for v in rng.GetMax())}"
        )
    print("\nprim còn lại dưới /World:")
    for child in stage.GetPrimAtPath("/World").GetChildren():
        print(f"  {child.GetPath()} ({child.GetTypeName()})")


def main() -> None:
    if not os.path.exists(SOURCE_USD):
        raise FileNotFoundError(f"Không thấy USD gốc: {SOURCE_USD}")
    shutil.copyfile(SOURCE_USD, OUTPUT_USD)
    stage = Usd.Stage.Open(OUTPUT_USD)

    print(f"nguồn : {SOURCE_USD}")
    print(f"đích  : {OUTPUT_USD}\n")
    reverse_joint_parents(stage)
    anchor_root_to_world(stage)
    remove_clutter(stage)
    lift_above_ground(stage, args.clearance)
    add_joint_drives(stage)
    if args.density is not None:
        set_density(stage, args.density)

    stage.GetRootLayer().Save()
    print(f"\nĐã ghi {OUTPUT_USD}")
    if args.verify:
        report(stage)


if __name__ == "__main__":
    main()
    simulation_app.close()
