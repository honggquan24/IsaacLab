# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Vá USD gốc của robot bipedal wheel (export từ Onshape) thành bản chạy được.

Bản export thô thiếu 4 thứ mà PhysX cần, script này vá tất cả trong một phiên
stage duy nhất và ghi ra ``usd/wheeled_biped_fixed.usd`` (không đụng bản gốc):

1. ``detach_wheels_from_loop`` — ``close_loop_linear`` neo vào body BÁNH XE nên
   khớp quay bánh bị kẹt trong vòng kín 5-bar và không quay tự do được. Chuyển
   body0 sang coupler (knee) và tính lại localPos0/localRot0 để giữ nguyên điểm
   neo trong world (hình học không đổi).
2. ``add_wheel_drive`` — Onshape không tạo ``UsdPhysics.DriveAPI`` cho khớp bánh
   nên ``ImplicitActuatorCfg`` velocity-drive không sinh mô-men.
3. ``limit_hip_joints`` — giới hạn ±60° cho 4 khớp hip (KHÔNG đụng khớp bánh:
   phải để -inf/inf, và không đụng ``close_loop_*``).
4. ``add_contact_reporting`` — thêm ``RigidBodyAPI`` + ``PhysxContactReportAPI``
   cho link hip/knee để ``ContactSensorCfg`` đọc được lực va chạm.

Mọi bước đều idempotent — chạy lại nhiều lần cho cùng kết quả.

Chạy:
    ./isaaclab.sh -p scripts/ute/wheeled_biped/prepare_usd.py
    ./isaaclab.sh -p scripts/ute/wheeled_biped/prepare_usd.py --verify
"""

import argparse
import os
import shutil

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--verify", action="store_true", help="In lại trạng thái USD sau khi vá.")
parser.add_argument("--hip-limit-deg", type=float, default=60.0, help="Giới hạn góc hip [deg].")
args = parser.parse_args()

from isaacsim import SimulationApp  # noqa: E402

simulation_app = SimulationApp({"headless": True})

from pxr import Gf, PhysxSchema, Usd, UsdGeom, UsdPhysics  # noqa: E402

PACKAGE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
    "source",
    "isaaclab_assets",
    "isaaclab_assets",
    "wheeled_biped",
)
SOURCE_USD = os.path.join(PACKAGE_DIR, "usd", "wheeled_biped.usd")
OUTPUT_USD = os.path.join(PACKAGE_DIR, "usd", "wheeled_biped_fixed.usd")

# (khớp vòng kín, body bánh hiện đang neo, coupler đích)
LOOP_JOINT_REMAPS = [
    ("right_close_loop_linear", "wheel_01", "knee_01"),
    ("left_close_loop_linear", "wheel", "knee"),
]
WHEEL_JOINTS = ["right_wheel_joint", "left_wheel_joint"]
HIP_JOINTS = ["right_hip_joint", "right_hip_joint_mimic", "left_hip_joint", "left_hip_joint_mimic"]
CONTACT_LINKS = {"hip", "hip_01", "hip_02", "hip_03", "knee", "knee_01", "knee_02", "knee_03"}


def find_prim(stage: Usd.Stage, name: str) -> Usd.Prim | None:
    """Trả về prim đầu tiên có tên ``name``, hoặc ``None`` nếu không tìm thấy."""
    for prim in stage.Traverse():
        if prim.GetName() == name:
            return prim
    return None


def joint_frame_matrix(pos: Gf.Vec3f, quat: Gf.Quatf) -> Gf.Matrix4d:
    """Dựng ma trận joint-frame theo quy ước row-vector: ``worldPt = jointPt * M``."""
    matrix = Gf.Matrix4d()
    matrix.SetRotate(Gf.Quatd(quat.GetReal(), Gf.Vec3d(*quat.GetImaginary())))
    matrix.SetTranslateOnly(Gf.Vec3d(pos[0], pos[1], pos[2]))
    return matrix


def detach_wheels_from_loop(stage: Usd.Stage) -> None:
    """Chuyển neo của khớp vòng kín từ bánh xe sang coupler, giữ nguyên điểm neo world."""
    xform_cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    for joint_name, old_body, new_body in LOOP_JOINT_REMAPS:
        joint_prim = find_prim(stage, joint_name)
        if joint_prim is None:
            print(f"  [SKIP] {joint_name}: không có trong USD")
            continue
        joint = UsdPhysics.Joint(joint_prim)
        new_prim = find_prim(stage, new_body)
        if [target.name for target in joint.GetBody0Rel().GetTargets()] == [new_body]:
            print(f"  [OK]   {joint_name}: body0 đã là {new_body}")
            continue

        world_old = xform_cache.GetLocalToWorldTransform(find_prim(stage, old_body))
        world_new = xform_cache.GetLocalToWorldTransform(new_prim)
        local_frame = joint_frame_matrix(joint.GetLocalPos0Attr().Get(), joint.GetLocalRot0Attr().Get())
        anchor_before = (local_frame * world_old).ExtractTranslation()

        remapped = local_frame * world_old * world_new.GetInverse()
        position = remapped.ExtractTranslation()
        rotation = remapped.ExtractRotationQuat()
        anchor_after = (remapped * world_new).ExtractTranslation()

        joint.GetBody0Rel().SetTargets([new_prim.GetPath()])
        joint.GetLocalPos0Attr().Set(Gf.Vec3f(*[float(v) for v in position]))
        joint.GetLocalRot0Attr().Set(
            Gf.Quatf(float(rotation.GetReal()), Gf.Vec3f(*[float(v) for v in rotation.GetImaginary()]))
        )
        print(f"  [FIX]  {joint_name}: body0 {old_body} -> {new_body}")
        print(f"         anchor world {anchor_before} -> {anchor_after}")


def add_wheel_drive(stage: Usd.Stage) -> None:
    """Thêm ``DriveAPI`` góc (chỉ damping) cho khớp bánh để velocity-drive hoạt động."""
    for joint_name in WHEEL_JOINTS:
        joint_prim = find_prim(stage, joint_name)
        if joint_prim is None:
            print(f"  [SKIP] {joint_name}: không có trong USD")
            continue
        existed = joint_prim.HasAPI(UsdPhysics.DriveAPI, "angular")
        drive = UsdPhysics.DriveAPI.Apply(joint_prim, "angular")
        drive.CreateTypeAttr().Set("force")
        drive.CreateStiffnessAttr().Set(0.0)
        drive.CreateDampingAttr().Set(25.0)
        drive.CreateMaxForceAttr().Set(300.0)
        drive.CreateTargetVelocityAttr().Set(0.0)
        print(
            f"  [{'OK' if existed else 'ADD'}]{'   ' if existed else '  '}{joint_name}: DriveAPI(angular) k=0 d=25 maxForce=300"  # noqa: E501
        )


def limit_hip_joints(stage: Usd.Stage, limit_deg: float) -> None:
    """Đặt giới hạn góc đối xứng ``±limit_deg`` cho các khớp hip."""
    for joint_name in HIP_JOINTS:
        joint_prim = find_prim(stage, joint_name)
        if joint_prim is None or not joint_prim.IsA(UsdPhysics.RevoluteJoint):
            print(f"  [SKIP] {joint_name}: không phải revolute joint")
            continue
        revolute = UsdPhysics.RevoluteJoint(joint_prim)
        before = (revolute.GetLowerLimitAttr().Get(), revolute.GetUpperLimitAttr().Get())
        revolute.CreateLowerLimitAttr().Set(-limit_deg)
        revolute.CreateUpperLimitAttr().Set(+limit_deg)
        print(f"  [SET]  {joint_name}: limit {before} -> (-{limit_deg}, +{limit_deg}) deg")


def add_contact_reporting(stage: Usd.Stage) -> None:
    """Thêm ``RigidBodyAPI`` + ``PhysxContactReportAPI`` cho link hip/knee."""
    for prim in stage.Traverse():
        if prim.GetName() not in CONTACT_LINKS:
            continue
        has_rigid = prim.HasAPI(UsdPhysics.RigidBodyAPI)
        has_contact = prim.HasAPI(PhysxSchema.PhysxContactReportAPI)
        if not has_rigid:
            UsdPhysics.RigidBodyAPI.Apply(prim)
            UsdPhysics.RigidBodyAPI(prim).GetKinematicEnabledAttr().Set(True)
        if not has_contact:
            PhysxSchema.PhysxContactReportAPI.Apply(prim)
        state = "OK" if has_rigid and has_contact else "ADD"
        print(f"  [{state}]{'   ' if state == 'OK' else '  '}{prim.GetPath()}")


def verify(stage: Usd.Stage) -> None:
    """In tóm tắt khớp và schema sau khi vá."""
    print("\n--- Khớp ---")
    for prim in stage.Traverse():
        if not prim.IsA(UsdPhysics.Joint):
            continue
        bodies = [target.name for target in UsdPhysics.Joint(prim).GetBody1Rel().GetTargets()]
        print(f"  {prim.GetName():<26} type={prim.GetTypeName():<22} body1={bodies}")
    print("\n--- Link có contact reporting ---")
    for prim in stage.Traverse():
        if prim.GetName() in CONTACT_LINKS:
            print(f"  {prim.GetPath()}  schemas={prim.GetAppliedSchemas()}")


def main() -> None:
    if not os.path.isfile(SOURCE_USD):
        raise FileNotFoundError(f"Không tìm thấy USD gốc: {SOURCE_USD}")
    shutil.copy(SOURCE_USD, OUTPUT_USD)
    stage = Usd.Stage.Open(OUTPUT_USD)

    print(f"USD gốc : {SOURCE_USD}")
    print(f"USD đích: {OUTPUT_USD}\n")
    print("[1/4] Gỡ bánh xe khỏi vòng kín 5-bar")
    detach_wheels_from_loop(stage)
    print("\n[2/4] Thêm DriveAPI cho khớp bánh")
    add_wheel_drive(stage)
    print(f"\n[3/4] Giới hạn khớp hip ±{args.hip_limit_deg}°")
    limit_hip_joints(stage, args.hip_limit_deg)
    print("\n[4/4] Bật contact reporting cho hip/knee")
    add_contact_reporting(stage)

    stage.GetRootLayer().Save()
    print(f"\n✓ Đã ghi: {OUTPUT_USD}")
    if args.verify:
        verify(Usd.Stage.Open(OUTPUT_USD))


main()
simulation_app.close()
