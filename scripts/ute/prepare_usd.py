# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Vá USD export từ Onshape thành bản chạy được với Isaac Lab.

Dùng chung cho mọi robot trong dự án — họ con lắc trên xe đẩy (đơn/kép/ba) và xe hai bánh tự
cân bằng: script tự dò cấu trúc trong stage chứ không ghim sẵn tên prim, nên thêm một khâu vào
CAD cũng không phải sửa gì ở đây.

Hai kiểu robot, khác nhau ở chỗ có neo xuống world hay không:

* **nền cố định** (con lắc): một thân được cố định xuống world, script tự dò khớp neo;
* **thân nổi** (xe cân bằng): chạy với ``--floating-base --base-body <thân>``, script bỏ qua
  bước neo và bước nâng khỏi sàn.

Đọc ``usd/<package>_base.usd``, ghi ra ``usd/<package>_cfg.usd`` (không đụng bản gốc). Bốn
việc, tất cả đều idempotent — chạy lại nhiều lần cho cùng kết quả:

1. ``orient_joint_tree`` — Onshape ghi ``body0``/``body1`` theo thứ tự mate người vẽ chọn nên
   bản export hay có CON làm CHA (con lắc đơn: cart là cha của rack). Script dò cây khớp bằng
   BFS từ thân gốc rồi đảo những khớp ngược chiều, đảo kèm ``localPos``/``localRot`` để khung
   khớp trong world không đổi — hình học giữ nguyên, chỉ dấu của bậc tự do về chiều thuận.
2. ``anchor_root_to_world`` — bản export neo thân gốc vào ``/World/Plane`` (một quad 10 mm
   không có API vật lý nào) bằng D6 khoá cả 6 trục. Thay bằng ``UsdPhysics.FixedJoint`` nối
   thẳng world; đây đúng thứ mà :func:`find_global_fixed_joint_prim` của Isaac Lab tìm để
   nhận ra articulation nền cố định, và không phải nhân bản cái Plane vào từng env.
3. ``lift_above_ground`` — mô hình CAD có đáy con lắc nằm dưới mặt sàn. Nâng cả cụm lên cho
   đáy cách sàn ``--clearance``, nâng luôn khung world của FixedJoint để hai bên không đá nhau.
4. ``add_joint_drives`` — Onshape không tạo ``UsdPhysics.DriveAPI``, thiếu nó thì
   ``ImplicitActuatorCfg`` không sinh được lực. Thêm drive lực cho mọi khớp động; Isaac Lab
   ghi đè hệ số lúc chạy.
5. ``limit_joint_velocity`` — đặt ``physxJoint:maxJointVelocity`` cho từng khớp. Chuỗi càng
   dài thì đầu mút càng dễ đạt tốc độ lớn trong một bước sim, solver không hội tụ và khớp bị
   giãn hoặc văng; chặn ở mức DOF rẻ hơn nhiều so với giảm ``sim.dt``.

Đặt tên trong Onshape
---------------------
Code trong ``isaaclab_assets`` tra khớp theo tên, mà Isaac Sim lấy tên khớp **từ tên mate**.
Vì vậy mate phải đặt là ``Slider_1`` cho khớp trượt và ``Revolute_1``, ``Revolute_2``, ...
cho các khâu con lắc tính từ xe ra. Tên thân thì tuỳ.

Chạy:
    ./isaaclab.sh -p scripts/ute/prepare_usd.py --package cart_pendulum --verify
    ./isaaclab.sh -p scripts/ute/prepare_usd.py --package cart_pendulum_double
    ./isaaclab.sh -p scripts/ute/prepare_usd.py --package cart_pendulum_triple
    ./isaaclab.sh -p scripts/ute/prepare_usd.py --package balance_car \
        --floating-base --base-body Group_1 --max-angular-velocity 40 --verify
"""

import argparse
import os
import shutil
from collections import deque

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument(
    "--package",
    default="cart_pendulum",
    help="Tên package trong isaaclab_assets, cũng là tiền tố của file USD.",
)
parser.add_argument("--input", default=None, help="Ghi đè đường dẫn USD gốc.")
parser.add_argument("--output", default=None, help="Ghi đè đường dẫn USD đích.")
parser.add_argument("--verify", action="store_true", help="In lại trạng thái USD sau khi vá.")
parser.add_argument("--clearance", type=float, default=0.04, help="Khoảng hở giữa đáy con lắc và sàn [m].")
parser.add_argument("--base-body", default=None, help="Chỉ định thân gốc nếu script dò không ra.")
parser.add_argument(
    "--floating-base",
    action="store_true",
    help=(
        "Robot KHÔNG neo xuống world (xe cân bằng, robot chân...). Bỏ bước neo FixedJoint và bỏ"
        " bước nâng khỏi sàn — với thân nổi thì Isaac Lab ghi đè tư thế gốc bằng init_state.pos,"
        " nâng trong USD chỉ tạo ảo giác."
    ),
)
parser.add_argument(
    "--max-angular-velocity",
    type=float,
    default=15.0,
    help="Trần tốc độ cho khớp quay [rad/s]. 0 = không đặt.",
)
parser.add_argument(
    "--max-linear-velocity",
    type=float,
    default=20.0,
    help="Trần tốc độ cho khớp trượt [m/s]. 0 = không đặt.",
)
parser.add_argument(
    "--density",
    type=float,
    default=None,
    help="Ghi physics:density [kg/m^3] cho các thân. Bỏ trống thì để PhysX tự tính (mặc định 1000).",
)
args = parser.parse_args()

# Script chỉ đụng tới `pxr`, không gọi API nào của Kit. Nếu `pxr` đã import được sẵn (chạy
# trong môi trường có USD trên PYTHONPATH) thì bỏ hẳn việc khởi động Isaac Sim — tiết kiệm
# ~8 giây boot và một đống warning lúc tắt. Chạy bằng `./isaaclab.sh -p` thì `pxr` chưa sẵn,
# lúc đó mới boot như cũ.
simulation_app = None
try:
    from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdPhysics  # noqa: E402
except ImportError:
    from isaacsim import SimulationApp  # noqa: E402

    simulation_app = SimulationApp({"headless": True})

    from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdPhysics  # noqa: E402, F811

# script nằm ở <repo>/scripts/ute/prepare_usd.py — leo 3 cấp là tới gốc repo. Đừng đếm tay:
# lần chuyển script từ scripts/ute/cart_pendulum/ ra scripts/ute/ đã làm sai số cấp một lần rồi.
REPO_DIR = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
PACKAGE_DIR = os.path.join(REPO_DIR, "source", "isaaclab_assets", "isaaclab_assets", args.package)
SOURCE_USD = args.input or os.path.join(PACKAGE_DIR, "usd", f"{args.package}_base.usd")
OUTPUT_USD = args.output or os.path.join(PACKAGE_DIR, "usd", f"{args.package}_cfg.usd")

MOVABLE_JOINT_TYPES = ("PhysicsPrismaticJoint", "PhysicsRevoluteJoint")
JOINT_TYPES = MOVABLE_JOINT_TYPES + ("PhysicsFixedJoint", "PhysicsJoint", "PhysicsD6Joint")
# tên trục của DriveAPI theo loại khớp
DRIVE_AXIS = {"PhysicsPrismaticJoint": "linear", "PhysicsRevoluteJoint": "angular"}
CLUTTER_PATHS = ["/Environment", "/Render"]
ANCHOR_JOINT_NAME = "FixedJoint"


class Model:
    """Những gì dò được từ stage: thân gốc, các thân, các khớp, khớp neo."""

    def __init__(self, stage: Usd.Stage):
        self.stage = stage
        self.root = self._find_articulation_root()
        self.bodies = [
            prim.GetPath().pathString for prim in Usd.PrimRange(self.root) if prim.HasAPI(UsdPhysics.RigidBodyAPI)
        ]
        self.joints = [prim for prim in Usd.PrimRange(self.root) if prim.GetTypeName() in JOINT_TYPES]
        self._check_physics()
        self.anchor, self.base = self._find_anchor()

    def _check_physics(self) -> None:
        """Bắt sớm cái lỗi export hay gặp nhất: file chỉ có hình, không có vật lý.

        Không có hai thứ này thì script bó tay — nó vá được chiều cha-con và thêm drive, chứ
        không dựng ra được thân cứng hay khớp: trục quay của khớp chỉ Onshape mới biết.
        """
        meshes = [p for p in Usd.PrimRange(self.root, Usd.TraverseInstanceProxies()) if p.IsA(UsdGeom.Mesh)]
        if not self.bodies:
            raise RuntimeError(
                f"USD không có thân cứng nào (thiếu UsdPhysics.RigidBodyAPI), dù tìm thấy {len(meshes)} mesh."
                " Đây là file CHỈ CÓ HÌNH, chưa có vật lý — export lại từ Onshape với phần khớp/mate,"
                " đừng dùng bản flatten hay bản save lại từ viewport."
            )
        if not [p for p in self.joints if p.GetTypeName() in MOVABLE_JOINT_TYPES]:
            raise RuntimeError(
                f"USD có {len(self.bodies)} thân cứng nhưng KHÔNG có khớp động nào."
                " Onshape sinh khớp từ mate, nên bản export này thiếu mate hoặc bị bỏ khi export."
                " Script không dựng khớp thay được: trục quay chỉ bên CAD mới biết."
            )

    def _find_articulation_root(self) -> Usd.Prim:
        for prim in self.stage.Traverse():
            if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
                return prim
        raise RuntimeError("Không thấy prim nào có ArticulationRootAPI trong stage.")

    @staticmethod
    def targets(prim: Usd.Prim, which: int) -> list[str]:
        joint = UsdPhysics.Joint(prim)
        rel = joint.GetBody0Rel() if which == 0 else joint.GetBody1Rel()
        return [str(t) for t in rel.GetTargets()]

    def _find_anchor(self) -> tuple[Usd.Prim | None, str]:
        """Khớp neo là khớp có một đầu không phải thân của articulation (hoặc để trống)."""
        for prim in self.joints:
            body0 = self.targets(prim, 0)
            body1 = self.targets(prim, 1)
            outside0 = not body0 or body0[0] not in self.bodies
            outside1 = not body1 or body1[0] not in self.bodies
            if outside0 and not outside1:
                return prim, body1[0]
            if outside1 and not outside0:
                return prim, body0[0]
        if args.base_body:
            base = f"{self.root.GetPath()}/{args.base_body}"
            if base not in self.bodies:
                raise RuntimeError(f"--base-body '{args.base_body}' không phải thân trong articulation.")
            return None, base
        if args.floating_base:
            # thân nổi thì không có khớp neo để dò; lấy thân nối vào NHIỀU khớp nhất làm gốc —
            # với xe hai bánh đó là khung xe (2 khớp) chứ không phải bánh (1 khớp)
            degree: dict[str, int] = {body: 0 for body in self.bodies}
            for prim in self.joints:
                for which in (0, 1):
                    for target in self.targets(prim, which):
                        if target in degree:
                            degree[target] += 1
            base = max(degree, key=lambda b: degree[b])
            print(f"  [dò]     thân gốc = {base.rsplit('/', 1)[-1]} ({degree[base]} khớp)")
            return None, base
        raise RuntimeError(
            "Không dò được khớp neo xuống world. Trong Onshape phải cố định một thân (ví dụ ray)"
            " xuống mặt phẳng gốc, hoặc chạy lại với --base-body <tên thân>,"
            " hoặc --floating-base nếu robot vốn không neo (xe cân bằng, robot chân)."
        )

    def movable_joints(self) -> list[Usd.Prim]:
        return [p for p in self.joints if p.IsValid() and p.GetTypeName() in MOVABLE_JOINT_TYPES and p != self.anchor]

    def adjacency(self) -> dict[str, list[tuple[Usd.Prim, str]]]:
        graph: dict[str, list[tuple[Usd.Prim, str]]] = {body: [] for body in self.bodies}
        for prim in self.movable_joints():
            body0 = self.targets(prim, 0)
            body1 = self.targets(prim, 1)
            if not body0 or not body1:
                continue
            graph[body0[0]].append((prim, body1[0]))
            graph[body1[0]].append((prim, body0[0]))
        return graph


def short(path: str) -> str:
    return path.rsplit("/", 1)[-1]


def orient_joint_tree(model: Model) -> None:
    """Duyệt BFS từ thân gốc, đảo những khớp đang ghi con làm cha."""
    graph = model.adjacency()
    seen = {model.base}
    queue = deque([model.base])
    chain = [short(model.base)]
    while queue:
        parent = queue.popleft()
        for prim, child in graph[parent]:
            if child in seen:
                continue
            seen.add(child)
            queue.append(child)
            chain.append(f"{prim.GetName()} -> {short(child)}")
            if model.targets(prim, 0) == [parent]:
                print(f"  [bỏ qua] {prim.GetName()}: đã đúng chiều {short(parent)} -> {short(child)}")
                continue
            joint = UsdPhysics.Joint(prim)
            pos0 = prim.GetAttribute("physics:localPos0").Get()
            pos1 = prim.GetAttribute("physics:localPos1").Get()
            rot0 = prim.GetAttribute("physics:localRot0").Get()
            rot1 = prim.GetAttribute("physics:localRot1").Get()
            joint.GetBody0Rel().SetTargets([Sdf.Path(parent)])
            joint.GetBody1Rel().SetTargets([Sdf.Path(child)])
            prim.GetAttribute("physics:localPos0").Set(pos1)
            prim.GetAttribute("physics:localPos1").Set(pos0)
            prim.GetAttribute("physics:localRot0").Set(rot1)
            prim.GetAttribute("physics:localRot1").Set(rot0)
            print(f"  [đảo]    {prim.GetName()}: {short(parent)} -> {short(child)}")

    missing = set(model.bodies) - seen
    if missing:
        raise RuntimeError(f"Các thân này không nối được vào cây khớp: {sorted(short(m) for m in missing)}")
    print(f"  chuỗi   : {' | '.join(chain)}")


def anchor_root_to_world(model: Model) -> str:
    """Thay khớp neo bằng FixedJoint nối thẳng thân gốc với world."""
    anchor_path = f"{model.root.GetPath()}/{ANCHOR_JOINT_NAME}"
    if model.anchor is not None and model.anchor.GetTypeName() == "PhysicsFixedJoint":
        if not model.targets(model.anchor, 0):
            print(f"  [bỏ qua] {model.anchor.GetName()}: đã là FixedJoint nối world")
            return model.anchor.GetPath().pathString

    local_pos, local_rot = None, None
    if model.anchor is not None:
        local_pos = model.anchor.GetAttribute("physics:localPos0").Get()
        local_rot = model.anchor.GetAttribute("physics:localRot0").Get()
        name = model.anchor.GetName()
        # prim đã xoá vẫn nằm trong danh sách joints và sẽ nổ khi đụng tới, nên bỏ ra trước
        model.joints = [p for p in model.joints if p != model.anchor]
        model.stage.RemovePrim(model.anchor.GetPath())
        model.anchor = None
        print(f"  [xoá]    {name} (khớp neo cũ)")
    if local_pos is None:
        matrix = UsdGeom.XformCache().GetLocalToWorldTransform(model.stage.GetPrimAtPath(model.base))
        local_pos = Gf.Vec3f(matrix.ExtractTranslation())
        quat = matrix.ExtractRotationQuat()
        local_rot = Gf.Quatf(quat.GetReal(), Gf.Vec3f(quat.GetImaginary()))

    fixed = UsdPhysics.FixedJoint.Define(model.stage, anchor_path)
    # body0 để trống = world; đây là dấu hiệu Isaac Lab dùng để nhận ra nền cố định
    fixed.GetBody1Rel().SetTargets([Sdf.Path(model.base)])
    fixed.GetLocalPos0Attr().Set(local_pos)
    fixed.GetLocalRot0Attr().Set(local_rot)
    fixed.GetLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
    fixed.GetLocalRot1Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    print(f"  [thêm]   {ANCHOR_JOINT_NAME}: world -> {short(model.base)}")
    return anchor_path


def remove_clutter(model: Model) -> None:
    """Xoá đèn, scope Render và mọi thứ dưới /World không thuộc cây robot."""
    for path in CLUTTER_PATHS:
        if model.stage.GetPrimAtPath(path):
            model.stage.RemovePrim(path)
            print(f"  [xoá]    {path}")

    world = model.stage.GetPrimAtPath("/World")
    if not world:
        return
    root_path = model.root.GetPath().pathString
    for child in list(world.GetChildren()):
        keep = root_path == child.GetPath().pathString or root_path.startswith(child.GetPath().pathString + "/")
        if not keep:
            model.stage.RemovePrim(child.GetPath())
            print(f"  [xoá]    {child.GetPath()}")


def robot_xform(model: Model) -> Usd.Prim:
    """Prim con trực tiếp của /World chứa articulation — chỗ đặt phép nâng."""
    prim = model.root
    while prim.GetParent() and prim.GetParent().GetPath().pathString != "/World":
        prim = prim.GetParent()
    return prim


def lowest_z(model: Model) -> float:
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
    return min(
        cache.ComputeWorldBound(model.stage.GetPrimAtPath(body)).ComputeAlignedRange().GetMin()[2]
        for body in model.bodies
    )


def lift_above_ground(model: Model, anchor_path: str, clearance: float) -> None:
    """Nâng cụm robot để đáy thân thấp nhất cao hơn sàn ``clearance`` mét."""
    xform = UsdGeom.Xformable(robot_xform(model))
    translate_op = next(
        (op for op in xform.GetOrderedXformOps() if op.GetOpType() == UsdGeom.XformOp.TypeTranslate),
        None,
    )
    if translate_op is None:
        translate_op = xform.AddTranslateOp()
        translate_op.Set(Gf.Vec3d(0.0, 0.0, 0.0))

    lowest = lowest_z(model)
    delta = clearance - lowest
    if abs(delta) < 1e-6:
        print(f"  [bỏ qua] chiều cao: đáy đã ở z={lowest:.4f} m")
        return

    current = translate_op.Get()
    translate_op.Set(Gf.Vec3d(current[0], current[1], current[2] + delta))
    # khung world của FixedJoint phải đi theo, nếu không nó kéo thân gốc về chỗ cũ
    anchor = model.stage.GetPrimAtPath(anchor_path)
    pos0 = anchor.GetAttribute("physics:localPos0").Get()
    anchor.GetAttribute("physics:localPos0").Set(Gf.Vec3f(pos0[0], pos0[1], pos0[2] + delta))
    print(f"  [nâng]   {delta:+.4f} m — đáy từ z={lowest:.4f} lên z={clearance:.4f}")


def add_joint_drives(model: Model) -> None:
    """Thêm DriveAPI dạng lực cho mọi khớp động."""
    for prim in model.movable_joints():
        axis = DRIVE_AXIS[prim.GetTypeName()]
        if prim.HasAPI(UsdPhysics.DriveAPI, axis):
            print(f"  [bỏ qua] {prim.GetName()}: đã có DriveAPI:{axis}")
            continue
        drive = UsdPhysics.DriveAPI.Apply(prim, axis)
        drive.CreateTypeAttr().Set("force")
        # hệ số để 0: Isaac Lab ghi đè theo ImplicitActuatorCfg lúc khởi tạo
        drive.CreateStiffnessAttr().Set(0.0)
        drive.CreateDampingAttr().Set(0.0)
        drive.CreateMaxForceAttr().Set(1.0e6)
        drive.CreateTargetPositionAttr().Set(0.0)
        drive.CreateTargetVelocityAttr().Set(0.0)
        print(f"  [thêm]   {prim.GetName()}: DriveAPI:{axis}")


def limit_joint_velocity(model: Model, angular: float, linear: float) -> None:
    """Đặt trần tốc độ cho từng khớp qua ``physxJoint:maxJointVelocity``.

    Chuỗi con lắc càng dài thì đầu mút càng dễ đạt tốc độ lớn trong một bước sim, tới mức
    solver không hội tụ và khớp bị giãn hoặc văng. Trần này chặn ngay ở mức DOF nên rẻ hơn
    nhiều so với việc giảm ``sim.dt``. Đơn vị theo loại khớp: rad/s cho khớp quay, m/s cho
    khớp trượt.
    """
    for prim in model.movable_joints():
        value = angular if prim.GetTypeName() == "PhysicsRevoluteJoint" else linear
        if value <= 0.0:
            continue
        physx_joint = PhysxSchema.PhysxJointAPI.Apply(prim)
        physx_joint.CreateMaxJointVelocityAttr().Set(value)
        unit = "rad/s" if prim.GetTypeName() == "PhysicsRevoluteJoint" else "m/s"
        print(f"  [đặt]    {prim.GetName()}: maxJointVelocity={value} {unit}")


def set_density(model: Model, density: float) -> None:
    """Ghi khối lượng riêng cho các thân để PhysX tính khối lượng theo thể tích thật."""
    for body in model.bodies:
        UsdPhysics.MassAPI.Apply(model.stage.GetPrimAtPath(body)).CreateDensityAttr().Set(density)
        print(f"  [đặt]    {short(body)}: density={density} kg/m^3")


def report(model: Model) -> None:
    print("\n--- trạng thái sau khi vá ---")
    print(f"articulation root : {model.root.GetPath()}")
    print(f"thân gốc          : {short(model.base)}")
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
    for prim in model.joints:
        if not prim.IsValid():
            continue
        joint = UsdPhysics.Joint(prim)
        low = prim.GetAttribute("physics:lowerLimit")
        high = prim.GetAttribute("physics:upperLimit")
        drives = [s for s in prim.GetAppliedSchemas() if "DriveAPI" in s]
        print(f"\n{prim.GetName()} ({prim.GetTypeName()})")
        print(f"  body0 (cha) : {[short(str(t)) for t in joint.GetBody0Rel().GetTargets()] or 'world'}")
        print(f"  body1 (con) : {[short(str(t)) for t in joint.GetBody1Rel().GetTargets()]}")
        axis = prim.GetAttribute("physics:axis")
        if axis and axis.HasAuthoredValue():
            print(f"  axis        : {axis.Get()}")
        if low and low.HasAuthoredValue():
            print(f"  limit       : [{low.Get():.4f}, {high.Get():.4f}]")
        elif prim.GetTypeName() in MOVABLE_JOINT_TYPES:
            print("  limit       : không giới hạn")
        if prim.GetTypeName() in MOVABLE_JOINT_TYPES:
            print(f"  drive       : {drives if drives else 'KHÔNG CÓ'}")
            max_vel = prim.GetAttribute("physxJoint:maxJointVelocity")
            print(f"  max vel     : {max_vel.Get() if max_vel and max_vel.HasAuthoredValue() else 'KHÔNG ĐẶT'}")
    print("\nbounding box world:")
    for body in model.bodies:
        rng = cache.ComputeWorldBound(model.stage.GetPrimAtPath(body)).ComputeAlignedRange()
        print(
            f"  {short(body):12} min={tuple(round(v, 4) for v in rng.GetMin())}"
            f"  max={tuple(round(v, 4) for v in rng.GetMax())}"
        )
    print("\nprim còn lại dưới /World:")
    for child in model.stage.GetPrimAtPath("/World").GetChildren():
        print(f"  {child.GetPath()} ({child.GetTypeName()})")


def main() -> None:
    if not os.path.exists(SOURCE_USD):
        raise FileNotFoundError(f"Không thấy USD gốc: {SOURCE_USD}")
    shutil.copyfile(SOURCE_USD, OUTPUT_USD)
    stage = Usd.Stage.Open(OUTPUT_USD)

    print(f"nguồn : {SOURCE_USD}")
    print(f"đích  : {OUTPUT_USD}\n")
    try:
        model = Model(stage)
    except Exception:
        # bản copy chưa vá mà vẫn mang đúng tên file đích là cái bẫy: env sẽ nạp nó và chạy
        # với robot không có drive, không báo lỗi gì. Dọn đi để lần chạy sau không nhầm.
        stage = None
        os.remove(OUTPUT_USD)
        print(f"[dọn]    xoá {OUTPUT_USD} (bản copy chưa vá)\n")
        raise
    print(f"  thân    : {[short(b) for b in model.bodies]}")
    print(f"  khớp    : {[p.GetName() for p in model.joints]}")
    print(f"  gốc     : {short(model.base)}\n")

    orient_joint_tree(model)
    if args.floating_base:
        print("  [bỏ qua] neo world + nâng khỏi sàn (--floating-base)")
        remove_clutter(model)
    else:
        anchor_path = anchor_root_to_world(model)
        remove_clutter(model)
        lift_above_ground(model, anchor_path, args.clearance)
    add_joint_drives(model)
    limit_joint_velocity(model, args.max_angular_velocity, args.max_linear_velocity)
    if args.density is not None:
        set_density(model, args.density)

    stage.GetRootLayer().Save()
    print(f"\nĐã ghi {OUTPUT_USD}")
    if args.verify:
        report(Model(stage))


if __name__ == "__main__":
    main()
    if simulation_app is not None:
        simulation_app.close()
