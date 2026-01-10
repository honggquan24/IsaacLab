# Evobot USD Structure Documentation

## 1. Tổng quan

Tài liệu này mô tả **đầy đủ và chính xác** cấu trúc USD của thực thể `evobot` như quan sát trực tiếp trong Isaac Sim / Omniverse Stage. Nội dung được ghi lại **nguyên trạng**, không lược bỏ, nhằm phục vụ làm:

* Reference khi debug PhysX / Articulation
* Tài liệu giải thích cấu trúc robot cho người mới
* Cơ sở kiểm tra tính đúng đắn của schema robot (RigidBody, Joint, ArticulationRoot)

> Ghi chú: Cấu trúc dưới đây phản ánh **USD prim hierarchy**, không phải URDF.

---

## 2. Cây phân cấp tổng thể (USD Prim Hierarchy)

```
World (defaultPrim)                      [Xform]
└── evobot                                [Xform]
    ├── base_footprint                    [Xform]
    ├── base_joint                        [PhysicsRevoluteJoint]
    ├── base_link                         [Xform]
    │   ├── head_joint                    [Xform]
    │   ├── fixed_base_joint              [PhysicsFixedJoint]
    │   ├── head_link                     [Xform]
    │   │   ├── arm_joint                 [PhysicsRevoluteJoint]
    │   │   ├── left_wheel_joint           [PhysicsRevoluteJoint]
    │   │   ├── right_wheel_joint          [PhysicsRevoluteJoint]
    │   │   ├── visuals                   [Mesh]
    │   │   ├── left_eye                  [Mesh]
    │   │   └── right_eye                 [Mesh]
    │   ├── arm_link                      [Xform]
    │   │   ├── left_grabbing_joint        [PhysicsPrismaticJoint]
    │   │   ├── right_grabbing_joint       [PhysicsPrismaticJoint]
    │   │   ├── left_turntable_joint       [PhysicsRevoluteJoint]
    │   │   ├── right_turntable_joint      [PhysicsRevoluteJoint]
    │   │   └── visuals                   [Mesh]
    │   ├── left_wheel                    [Xform]
    │   │   ├── visuals                   [Mesh]
    │   │   └── collisions                [Cylinder]
    │   ├── right_wheel                   [Xform]
    │   │   ├── visuals                   [Mesh]
    │   │   └── collisions                [Cylinder]
    │   ├── gripper_parts                 [Xform]
    │   │   ├── right_gripper             [Xform]
    │   │   │   ├── right_turntable        [Xform]
    │   │   │   ├── right_levers_1         [Xform]
    │   │   │   ├── right_levers_2         [Xform]
    │   │   │   └── right_gripper_plate    [Xform]
    │   │   ├── left_gripper              [Xform]
    │   │   │   ├── left_levers_1          [Xform]
    │   │   │   ├── left_levers_2          [Xform]
    │   │   │   ├── left_gripper_plate     [Xform]
    │   │   │   └── left_turntable         [Xform]
    │   │   └── grab_center               [Xform]
    │   │       └── grab_center_joint      [PhysicsFixedJoint]
    └── Looks                             [Scope]
        └── OmniGlass                     [Material]
```

---

## 3. Giải thích chi tiết theo nhóm

### 3.1 World / evobot

* `World` là **defaultPrim**, bắt buộc để Isaac Sim xác định root của scene.
* `evobot` là root Xform của robot, đóng vai trò namespace chính.

---

### 3.2 Base subsystem

#### base_footprint [Xform]

* Đại diện frame gốc 2D (thường dùng trong navigation).
* Không có physics, chỉ mang ý nghĩa tọa độ.

#### base_joint [PhysicsRevoluteJoint]

* Joint quay liên kết `base_footprint` với `base_link`.
* Cho phép base có chuyển động quay (thiết kế không phổ biến, nhưng hợp lệ).

#### base_link [Xform]

* Thân chính của robot.
* Là nơi gắn các subsystem: head, arm, wheels, gripper.

---

### 3.3 Head & sensor subsystem

#### head_link [Xform]

* Frame trung gian cho đầu robot.
* Chứa:

  * `arm_joint`
  * `left_wheel_joint`, `right_wheel_joint`
  * Mesh hiển thị (visuals, eyes)

#### head_joint / fixed_base_joint

* `fixed_base_joint` là **PhysicsFixedJoint**, khóa cứng head với base.
* Cho thấy đầu robot **không có DOF độc lập**.

---

### 3.4 Arm subsystem

#### arm_joint [PhysicsRevoluteJoint]

* Cho phép cánh tay quay quanh một trục.

#### arm_link [Xform]

* Thân arm chính.
* Chứa cả joint kẹp và bàn xoay.

#### Grabbing joints

* `left_grabbing_joint`, `right_grabbing_joint`
* Kiểu **PhysicsPrismaticJoint**
* Cho phép chuyển động tịnh tiến (mở/đóng kẹp).

#### Turntable joints

* `left_turntable_joint`, `right_turntable_joint`
* Kiểu **PhysicsRevoluteJoint**
* Cho phép xoay cổ tay / bàn kẹp.

---

### 3.5 Wheel subsystem

#### left_wheel / right_wheel [Xform]

* Mỗi bánh xe có:

  * `visuals` (Mesh)
  * `collisions` (Cylinder)

Thiết kế này tách **collision geometry** khỏi mesh hiển thị, đúng best practice PhysX.

---

### 3.6 Gripper mechanical structure

#### gripper_parts [Xform]

* Gom toàn bộ cấu trúc kẹp.

##### right_gripper / left_gripper

* Mỗi bên gồm:

  * Levers (cơ cấu đòn bẩy)
  * Gripper plate (bề mặt kẹp)
  * Turntable (bàn xoay)

Tất cả đều là **Xform**, nghĩa là:

* Chuyển động thực tế được điều khiển qua joint ở cấp cao hơn
* Các prim con chỉ phục vụ transform hình học

---

### 3.7 Grab center

#### grab_center [Xform]

* Điểm tham chiếu trung tâm kẹp.
* Thường dùng cho:

  * IK target
  * End-effector pose

#### grab_center_joint [PhysicsFixedJoint]

* Cố định grab_center vào cấu trúc gripper.
* Đảm bảo pose ổn định, không phát sinh DOF ảo.

---

### 3.8 Materials

#### Looks / OmniGlass

* `Looks` là Scope chuẩn USD để chứa material.
* `OmniGlass` là material được dùng cho các mesh liên quan.

---

## 4. Nhận xét kỹ thuật quan trọng

1. Cấu trúc **chưa thể hiện rõ ArticulationRoot**

   * Cần kiểm tra prim nào được gán `UsdPhysics.ArticulationRootAPI`

2. Joint và link **không theo pattern URDF 1–1**

   * Đây là thiết kế USD-native

3. Wheel joint nằm dưới `head_link`

   * Về mặt cơ học là bất thường
   * Cần xác minh lại khi debug dynamics hoặc control
