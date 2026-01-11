Dưới đây là **phiên bản viết lại và hoàn chỉnh hóa tài liệu**, giữ **100% nội dung gốc**, đồng thời **bổ sung giải thích “là gì” và “vì sao”**, chuẩn hóa thuật ngữ theo **USD / PhysX / Isaac Sim / Isaac Lab**, đủ dùng làm **giáo trình – reference – documentation kỹ thuật**.

---

# Tài liệu Cấu trúc USD Evobot

## 1. Tổng quan

Tài liệu này mô tả **cấu trúc USD nguyên trạng (authoritative structure)** của robot **`evobot`** trong môi trường **NVIDIA Isaac Sim / Omniverse**.
Mục tiêu của tài liệu là:

* Làm **tài liệu tham chiếu chính xác** cho:

  * Debug PhysX
  * Debug Articulation
  * Kiểm tra mapping link–joint
* Làm nền tảng cho:

  * Tích hợp **Isaac Lab**
  * Viết `ArticulationCfg`, `SceneEntityCfg`
  * Huấn luyện RL / Navigation / Manipulation

Tài liệu **không mô tả logic điều khiển**, **không mô tả controller**, mà **chỉ tập trung vào cấu trúc USD và ý nghĩa kỹ thuật của từng thành phần**.

---

## 2. Cây phân cấp USD (Prim Hierarchy)

### 2.1 Hierarchy đầy đủ (Cấu trúc thực tế từ USD)

```text
World (defaultPrim)
└── evobot                                  (Xform, namespace container)
    └── evobot                              (Xform, ArticulationRoot)
        │
        ├── gripper                         (Xform) - Kẹp trái
        ├── gripper_01                      (Xform) - Kẹp phải
        │
        ├── head_link                       (Xform) - Đầu robot
        │   └── arm_link                    (Xform) - Cánh tay
        │       ├── part_below              (Xform)
        │       └── part_below_01           (Xform)
        │
        ├── leg_link                        (Xform) - Chân robot
        │   ├── leg                         (Xform)
        │   └── leg_01                      (Xform)
        │
        ├── top_link                        (Xform) - Thân trên (gắn IMU)
        │   ├── name                        (Xform)
        │   │   ├── name                    (Xform)
        │   │   └── cylinder_link           (Xform)
        │
        ├── wheel                           (Xform) - Bánh phải
        ├── wheel_01                        (Xform) - Bánh trái
        │
        ├── base_joint                      (PhysicsRevoluteJoint)
        ├── arm_joint                       (PhysicsRevoluteJoint)
        ├── left_wheel_joint                (PhysicsRevoluteJoint)
        ├── right_wheel_joint               (PhysicsRevoluteJoint)
        ├── left_grabbing_joint             (PhysicsPrismaticJoint)
        └── right_grabbing_joint            (PhysicsPrismaticJoint)
```

### 2.2 Đặc điểm cấu trúc

* Có **2 cấp `evobot`**:

  * `World/evobot`: **namespace container**
  * `World/evobot/evobot`: **robot thực**, được gắn `ArticulationRoot`
* **Toàn bộ link và joint** nằm dưới **một ArticulationRoot duy nhất**
* **Tất cả joint** được đặt **cùng cấp** với các link, **không lồng trong link**
* Không có prim collision riêng biệt:

  * Collision (nếu có) được **gắn trực tiếp lên visual mesh**
* Cấu trúc tuân thủ **best practice của PhysX Articulation trong Isaac Sim**

---

## 3. Thành phần chính

### 3.1 Chuỗi động học (Kinematic Chain)

Chuỗi động học logic của robot có thể được diễn giải như sau:

```text
top_link (root, thân trên - gắn IMU)
    → head_link → arm_link (cánh tay)
    → leg_link (chân)
    → wheel (bánh phải) via right_wheel_joint
    → wheel_01 (bánh trái) via left_wheel_joint
    → gripper (kẹp trái) via left_grabbing_joint
    → gripper_01 (kẹp phải) via right_grabbing_joint
```

**Giải thích:**

* `top_link` là **root link** của toàn bộ hệ Articulation (thân trên robot)
* `head_link → arm_link` tạo thành **chuỗi nối tiếp (serial chain)** cho tay máy
* `leg_link` chứa các thành phần chân của robot
* Hai bánh xe và hai kẹp được:
  * Liên kết động học thông qua **PhysicsJoint**
* Cách tổ chức này giúp:
  * PhysX giải Articulation ổn định
  * Tránh inertia propagation sai
  * Dễ debug joint độc lập

**Lưu ý quan trọng:**
* **IMU nên gắn vào `top_link`** vì đây là thân chính của robot
* **Contact sensor** có thể gắn vào `head_link` hoặc `arm_link` để phát hiện va chạm

---

### 3.2 Các Joint (5 DOF)

| Joint name             | Loại      | Ý nghĩa           |
| ---------------------- | --------- | ----------------- |
| `left_wheel_joint`     | Revolute  | Quay bánh xe trái |
| `right_wheel_joint`    | Revolute  | Quay bánh xe phải |
| `arm_joint`            | Revolute  | Quay cánh tay     |
| `left_grabbing_joint`  | Prismatic | Trượt kẹp trái    |
| `right_grabbing_joint` | Prismatic | Trượt kẹp phải    |

**Lưu ý kỹ thuật quan trọng:**

* Mỗi joint phải khai báo:

  * `body0`: parent link
  * `body1`: child link
* Joint **không phải là prim cha của link**
* Đây là yêu cầu bắt buộc để:

  * PhysX nhận đúng DOF
  * Isaac Lab đọc được joint state

---

### 3.3 Các Link / Rigid Body

| Link                          | Vai trò                                    |
| ----------------------------- | ------------------------------------------ |
| `top_link`                    | **Thân chính, root của Articulation (gắn IMU)** |
| `head_link`                   | Khối đầu robot                             |
| `arm_link`                    | Cánh tay chính                             |
| `leg_link`                    | Chân robot (chứa leg, leg_01)             |
| `wheel`, `wheel_01`           | Bánh xe phải / trái                        |
| `gripper`, `gripper_01`       | Kẹp phải / trái                            |
| `part_above`, `part_above_01` | Thành phần phụ thân trên                   |
| `part_below`, `part_below_01` | Thành phần phụ thân dưới                   |
| `cylinder_link`               | Thành phần hình trụ (trong top_link)       |

---

## 4. Cấu hình cho Isaac Lab

### 4.1 `joint_names` cho `ArticulationCfg`

Danh sách joint được dùng để:

* Gán actuator
* Đọc trạng thái joint
* Áp dụng action trong RL

```python
joint_names = [
    "left_wheel_joint",
    "right_wheel_joint",
    "arm_joint",
    "left_grabbing_joint",
    "right_grabbing_joint"
]
```

**Yêu cầu:**

* Tên **phải khớp chính xác** với prim name trong USD
* Thứ tự ảnh hưởng trực tiếp tới:

  * Action vector
  * Observation vector

---

### 4.2 `body_names` cho `SceneEntityCfg`

Danh sách body dùng cho:

* Contact sensor
* Force sensor
* Observation (pose, velocity)

```python
body_names = [
    "top_link",       # Root link (thân chính)
    "head_link",      # Đầu robot
    "arm_link",       # Cánh tay
    "leg_link",       # Chân
    "wheel",          # Bánh phải
    "wheel_01",       # Bánh trái
    "gripper",        # Kẹp trái
    "gripper_01",     # Kẹp phải
    "cylinder_link",  # Thành phần hình trụ
]
```

### 4.3 Sensor Configuration Examples

```python
# IMU - gắn vào thân chính (top_link)
imu = ImuCfg(
    prim_path="/World/envs/env_.*/Robot/evobot/evobot/top_link",
    update_period=0.02,  # 50Hz
    gravity_bias=(0.0, 0.0, 0.0),
)

# Contact sensor - gắn vào đầu robot để phát hiện va chạm
contact_sensor = ContactSensorCfg(
    prim_path="/World/envs/env_.*/Robot/evobot/evobot/head_link",
    update_period=0.01,  # 100Hz
)
```

**Lưu ý:**

* `body_names` **chỉ tham chiếu link**
* Không được đưa joint vào danh sách này
* Mọi body phải:

  * Có RigidBody API
  * Có mass hợp lệ

---