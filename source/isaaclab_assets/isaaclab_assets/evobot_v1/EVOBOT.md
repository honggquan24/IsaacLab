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

# Tóm tắt Cấu trúc Thư mục Codebase Evobot

## Tổng quan
Codebase này triển khai môi trường RL cho robot **Evobot V1** trong Isaac Lab/Isaac Sim, bao gồm 2 task chính:
- **Balance Task**: Giữ thăng bằng (self-balancing robot)
- **Navigation Task**: Di chuyển đến mục tiêu (với 2 cách tiếp cận)

## Cấu trúc thư mục chi tiết

```
evobot_v1/
├── .claude/                              # Cấu hình Claude AI
│   └── settings.local.json              # Permissions cho Claude
│
├── config/                              # ═══ CẤU HÌNH CHÍNH ═══
│   ├── __init__.py                      # Export tất cả config modules
│   │
│   ├── robot/                           # Robot configuration
│   │   ├── __init__.py
│   │   └── evobot_v1_cfg.py            # ★ Robot articulation config
│   │                                    #   - USD file path
│   │                                    #   - Actuators (wheels, arm, grabbers)
│   │                                    #   - Initial joint positions
│   │
│   ├── balance/                         # ═══ BALANCE TASK ═══
│   │   ├── __init__.py
│   │   ├── evobot_v1_env_cfg_balance.py # ★ Balance environment config
│   │   │                                #   - Scene: robot + ground + sensors
│   │   │                                #   - Actions: 5 DOF joint effort
│   │   │                                #   - Observations: IMU + joints + pose
│   │   │                                #   - Rewards: rpy_alignment, alive, etc.
│   │   │                                #   - Terminations: fall, bad orientation
│   │   └── agents/                      # RL training config
│   │       ├── __init__.py
│   │       └── rsl_rl_ppo_cfg.py       # PPO hyperparameters for balance
│   │
│   └── navigation/                      # ═══ NAVIGATION TASK ═══
│       ├── __init__.py
│       │
│       ├── evobot_v1_navigation_env_cfg.py  # ★ Approach 1: End-to-End
│       │                                     #   - Train balance + navigation together
│       │                                     #   - Direct wheel control from policy
│       │                                     #   - Extends balance config with commands
│       │
│       ├── evobot_v1_navigation_pretrained_env_cfg.py  # ★ Approach 2: Hierarchical
│       │                                               #   - Uses pre-trained balance policy
│       │                                               #   - High-level: [vx, vy, omega]
│       │                                               #   - Low-level: balance controller
│       │
│       ├── agents/                      # RL training config
│       │   ├── __init__.py
│       │   └── rsl_rl_ppo_cfg.py       # PPO hyperparameters for navigation
│       │
│       └── mdp/                         # Navigation-specific MDP components
│           ├── __init__.py
│           ├── pre_trained_policy_action.py  # ★ Hierarchical action wrapper
│           │                                  #   - Loads balance policy checkpoint
│           │                                  #   - Converts velocity commands → wheel torques
│           └── rewards.py               # Navigation reward functions
│                                        #   - position_command_error_tanh
│                                        #   - heading_command_error_abs
│                                        #   - position_reached_bonus
│
├── mdp/                                 # ═══ SHARED MDP COMPONENTS ═══
│   ├── __init__.py
│   ├── observations.py                  # Custom observation functions
│   │                                    #   - obs_body_roll/pitch/yaw
│   │                                    #   - lin_vel_b, angl_vel_b
│   │                                    #   - obs_pos_world
│   ├── rewards.py                       # Custom reward functions (balance task)
│   │                                    #   - rpy_alignment_imu
│   │                                    #   - height_reward
│   │                                    #   - angular_velocity_reward
│   └── terminations.py                  # Custom termination conditions
│                                        #   - reset_when_fall
│
├── tests/                               # ═══ TEST SCRIPTS ═══
│   └── run_robot_rl_env.py             # Test environment loading
│
├── usd_file/                            # ═══ USD ROBOT FILES ═══
│   ├── evobot_cfg.usd                   # USD variant 1 (5KB)
│   ├── evoBOT_cfg.usd                   # USD variant 2 (7KB)
│   └── evobot_v1_cfg.usd               # ★ Main USD file (41MB, with meshes)
│
├── __init__.py                          # ★ GYM ENVIRONMENT REGISTRATION
│                                        #   - Isaac-Evobot-V1-Balance
│                                        #   - Isaac-Evobot-V1-Navigation
│                                        #   - Isaac-Evobot-V1-Navigation-Play
│                                        #   - Isaac-Evobot-V1-Navigation-Pretrained
│                                        #   - Isaac-Evobot-V1-Navigation-Pretrained-Play
│
└── EVOBOT.md                           # ★★★ DOCUMENTATION (THIS FILE)
                                        #   - USD structure reference
                                        #   - Kinematic chain explanation
                                        #   - Isaac Lab integration guide
```

---

## Mô tả chi tiết các thành phần quan trọng

### 1. Robot Configuration ([config/robot/evobot_v1_cfg.py](config/robot/evobot_v1_cfg.py))

**Vai trò**: Định nghĩa cấu trúc vật lý và cơ học của robot

**Nội dung chính**:
```python
EVOBOT_V1_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=UsdFileCfg(
        usd_path="usd_file/evoBOT_cfg.usd",  # USD file path
        ...
    ),
    actuators={
        "wheels": ImplicitActuatorCfg(joint_names=["left_wheel_joint", "right_wheel_joint"], ...),
        "arm": ImplicitActuatorCfg(joint_names=["arm_joint"], ...),
        "grabbers": ImplicitActuatorCfg(joint_names=["left_grabbing_joint", "right_grabbing_joint"], ...),
    },
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={"left_wheel_joint": 0.0, "right_wheel_joint": 0.0, ...},
    ),
)
```

**5 DOF joints**:
- `left_wheel_joint`, `right_wheel_joint`: Revolute (bánh xe)
- `arm_joint`: Revolute (cánh tay)
- `left_grabbing_joint`, `right_grabbing_joint`: Prismatic (kẹp)

---

### 2. Balance Task ([config/balance/evobot_v1_env_cfg_balance.py](config/balance/evobot_v1_env_cfg_balance.py))

**Mục tiêu**: Train robot giữ thăng bằng ở tư thế đứng thẳng

**Components**:
- **Scene**: Robot + ground plane + IMU sensor (gắn trên `top_link`)
- **Actions**: Joint effort control cho 5 DOF (scale: wheels=300, arm=100, grabbers=50)
- **Observations** (Policy):
  - IMU: linear acceleration, angular velocity, orientation, projected gravity
  - Body pose (7D: position + quaternion)
  - Joint states: position, velocity, effort
  - Previous actions
- **Rewards** (7 terms):
  - `alive` (+2.0): Survival bonus
  - `terminating` (-100.0): Penalty for falling
  - `rpy_alignment` (+10.0): Main balance reward (upright orientation)
  - `action_rate` (-1.0): Action smoothness
  - `joint_vel` (-2e-4): Joint velocity penalty
  - `ang_vel_xy` (-0.005): Angular stability
  - `joint_acc_l2` (-1e-8): Joint acceleration penalty
- **Terminations**:
  - Time out (10s)
  - Root height < 0.35m
  - Orientation > 60° from vertical
  - Head contact force > 30N

**Training command**:
```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task=Isaac-Evobot-V1-Balance \
    --num_envs 1024 \
    --headless --rendering_mode performance
```

---

### 3. Navigation Task - End-to-End ([config/navigation/evobot_v1_navigation_env_cfg.py](config/navigation/evobot_v1_navigation_env_cfg.py))

**Approach**: Train trực tiếp từ observations → wheel actions (không cần pre-trained balance policy)

**Extends Balance Config**:
- Kế thừa tất cả balance observations + rewards
- **Thêm command manager**: `UniformPose2dCommand` (target position x, y, heading)
- **Thêm observations**: `pose_command` (3D: target_x, target_y, target_heading)
- **Thêm navigation rewards**:
  - `position_tracking` (+3.0): Coarse position tracking (std=1.5m)
  - `position_tracking_fine` (+2.0): Fine position tracking (std=0.3m)
  - `heading_tracking` (-0.3): Heading alignment penalty
  - `position_reached` (+5.0): Bonus when within 0.3m of target

**Ưu điểm**: Đơn giản, không cần train 2 bước
**Nhược điểm**: Phải học balance + navigation cùng lúc (khó hơn)

**Training command**:
```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task=Isaac-Evobot-V1-Navigation \
    --num_envs 1024 \
    --headless --rendering_mode performance
```

---

### 4. Navigation Task - Hierarchical ([config/navigation/evobot_v1_navigation_pretrained_env_cfg.py](config/navigation/evobot_v1_navigation_pretrained_env_cfg.py))

**Approach**: Dùng **pre-trained balance policy** làm low-level controller

**Architecture**:
```
High-level Policy (navigation)
    ↓ [vx, vy, omega] (velocity commands)
PreTrainedBalancePolicyAction wrapper
    ↓ balance observations (40D)
Low-level Policy (balance checkpoint)
    ↓ wheel torques
Robot
```

**Action Space**:
- **High-level input**: 3D [forward_velocity, lateral_velocity, turn_rate]
- **Low-level processing**: `PreTrainedBalancePolicyAction`
  - Loads balance policy from `logs/rsl_rl/evobot_v1_ppo_balance/.../exported/policy.pt`
  - Computes balance observations (IMU, joints, pose, last_action)
  - Runs balance policy → base wheel efforts
  - Adds velocity modulation:
    ```python
    left_wheel = balance_effort + (vx * vel_scale - omega * turn_scale)
    right_wheel = balance_effort + (vx * vel_scale + omega * turn_scale)
    ```

**Observations** (High-level policy only):
- Base linear/angular velocity
- Projected gravity
- Pose command (target position)

**Ưu điểm**:
- Balance policy đã được train tốt → stable low-level control
- High-level policy chỉ cần học navigation (đơn giản hơn)

**Nhược điểm**:
- Cần train balance task trước
- Phải export balance policy (`.pt` file)

**Workflow**:
1. Train balance: `--task=Isaac-Evobot-V1-Balance`
2. Export policy:
   ```bash
   ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
       --task Isaac-Evobot-V1-Balance --num_envs 1 \
       'agent.load_run=2026-01-12_14-45-12' \
       'agent.load_checkpoint="model_2000.pt"'
   ```
   → Policy exported to `logs/rsl_rl/.../exported/policy.pt`
3. Update `policy_path` in [evobot_v1_navigation_pretrained_env_cfg.py](config/navigation/evobot_v1_navigation_pretrained_env_cfg.py):105
4. Train navigation: `--task=Isaac-Evobot-V1-Navigation-Pretrained`

---

### 5. Pre-trained Policy Action Wrapper ([config/navigation/mdp/pre_trained_policy_action.py](config/navigation/mdp/pre_trained_policy_action.py))

**Class**: `PreTrainedBalancePolicyAction(ActionTerm)`

**Key responsibilities**:
1. **Load balance policy**: `torch.jit.load(policy_path)` (TorchScript model)
2. **Observation matching**:
   - Creates `ObservationManager` for low-level policy
   - Computes 40D balance observations (IMU, joints, pose, last_action)
3. **Action conversion**:
   - Receives high-level `[vx, vy, omega]` commands
   - Runs balance policy with low-level observations
   - Adds velocity modulation to balance actions
4. **Debug visualization**: Arrow markers for velocity commands

**Configuration**:
```python
PreTrainedBalancePolicyActionCfg(
    asset_name="robot",
    policy_path="logs/.../exported/policy.pt",  # ★ Must update this
    low_level_decimation=1,
    low_level_actions=BalanceActionCfg().wheel_effort,  # Wheel joint effort
    low_level_observations=LowLevelObservationsCfg(),  # 40D balance obs
    velocity_scale=0.5,  # Forward velocity scaling
    turn_scale=0.5,      # Turn rate scaling
)
```

---

### 6. Gym Environment Registration ([__init__.py](__init__.py))

**Đăng ký 5 environments**:

1. **`Isaac-Evobot-V1-Balance`**
   - Entry point: `EvobotV1EnvCfgBalance`
   - PPO config: `EvobotPPORunnerCfgBalance`

2. **`Isaac-Evobot-V1-Navigation`** (End-to-end)
   - Entry point: `EvobotV1NavigationEnvCfg`
   - PPO config: `EvobotNavigationPPORunnerCfg`

3. **`Isaac-Evobot-V1-Navigation-Play`** (End-to-end evaluation)
   - Entry point: `EvobotV1NavigationEnvCfgPlay`
   - Resampling time: 2s (fixed for evaluation)

4. **`Isaac-Evobot-V1-Navigation-Pretrained`** (Hierarchical)
   - Entry point: `EvobotV1NavigationPretrainedEnvCfg`
   - Uses pre-trained balance policy

5. **`Isaac-Evobot-V1-Navigation-Pretrained-Play`** (Hierarchical evaluation)
   - Entry point: `EvobotV1NavigationPretrainedEnvCfgPlay`
   - 16 envs, 6m spacing

---

## Luồng huấn luyện đầy đủ

### Option 1: End-to-End Navigation (Đơn giản)
```bash
# Train navigation trực tiếp (không cần balance policy)
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task=Isaac-Evobot-V1-Navigation \
    --num_envs 1024 \
    --headless --rendering_mode performance

# Evaluate
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
    --task Isaac-Evobot-V1-Navigation-Play \
    --num_envs 16 \
    'agent.load_run=<run_name>' \
    'agent.load_checkpoint="model_500.pt"'
```

### Option 2: Hierarchical Navigation (Khuyến nghị)
```bash
# STEP 1: Train balance task
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task=Isaac-Evobot-V1-Balance \
    --num_envs 1024 \
    --headless --rendering_mode performance

# STEP 2: Export balance policy
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
    --task Isaac-Evobot-V1-Balance --num_envs 1 \
    'agent.load_run=2026-01-12_14-45-12' \
    'agent.load_checkpoint="model_2000.pt"'
# → Policy saved to logs/rsl_rl/.../exported/policy.pt

# STEP 3: Update policy_path in evobot_v1_navigation_pretrained_env_cfg.py:105

# STEP 4: Train navigation with pre-trained balance
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task=Isaac-Evobot-V1-Navigation-Pretrained \
    --num_envs 512 \
    --headless --rendering_mode performance

# STEP 5: Evaluate
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
    --task Isaac-Evobot-V1-Navigation-Pretrained-Play \
    --num_envs 16 \
    'agent.load_run=<nav_run_name>' \
    'agent.load_checkpoint="model_500.pt"'
```

---

## Key Technical Features

### 1. Memory Optimization
- **Problem**: Memory leak trong critic observations (16 terms → excessive buffer usage)
- **Solution**: Reduced to 8 essential terms in `ObservationsCfg.CriticCfg`
- **Episode length**: 10s (600 steps @ 60Hz) để balance với buffer size

### 2. Hierarchical RL Architecture
- **Low-level**: Balance policy (giữ thăng bằng) - 5 DOF joint control
- **High-level**: Navigation policy (di chuyển đến target) - 3D velocity commands
- **Decoupling**: High-level không cần quan tâm low-level balance control

### 3. Custom MDP Components
- **Shared MDP** ([mdp/](mdp/)): Observations, rewards, terminations dùng chung
- **Navigation MDP** ([config/navigation/mdp/](config/navigation/mdp/)): Specialized cho navigation task
- **Modular design**: Dễ dàng thêm task mới (e.g., manipulation, obstacle avoidance)

### 4. USD Structure Best Practices
- **Single ArticulationRoot**: Tất cả links/joints dưới một root
- **Flat joint hierarchy**: Joints cùng cấp với links (không lồng nhau)
- **Sensor placement**: IMU trên `top_link` (root body), contact sensor trên `head_link`

---

## Troubleshooting & Common Issues

### Issue 1: Policy file not found
```
FileNotFoundError: Policy file 'logs/.../exported/policy.pt' does not exist.
```
**Solution**: Export balance policy trước khi train navigation-pretrained

### Issue 2: Observation dimension mismatch
```
RuntimeError: Expected 40D observations, got 35D
```
**Solution**: Kiểm tra `LowLevelObservationsCfg` phải khớp với balance policy training observations

### Issue 3: Robot falls immediately
**Causes**:
- Balance policy chưa train đủ (< 300 iterations)
- `velocity_scale`/`turn_scale` quá lớn (default: 0.5)
- Initial joint positions không khớp với reset positions

**Solution**: Train balance task đến convergence (~2000 iterations) trước khi dùng cho navigation