# Biped RL → PID Cascade (Unitree H1)

Áp dụng kiến trúc **RL → PID gains** (từ UAV project) sang bipedal robot.  
RL không điều khiển trực tiếp — thay vào đó RL **học PID gains** cho từng lớp cascade.

---

## Ý tưởng cốt lõi

Thay vì để RL output joint torques trực tiếp, mỗi lớp RL học **3 gains [kp, ki, kd]** cho một bộ PID cụ thể. Gains được ánh xạ từ không gian [-1, 1] sang không gian vật lý:

```
raw_action ∈ [-1, 1]  →  gains = raw_action * scale + bias
```

`scale` và `bias` được tính từ `[gains_min, gains_max]` với **bounds đối xứng quanh reference gains** → `raw_action = 0` cho ra gains tham chiếu đã được tune tay → robot ổn định từ bước đầu tiên mà không cần warm-up.

---

## Cascade 3 lớp

```
Velocity RL (Layer 3)
    RL tunes [vel_kp, vel_ki, vel_kd]
    obs (19): vel_err(3) + lin_vel_b(3) + ang_vel(3) + gravity(3) + quat(4) + last_action(3)
    │
    ▼  vel_err → lean_angle_des [pitch_des, roll_des]
    │
Balance PID (Layer 2)  ← pretrained, frozen
    RL tunes [bal_kp, bal_ki, bal_kd]
    obs (16): balance_err(3) + ang_vel(3) + gravity(3) + quat(4) + last_action(3)
    │
    ▼  orientation_err → joint position corrections (via allocation matrix)
    │
Joint PID (Layer 1)  ← pretrained, frozen
    RL tunes [kp, ki, kd]  shared across all 10 leg joints
    obs (29): joint_err(10) + joint_vel(10) + gravity(3) + ang_vel(3) + last_action(3)
    │
    ▼  tau_j = kp*(q_des - q) + ki*integral + kd*(-dq)
    │
set_joint_effort_target()
```

**Training order (bắt buộc):** Joint PID → Balance PID → Velocity RL

---

## Robot: Unitree H1

- **10 leg joints** (control): hip_yaw × 2, hip_roll × 2, hip_pitch × 2, knee × 2, ankle × 2
- **8 arm joints** (fixed PD hold): shoulders × 6, elbows × 2
- **Standing pose** (mặc định): hip_pitch = -0.4 rad, knee = 0.8 rad, ankle = -0.4 rad
- **Actuator mode**: effort (stiffness=0, damping=0) → action term tự tính PID torques
- **Physics**: 200 Hz; **Policy**: 50 Hz (decimation = 4)

---

## Cấu trúc thư mục

```
biped/
├── biped_cfg.py                  # H1 ArticulationCfg + joint lists + allocation vectors
├── mdp/
│   ├── actions.py                # BipedJointPIDGainsAction
│   │                             # BipedBalancePIDGainsAction
│   │                             # BipedVelPIDGainsAction
│   ├── commands.py               # BipedTargetJointPosCommand
│   │                             # BipedTargetAttitudeCommand
│   │                             # BipedTargetVelCommand
│   ├── observations.py           # joint_pos_error, balance_error_b, vel_error_b
│   └── rewards.py                # tracking rewards cho từng layer
├── agents/
│   └── rsl_rl_ppo_cfg.py         # PPO configs (3 tasks)
└── rl_control/
    ├── joint_pid_env_cfg.py      # Task: Biped-Joint-PID
    ├── balance_env_cfg.py        # Task: Biped-Balance-PID
    ├── velocity_env_cfg.py       # Task: Biped-Velocity-RL
    └── model/
        ├── joint_pid/rl_model.pt # pretrained Layer 1 (sau khi train)
        └── balance/rl_model.pt   # pretrained Layer 2 (sau khi train)
```

---

## Hướng dẫn training

### Bước 1 — Train Joint PID (Layer 1)

```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task Biped-Joint-PID --num_envs 256 --headless
```

RL học gains [kp, ki, kd] shared cho tất cả leg joints. q_des được randomize ±0.15 rad quanh standing pose để inner policy có thể **generalize** khi được dùng trong Layer 2.

Sau khi train xong, export TorchScript:
```python
import torch
runner.alg.actor_critic.actor  # lấy actor network
torch.jit.save(torch.jit.script(actor), "rl_control/model/joint_pid/rl_model.pt")
```

### Bước 2 — Train Balance PID (Layer 2)

```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task Biped-Balance-PID --num_envs 256 --headless
```

RL học gains [bal_kp, bal_ki, bal_kd] cho body orientation outer-loop. Layer 1 (joint policy) được load từ `model/joint_pid/rl_model.pt` và **đóng băng (frozen)**.

Flow trong `apply_actions`:
1. Tính orientation error [roll_err, pitch_err, yaw_err]
2. Balance PID → pitch/roll correction angles
3. Correction → delta_q qua allocation matrix
4. q_des = default_standing + delta_q
5. Run inner joint policy → [kp, ki, kd]
6. Apply joint PID torques

Sau khi train, export:
```python
torch.jit.save(torch.jit.script(actor), "rl_control/model/balance/rl_model.pt")
```

### Bước 3 — Train Velocity RL (Layer 3)

```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task Biped-Velocity-RL --num_envs 256 --headless
```

RL học gains [vel_kp, vel_ki, vel_kd] cho velocity outer-loop. Layer 1 + 2 đều frozen.

Flow trong `apply_actions`:
1. Velocity error → lean angle desired [pitch_des, roll_des]
2. Run pretrained balance policy (obs 16) → bal_gains
3. Balance PID → delta_q corrections
4. Run pretrained joint policy (obs 29) → joint_gains
5. Apply joint PID torques

### Play

```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
    --task Biped-Velocity-RL --num_envs 1
```

---

## Thông số gains tham chiếu

| Layer | Gains | Min | Max | Ref (bias) |
|---|---|---|---|---|
| Joint PID | [kp, ki, kd] | (20, 0, 0) | (180, 10, 10) | (100, 5, 5) |
| Balance outer | [bal_kp, bal_ki, bal_kd] | (2, 0, 0) | (14, 1, 3) | (8, 0.5, 1.5) |
| Velocity outer | [vel_kp, vel_ki, vel_kd] | (0.05, 0, 0) | (0.55, 0.1, 0.1) | (0.3, 0.05, 0.05) |

---

## Balance → Joint Allocation Matrix

Khi body pitch/roll, balance PID tính một correction angle và phân bổ sang joint corrections:

```python
# biped_cfg.py
PITCH_ALLOC = [0, 0, -1.0,  0.5, -0.5,   # left:  hip_yaw, hip_roll, hip_pitch, knee, ankle
               0, 0, -1.0,  0.5, -0.5]   # right: symmetric

ROLL_ALLOC  = [0, 0.5, 0, 0, 0,           # left hip_roll +
               0,-0.5, 0, 0, 0]           # right hip_roll -

YAW_ALLOC   = [0.2, 0, 0, 0, 0,           # left hip_yaw +
              -0.2, 0, 0, 0, 0]           # right hip_yaw -
```

Đây là giá trị khởi đầu dựa trên kinematics đơn giản. **Cần tune lại** cho H1 thực tế (có thể học bằng domain randomization hoặc System ID).

---

## Registered Tasks

| Task ID | Layer | Action class | Obs | Action | Cần pretrained |
|---|---|---|---|---|---|
| `Biped-Joint-PID` | 1 | `BipedJointPIDGainsAction` | 29 | 3 | — |
| `Biped-Balance-PID` | 2 | `BipedBalancePIDGainsAction` | 16 | 3 | joint_pid |
| `Biped-Velocity-RL` | 3 | `BipedVelPIDGainsAction` | 19 | 3 | balance + joint_pid |

---

## So sánh với UAV cascade

| UAV | Biped | Mô tả |
|---|---|---|
| Rate PID | **Joint PID** | Innermost: error → corrective output |
| Attitude PID | **Balance PID** | Outer: orientation → inner setpoint |
| Velocity PID | **Velocity RL** | Outermost: vel → orientation setpoint |
| `permanent_wrench_composer` | `set_joint_effort_target` | Force application API |
| Body moments (4 DOF) | Joint torques (10 DOF) | Control output |
| Obs: 16 dims | Obs: 29 dims (joint space lớn hơn) | Inner policy input |

---

## Reward design

### Layer 1 (Joint PID)
- `joint_pos_tracking_exp` (+10, std=0.3): reward chính, exp kernel
- `joint_pos_tracking_l2` (−2): penalty sai số
- `joint_vel_l2` (−0.001): phạt velocity không cần thiết
- `action_rate_l2` (−0.01): phạt gains thay đổi đột ngột

### Layer 2 (Balance PID)
- `balance_tracking_exp` (+8, std=0.2): reward orientation tracking
- `balance_tracking_l2` (−2): penalty orientation error
- `body_height_l2` (−0.5): phạt height drift (target 0.98 m)
- `ang_vel_l2` (−0.01): phạt dao động góc

### Layer 3 (Velocity RL)
- `vel_tracking_exp` (+8, std=0.4): reward velocity tracking (vx, vy)
- `vel_tracking_l2` (−2): penalty velocity error
- `body_height_l2` (−0.5): phạt height drift
- `lin_vel_z_l2` (−0.1): phạt drift Z riêng (không ảnh hưởng vx/vy tracking)
- `action_rate_l2` (−0.01)

**Lưu ý**: Không dùng `rpy_alignment` trong Balance task vì target là non-zero lean angle.

---

## Setup

Thêm vào `isaaclab_assets/__init__.py`:
```python
from . import biped
```

Đảm bảo H1 USD có tại `Isaac/Robots/Unitree/H1/h1.usd` trên Nucleus server.
