# IsaacLab RL Flow: `locomanipulation_g1_env_cfg.py`

Flow đầy đủ từ action input đến lực/torque trong PhysX, dựa trên file:  
`source/isaaclab_tasks/.../locomanipulation/pick_place/locomanipulation_g1_env_cfg.py`

---

## Tổng quan: env này có 2 ActionTerm song song

```python
# locomanipulation_g1_env_cfg.py — ActionsCfg
class ActionsCfg:
    upper_body_ik        = G1_UPPER_BODY_IK_ACTION_CFG          # PinkInverseKinematicsAction
    lower_body_joint_pos = AgileBasedLowerBodyActionCfg(...)     # AgileBasedLowerBodyAction
```

Hai term này chạy **song song**, mỗi term điều khiển nhóm joints khác nhau:

| Term | Joints | Input | Cơ chế |
|---|---|---|---|
| `upper_body_ik` | shoulders, elbows, wrists, hands (17+14 joints) | EEF pose L/R + hand joints | Pink IK solver → q_des |
| `lower_body_joint_pos` | hips, knees (8 joints) | `[vx, vy, wz, hip_height]` (4 dims) | Agile locomotion policy → q_des |

Kết quả của cả hai đều là `set_joint_position_target` — tức là output cuối cùng là **position target cho joints**, không phải torques trực tiếp.

---

## Robot G1 có 5 nhóm actuator khác nhau

Từ `source/isaaclab_assets/isaaclab_assets/robots/unitree.py` — `G1_29DOF_CFG`:

```python
actuators={
    "legs":  DCMotorCfg(
        joint_names_expr=[".*_hip_yaw_joint", ".*_hip_roll_joint",
                          ".*_hip_pitch_joint", ".*_knee_joint"],
        stiffness={".*_hip_.*": 100.0, ".*_knee_joint": 200.0},
        damping  ={".*_hip_.*": 2.5,   ".*_knee_joint": 5.0},
        saturation_effort=180.0,
    ),
    "feet":  DCMotorCfg(
        joint_names_expr=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"],
        stiffness={".*_ankle_pitch_joint": 20.0, ".*_ankle_roll_joint": 20.0},
        damping  ={".*_ankle_pitch_joint": 0.2,  ".*_ankle_roll_joint": 0.1},
        saturation_effort=80.0,
    ),
    "waist": ImplicitActuatorCfg(
        joint_names_expr=["waist_.*_joint"],
        stiffness={"waist_.*_joint": 5000.0},
        damping  ={"waist_.*_joint": 5.0},
    ),
    "arms":  ImplicitActuatorCfg(
        joint_names_expr=[".*_shoulder_.*", ".*_elbow_.*", ".*_wrist_.*"],
        stiffness=3000.0,
        damping  =10.0,
    ),
    "hands": ImplicitActuatorCfg(
        joint_names_expr=[".*_index_.*", ".*_middle_.*", ".*_thumb_.*"],
        stiffness=20.0,
        damping  =2.0,
    ),
}
```

**2 loại actuator** — hoạt động hoàn toàn khác nhau:
- `DCMotorCfg` (legs, feet): **explicit** — Python tính PD + clip torque-speed curve → gửi torque vào PhysX
- `ImplicitActuatorCfg` (waist, arms, hands): **implicit** — Python pass-through → PhysX tự tính PD bằng stiffness/damping

---

## env.step(action) — vòng lặp chính

File: `isaaclab/envs/manager_based_rl_env.py`:

```python
def step(self, action: torch.Tensor):
    # Bước 1: process_actions gọi MỘT LẦN
    self.action_manager.process_action(action)
    #   ├── upper_body_ik.process_actions(upper_actions)
    #   └── lower_body_joint_pos.process_actions(lower_actions)  ← [vx, vy, wz, hip_height]

    # Bước 2: lặp decimation=4 lần (200 Hz / 50 Hz = 4)
    for _ in range(4):
        self.action_manager.apply_action()
        #   ├── upper_body_ik.apply_actions()        → set_joint_position_target (arms)
        #   └── lower_body_joint_pos.apply_actions() → set_joint_position_target (legs)

        self.scene.write_data_to_sim()
        # → _apply_actuator_model() cho từng actuator group
        # → set_dof_position_targets / set_dof_actuation_forces → PhysX

        self.sim.step()     # PhysX tích phân 1 substep dt=0.005s
        self.scene.update() # đọc q, dq mới từ PhysX về Python

    # Bước 3: termination, obs
    ...
```

---

## Term 1: AgileBasedLowerBodyAction (lower_body_joint_pos)

### process_actions() — gọi 1 lần / env step

File: `locomanipulation/pick_place/mdp/actions.py`

```python
def process_actions(self, actions: torch.Tensor):
    # actions = [vx, vy, wz, hip_height]  shape (N, 4)
    base_command = actions

    # Lấy obs group "lower_body_policy" đã được tính từ trước
    # obs gồm: base_lin_vel(3) + base_ang_vel(3) + projected_gravity(3)
    #         + joint_pos_rel(N_joints) + joint_vel_rel(N_joints) + last_action(N_joints)
    obs_tensor = self._env.obs_buf["lower_body_policy"]  # (N, obs_dim)

    # Ghép command vào obs (lặp lại theo history_length)
    # repeated_commands shape: (N, history_length * 4)
    repeated_commands = base_command.unsqueeze(1).repeat(1, history_length, 1).reshape(N, -1)
    policy_input = torch.cat([repeated_commands, obs_tensor], dim=-1)

    # Chạy pretrained agile locomotion policy (TorchScript)
    joint_actions = self._policy.forward(policy_input)  # (N, 8) — 8 leg joints

    self._raw_actions[:] = joint_actions

    # Scale + offset: q_des = policy_output * 0.25 + default_joint_pos
    self._processed_actions = joint_actions * 0.25 + self._policy_output_offset
    #                                          ↑
    #                    policy_output_scale=0.25 từ AgileBasedLowerBodyActionCfg
```

`self._policy_output_offset` = `default_joint_pos` của các leg joints lấy lúc khởi tạo — giống pattern UAV "action=0 → reference state".

### apply_actions() — gọi 4 lần / env step

```python
def apply_actions(self):
    # Chỉ ghi q_des vào buffer — chưa gửi PhysX
    self._asset.set_joint_position_target(
        self._processed_actions,   # (N, 8) — q_des cho 8 leg joints
        joint_ids=self._joint_ids,
    )
    # → robot._data.joint_pos_target[:, leg_joint_ids] = q_des
```

**Không có PID ở đây.** PD được tính sau đó trong `_apply_actuator_model()`.

---

## Term 2: PinkInverseKinematicsAction (upper_body_ik)

### process_actions() — gọi 1 lần / env step

File: `isaaclab/envs/mdp/actions/pink_task_space_actions.py`

```python
def process_actions(self, actions: torch.Tensor):
    # actions = [left_wrist_pos(3) + left_wrist_quat(4) +
    #            right_wrist_pos(3) + right_wrist_quat(4) +
    #            hand_joints(14)]   shape (N, 28)
    self._raw_actions[:] = actions

    # Tách hand joints ra
    self._target_hand_joint_positions = actions[:, -14:]  # (N, 14)

    # Lấy pose pelvis trong world frame
    self.base_link_frame_in_world_rf = self._get_base_link_frame_transform()

    # Chuyển EEF poses từ world frame sang pelvis-local frame
    controlled_frame_poses = self._extract_controlled_frame_poses(actions[:, :14])
    transformed_poses = self._transform_poses_to_base_link_frame(controlled_frame_poses)

    # Set targets cho Pink IK tasks (left wrist, right wrist, posture)
    self._set_task_targets(transformed_poses)
    # IK solver chưa chạy ở đây — chỉ set targets
```

### apply_actions() — gọi 4 lần / env step

```python
def apply_actions(self):
    # Chạy Pink IK solver — giải bài toán Jacobian-based IK
    # Cho từng env riêng (CPU, sequential):
    ik_joint_positions = self._compute_ik_solutions()
    # → joint_pos_des (N, 17) cho shoulder/elbow/wrist joints

    # Ghép với hand joints
    all_joint_positions = torch.cat([ik_joint_positions,
                                     self._target_hand_joint_positions], dim=1)
    # all_joint_positions: (N, 17+14=31)

    self._processed_actions = all_joint_positions

    # Apply gravity compensation: lấy từ PhysX, ghi vào effort target
    if self.cfg.enable_gravity_compensation:
        gravity = robot.root_physx_view.get_gravity_compensation_forces()
        #         ↑ PhysX tính sẵn torque để bù trọng lực từng joint
        robot.set_joint_effort_target(gravity, joint_ids=arm_joint_ids)
        # → robot._data.joint_effort_target[:, arm_ids] = gravity_torque

    # Ghi q_des vào buffer
    robot.set_joint_position_target(all_joint_positions, joint_ids=arm_joint_ids)
    # → robot._data.joint_pos_target[:, arm_ids] = q_des
```

Arm joints có **cả hai**: `joint_pos_target` (q_des) và `joint_effort_target` (gravity feedforward). Cả hai sẽ được dùng trong `ImplicitActuator.compute()`.

---

## write_data_to_sim() — biến buffer thành PhysX commands

Gọi mỗi substep, sau `apply_actions()`:

```python
def write_data_to_sim(self):
    self._apply_actuator_model()   # xử lý từng actuator group
    self.root_physx_view.set_dof_actuation_forces(self._joint_effort_target_sim, ...)
    if self._has_implicit_actuators:   # True vì có waist/arms/hands
        self.root_physx_view.set_dof_position_targets(self._joint_pos_target_sim, ...)
        self.root_physx_view.set_dof_velocity_targets(self._joint_vel_target_sim, ...)
```

### _apply_actuator_model(): xử lý 5 nhóm

```python
for actuator in self.actuators.values():
    control_action = ArticulationActions(
        joint_positions  = joint_pos_target[:, actuator.joints],   # q_des
        joint_velocities = joint_vel_target[:, actuator.joints],   # dq_des (= 0 mặc định)
        joint_efforts    = joint_effort_target[:, actuator.joints], # tau_ff
    )
    control_action = actuator.compute(control_action, q_current, dq_current)
    # → cập nhật _joint_pos_target_sim / _joint_effort_target_sim
```

---

## DCMotorCfg: legs và feet

`DCMotorCfg` kế thừa `IdealPDActuator` + thêm torque-speed clipping.

```python
# actuator_pd.py — IdealPDActuator.compute() (base của DCMotor)
def compute(self, control_action, joint_pos, joint_vel):
    error_pos = control_action.joint_positions  - joint_pos   # q_des - q
    error_vel = control_action.joint_velocities - joint_vel   # 0 - dq = -dq
    # Python tính PD torque
    self.computed_effort = (self.stiffness * error_pos
                          + self.damping   * error_vel
                          + control_action.joint_efforts)   # tau_ff (= 0)
    # DCMotor: clip theo torque-speed curve
    self.applied_effort = self._clip_effort_dcmotor(self.computed_effort, joint_vel)
    # Ghi đè: chỉ gửi effort, xóa pos/vel targets
    control_action.joint_efforts    = self.applied_effort   # torque cuối
    control_action.joint_positions  = None   # PhysX không nhận
    control_action.joint_velocities = None   # PhysX không nhận
    return control_action
```

**Ví dụ với knee joint** (stiffness=200, damping=5):
```
q_des = agile_policy_output * 0.25 + default_q   (từ process_actions)
error_pos = q_des - q_current
error_vel = 0 - dq_current = -dq_current
tau = 200 * error_pos + 5 * (-dq_current)
tau = clip(tau, torque_speed_min, torque_speed_max)  ← DCMotor saturation
→ PhysX nhận tau dưới dạng actuation_force
```

**PhysX nhận**:
- `set_dof_actuation_forces(tau)` ✓
- `set_dof_position_targets(...)` ✗ (bị xóa)
- `set_dof_velocity_targets(...)` ✗ (bị xóa)
- Stiffness/damping trong PhysX = **0** (forced khi init, vì explicit actuator)

**PhysX tính**: `tau_joint = 0*(q-0) + 0*(dq-0) + tau = tau` — chỉ apply torque thuần.

---

## ImplicitActuatorCfg: waist, arms, hands

```python
# actuator_pd.py — ImplicitActuator.compute()
def compute(self, control_action, joint_pos, joint_vel):
    # Chỉ tính để LOG — không thay đổi control_action
    self.computed_effort = (self.stiffness * (control_action.joint_positions - joint_pos)
                          + self.damping   * (control_action.joint_velocities - joint_vel)
                          + control_action.joint_efforts)
    return control_action   # ← NGUYÊN XI
```

control_action được truyền thẳng vào PhysX với:
- `joint_positions = q_des` (từ `set_joint_position_target`)
- `joint_efforts = gravity_torque` (chỉ arms, từ `set_joint_effort_target`)

**PhysX nhận** (vì `_has_implicit_actuators = True`):
- `set_dof_position_targets(q_des)`
- `set_dof_velocity_targets(0)` ← dq_des=0
- `set_dof_actuation_forces(tau_ff)` ← gravity torque (arms) hoặc 0 (waist, hands)
- Stiffness/damping = giá trị từ config (ghi vào PhysX lúc init)

**PhysX tính**:
```
# Arms  (stiffness=3000, damping=10):
tau_joint = 3000*(q_des - q) + 10*(0 - dq) + gravity_torque
          ↑ PD giữ arm đúng IK solution    ↑ feedforward bù trọng lực

# Waist (stiffness=5000, damping=5):
tau_joint = 5000*(q_des - q) + 5*(0 - dq) + 0
          ↑ rất cứng — giữ nguyên vị trí lệnh

# Hands (stiffness=20, damping=2):
tau_joint = 20*(q_des - q) + 2*(0 - dq) + 0
          ↑ nhẹ — retargeting từ hand tracking
```

---

## Bức tranh tổng hợp: một env step hoàn chỉnh

```
Teleop / XR controller
    │  actions = [upper_body_EEF_poses(14) + hand_joints(14)  |  vx, vy, wz, hip_height(4)]
    │            ◄──────────── upper_body_ik ───────────────►  ◄── lower_body_joint_pos ──►
    ↓
process_actions() [1 lần, 50 Hz]
    ├─ PinkIK: chuyển EEF poses → pelvis frame, set IK task targets
    └─ AgilePolicy: [vx,vy,wz,h] + obs → pretrained policy → q_des legs * 0.25 + default_q

for _ in range(4):   ← 200 Hz physics loop
    apply_actions()
    ├─ PinkIK.apply_actions():
    │       Pink IK solver chạy → q_des arms (17 joints)
    │       set_joint_effort_target(gravity_torque, arm_ids)   ← feedforward
    │       set_joint_position_target(q_des_arms, arm_ids)
    └─ AgilePolicy.apply_actions():
            set_joint_position_target(q_des_legs, leg_ids)

    write_data_to_sim()
    ├─ DCMotor "legs" (hip×4, knee×2, mỗi bên):
    │       tau = 100*(q_des-q) + 2.5*(-dq)   → clip DCMotor → effort_target_sim
    │       pos/vel targets → None (không gửi PhysX)
    ├─ DCMotor "feet" (ankle×2, mỗi bên):
    │       tau = 20*(q_des-q) + 0.2*(-dq)    → clip DCMotor → effort_target_sim
    ├─ Implicit "waist":
    │       pass-through → pos_target_sim=q_des, effort_target_sim=0
    ├─ Implicit "arms":
    │       pass-through → pos_target_sim=q_des_IK, effort_target_sim=gravity
    └─ Implicit "hands":
            pass-through → pos_target_sim=q_des_hand, effort_target_sim=0

    set_dof_actuation_forces([tau_legs, tau_feet, 0_waist, gravity_arms, 0_hands])
    set_dof_position_targets([0_legs, 0_feet, q_waist, q_arms_IK, q_hands])
    set_dof_velocity_targets([0, 0, 0, 0, 0])

    sim.step()   ← PhysX tích phân
    ├─ legs:  K=0, D=0 trong PhysX → tau_joint = tau_DCMotor  (Python đã tính)
    ├─ feet:  K=0, D=0 trong PhysX → tau_joint = tau_DCMotor
    ├─ waist: K=5000, D=5  → tau = 5000*(q_des-q) + 5*(0-dq) + 0
    ├─ arms:  K=3000, D=10 → tau = 3000*(q_des-q) + 10*(0-dq) + gravity
    └─ hands: K=20,   D=2  → tau = 20*(q_des-q) + 2*(0-dq) + 0
    → integrate: M*ddq = tau_joints + tau_gravity + tau_contact
    → q_new, dq_new

    scene.update()   ← đọc q_new, dq_new từ PhysX về Python
```

---

## Tóm tắt: ai làm gì với stiffness/damping

| Nhóm joints | Actuator | Stiffness/Damping | Ai tính PD | Gửi gì vào PhysX |
|---|---|---|---|---|
| legs (hip, knee) | `DCMotorCfg` | K=100-200, D=2.5-5 **trong Python** | **Python** (explicit) | chỉ `effort` (torque) |
| feet (ankle) | `DCMotorCfg` | K=20, D=0.1-0.2 **trong Python** | **Python** (explicit) | chỉ `effort` (torque) |
| waist | `ImplicitActuatorCfg` | K=5000, D=5 **trong PhysX** | **PhysX** (implicit) | `pos_target` + `effort=0` |
| arms | `ImplicitActuatorCfg` | K=3000, D=10 **trong PhysX** | **PhysX** (implicit) | `pos_target` + `effort=gravity` |
| hands | `ImplicitActuatorCfg` | K=20, D=2 **trong PhysX** | **PhysX** (implicit) | `pos_target` + `effort=0` |

**DCMotor**: stiffness/damping tồn tại trong Python object (`self.stiffness`, `self.damping`), nhưng PhysX bị force về 0 khi init — PhysX không biết đến chúng.

**ImplicitActuator**: stiffness/damping được `set_dof_stiffnesses()` / `set_dof_dampings()` vào PhysX khi init — PhysX dùng chúng mỗi substep để tính PD.

---

## Lỗi trong FlowRL.md cũ và lý do sai

File cũ viết `gains (N, 3) = kp, ki, kd` — đó là pattern của biped/UAV project (RL outputs PID gains). Trong `locomanipulation_g1_env_cfg.py` **không có RL-PID**:

- `lower_body_joint_pos`: input là `[vx, vy, wz, hip_height]` (velocity commands), output của policy là **joint positions** — không phải gains
- `upper_body_ik`: input là **EEF poses** từ XR teleop, không có RL nào ở đây
- PD control ở đây là **cố định** (hardcoded trong `G1_29DOF_CFG`), không được tuned bởi RL
