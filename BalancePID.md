# PID Flow: Xe 2 Bánh Tự Cân Bằng (Self-Balancing Robot) trong IsaacLab

---

## Bài toán

Xe 2 bánh tự cân bằng (kiểu Segway) có xu hướng ngã về phía trước/sau do trọng tâm cao hơn trục bánh. PID phải liên tục quay bánh để giữ xe thẳng đứng — giống như bạn giữ thước trên ngón tay.

**State cần kiểm soát**: pitch angle (góc nghiêng thân xe)  
**Actuator output**: torque cho 2 bánh xe  

---

## Cấu trúc cascade 2 lớp

```
Lớp ngoài — Velocity PID (10-25 Hz)
    Input:  v_des (m/s) — lệnh tốc độ người dùng
    Output: pitch_des (rad) — góc nghiêng mong muốn
        ↓
Lớp trong — Balance PID (100-200 Hz)
    Input:  pitch_des, pitch_actual, pitch_rate
    Output: tau_balance (Nm) — torque cân bằng
        ↓
    Tách thành 2 bánh:
    tau_left  = tau_balance + tau_yaw
    tau_right = tau_balance - tau_yaw
        ↓
set_joint_effort_target([tau_left, tau_right])
        ↓
ImplicitActuator(stiffness=0, damping=0) — pass-through
        ↓
PhysX: tau_joint = 0 + 0 + tau = tau
```

**Tại sao cần 2 lớp?**  
Nếu chỉ dùng 1 PID để giữ pitch=0, xe sẽ đứng yên tại chỗ. Muốn di chuyển về phía trước, thân xe phải nghiêng về phía trước một chút — outer loop tính góc nghiêng cần thiết dựa vào lệnh tốc độ.

---

## Lấy state từ robot trong IsaacLab

```python
# Pitch (góc nghiêng thân xe, đơn vị radian)
# Cách 1: từ projected_gravity (đơn giản, ổn định hơn)
gravity_b = robot.data.projected_gravity_b      # (N, 3) — gravity trong body frame
# Khi xe thẳng đứng: gravity_b = [0, 0, -1]
# Khi xe nghiêng về trước (pitch+): gravity_b[0] > 0
pitch = torch.atan2(gravity_b[:, 0], -gravity_b[:, 2])  # (N,) — rad

# Cách 2: từ euler angles (cần cẩn thận về thứ tự trục)
euler = robot.data.root_euler_angles_w           # (N, 3) — roll, pitch, yaw
pitch = euler[:, 1]                              # (N,)

# Pitch rate (tốc độ thay đổi góc nghiêng)
ang_vel_b = robot.data.root_ang_vel_b            # (N, 3) — angular vel trong body frame
pitch_rate = ang_vel_b[:, 1]                     # (N,) — quanh trục Y body

# Vận tốc bánh xe (rad/s)
wheel_vel = robot.data.joint_vel[:, wheel_ids]   # (N, 2) — [left, right]

# Vận tốc tịnh tiến (m/s)
WHEEL_RADIUS = 0.1   # m — bán kính bánh
v_actual = (wheel_vel[:, 0] + wheel_vel[:, 1]) / 2.0 * WHEEL_RADIUS  # (N,)

# Vận tốc quay (yaw rate)
yaw_rate = robot.data.root_ang_vel_b[:, 2]       # (N,) — quanh trục Z body
```

---

## Lớp trong — Balance PID

Nhiệm vụ: giữ `pitch == pitch_des` bằng cách điều chỉnh torque bánh.

**Tại sao pitch nghiêng về trước → bánh phải quay về trước?**  
Khi xe nghiêng về trước (pitch > 0), bánh quay về trước sẽ "đuổi kịp" điểm rơi của trọng tâm, kéo xe về vị trí thẳng.

```python
# Gọi ở apply_actions(), dt = physics_dt = 1/200 s
dt = env.physics_dt

# 1. Tính error
pitch_error = pitch_des - pitch          # (N,)   dương = xe đang ngã về sau

# 2. Tích phân (integral) — anti-windup
self._bal_integral += pitch_error * dt
self._bal_integral  = self._bal_integral.clamp(-0.5, 0.5)  # giới hạn windup

# 3. Đạo hàm — dùng pitch_rate trực tiếp (không dùng d(error)/dt)
#    Lý do: d(error)/dt = d(pitch_des - pitch)/dt ≈ -pitch_rate
#    (vì pitch_des thay đổi chậm hơn nhiều so với pitch)
#    Dùng -pitch_rate tránh "derivative kick" khi pitch_des thay đổi đột ngột
d_term = -pitch_rate                     # (N,)

# 4. Tính torque cân bằng
tau_balance = (kp_bal * pitch_error
             + ki_bal * self._bal_integral
             + kd_bal * d_term)          # (N,)

# Clamp torque tối đa
tau_balance = tau_balance.clamp(-MAX_TORQUE, MAX_TORQUE)
```

**Ý nghĩa từng term**:

| Term | Tác dụng | Nếu thiếu |
|---|---|---|
| `kp * pitch_error` | Lực tỉ lệ với mức nghiêng | Xe phục hồi chậm, dao động |
| `ki * integral` | Bù sai số tĩnh (ví dụ trọng tâm lệch) | Xe đứng hơi nghiêng, không về đúng 0 |
| `kd * (-pitch_rate)` | Cản lại chuyển động nghiêng (damping) | Xe dao động liên tục, không tắt |

---

## Lớp ngoài — Velocity PID

Nhiệm vụ: tính `pitch_des` để đạt tốc độ mong muốn.

**Trực giác**: Muốn đi nhanh về trước → phải nghiêng về trước (lean forward). Outer PID tính lean angle cần thiết.

```python
# Gọi ít tần suất hơn inner (hoặc cùng tần suất với decimation riêng)

# 1. Tính velocity error
vel_error = v_des - v_actual             # (N,)   dương = cần đi nhanh hơn

# 2. Tích phân
self._vel_integral += vel_error * dt
self._vel_integral  = self._vel_integral.clamp(-1.0, 1.0)

# 3. Đạo hàm
vel_deriv = (vel_error - self._vel_prev_error) / dt
vel_deriv = vel_deriv.clamp(-10.0, 10.0)
self._vel_prev_error = vel_error.clone()

# 4. Tính pitch_des
pitch_des = (kp_vel * vel_error
           + ki_vel * self._vel_integral
           + kd_vel * vel_deriv)         # (N,)

# Giới hạn lean angle — xe không thể nghiêng quá ~15°
MAX_LEAN = 0.26  # rad ≈ 15°
pitch_des = pitch_des.clamp(-MAX_LEAN, MAX_LEAN)
```

---

## Điều khiển yaw (quay đầu) — tách biệt với balance

```python
# Từ lệnh người dùng: yaw_rate_des (rad/s)
yaw_rate_error = yaw_rate_des - yaw_rate  # (N,)

self._yaw_integral += yaw_rate_error * dt
self._yaw_integral  = self._yaw_integral.clamp(-0.3, 0.3)

tau_yaw = kp_yaw * yaw_rate_error + ki_yaw * self._yaw_integral
tau_yaw = tau_yaw.clamp(-MAX_YAW_TORQUE, MAX_YAW_TORQUE)
```

---

## Ghép torque vào 2 bánh

```python
# tau_balance: cùng chiều cho cả 2 bánh → giữ thẳng / tiến / lùi
# tau_yaw:     ngược chiều 2 bánh → quay đầu

tau_left  = tau_balance + tau_yaw   # (N,)
tau_right = tau_balance - tau_yaw   # (N,)

torques = torch.stack([tau_left, tau_right], dim=1)  # (N, 2)

robot.set_joint_effort_target(torques, joint_ids=wheel_ids)
```

Quy ước:
- `tau_balance > 0` → cả 2 bánh quay về phía trước → xe tiến, hoặc đuổi kịp khi ngã về trước
- `tau_yaw > 0` → left nhanh hơn right → xe quay sang phải

---

## Cấu hình actuator bắt buộc

```python
# robot_cfg.py
actuators={
    "wheels": ImplicitActuatorCfg(
        joint_names_expr=["left_wheel_joint", "right_wheel_joint"],
        stiffness=0.0,   # ← PHẢI = 0: tắt PD của PhysX
        damping=0.0,     # ← PHẢI = 0: tắt damping của PhysX
        effort_limit=10.0,  # Nm — tùy robot
    ),
}
```

Nếu để stiffness > 0, PhysX sẽ tự thêm lực `Kp*(q_des - q)` vào — can thiệp vào torque của Python PID, gây mất ổn định.

---

## Toàn bộ flow trong một env step

```
t=0ms   process_actions([v_des, yaw_rate_des])
        Outer velocity PID:
            vel_error = v_des - v_actual
            pitch_des = kp_vel*vel_error + ki_vel*vel_int + kd_vel*vel_deriv
            pitch_des = clamp(pitch_des, ±0.26 rad)

        ┌── physics substep 1 (t=0ms, dt=0.005s)
        │   apply_actions():
        │       pitch = atan2(gravity_b[0], -gravity_b[2])
        │       pitch_rate = ang_vel_b[1]
        │       pitch_error = pitch_des - pitch
        │       bal_integral += pitch_error * 0.005
        │       tau_balance = kp*pitch_error + ki*bal_integral + kd*(-pitch_rate)
        │       tau_yaw = kp_yaw*yaw_err + ki_yaw*yaw_int
        │       tau_left  = tau_balance + tau_yaw
        │       tau_right = tau_balance - tau_yaw
        │       set_joint_effort_target([tau_left, tau_right])
        │           → joint_effort_target buffer
        │   write_data_to_sim():
        │       ImplicitActuator(K=0).compute() → pass-through
        │       set_dof_actuation_forces([tau_left, tau_right])
        │       set_dof_position_targets([0, 0])    ← K=0 → không ảnh hưởng
        │       set_dof_velocity_targets([0, 0])    ← D=0 → không ảnh hưởng
        │   sim.step():
        │       tau_joint = 0 + 0 + [tau_left, tau_right]
        │       F_net = tau_joint + gravity + contact_friction
        │       integrate → pitch_new, wheel_vel_new
        │   scene.update() → pitch, wheel_vel mới
        │
        ├── substep 2 (t=5ms): apply_actions() với pitch_new → tau mới → ...
        ├── substep 3 (t=10ms): ...
        └── substep 4 (t=15ms): ...

t=20ms  process_actions([v_des_mới, ...])
        Outer PID tính pitch_des mới
        ...
```

---

## Tham số PID tham khảo (điểm khởi đầu)

| PID | kp | ki | kd | Ghi chú |
|---|---|---|---|---|
| Balance (pitch) | 50–150 | 0.5–5 | 3–15 | kd quan trọng nhất — thiếu kd xe dao động |
| Velocity | 0.1–0.5 | 0.01–0.1 | 0.0–0.05 | ki giúp đạt đúng tốc độ mong muốn |
| Yaw rate | 5–20 | 0.1–1 | 0.0–0.5 | Thường không cần kd |

**Thứ tự tune**:
1. Tắt outer loop, set `pitch_des = 0`, tune `kp_bal` trước: tăng dần đến khi xe dao động nhẹ
2. Thêm `kd_bal`: tắt dao động
3. Thêm `ki_bal`: xe về đúng vị trí thẳng đứng
4. Bật outer loop, tune `kp_vel` trước, sau đó `ki_vel`

---

## Sai lầm phổ biến

**1. Dùng `d(pitch_error)/dt` thay vì `-pitch_rate`**  
Khi `pitch_des` thay đổi đột ngột (ví dụ velocity command mới), `d(pitch_error)/dt` sẽ có spike lớn → torque giật. Dùng `-pitch_rate` (gyro trực tiếp) mượt hơn.

**2. Không có anti-windup cho integral**  
Khi xe ngã quá xa và không thể phục hồi, integral tích lũy rất lớn (windup). Khi xe được đặt lại thẳng, integral lớn sẽ làm xe overshooting mạnh. Clamp integral giải quyết việc này.

**3. Để stiffness > 0 trong actuator config**  
PhysX sẽ thêm lực `Kp*(q_current - q_des_prev)` vào — can thiệp vào Python PID. Kết quả không thể dự đoán được.

**4. Outer loop chạy cùng tần suất inner loop**  
Outer loop (velocity) nên chạy chậm hơn inner loop (balance) ít nhất 4-10x. Nếu outer loop quá nhanh, nó liên tục thay đổi `pitch_des` trước khi inner loop kịp phản ứng → mất ổn định.
