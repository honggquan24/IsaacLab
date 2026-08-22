# Xe hai bánh tự cân bằng — viết lại toàn bộ theo mẫu Isaac Lab

> Ngày: 2026-08-22 · Package: `isaaclab_assets/balance_car` (file này nằm ngay trong package)
> Nhánh: `demo` · Isaac Lab 2.3.2 · Isaac Sim 5.1.0

|  | Trước | Sau |
|---|---|---|
| Tổng số dòng | 2707 | 1893 |
| Hàm MDP tự viết ở **tầng thấp** | 18 hàm + 2 class | **0** |
| Hàm reward tự viết ở navigation | 17 | 4 |
| Package `balance_car/mdp/` | 5 file | **đã xoá** |
| Nhịp điều khiển | 30 Hz | 50 Hz |
| Nhiễu miền (domain randomization) | không có | ma sát + khối lượng + trọng tâm + xô đẩy |

Mẫu đã học theo:

| Task | Mẫu trong Isaac Lab |
|---|---|
| `Isaac-Balance-Car` | `isaaclab_tasks/manager_based/locomotion/velocity/velocity_env_cfg.py` |
| `Isaac-Balance-Car-Navigation-Pretrained` | `isaaclab_tasks/manager_based/navigation/config/anymal_c/navigation_env_cfg.py` |

Lý do chọn hai mẫu này: xe cân bằng **đúng là** bài "bám lệnh `(vx, vy, wz)` với thân nổi" mà
mẫu locomotion giải. Nó chỉ khác ở chỗ robot có 2 bậc tự do thay vì 12 và nó bất ổn định hở
vòng. Mọi thứ nó cần — observation, reward, termination, command, event — đều đã có sẵn trong
`isaaclab.envs.mdp`.

---

## 1. Sự thật lấy từ USD

Mọi con số dưới đây quét trực tiếp từ `usd/balance_car_base.usd` (bản Onshape thô) và
`usd/balance_car_cfg.usd` (bản đã vá) — **hai file cho kết quả giống hệt nhau**, `prepare_usd.py`
không đụng tới khối lượng.

### 1.1. Cách quét (không cần boot Isaac Sim)

```bash
SP=~/miniconda3/envs/env_ute/lib/python3.11/site-packages/isaacsim
U=$SP/extscache/omni.usd.libs-1.0.1+69cbf6ad.lx64.r.cp311
P=$SP/extscache/omni.usd.schema.physx-107.3.26+107.3.3.lx64.r.cp311.u353
export PYTHONPATH=$U:$P
export LD_LIBRARY_PATH=$U/bin:$P/bin:$CONDA_PREFIX/lib
export PXR_PLUGINPATH_NAME=$(ls -d $P/plugins/*/resources | tr '\n' ':')
python -c "from pxr import Usd, UsdPhysics; ..."
```

> **Bẫy:** export từ Onshape đánh dấu các part là `instanceable`, nên `Stage.Traverse()`
> **không** đi vào mesh — nó sẽ báo "USD không có va chạm, không có khối lượng". Phải dùng
> `Usd.PrimRange.Stage(stage, Usd.TraverseInstanceProxies(Usd.PrimAllPrimsPredicate))`.

### 1.2. Khối lượng và hình học

| Thân | Khối lượng | Trọng tâm (world, gốc thân = mặt trên nắp) |
|---|---|---|
| `base` | 36.537 kg | z = −0.3177 |
| `cover` | 10.603 kg | z = −0.0100 |
| `wheel` × 2 | 2.495 kg mỗi bánh | z = −0.4200, x = ±0.196 |
| **Tổng** | **52.130 kg** | z = −0.2649 |

| Đại lượng | Giá trị |
|---|---|
| Bán kính bánh | 0.2050 m |
| Bề dày bánh | 0.09 m |
| Vệt bánh (tâm–tâm) | 0.392 m |
| Cao độ trục bánh (hệ thân) | −0.42 m |
| Đáy bánh | z = −0.625 → thả ở z = 0.63 |
| **Khối lượng khung** (base+cover) | **47.140 kg** |
| **Trọng tâm khung so với trục bánh** | **0.1715 m** ← chiều dài con lắc ngược |
| Va chạm | `convexHull` cho mọi mesh, kể cả bánh (88 đỉnh) |
| Trục world của cả hai khớp | **(−1, 0, 0)** |

Ghi chú về `convexHull`: bánh xe trong sim là **đa giác 88 đỉnh**, không phải hình tròn trơn.
Lăn ở tốc độ thấp sẽ có gợn nhẹ. Muốn tròn thật phải đổi approximation trong CAD hoặc thay
bằng primitive cylinder.

### 1.3. Cực bất ổn hở vòng

Tuyến tính hoá quanh tư thế đứng (khung 47.140 kg, `l` = 0.1715 m, `I_com` = 1.5667 kg·m²,
hai bánh lăn cho khối lượng tịnh tiến hiệu dụng `2m + 2I/R²` = 7.486 kg):

```
λ = 6.719 rad/s     τ = 1/λ = 148.8 ms
```

| Nhịp điều khiển | `e^(λ·T)` — sai lệch phồng bao nhiêu giữa hai bước | số bước mỗi τ |
|---|---|---|
| 30 Hz (TRƯỚC) | **1.251** (+25%) | 4.46 |
| 50 Hz (SAU) | **1.144** (+14%) | 7.44 |
| *(con lắc đơn @60 Hz để so sánh)* | *1.164* | *6.59* |

---

## 2. Từng lỗi — trước, vì sao sai, sau

### Lỗi 1 — Policy bị chấm điểm theo thứ nó không nhìn thấy

**TRƯỚC** — `mdp/observations.py`:

```python
def lin_vel_b(env, asset_cfg):
    imu = env.scene[asset_cfg.name]
    return imu.data.lin_vel_b[:, 1].unsqueeze(-1)      # CHỈ 1 số: vận tốc tiến

def angl_vel_b(env, asset_cfg):
    imu = env.scene[asset_cfg.name]
    return imu.data.ang_vel_b[:, 0].unsqueeze(-1)      # CHỈ 1 số: tốc độ ROLL
```

Trong khi reward chấm điểm theo **tốc độ quay quanh Z**:

```python
def track_ang_vel_exp(env, command_name="base_velocity", std=0.5, turn_sign=1.0):
    ang_vel = turn_sign * robot.data.root_ang_vel_w[:, 2]     # yaw rate
    return torch.exp(-torch.square(command[:, 2] - ang_vel) / std**2)
```

**Vì sao sai:** vector quan sát **không hề chứa yaw rate**. Nó có góc yaw tuyệt đối
(`obs_body_yaw`) nhưng mạng là MLP không bộ nhớ, không sai phân được hai bước liên tiếp. Policy
bị yêu cầu bám một đại lượng nó không quan sát được.

**Bằng chứng trong log:** `Metrics/base_velocity/error_vel_yaw = 1.2317` trên dải lệnh ±1.0
rad/s — sai số lớn hơn cả biên độ lệnh, và không giảm theo thời gian train.

**SAU** — dùng thẳng term có sẵn, đủ cả 3 trục:

```python
base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(-0.1, 0.1))   # 3 số
base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(-0.2, 0.2))   # 3 số, có wz
```

---

### Lỗi 2 — Khối lượng sai gần gấp đôi

**TRƯỚC** — `balance_car_cfg.py`:

```python
"""Số liệu dưới đây đọc thẳng từ ``usd/balance_car_base.usd`` để khỏi phải mở stage mới biết.
...
* khối lượng: khung 22.46 kg (base 11.86 + cover 10.60), mỗi bánh 2.50 kg. Tổng 27.45 kg.
"""
BALANCE_CAR_MASS = 27.45
```

File USD mà docstring viện dẫn khai báo **52.130 kg**.

**Vì sao không ai bắt được:** trong bốn con số thì **ba con số đúng**:

| | Docstring | USD | |
|---|---|---|---|
| cover | 10.60 | 10.603 | ✅ |
| bánh (mỗi cái) | 2.50 | 2.495 | ✅ |
| trọng tâm so với trục | 0.172 | 0.1715 | ✅ |
| **base** | **11.86** | **36.537** | ❌ **sai 3.1 lần** |

Bản CAD được vẽ lại, phần `base` nặng lên hơn ba lần, và đúng một dòng đó không được cập nhật.
Docstring vẫn ghi "đọc thẳng từ USD" trong khi USD chưa hề được đọc.

**Hậu quả:** mọi đại lượng dẫn xuất sai theo — mô-men trượt, mô-men gượng dậy, cỡ của action.

**SAU:**

```python
BALANCE_CAR_MASS = 52.130
"""Tổng khối lượng [kg] — khung 47.140 (base 36.537 + cover 10.603) + hai bánh 4.990."""

BALANCE_CAR_COM_HEIGHT = 0.1715
"""Trọng tâm KHUNG so với trục bánh [m] — chiều dài con lắc ngược thật của bài toán.

Dùng trọng tâm khung chứ không phải trọng tâm cả xe (0.1551 m): bánh xe quay quanh trục nên
không góp phần vào mô-men lật, chỉ khung mới lật.
"""
```

Kèm bảng khối lượng từng thân ngay trong docstring, để lần sau đối chiếu là thấy ngay.

---

### Lỗi 3 — Ba hằng số ma sát cãi nhau trong cùng một file

**TRƯỚC:**

```python
BALANCE_CAR_GROUND_FRICTION = 0.3          # ← giá trị thật sự được dùng
"""...
======  ==================  =================  ===========================
μ       mô-men trượt/bánh   gia tốc tối đa     góc cứu được tối đa
======  ==================  =================  ===========================
0.5     13.8 N·m            4.9 m/s²           26.6° ← mặc định, quá trơn
1.0     27.6 N·m            9.8 m/s²           45.0°
1.2     33.1 N·m            11.8 m/s²          50.2° ← đang dùng      # ← nói 1.2
======  ==================  =================  ===========================
"""
```

Ba giá trị cùng tồn tại: hằng số là **0.3**, bảng ghi "**1.2** đang dùng", và các con số mô-men
trong bảng (13.8 / 27.6 / 33.1) tính từ **khối lượng 27.45 kg đã sai**.

**Vì sao nghiêm trọng:** ma sát quyết định **góc nghiêng lớn nhất còn cứu được**, qua
`tan θ_max = μ`:

| μ | mô-men trượt/bánh (m=52.13) | gia tốc tối đa | góc cứu được tối đa |
|---|---|---|---|
| **0.3** ← đang chạy | 15.73 N·m | 2.94 m/s² | **16.70°** |
| 0.5 (mặc định Isaac Lab) | 26.21 N·m | 4.91 m/s² | 26.57° |
| **1.0** ← chọn | **52.42 N·m** | 9.81 m/s² | **45.00°** |
| 1.2 | 62.90 N·m | 11.77 m/s² | 50.19° |

Nhưng termination cho phép nghiêng tới **40°** mới kết thúc episode. Nghĩa là toàn bộ vùng
**16.7° → 40°** là những tư thế mà **vật lý không cho cứu**, env vẫn giữ xe sống và vẫn chấm
điểm policy trong đó. Policy bị bắt học từ những ván thua sẵn, và không có gì trong log chỉ ra
điều đó.

Đối chiếu cụ thể: gượng dậy từ 30° cần gia tốc `g·tan30° = 5.66 m/s²`. Ở μ = 0.3 gia tốc tối đa
là **2.94 m/s²** — không đủ, dù bơm bao nhiêu mô-men cũng vô ích.

**SAU:**

```python
BALANCE_CAR_GROUND_FRICTION = 1.0
```

Chọn 1.0 để **khớp với ngưỡng ngã 40°**: mọi tư thế mà env còn cho tồn tại đều là tư thế cứu
được. Đặt ở ground plane với `friction_combine_mode="multiply"`, còn ma sát bánh được random
0.7–1.3 ở `EventCfg` → μ hiệu dụng 0.7–1.3.

> Đổi `"max"` → `"multiply"`: chế độ `"max"` của bản cũ **nuốt luôn phần random**, mọi env chạy
> đúng một hệ số. `"multiply"` là chế độ mà mẫu locomotion dùng.

---

### Lỗi 4 — Action scale gấp 10 lần ngưỡng trượt

**TRƯỚC:**

```python
BALANCE_CAR_TRACTION_TORQUE = 150.0
"""Mô-men [N·m] mà một bánh bắt đầu trượt, ở ``BALANCE_CAR_GROUND_FRICTION``. Hiện ≈ 33.1.

Tính thẳng từ ``μ · (m·g/2) · R`` chứ không gõ số cứng: chỉnh ma sát thì trần mô-men tự đi
theo, khỏi phải nhớ sửa hai chỗ.
"""
```

Docstring nói "không gõ số cứng" ngay bên trên một số gõ cứng, và nói "≈ 33.1" bên cạnh giá trị
150.0.

Con số thật theo chính công thức đó:

| Dùng | μ | m | Kết quả |
|---|---|---|---|
| Công thức, hằng số thật của file | 0.3 | 27.45 (sai) | **8.28 N·m** |
| Công thức, μ thật + m đúng | 0.3 | 52.13 | **15.73 N·m** |
| Con số docstring viện dẫn | 1.2 | 27.45 (sai) | 33.12 N·m |
| **Giá trị gõ cứng** | — | — | **150.0** |

**Vì sao sai:** trên ngưỡng trượt, bánh chỉ **quay trượt** chứ không thêm gia tốc. Với scale
150 và ngưỡng thật 15.73, **89.5% dải action là vùng chết** — mọi giá trị trong đó cho cùng một
kết quả vật lý, gradient bằng 0, mà policy vẫn phải mò trong đó.

**SAU:**

```python
BALANCE_CAR_TRACTION_TORQUE = BALANCE_CAR_GROUND_FRICTION * (BALANCE_CAR_MASS * 9.81 / 2) * BALANCE_CAR_WHEEL_RADIUS
# = 1.0 × (52.130 × 9.81 / 2) × 0.205 = 52.42 N·m
```

Đối chiếu để thấy 52.42 là đủ:

* giữ **tĩnh** ở 10°: `m·g·l·sin10°` = 15.2 N·m cho cả hai bánh → 7.6 mỗi bánh;
* **gượng dậy từ 30°** (con số định cỡ): `m·a·R/2` = **30.26 N·m** mỗi bánh — còn dư 42%.

---

### Lỗi 5 — Hướng tiến là +Y, và 159 dòng viết ra để chống lại điều đó

**TRƯỚC** — lệnh tiến đặt vào `lin_vel_x`, rồi viết một command class mới để sửa hậu quả:

```python
# CommandsCfg
base_velocity = BalanceCarVelocityCommandCfg(
    ranges=BalanceCarVelocityCommandCfg.Ranges(
        lin_vel_x=(-1.5, 1.5),      # lệnh TIẾN đặt ở X
        lin_vel_y=(0.0, 0.0),
        ang_vel_z=(-1.0, 1.0),
    ),
)
```

`mdp/commands.py` (159 dòng) tồn tại chỉ để sửa lại metric và mũi tên debug cho đúng trục.
Kèm theo là `track_lin_vel_exp` tự viết đọc `imu.data.lin_vel_b[:, 1]`.

**Vì sao sai:** thân xe tiến theo **+Y**. Quy ước của Isaac Lab là forward = +X, nên:

* `mdp.track_lin_vel_xy_exp` so `command[:, :2]` với `root_lin_vel_b[:, :2]` → so lệnh tiến với
  vận tốc NGANG. Policy bám hoàn hảo vẫn cho `error_vel_xy = |cmd|·√2`;
* hàm vẽ mũi tên dùng `atan2(v[1], v[0])`; với lệnh `(vx, 0)` nó ra `atan2(0, vx) = 0` → **mũi
  tên XANH LÁ chỉ ngang hông xe**, lệch 90° so với hướng được lệnh chạy.

**SAU** — không viết class nào, chỉ đổi chỗ lệnh:

```python
ranges=mdp.UniformVelocityCommandCfg.Ranges(
    lin_vel_x=(0.0, 0.0),                     # thành phạt trượt ngang miễn phí
    lin_vel_y=(-1.5, 1.5),                    # ← lệnh TIẾN nằm đây
    ang_vel_z=(-1.0, 1.0),
)
```

Một dòng đó làm đúng ngay **ba** thứ cùng lúc:

1. `track_lin_vel_xy_exp` so đúng trục — không cần bọc lại;
2. `error_vel_xy` thành số **đọc được**, tiến về 0 khi bám tốt;
3. mũi tên xanh lá `atan2(vy, 0) = 90°` → chỉ đúng dọc **body +Y**.

Và thành phần X (luôn bằng 0) trở thành **phạt trượt ngang miễn phí** — đúng thứ ta muốn cho xe
vi sai không đi ngang được.

Xoá `mdp/commands.py` (159 dòng).

#### Dấu của action — suy ra được, không cần thử

Trước đây dấu này là câu hỏi treo, phải chạy `play.py` nhìn tận mắt. Thật ra suy được từ USD:

```
cả hai khớp cùng trục world (−1, 0, 0)
ω = (−ω, 0, 0),  r (tâm → điểm tiếp xúc) = (0, 0, −R)
lăn không trượt:  v_tiếp_xúc = v_tâm + ω × r = 0
                  v_tâm = −(ω × r) = (0, +ωR, 0)
```

→ **mô-men dương = xe tiến (+Y)**, và vì hai khớp cùng trục nên **cả hai `scale` cùng dương**.

> ⚠️ Onshape ghi khung khớp theo thứ tự người vẽ chọn mate, nên **chiều trục đổi giữa các lần
> export kể cả khi hình học không đổi**. Một bản export trước của chính file CAD này cho
> `Revolute_1 = (−1,0,0)` nhưng `Revolute_2 = (+1,0,0)`. Sau **mỗi** lần sinh USD phải quét lại:
> cùng dấu → hai scale cùng dương, khác dấu → một trong hai phải âm.

---

### Lỗi 6 — Quan sát: yaw tuyệt đối, góc bánh vô hạn, critic trùng policy

**TRƯỚC** (12 chiều):

```python
joint_pos      = ObsTerm(func=mdp.joint_pos, ...)      # 2 — góc bánh THÔ
joint_vel      = ObsTerm(func=mdp.joint_vel, ...)      # 2
pitch_angl_p   = ObsTerm(func=obs_body_pitch, ...)     # 1 — Euler
pitch_angl_r   = ObsTerm(func=obs_body_roll, ...)      # 1 — Euler
pitch_angl_y   = ObsTerm(func=obs_body_yaw, ...)       # 1 — YAW TUYỆT ĐỐI
l_vel          = ObsTerm(func=lin_vel_b, ...)          # 1
a_vel          = ObsTerm(func=angl_vel_b, ...)         # 1
velocity_commands = ObsTerm(func=mdp.generated_commands, ...)  # 3
```

Cộng thêm một nhóm `critic` **giống nhóm `policy` từng ký tự**.

Ba vấn đề:

1. **Yaw tuyệt đối** là một hướng cố định trong world. Policy học bám hướng đó thì không rẽ
   được. Isaac Lab dùng `projected_gravity` thay Euler: bắt cả roll lẫn pitch, không gimbal,
   không mang thông tin yaw.
2. **`joint_pos` của bánh** là toạ độ cyclic — động lực học của xe không phụ thuộc vào nó — mà
   lại **tăng vô hạn** khi xe chạy. Đưa vào mạng là đưa một đại lượng không dừng; chuẩn hoá
   quan sát cũng không cứu được vì phân phối dịch đi mãi trong lúc train.
3. **Nhóm `critic` trùng `policy`** không mua thêm gì, chỉ nhân đôi chỗ để hai bên trôi khỏi
   nhau.

Ngoài ra `enable_corruption = False` — **không có nhiễu quan sát nào**. Policy học trên số đo
hoàn hảo, gãy ngay khi gặp cảm biến thật, và dễ bám vào chi tiết vi mô của solver thay vì vào
vật lý.

**SAU** (16 chiều, đúng bộ của mẫu locomotion, bỏ height scan vì sàn phẳng):

```python
base_lin_vel      = ObsTerm(func=mdp.base_lin_vel,      noise=Unoise(-0.1,  0.1))   # 3
base_ang_vel      = ObsTerm(func=mdp.base_ang_vel,      noise=Unoise(-0.2,  0.2))   # 3
projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(-0.05, 0.05))  # 3
velocity_commands = ObsTerm(func=mdp.generated_commands, params={...})              # 3
joint_vel         = ObsTerm(func=mdp.joint_vel_rel,     noise=Unoise(-1.5,  1.5))   # 2
actions           = ObsTerm(func=mdp.last_action)                                   # 2
# enable_corruption = True
```

`joint_pos` **cố ý bỏ** (xem lý do 2 ở trên). Nhóm `critic` bỏ hẳn.

> Tên term `actions` và `velocity_commands` là **bắt buộc**: tầng navigation ghi đè đúng hai
> term này để tiêm lệnh của nó xuống.

---

### Lỗi 7 — Termination chỉ đo roll, bỏ sót lật ngang

**TRƯỚC** — `mdp/terminations.py`:

```python
def reset_when_fall(env):
    roll, _, _ = euler_xyz_from_quat(robot.data.root_quat_w)
    return torch.abs(roll - 0.0) > math.pi / 180 * 40
```

Chỉ đo **roll**. Xe lật ngang (pitch) thì roll vẫn bằng 0 và episode vẫn tiếp tục. Euler còn có
điểm gãy và gimbal.

**SAU** — term có sẵn, đo góc giữa trục z của thân và phương thẳng đứng:

```python
base_fell = DoneTerm(func=mdp.bad_orientation, params={"limit_angle": BALANCE_CAR_FALL_ANGLE})
# mdp.bad_orientation = torch.acos(-projected_gravity_b[:, 2]).abs() > limit_angle
```

Bắt cả lật trước/sau lẫn lật ngang, không có điểm gãy.

---

### Lỗi 8 — Reward tự viết trùng từng dòng với term có sẵn

**TRƯỚC** — `mdp/rewards.py`:

```python
def cover_flat_l2(env, asset_cfg=SceneEntityCfg("robot")):
    asset = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.projected_gravity_b[:, :2]), dim=1)
```

Isaac Lab, `isaaclab/envs/mdp/rewards.py`:

```python
def flat_orientation_l2(env, asset_cfg=SceneEntityCfg("robot")):
    asset: RigidObject = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.projected_gravity_b[:, :2]), dim=1)
```

**Giống từng ký tự.** Kèm theo là 10 hàm khác trong cùng file, trong đó 6 hàm **không được dùng
ở đâu** nhưng vẫn mang docstring nói về bài toán cũ:

| Hàm chết | Docstring nói về |
|---|---|
| `reward_angle_r` | tư thế đứng là `\|roll\| = 90°` (CAD cũ, đã vẽ lại) |
| `reward_angle_y` | bám một hướng yaw TUYỆT ĐỐI |
| `reward_vel` | thưởng vận tốc bánh về 0 |
| `bonus_reward` | dải cứng ±2°, không có đạo hàm |
| `penalty_when_center_of_env_l2` | kéo xe về giữa env |
| `reward_li_vel` | thưởng vận tốc tiến về 0 |

Ba hàm cuối thưởng cho việc **đứng yên**, tức chống lại chính lệnh vận tốc mà env đang dạy.

Còn `track_lin_vel_exp` và `track_ang_vel_exp` mang docstring:

> *"Thân xe bị xoay 90° quanh trục roll (tư thế đứng là `|roll| = 90°`)"*

— mô tả bản CAD **cũ**, trong khi `balance_car_cfg.py` cùng lúc khẳng định bản mới đứng ở
quaternion đơn vị. Hai file nói ngược nhau về cùng một robot.

**SAU** — toàn bộ là term có sẵn:

```python
track_lin_vel_xy_exp = RewTerm(func=mdp.track_lin_vel_xy_exp, weight=2.0,  params={"std": 0.5})
track_ang_vel_z_exp  = RewTerm(func=mdp.track_ang_vel_z_exp,  weight=1.0,  params={"std": 0.5})
flat_orientation_l2  = RewTerm(func=mdp.flat_orientation_l2,  weight=-2.0)
ang_vel_xy_l2        = RewTerm(func=mdp.ang_vel_xy_l2,        weight=-0.05)
lin_vel_z_l2         = RewTerm(func=mdp.lin_vel_z_l2,         weight=-2.0)
dof_torques_l2       = RewTerm(func=mdp.joint_torques_l2,     weight=-1.0e-5)
dof_acc_l2           = RewTerm(func=mdp.joint_acc_l2,         weight=-2.5e-7)
action_rate_l2       = RewTerm(func=mdp.action_rate_l2,       weight=-0.01)
termination_penalty  = RewTerm(func=mdp.is_terminated,        weight=-100.0)
```

#### Vì sao bỏ term thưởng "phẳng tuyệt đối"

Bản cũ có `cover_flat_bonus` = đỉnh nhọn `exp(-θ²/std²)`. Ở một lần chỉnh trước đó nó từng để
**std 0.05 (~±3°) với trọng số 3.5**, và đó chính là nghiệm suy biến:

> Xe cân bằng **bắt buộc phải nghiêng để tăng tốc** — độ nghiêng chính là tín hiệu điều khiển,
> `θ = atan(a/g)`. Với đỉnh nhọn ±3° và trọng số 3.5, chỉ cần gia tốc 0.5 m/s² (nghiêng 2.9°)
> là mất 2.24 điểm/giây, trong khi bám lệnh giỏi nhất cũng chỉ được 2.0. **"Đứng im và phẳng
> lì" trở thành nghiệm TỐI ƯU.**

Nay chỉ giữ `flat_orientation_l2` ở **−2.0**: nghiêng 5.8° (gia tốc 1 m/s²) chỉ tốn 0.02
điểm/giây. Thứ làm nắp xe hết "rung lag" trong video là `ang_vel_xy_l2` (damp nhịp lắc), không
phải phạt góc mạnh hơn.

#### Đọc trọng số reward cho đúng

Isaac Lab nhân reward với `step_dt` trong `RewardManager.compute`, nên **mọi trọng số là tốc độ
theo GIÂY, không phải theo bước**. `track_lin_vel_xy_exp` trọng số 2.0 = tối đa 2.0 điểm mỗi
giây = 0.02 mỗi bước ở 50 Hz. Đây cũng là lý do `Episode_Reward/<term>` trong log đọc được
trực tiếp như "điểm trung bình mỗi giây".

---

### Lỗi 9 — Nhịp điều khiển 30 Hz

**TRƯỚC:** `self.decimation = 2`, `self.sim.dt = 1/60` → **30 Hz**.

**SAU:** `self.decimation = 4`, `self.sim.dt = 1/200` → **50 Hz** (giống hệt mẫu locomotion).

Xem bảng ở §1.3: 30 Hz cho `e^(λT) = 1.251` — sai lệch phồng 25% giữa hai bước điều khiển và
chỉ 4.46 bước cho mỗi lần biên độ nhân e. 50 Hz đưa về 1.144 / 7.44 bước, ngang mức mà con lắc
đơn (bài đã chạy tốt) đang có ở 60 Hz.

---

### Lỗi 10 — Không có nhiễu miền nào

**TRƯỚC** — `EventCfg` chỉ có hai mục reset:

```python
reset_base   = EventTerm(func=mdp.events.reset_root_state_uniform, ...)
reset_wheels = EventTerm(func=mdp.events.reset_joints_by_offset, ...)
```

Không random ma sát, không random khối lượng, không random trọng tâm, không xô đẩy. Mọi env là
**cùng một con robot trên cùng một mặt sàn**, và policy được phép học một nghiệm khớp chính xác
với đúng bộ tham số đó.

**SAU** — thêm ba mục `startup` và một mục `interval`:

```python
physics_material = EventTerm(func=mdp.randomize_rigid_body_material, mode="startup",
    params={"asset_cfg": SceneEntityCfg("robot", body_names=["wheel", "wheel_01"]),
            "static_friction_range": (0.7, 1.3), "dynamic_friction_range": (0.6, 1.1),
            "restitution_range": (0.0, 0.0), "num_buckets": 64})

add_frame_mass = EventTerm(func=mdp.randomize_rigid_body_mass, mode="startup",
    params={"asset_cfg": SceneEntityCfg("robot", body_names="Group_1"),
            "mass_distribution_params": (-5.0, 5.0), "operation": "add"})

frame_com = EventTerm(func=mdp.randomize_rigid_body_com, mode="startup",
    params={"asset_cfg": SceneEntityCfg("robot", body_names="Group_1"),
            "com_range": {"x": (-0.02, 0.02), "y": (-0.02, 0.02), "z": (-0.02, 0.02)}})

push_robot = EventTerm(func=mdp.push_by_setting_velocity, mode="interval",
    interval_range_s=(6.0, 12.0),
    params={"velocity_range": {"x": (-0.3, 0.3), "y": (-0.3, 0.3)}})
```

Ba tham số này đúng là ba thứ sai nhiều nhất khi mang sang phần cứng thật. Đặc biệt
`frame_com`: **lệch trọng tâm là sai số hiệu chỉnh kinh điển của xe cân bằng** — xe đứng yên ở
một góc nghiêng khác 0. ±2 cm trên cánh tay 17 cm là ±6.6° điểm cân bằng.

Nhiễu reset cũng mạnh lên: roll ±0.15 rad (±8.6°) thay vì ±0.1, thêm pitch ±0.05, thêm vận tốc
ban đầu.

---

### Lỗi 11 — Hai nguồn dữ liệu nói hai chuyện khác nhau

**TRƯỚC** — một cảm biến IMU gắn ở trục bánh, với đường dẫn prim ghim cứng theo CAD:

```python
imu = ImuCfg(
    prim_path="/World/envs/env_.*/Robot/robot/robot/Group_1",
    offset=ImuCfg.OffsetCfg(pos=(0.0, 0.0, BALANCE_CAR_AXLE_OFFSET), rot=(1.0, 0.0, 0.0, 0.0)),
)
```

Và các term đọc **ba nguồn khác nhau** cho cùng một robot:

| Term | Đọc từ |
|---|---|
| `cover_flat_l2` | `robot.data.projected_gravity_b` (root) |
| `reward_roll_rate` | `imu.data.ang_vel_b` (IMU) |
| `track_lin_vel_exp` | `imu.data.lin_vel_b` (IMU) |
| `track_ang_vel_exp` | `robot.data.root_ang_vel_w` (world) |

Trọng tâm cao hơn trục bánh 0.17 m, nên khi thân lắc với tốc độ góc ω thì hai điểm chênh nhau
`l·ω` — ở ω = 0.6 rad/s là **0.25 m/s**, đủ để metric và reward kể hai câu chuyện khác hẳn nhau
về cùng một policy.

**SAU** — bỏ hẳn IMU. Một nguồn duy nhất là root state, đúng như mọi task locomotion của Isaac
Lab. Bỏ luôn chuỗi prim path ghim cứng — thứ phải sửa mỗi lần đổi tên thân trong CAD.

> Mất mát: `root_lin_vel_b` là vận tốc **trọng tâm**, có lẫn phần lắc. Nhưng đó là vật lý thật
> (trọng tâm thật sự chuyển động như vậy) và là đại lượng mà **toàn bộ** term của Isaac Lab
> dùng thống nhất. Đổi lại được sự nhất quán giữa quan sát, reward và metric.

---

### Lỗi 12 — Navigation: chép tay quan sát, 13/17 reward chết, 357 dòng trùng lặp

#### 12.1. Chép tay danh sách quan sát của tầng thấp

**TRƯỚC** — `navigation_pretrained_env_cfg.py` khai lại toàn bộ 8 term:

```python
@configclass
class LowLevelObservationsCfg(ObsGroup):
    """This must match the observation space that the balance policy was trained on."""
    joint_pos = ObsTerm(func=mdp.joint_pos, ...)
    joint_vel = ObsTerm(func=mdp.joint_vel, ...)
    pitch_angl_p = ObsTerm(func=obs_body_pitch, ...)
    ...
    # Thứ tự term phải khớp đúng PolicyCfg của tầng thấp.
```

Đây là một ràng buộc **không ai kiểm được**. Sửa quan sát ở `balance_env_cfg.py` mà quên sửa ở
đây thì policy nhận vào một vector trộn sai thứ tự, chạy trơn tru và cho kết quả vô nghĩa —
không có gì báo lỗi.

**SAU** — dùng thẳng chính nhóm đó, đúng cách mẫu anymal_c làm:

```python
low_level_observations=LOW_LEVEL_ENV_CFG.observations.policy
```

Không còn gì để lệch.

#### 12.2. Hai dòng ghi đè bị comment

```python
# cfg.low_level_observations.actions.func = lambda dummy_env: last_action()
# cfg.low_level_observations.actions.params = dict()
```

Lúc đó **comment như vậy là đúng** — quan sát tầng thấp của bản cũ không hề có term `actions`.
Nhưng đó chính là cái bẫy của việc chép tay: hôm nào thêm một term vào tầng thấp mà quên bỏ
comment ở đây thì `mdp.last_action` sẽ trả về action **3 chiều của tầng cao** thay vì mô-men
**2 chiều** của tầng thấp.

Nay quan sát tầng thấp có `actions` (theo mẫu Isaac Lab) nên hai dòng này đã được bỏ comment và
là bắt buộc.

#### 12.3. 13 trong 17 hàm reward là di sản của thiết kế đã bỏ

`navigation/mdp/rewards.py` có 17 hàm, chỉ 3 hàm được dùng. Mười ba hàm còn lại thuộc thiết kế
"chạy tới một đích đứng yên" đã bỏ từ lâu: `goal_progress_reward`, `position_reached_bonus`,
`velocity_towards_goal`, `heading_alignment_reward`, `upright_reward`, `tilt_penalty`,
`navigation_velocity_reward`, `forward_velocity_tracking`, `lateral_velocity_penalty`,
`velocity_goal_alignment`, `yaw_rate_penalty`, `joint_velocity_penalty`, `heading_command_error_abs`.

`goal_progress_reward` còn giữ trạng thái trong `env.extras["prev_dist"]` — một dict **dùng
chung cho mọi env** và không được dọn lúc reset, nên ngay sau mỗi lần reset nó cho một cú
thưởng/phạt rác.

**SAU** — còn 4 hàm, đúng 4 hàm đang dùng: `path_position_exp`, `path_heading_exp`,
`path_lateral_l2`, `position_command_error_tanh`.

#### 12.4. `navigation_env_cfg.py` — 357 dòng chép lại scene

Bản cũ khai lại **toàn bộ** scene, ground plane, IMU, actions, và hai nhóm observation cho task
nav học-từ-đầu. Hai bản chép sau đó trôi khỏi nhau: ma sát, nhịp điều khiển, ngưỡng ngã ở hai
nơi không còn giống nhau.

**SAU** — 122 dòng, là **một biến thể của** `BalanceCarEnvCfg`: cùng scene, cùng action, cùng
event, cùng termination. Chỉ đổi ba thứ — lệnh (`UniformPose2dCommandCfg`), quan sát, reward.

> Không có term thưởng **hướng cuối** ở task này. `UniformPose2dCommand` tính sai số hướng theo
> `data.heading_w` — góc của **body +X** trong world — mà xe này tiến theo **body +Y**, nên sai
> số đó luôn lệch một hằng số 90°. Đích chỉ có vị trí, không có hướng.

#### 12.5. Nhịp tầng cao

| | Trước | Sau |
|---|---|---|
| `decimation` | `LOW_LEVEL.decimation × 5` = 10 | `LOW_LEVEL.decimation × 5` = 20 |
| Nhịp thật | 60/10 = **6 Hz** | 200/20 = **10 Hz** |

Mẫu anymal_c dùng ×10 (5 Hz); ở đây ×5 vì mục tiêu chạy liên tục chứ không đứng yên, lệnh cập
nhật thưa quá thì xe cắt cua.

---

### Lỗi 13 — Bộ tham số PPO chậm hơn mặc định 5–10 lần

**TRƯỚC / SAU:**

| Tham số | Trước | Sau | Vì sao |
|---|---|---|---|
| `num_learning_epochs` | 1 | **5** | mỗi rollout chỉ được dùng 1 lần → tốn gấp 5 lần số mẫu cho cùng tiến bộ |
| `num_mini_batches` | 64 | **4** | batch quá nhỏ → gradient nhiễu |
| `learning_rate` | 1e-4 | **1e-3** | chậm gấp 10 lần |
| `schedule` | `"adam"` | **`"adaptive"`** | xem ghi chú dưới |
| `init_noise_std` | 0.2 | **1.0** | 0.2 gần như tắt thăm dò ngay từ đầu |
| `num_steps_per_env` | 100 | **24** | rollout dài không cần thiết khi episode 20 s |
| `max_iterations` | 400 | **1000** | |
| `actor_hidden_dims` | [256, 512, 256] | **[128, 128, 128]** | theo `AnymalCFlatPPORunnerCfg` |
| `critic_hidden_dims` | [256, 512, 512, 256] | **[128, 128, 128]** | |
| `activation` | relu | **elu** | |
| `entropy_coef` | 0.01 | **0.005** | |

> **`schedule="adam"` KHÔNG phải adaptive.** `rsl_rl/algorithms/ppo.py:259` viết
> `if self.desired_kl is not None and self.schedule == "adaptive":` — với bất kỳ giá trị nào
> khác, learning rate đứng yên và `desired_kl=0.01` là **dòng config chết**.

Giữ nguyên `actor_obs_normalization = True` / `critic_obs_normalization = True` — chỗ này
**khác** mẫu anymal (để `False`) và là có lý do: quan sát ở đây trộn hai thang rất lệch nhau,
tốc độ bánh tới ±40 rad/s bên cạnh `projected_gravity` trong [−1, 1]. Không chuẩn hoá thì lớp
đầu bị tốc độ bánh át.

`experiment_name = "carbalance_ppo"` **giữ nguyên** — tầng navigation dò policy đã export theo
đường dẫn đó.

---

## 3. Cây file trước / sau

```
TRƯỚC (2707 dòng)                           SAU (1893 dòng)
balance_car/                                balance_car/
├── balance_car_cfg.py            190       ├── balance_car_cfg.py            215
├── balance_env_cfg.py            366       ├── balance_env_cfg.py            405
├── __init__.py                   127       ├── __init__.py                   171
├── mdp/                     ← XOÁ HẲN      ├── agents/rsl_rl_ppo_cfg.py       63
│   ├── commands.py               159       └── navigation/
│   ├── observations.py            68           ├── navigation_env_cfg.py     122
│   ├── rewards.py                235           ├── navigation_pretrained_…   185
│   ├── terminations.py            35           ├── agents/rsl_rl_ppo_cfg.py   73
│   └── __init__.py                 9           └── mdp/
├── agents/rsl_rl_ppo_cfg.py       38               ├── commands.py           290
└── navigation/                                     ├── rewards.py             96
    ├── navigation_env_cfg.py     357               └── pre_trained_policy…   234
    ├── navigation_pretrained_…   279
    ├── agents/rsl_rl_ppo_cfg.py   62
    └── mdp/
        ├── commands.py           285
        ├── rewards.py            261
        └── pre_trained_policy…   202
```

Phần tự viết còn lại **đúng hai thứ**, và cả hai đều là thứ Isaac Lab không có:

1. `navigation/mdp/commands.py` — `PathCommand`: mục tiêu **chạy liên tục** trên đường tròn hoặc
   hình số 8, khác hẳn `UniformPose2dCommand` (đích đứng yên);
2. `navigation/mdp/rewards.py` — bốn hàm reward bám quỹ đạo đi kèm.

Cộng thêm `navigation/mdp/pre_trained_policy_action.py` — **bản sao có chủ đích** của
`isaaclab_tasks.manager_based.navigation.mdp.PreTrainedPolicyAction`. Không import thẳng được vì
package này nằm trong `isaaclab_assets`, mà `isaaclab_tasks` lại import `isaaclab_assets` lúc
khởi tạo → import chéo sẽ tạo vòng.

---

## 4. Bảng đối chiếu hằng số

| Hằng số | Trước | Sau | Ghi chú |
|---|---|---|---|
| `BALANCE_CAR_MASS` | 27.45 | **52.130** | đọc từ USD |
| `BALANCE_CAR_COM_HEIGHT` | 0.172 | 0.1715 | (đã đúng, làm rõ là COM **khung**) |
| `BALANCE_CAR_GROUND_FRICTION` | 0.3 | **1.0** | khớp ngưỡng ngã 40° |
| `BALANCE_CAR_TRACTION_TORQUE` | 150.0 (gõ cứng) | **52.42** (công thức) | μ·(m·g/2)·R |
| `BALANCE_CAR_TRACK_WIDTH` | — | 0.392 | mới |
| `BALANCE_CAR_MAX_WHEEL_SPEED` | 40.0 (rải rác) | 40.0 (hằng số) | khớp `physxJoint:maxJointVelocity` |
| `BALANCE_CAR_FALL_ANGLE` | 40° (chôn trong hàm) | 40° (hằng số) | giờ đo cả roll lẫn pitch |
| `BALANCE_CAR_MAX_SPEED` | — | 1.5 m/s | mới |
| `friction_combine_mode` | `"max"` | `"multiply"` | `"max"` nuốt phần random |
| `sim.dt` / `decimation` | 1/60 / 2 → 30 Hz | **1/200 / 4 → 50 Hz** | |
| actuator | 2 cái (`wheel_left`, `wheel_right`) | 1 cái (`wheels`) | hai bánh cùng trục, cùng tham số |
| Chiều quan sát | 12 (+ critic trùng) | 16 | |

---

## 5. Ảnh hưởng khi chạy lại

* **Phải train lại từ đầu.** Không gian quan sát (12 → 16) và ngữ nghĩa action đều đổi;
  checkpoint cũ không nạp được.
* **Thư mục log của navigation đổi tên:**
  `cart_v1_navigation` → `balance_car_nav_scratch`,
  `cart_v1_navigation_pretrained` → `balance_car_nav_pretrained`.
  Run cũ vẫn còn trên đĩa nhưng `--load_run` sẽ không thấy nữa (mà cũng không dùng được).
* **Tầng thấp giữ nguyên `carbalance_ppo`** — đây là chủ ý, `latest_exported_policy()` dò theo tên đó.
* **Không phải sinh lại USD.** Không có thay đổi nào ở file USD.
* Thêm task mới `Isaac-Balance-Car-Play` (16 env, tắt nhiễu và xô đẩy, giữ random ma sát/khối
  lượng để thấy policy có thật sự bền hay chỉ khớp một bộ tham số).

### Thứ tự chạy

```bash
# 1. Tầng thấp
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task Isaac-Balance-Car --num_envs 4096 --headless

# 2. play.py — BẮT BUỘC, đây mới là bước sinh ra exported/policy.pt
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
    --task Isaac-Balance-Car-Play --num_envs 16 --load_run <tên_run>

# 3. Tầng cao bám quỹ đạo
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task Isaac-Balance-Car-Navigation-Pretrained --num_envs 2048 --headless
```

---

## 6. Cách kiểm chứng

### 6.1. Số cần theo dõi trong log

| Số | Ý nghĩa | Mong đợi |
|---|---|---|
| `Metrics/base_velocity/error_vel_xy` | **giờ đọc được** — gộp sai số tiến và trượt ngang | tiến về 0 |
| `Episode_Termination/base_fell` | tỉ lệ ngã | giảm dần |
| `Episode_Reward/track_lin_vel_xy_exp` | điểm bám lệnh mỗi giây | tiến tới 2.0 |
| `Episode_Reward/flat_orientation_l2` | độ nghiêng | quanh −0.02, **không** cần về 0 |
| `Mean episode length` | | tiến tới 20 s |

> `flat_orientation_l2` về 0 tuyệt đối là **dấu hiệu xấu**: nghĩa là xe không nghiêng, tức
> không tăng tốc, tức đứng im.

#### ⚠️ `error_vel_xy` phải chia cho 4 mới là sai số thật

Đây là bẫy có sẵn trong Isaac Lab, không phải lỗi của package này.
`UniformVelocityCommand._update_metrics` cộng dồn mỗi bước rồi chia cho **chu kỳ đổi lệnh**:

```python
max_command_step = self.cfg.resampling_time_range[1] / self._env.step_dt   # 5.0 / 0.02 = 250
self.metrics["error_vel_xy"] += torch.norm(...) / max_command_step
```

nhưng metric chỉ được xoá ở `CommandTerm.reset()`, tức ở **cuối episode** (1000 bước), không
phải mỗi lần đổi lệnh. Với `episode_length_s = 20` và `resampling_time_range = (5, 5)` thì số
in ra bị nhân **1000/250 = 4 lần**.

| Đọc trong log | Sai số thật |
|---|---|
| `error_vel_xy = 1.2289` | 0.307 m/s |
| `error_vel_yaw = 0.5997` | 0.150 rad/s |

Kiểm chéo bằng reward: `track_lin_vel_xy_exp` = 1.4139 với trọng số 2.0 → giá trị hàm trung
bình 0.707 → `err = sqrt(-0.25·ln 0.707)` = **0.294 m/s**. Khớp với 0.307. Hai đường tính độc
lập cho cùng một kết quả, nên con số đọc được là đáng tin.

Muốn số in ra đúng luôn thì đặt `resampling_time_range = (20.0, 20.0)` bằng episode, đổi lại
mất đa dạng lệnh. Mẫu locomotion của Isaac Lab cũng dính (20 s / 10 s → nhân 2), nên ở đây giữ
nguyên và ghi lại hệ số.

### 6.2. Nhìn bằng mắt

Chạy `play.py` không có `--headless`. Hai mũi tên giờ cùng một hệ quy chiếu:

| Thấy gì | Kết luận |
|---|---|
| xanh lá và xanh dương **cùng hướng**, dài gần bằng nhau | bám tốt |
| hai mũi tên **ngược nhau** | sai dấu → đảo cả hai `scale` trong `ActionsCfg` |
| xanh dương ngắn tịt, xanh lá dài | xe đứng im, không bám lệnh |

Cả hai xoay theo thân xe — **đúng**, vì chúng là đại lượng trong hệ thân được vẽ ra world.

### 6.3. Kiểm tĩnh (không cần Isaac Sim)

```bash
ruff check  source/isaaclab_assets/isaaclab_assets/balance_car/
ruff format --check source/isaaclab_assets/isaaclab_assets/balance_car/
```

`import isaaclab` **không chạy được** ngoài môi trường Kit (thiếu `carb`), nên phần còn lại kiểm
bằng phân tích AST: mọi tên `mdp.*` phải resolve về `isaaclab/envs/mdp/**`, mọi field cfg phải
tồn tại, mọi relative import phải resolve.

---

## 7. Checklist sau mỗi lần đổi CAD

1. Chạy lại `prepare_usd.py`:
   ```bash
   ./isaaclab.sh -p scripts/ute/prepare_usd.py --package balance_car \
       --floating-base --base-body Group_1 --max-angular-velocity 40 --verify
   ```
2. **Quét lại trục world của hai khớp.** Cùng dấu → hai `scale` cùng dương; khác dấu → một
   trong hai phải âm. Đây là lỗi export lặp đi lặp lại, xem [`docs/ute/wheeled_biped_usd_fixes.md`](../../../../docs/ute/wheeled_biped_usd_fixes.md).
3. **Quét lại khối lượng từng thân** và cập nhật bảng trong docstring của `balance_car_cfg.py`.
   Đừng tin bảng cũ — lần này chỉ một trong bốn con số sai và nó tồn tại qua nhiều lần đọc.
4. Đo lại `AXLE_OFFSET`, bán kính bánh, `SPAWN_HEIGHT` (= |offset| + R + ~5 mm).
5. Tính lại `COM_HEIGHT` = trọng tâm **khung** − cao độ trục bánh, rồi kiểm `λ` và `e^(λ/f_ctrl)`.
   Vượt ~1.2 thì phải nâng nhịp điều khiển.
6. Kiểm tên thân (`Group_1`, `wheel`, `wheel_01`) — `EventCfg` tra theo tên này.

---

## 8. Liên quan

* [`../../../../docs/ute/wheeled_biped_usd_fixes.md`](../../../../docs/ute/wheeled_biped_usd_fixes.md) — các lỗi export Onshape lặp lại
* [`balance_env_cfg.py`](balance_env_cfg.py) — docstring module ghi lại phần "vì sao viết lại
  toàn bộ" và quy ước trục
* [`../cart_pendulum/cart_pendulum_env_cfg.py`](../cart_pendulum/cart_pendulum_env_cfg.py) — hằng
  số `CONTROL_RATE_HZ`, cùng phương pháp phân tích `e^(λT)` áp cho họ con lắc
