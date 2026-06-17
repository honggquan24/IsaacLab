# V5 — Sửa lỗi bánh xe không quay

> Ngày: 2026-06-14 · Robot: V5 bipedal wheel (5-bar, USD export thẳng từ Onshape, **không** qua URDF)

## 1. Triệu chứng

Bánh xe **không quay** dù đã cấu hình đúng trên giấy:

- Action: `JointVelocityActionCfg(joint_names=[right_wheel_joint, left_wheel_joint], scale=10/15)`
- Actuator: `ImplicitActuatorCfg(stiffness=0, damping=25)` (chuẩn velocity drive)
- DIAG cho thấy `vel_target=+10` đặt đúng, `damping=25`, nhưng `vel_actual≈0`.

## 2. Phương pháp chẩn đoán

Dùng test treo robot cố định ở h=1m (`test_action_v5.py --fix_base`) để cô lập hành vi khớp/bánh khỏi cân bằng & rơi. 5 phase: zero → hip sweep → wheel FWD → wheel BWD → combined.

Bảng quan sát quyết định (khi **treo**, h=1.000m):

| Tình huống | hip | lệnh bánh | bánh thực |
|---|---|---|---|
| P1 hip quét, bánh tắt | quét ±0.86 | 0 | ±11 (quán tính, **bình thường** cho bánh tự do) |
| P2/3 hip cố định, bánh chạy | 0 | ±10 | **≈0** ❌ |

## 3. Các giả thuyết SAI (đã loại)

| Giả thuyết | Bác bỏ bằng |
|---|---|
| Singular config tại hip=0 khóa bánh | hip=+0.86 (xa 0) bánh **vẫn** ≈0 |
| Quán tính bánh khổng lồ (lỗi đơn vị) | runtime: mass 0.45 kg, Idiag ~4e-4 — rất nhỏ |
| Sai loại actuator (Implicit vs IdealPD) | đúng một nửa — xem dưới |

## 4. Hai nguyên nhân thật (đều phải sửa)

### 4.1. Bánh xe bị nhét trong vòng kín 5-bar
`check_usd_joints.py` dump topology → `close_loop_linear` neo vào body **BÁNH XE**
(`wheel_01`/`wheel`) thay vì coupler (`knee_01`/`knee`):

```
TRƯỚC (sai): base→hip→knee_01→[wheel_joint]→wheel_01→[close_loop]→proxy→knee_02→...
                                    ▲ khớp bánh NẰM TRONG vòng kín → bị khóa
SAU  (đúng): vòng kín = knee_01 ↔ proxy ↔ knee_02 (coupler-coupler), bánh là LÁ tự do
```

→ Fix: `fix_loop_v5.py` đổi `body0` của close_loop từ bánh → coupler, **tính lại
localPos0/localRot0** để giữ nguyên điểm neo world (sai số ~1e-16). Xuất `robot_v5_fixed.usd`.

### 4.2. Khớp bánh THIẾU `DriveAPI` (nguyên nhân quyết định)
USD Onshape **không tạo `UsdPhysics.DriveAPI`** cho khớp bánh → PhysX không có
drive hoạt động → `ImplicitActuator` ghi target/damping nhưng **không có gì áp lực**.

Bằng chứng đắt giá (vòng sim thủ công, `inspect_runtime_v5.py`):
- Áp **effort thuần 50 N·m** + implicit d=25 → bánh vọt lên rail 80 r/s.
  (Nếu drive implicit chạy thì phải cân bằng tại `50 = 25·v → v≈2`.) → drive implicit **trơ**.
- Effort thuần làm bánh quay tới velocity_limit → **khớp tự do về cơ học**, không kẹt.

→ Fix: `add_wheel_drive_v5.py` apply `DriveAPI("angular", type=force, k=0, d=25, maxForce=300)`
vào 2 khớp bánh trong `robot_v5_fixed.usd`.

> Ghi chú: KHÔNG dùng `IdealPDActuator` (effort explicit) cho bánh — **mất ổn định**
> vì quán tính bánh quá nhỏ (4e-4): PD gain cao → overshoot mỗi bước → rail ±80.
> Phải dùng **Implicit** (PhysX giải implicit, ổn định mọi gain) + **bắt buộc có DriveAPI**.

## 5. Kết quả sau khi sửa

```
P2  vel_target=+10.00  vel_actual=+10.00/+10.00   ✓ bám chính xác, ổn định
P3  vel_target=-10.00  vel_actual=-10.00/-10.00   ✓
```

## 6. Trạng thái cấu hình hiện tại

- USD: **`usd_file/robot_v5_fixed.usd`** (gỡ bánh khỏi loop + có DriveAPI bánh)
- `legged_v5_cfg.py` → `LEGGED_ROBOT_V5_USD_PATH` trỏ `robot_v5_fixed.usd`
- Wheel: `ImplicitActuatorCfg(stiffness=0, damping=25, effort_limit_sim=300, velocity_limit_sim=80)`
- close_loop: `ImplicitActuatorCfg(stiffness=10000, damping=500)`

## 7. Quy trình tái tạo (nếu re-export USD từ Onshape)

Mỗi lần có USD mới từ Onshape phải chạy lại 2 bước vá:

```bash
# 1. Gỡ bánh khỏi vòng kín → robot_v5_fixed.usd
./isaaclab.sh -p source/isaaclab_assets/isaaclab_assets/legged_v3/tools/fix_loop_v5.py

# 2. Thêm DriveAPI cho khớp bánh (chỉnh trực tiếp robot_v5_fixed.usd)
./isaaclab.sh -p source/isaaclab_assets/isaaclab_assets/legged_v3/tools/add_wheel_drive_v5.py

# 3. Kiểm tra topology
./isaaclab.sh -p source/isaaclab_assets/isaaclab_assets/legged_v3/tools/check_usd_joints.py \
    source/isaaclab_assets/isaaclab_assets/legged_v3/usd_file/robot_v5_fixed.usd

# 4. Verify bánh quay (treo robot)
./isaaclab.sh -p source/isaaclab_assets/isaaclab_assets/legged_v3/tools/test_action_v5.py --fix_base --headless
```

> Lưu ý môi trường: cần `conda activate env_isaaclab` (python base không có isaacsim/pxr).
> pxr chỉ import được sau khi khởi tạo `SimulationApp`.

## 8. Scripts liên quan

| Script | Chức năng |
|---|---|
| `check_usd_joints.py [file.usd]` | Dump topology joint (parent/child, axis, limit) + truy vết chuỗi động học |
| `inspect_loop_v5.py` | In anchor localPose của close_loop + world transform body |
| `fix_loop_v5.py` | Gỡ bánh khỏi vòng kín, xuất `robot_v5_fixed.usd` |
| `inspect_mass_v5.py` | Soi mass/inertia/drive/limit khớp bánh (so 2 USD) |
| `inspect_runtime_v5.py` | Runtime mass/inertia + test áp effort/velocity-target thẳng |
| `add_wheel_drive_v5.py` | Thêm `DriveAPI(angular)` vào khớp bánh |
| `test_action_v5.py [--fix_base]` | Test 5 phase, treo robot để cô lập khớp/bánh |
