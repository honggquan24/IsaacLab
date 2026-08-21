# Nhánh `demo` — các dự án robot HUTECH trên Isaac Lab

Nhánh này gom toàn bộ dự án robot đang nằm rải rác ở các nhánh (`legged_v3`,
`rotary_pendulum_v2`, `uav`, …) về một chỗ, đổi tên cho thống nhất và sửa lại để
**chạy được** trên máy hiện tại — mục tiêu chính là quay video demo tổng hợp.

## 1. Vì sao nhánh này dựng trên Isaac Lab 2.3.2 chứ không phải 3.0

`main` đã được đồng bộ lên Isaac Lab 3.0.0 (`isaaclab` 4.5.16). Bản 3.0 yêu cầu
**Isaac Sim 6.0.0 + torch 2.10** (xem `docs/source/setup/installation/pip_installation.rst`),
trong khi máy đang cài **Isaac Sim 5.1.0 + torch 2.7.0** và ổ đĩa chỉ còn ~6 GB.
Task chuẩn `Isaac-Cartpole-v0` chạy trên `main` crash ngay ở
`PhysxManager._view = None`; cùng task đó chạy bình thường trên 2.3.2.

Nhánh `demo` vì vậy dựng từ nhánh `uav` (VERSION 2.3.2, `isaaclab` 0.54.3).
Khi nào giải phóng được ~30 GB và cài Isaac Sim 6.0 thì port tiếp lên 3.0
(việc chính sẽ là: quaternion WXYZ → XYZW, `.data.*` trả `wp.array` nên phải bọc
`wp.to_torch()`, và `write_*_to_sim` tách thành `_index` / `_mask`; Isaac Lab có
sẵn hai công cụ hỗ trợ ở `scripts/tools/find_quaternions.py` và
`scripts/tools/wrap_warp_to_torch.py`).

## 2. Môi trường

Dùng conda env `env_ute` (Python 3.11, Isaac Sim 5.1.0, torch 2.7.0+cu128,
rsl-rl-lib 3.1.2).

Trước đây `./isaaclab.sh -p` trong repo này nạp **lẫn lộn hai bản Isaac Lab**:
`isaaclab` lấy từ IsaacLabUTE (do `isaaclab.sh` của 3.0 tự thêm `source/isaaclab`
vào `PYTHONPATH`) còn `isaaclab_assets` / `isaaclab_tasks` lấy từ bản pip editable
trỏ sang `~/Documents/GitHub/IsaacLab` — nên các project trong repo này không bao
giờ được nạp, và lỗi hiện ra rất khó hiểu.

`isaaclab.sh` của nhánh này thêm **tất cả** extension trong `source/` vào
`PYTHONPATH`, nên checkout này luôn thắng bản pip editable. Không cần cài lại
pip, và bản `~/Documents/GitHub/IsaacLab` 2.3.2 vẫn dùng bình thường cho việc
khác (ví dụ dự án `uav`).


### Gói pip đã chỉnh trong `env_ute`

| Gói | Trước | Sau | Vì sao |
| --- | --- | --- | --- |
| `rsl-rl-lib` | 5.0.1 | **3.1.2** | 5.0.1 là bản của Isaac Lab 3.0 và dùng schema cấu hình mới (`cfg["actor"]`), làm `play.py`/`train.py` chết ngay với `KeyError: 'actor'`. `source/isaaclab_rl/setup.py` của 2.3.2 ghim đúng 3.1.2. Bản `~/Documents/GitHub/IsaacLab` (cũng 2.3.2) cần đúng bản này. |
| `lazy_loader` | *(chưa có)* | 0.5 | Isaac Lab 3.0 trên `main` cần; cài thêm không ảnh hưởng 2.3.2. |
| `ruff` | *(chưa có)* | 0.14.10 | đúng bản trong `.pre-commit-config.yaml`, dùng để lint/format. |

Không đụng tới bản pip editable đang trỏ sang `~/Documents/GitHub/IsaacLab`.

Kiểm tra nhanh:

```bash
./isaaclab.sh -p scripts/environments/list_envs.py | grep Isaac-Wheeled-Biped
```

## 3. Đổi tên

| Cũ (rải rác ở các nhánh) | Mới (nhánh `demo`) | Task id cũ → mới |
| --- | --- | --- |
| `legged_v5` | `wheeled_biped` | `Isaac-Legged-V5-*` → `Isaac-Wheeled-Biped-*` |
| `rotary_pendulum_v2` | `rotary_pendulum` | `Isaac-RotaryPendulum-V2-*` → `Isaac-Rotary-Pendulum-*` |
| `cartpole_v1` | `cart_pendulum` | `Isaac-Cartpole-V1-Run` → `Isaac-Cart-Pendulum` |
| `cartpole_v2` | `cart_pendulum_double` | `Isaac-Cartpole-V2-Run` → `Isaac-Cart-Pendulum-Double` |
| `balancecar_v1` | `balance_car` | `Isaac-Cartbalance-V1-*` → `Isaac-Balance-Car-*` |
| `evobot_v1` | `evobot` | `Isaac-Evobot-V1-*` → `Isaac-Evobot-*` |
| `legged_v1`, `legged_v2`, `legged_v3` | *(bỏ)* | vẫn còn trên nhánh `legged_v3`, `dev/robot_legged_v2` |

`experiment_name` trong các file agent cfg **giữ nguyên tên cũ** (ví dụ
`legged_v5_wheel_mimic`), nên checkpoint cũ trong `logs/rsl_rl/` vẫn nạp được.

Cấu trúc mỗi project giờ giống nhau:

```
source/isaaclab_assets/isaaclab_assets/<project>/
├── __init__.py              # gym.register(...)
├── <project>_cfg.py         # ArticulationCfg của robot
├── agents/rsl_rl_ppo_cfg.py # cấu hình PPO
├── mdp/                     # observation / reward / termination / action riêng
├── <task>/                  # env cfg theo từng bài toán
└── usd/                     # file USD (trước đây là usd_file/)
```

Những thứ **không phải code thư viện** đã được đưa ra ngoài package:

- script chạy tay, tool → `scripts/ute/`
- tài liệu → `docs/ute/`
- báo cáo NCKH (LaTeX + hình) → `report/nckh/`

## 4. Không còn vá vào Isaac Lab core

Nhánh `legged_v3` từng vá thẳng vào `source/isaaclab/`
(`BinaryGripperCommandCfg`, `track_lin_vel_xy_l2`, `track_ang_vel_z_l2`).
Đó chính là lý do mọi thứ vỡ khi `main` đồng bộ lên upstream. Nhánh này chuyển
hết về package của dự án:

- `isaaclab_assets/evobot/mdp/commands.py` — `BinaryGripperCommand(Cfg)`
- `isaaclab_assets/evobot/mdp/rewards_velocity.py` — `track_lin_vel_xy_l2`, `track_ang_vel_z_l2`

`source/isaaclab/` trên nhánh này **không có thay đổi nào** so với upstream.
(Hai thay đổi còn lại trên nhánh cũ chỉ là sửa chính tả docstring của
`imu_data.py` và thụt lề trong `velocity_env_cfg.py` — bỏ qua.)

## 5. Chuẩn bị USD cho robot bipedal wheel

Bản export Onshape (`wheeled_biped/usd/wheeled_biped.usd`) chưa chạy được ngay.
Chạy một lần để sinh `wheeled_biped_fixed.usd` (file này bị `.gitignore` bỏ qua
nên đã được `git add -f`):

```bash
./isaaclab.sh -p scripts/ute/wheeled_biped/prepare_usd.py --verify
```

Script vá 4 thứ, idempotent, gộp từ 4 script `usdfix_*` cũ:

1. gỡ bánh xe khỏi vòng kín 5 khâu (`close_loop_linear` neo lại vào coupler),
2. thêm `UsdPhysics.DriveAPI` cho khớp bánh (Onshape không xuất),
3. giới hạn hip ±60°,
4. bật contact reporting cho link hip/knee.

## 6. Chạy

Lệnh train + quay video của **từng task** nằm ngay trong docstring của
`isaaclab_assets/<project>/__init__.py`, đã quy đổi sẵn `--video_length` ra số step
theo tần số điều khiển của task đó — mở file, copy, chạy.


```bash
# huấn luyện
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task Isaac-Wheeled-Biped-Wheel --num_envs 4096 --headless

# xem lại policy đã train
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
    --task Isaac-Wheeled-Biped-Wheel --num_envs 4

# lái bằng bàn phím (pygame) — dùng để quay video
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play_teleop_wheeled_biped.py \
    --task Isaac-Wheeled-Biped-Wheel --num_envs 1 \
    --checkpoint logs/rsl_rl/legged_v5_wheel_mimic/2026-06-17_03-39-14/model_10799.pt

# ít env + khung nhìn gần, để quay video (bộ ghi viewport, không cần camera trên robot)
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
    --task Isaac-Wheeled-Biped-Wheel-Play --num_envs 4 --video --video_length 600
```

> Camera gắn trên robot (`CameraCfg`/`TiledCameraCfg` đặt dưới prim của robot) **không dùng
> được trên máy này**: Isaac Sim 5.1 báo `TypeError: Unable to write from unknown dtype,
> kind=f, size=0` khi `omni.syntheticdata` gắn annotator RGB, kể cả khi chạy kèm
> `--enable_cameras`. Vì vậy `Isaac-Wheeled-Biped-Wheel-Play` chỉ giảm số env và chỉnh khung
> nhìn; video lấy từ bộ ghi viewport của `play.py --video`.

Kiểm tra một task còn dựng được sau khi sửa code:

```bash
./isaaclab.sh -p scripts/ute/smoke_test.py --task Isaac-Wheeled-Biped-Wheel
bash scripts/ute/smoke_test_all.sh          # chạy lần lượt toàn bộ task
```

## 7. Tool

| Thư mục | Nội dung |
| --- | --- |
| `scripts/ute/smoke_test.py`, `smoke_test_all.sh` | dựng thử env, báo shape action/observation |
| `scripts/ute/wheeled_biped/prepare_usd.py` | vá USD robot bipedal wheel |
| `scripts/ute/wheeled_biped/inspect_*.py` | soi USD và trạng thái runtime (khớp, khối lượng, trục, contact, trọng lực, vòng kín, dấu bánh) |
| `scripts/ute/wheeled_biped/simtest_*.py` | chạy sim trần để thử spawn / motor / action |
| `scripts/ute/report/report_*.py` | vẽ đường cong huấn luyện, chọn epoch tốt nhất, cắt log |
| `scripts/ute/evobot/` | test PID, so sánh PID với RL, teleop bàn phím của evoBOT |
| `scripts/ute/balance_car/`, `cart_pendulum/`, `rotary_pendulum/` | script chạy tay từng robot |

## 8. Checkpoint có sẵn trên máy

| Experiment | Nơi lưu | Ghi chú |
| --- | --- | --- |
| `legged_v5_wheel_mimic` | `logs/rsl_rl/legged_v5_wheel_mimic/` | nhiều run tháng 6/2026, tốt nhất `2026-06-17_03-39-14/model_10799.pt` (actor 44→4, critic 77 — khớp đúng env sau khi đổi tên) |
| `legged_v5_obstacle_nav` | `logs/rsl_rl/legged_v5_obstacle_nav/` | 8 run |

Các project còn lại (`rotary_pendulum`, `cart_pendulum`, `cart_pendulum_double`,
`balance_car`, `evobot`) **chưa có checkpoint trên máy này** — muốn quay video thì
phải train lại.

## 9. Những thứ không mang sang nhánh này

| Thứ | Ở đâu | Vì sao |
| --- | --- | --- |
| `legged_v1`, `legged_v2`, `legged_v3` | nhánh `legged_v3`, `dev/robot_legged_v2` | chỉ dùng v5 theo yêu cầu |
| `scripts/environments/navigation/*_anymal.py`, `teleop_navigation.py` | nhánh `legged_v3` | teleop cho ANYmal của upstream, không thuộc dự án nào ở đây |
| `usdfix_*.py` (4 script rời) | nhánh `legged_v3` | đã gộp thành `scripts/ute/wheeled_biped/prepare_usd.py` |
| `mdp/rewards.py` cũ của `cartpole_v2` | nhánh `legged_v3` | env chỉ dùng `rewards1.py`; file này đã thành `mdp/rewards.py` với tên hàm snake_case |
| `--video` mặc định bật, `--seed 1` trong `train.py` | nhánh `legged_v3` | giữ mặc định của upstream; truyền cờ khi cần |
| Sửa chính tả docstring `imu_data.py` | nhánh `legged_v3` | giữ `source/isaaclab/` sạch, không khác upstream |

`uav/` vẫn nằm nguyên trên nhánh này (đến từ nhánh `uav`), chưa đổi tên và chưa
dọn vì không nằm trong phạm vi lần này.

## 10. Kết quả smoke test

Chạy `bash scripts/ute/smoke_test_all.sh --num_envs 2 --steps 5` (21/25 task dựng và bước được):

| Task | Kết quả |
| --- | --- |
| `Isaac-Wheeled-Biped-Wheel` | PASS — act=4, obs policy (44,) / critic (77,) |
| `Isaac-Wheeled-Biped-Wheel-Play` | PASS — act=4, obs như trên |
| `Isaac-Wheeled-Biped-Wheel-NoMimic` | PASS — act=6, obs (46,)/(79,) |
| `Isaac-Wheeled-Biped-Wheel-PIANN` | PASS — act=10, obs (50,)/(83,) |
| `Isaac-Wheeled-Biped-Navigation` | PASS — act=3, obs (16,) |
| `Isaac-Wheeled-Biped-Warehouse-Nav` | PASS — act=3, obs (106,) |
| `Isaac-Wheeled-Biped-Obstacle-Nav` | PASS — act=3, obs (106,) |
| `Isaac-Rotary-Pendulum-Balance` | PASS — act=1, obs (11,)/(13,) |
| `Isaac-Rotary-Pendulum-Balance-Stage1` | PASS — act=1, obs (7,)/(9,) |
| `Isaac-Rotary-Pendulum-Balance-Stage2` | PASS — act=1, obs (11,)/(13,) |
| `Isaac-Cart-Pendulum` | PASS — act=2, obs (4,) |
| `Isaac-Cart-Pendulum-Double` | PASS — act=1, obs (6,) |
| `Isaac-Balance-Car` | PASS — act=2, obs (12,)/(9,) |
| `Isaac-Balance-Car-Navigation` | PASS — act=2, obs (16,)/(16,) |
| `Isaac-Balance-Car-Navigation-Play` | PASS — act=2, obs (16,)/(16,) |
| `Isaac-Evobot-Balance` | PASS — act=5, obs (85,)/(43,) |
| `Isaac-Evobot-Velocity` | PASS — act=5, obs (60,)/(60,) |
| `Isaac-Evobot-Velocity-Play` | PASS — act=5, obs (60,)/(60,) |
| `Isaac-Evobot-Manipulation` | PASS — act=5, obs (59,)/(94,) |
| `Isaac-Evobot-Arm-FineTune` | PASS — act=5, obs (60,)/(60,) |
| `Isaac-Evobot-Gripper-FineTune` | PASS — act=5, obs (60,)/(60,) |
| `Isaac-Balance-Car-Navigation-Pretrained` | thiếu checkpoint tầng thấp |
| `Isaac-Balance-Car-Navigation-Pretrained-Play` | thiếu checkpoint tầng thấp |
| `Isaac-Evobot-Navigation` | thiếu checkpoint tầng thấp |
| `Isaac-Evobot-Navigation-Play` | thiếu checkpoint tầng thấp |

Bốn task cuối **không phải lỗi code**: chúng nạp policy tầng thấp đã export mà máy này chưa
có. Train task tầng thấp tương ứng (`Isaac-Balance-Car`, `Isaac-Evobot-Velocity`) rồi trỏ lại
`policy_path` trong env cfg là chạy được.

Ngoài ra đã kiểm tra riêng:

- `Isaac-Wheeled-Biped-Wheel` nạp được checkpoint thật
  (`model_10799.pt`, actor 44→4, critic 77) và chạy tiếp qua `play.py`.
- Toàn bộ 25 cấu hình agent (`rsl_rl_cfg_entry_point`) khởi tạo được với rsl-rl-lib 3.1.2.
