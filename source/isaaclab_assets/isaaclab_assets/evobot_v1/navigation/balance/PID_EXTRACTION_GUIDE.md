# Hướng dẫn Extract PID từ RL Policy

## Tổng quan

Công cụ này cho phép bạn chuyển đổi một RL policy đã train (PPO) thành các tham số PID cổ điển (Kp, Ki, Kd) để deploy trên robot thật hoặc để hiểu cách policy hoạt động.

## Phương pháp

### 1. Linear Regression Analysis

Policy RL học được có thể xấp xỉ bởi control law tuyến tính:

```
action = Kp * error + Ki * ∫error + Kd * (derror/dt)
```

Đối với balance robot (2 bánh):
- **Roll control** (nghiêng trái/phải): điều khiển differential wheel speed
- **Pitch control** (nghiêng trước/sau): điều khiển common wheel speed

Wheel mapping:
```
left_wheel = pitch_control + roll_control
right_wheel = pitch_control - roll_control
```

### 2. Extraction Process

1. **Collect trajectories**: Chạy trained policy với nhiều initial conditions khác nhau
2. **Compute features**: Tính error, integral, derivative cho roll và pitch
3. **Linear regression**: Fit model tuyến tính: `action = f(error, ∫error, derror/dt)`
4. **Extract gains**: Decompose coefficients thành Kp, Ki, Kd cho roll và pitch
5. **Validate**: So sánh performance PID vs RL policy

## Cách sử dụng

### Bước 1: Extract PID từ checkpoint

```bash
# Basic extraction (128 envs, 100 trajectories)
./isaaclab.sh -p source/isaaclab_assets/isaaclab_assets/evobot_v1/navigation/balance/extract_pid_from_rl.py \
    --checkpoint logs/rsl_rl/evobot_v1_velocity/2026-01-19_10-30-40/model_800.pt \
    --num_trajectories 100 \
    --trajectory_length 200

# With visualization
./isaaclab.sh -p source/isaaclab_assets/isaaclab_assets/evobot_v1/navigation/balance/extract_pid_from_rl.py \
    --checkpoint logs/rsl_rl/evobot_v1_velocity/2026-01-19_10-30-40/model_800.pt \
    --num_trajectories 200 \
    --trajectory_length 300 \
    --visualize

# More data for better accuracy
./isaaclab.sh -p source/isaaclab_assets/isaaclab_assets/evobot_v1/navigation/balance/extract_pid_from_rl.py \
    --checkpoint logs/rsl_rl/evobot_v1_velocity/2026-01-19_10-30-40/model_800.pt \
    --num_envs 256 \
    --num_trajectories 500 \
    --trajectory_length 400 \
    --visualize
```

### Bước 2: Kiểm tra output

Script sẽ tạo ra:

1. **Console output**: Extracted PID gains
   ```
   Extracted PID gains:
     Roll:  Kp=1.2345, Ki=0.0234, Kd=0.4567
     Pitch: Kp=1.5678, Ki=0.0456, Kd=0.6789

   Regression quality:
     Left wheel:  R²=0.9234
     Right wheel: R²=0.9156
   ```

2. **JSON file**: `logs/pid_extraction/pid_gains_YYYYMMDD_HHMMSS.json`
   ```json
   {
     "timestamp": "2026-01-21T12:34:56",
     "pid_gains": {
       "kp_roll": 1.2345,
       "ki_roll": 0.0234,
       "kd_roll": 0.4567,
       "kp_pitch": 1.5678,
       "ki_pitch": 0.0456,
       "kd_pitch": 0.6789
     },
     "regression_metrics": {
       "r2_left": 0.9234,
       "r2_right": 0.9156,
       "mse_left": 0.00123,
       "mse_right": 0.00145
     }
   }
   ```

3. **Plot (nếu --visualize)**: `logs/pid_extraction/pid_extraction_comparison.png`
   - So sánh RL policy actions vs PID reconstructed actions

### Bước 3: Test PID gains

```bash
# Test extracted PID gains
./isaaclab.sh -p source/isaaclab_assets/isaaclab_assets/evobot_v1/navigation/balance/test_pid_manual_tune.py \
    --kp_roll 1.2345 --ki_roll 0.0234 --kd_roll 0.4567 \
    --kp_pitch 1.5678 --ki_pitch 0.0456 --kd_pitch 0.6789 \
    --num_envs 1 \
    --test_duration 15.0
```

## Giải thích Metrics

### R² Score (Coefficient of Determination)
- **Range**: 0 đến 1
- **Ý nghĩa**: Tỷ lệ variance được giải thích bởi model
- **Interpretation**:
  - R² > 0.9: Excellent fit (PID approximates RL policy rất tốt)
  - R² > 0.8: Good fit (PID có thể thay thế RL)
  - R² > 0.7: Acceptable (PID cần điều chỉnh thêm)
  - R² < 0.7: Poor fit (RL policy quá phức tạp, PID không đủ)

### MSE (Mean Squared Error)
- **Range**: 0 trở lên (càng thấp càng tốt)
- **Ý nghĩa**: Sai số trung bình giữa RL action và PID action
- **Interpretation**:
  - MSE < 0.01: Excellent match
  - MSE < 0.05: Good match
  - MSE > 0.1: Poor match (cần thêm data hoặc tune alpha)

## Tuning Parameters

### `--alpha` (Ridge Regression Regularization)
- **Default**: 0.1
- **Effect**: Prevents overfitting by penalizing large coefficients
- **Guidelines**:
  - Alpha = 0: No regularization (may overfit)
  - Alpha = 0.01-0.1: Light regularization (recommended)
  - Alpha = 1.0: Strong regularization (smoother but less accurate)

### `--num_trajectories` và `--trajectory_length`
- **More data = better fit** nhưng tốn thời gian
- **Recommendations**:
  - Quick test: 50 trajectories × 100 steps
  - Standard: 100 trajectories × 200 steps
  - High accuracy: 500 trajectories × 400 steps

## Troubleshooting

### Problem: R² < 0.7 (Poor fit)

**Nguyên nhân có thể**:
1. RL policy học được control law phi tuyến
2. Không đủ data đa dạng
3. PID structure không phù hợp với task

**Giải pháp**:
```bash
# Thu thập nhiều data hơn với diverse initial conditions
./isaaclab.sh -p ... \
    --num_trajectories 500 \
    --trajectory_length 500

# Giảm regularization
./isaaclab.sh -p ... \
    --alpha 0.01

# Thử task đơn giản hơn (balance only)
./isaaclab.sh -p ... \
    --task Isaac-Evobot-V1-Balance
```

### Problem: Extracted PID không stable

**Nguyên nhân**:
- RL policy được train với observation normalization
- Extracted gains không scale đúng

**Giải pháp**:
1. Giảm gains xuống 50%: `Kp/2, Ki/2, Kd/2`
2. Tăng dần từng gain riêng lẻ
3. Dùng `--effort_scale 0.5` khi test

### Problem: Gains quá lớn/nhỏ

**Lý do**:
- Action scale khác nhau giữa RL và PID
- RL có clipping [-1, 1]

**Giải pháp**:
- Thử scale extracted gains: multiply by 0.5 hoặc 2.0
- Inspect plots để xem action magnitude

## Ưu điểm PID từ RL

1. **Interpretable**: Dễ hiểu hơn neural network
2. **Real-time**: CPU inference, không cần GPU
3. **Tunable**: Có thể fine-tune manually
4. **Robust**: Không cần observation normalization
5. **Deployable**: Dễ deploy lên embedded systems

## Hạn chế

1. **Approximation**: PID không thể học được policy phức tạp
2. **Linear assumption**: Chỉ tốt nếu RL policy gần tuyến tính
3. **Limited generalization**: PID có thể kém hơn RL ở edge cases

## Advanced: Weight Decomposition (Tương lai)

Ngoài linear regression, có thể phân tích trực tiếp weights của first layer:

```python
# Get first layer weights
first_layer = policy.actor.layers[0].weight  # Shape: (hidden_dim, obs_dim)

# Identify which obs indices correspond to roll_error, pitch_error, etc.
# Extract effective gains from weight matrix
```

Method này nhanh hơn nhưng cần biết chính xác observation structure.

## Ví dụ kết quả thực tế

Với evobot_v1_velocity model_800.pt:
```
Extracted PID gains:
  Roll:  Kp=0.8234, Ki=0.0112, Kd=0.3456
  Pitch: Kp=1.1234, Ki=0.0234, Kd=0.4567

Regression quality:
  Left wheel:  R²=0.9123
  Right wheel: R²=0.9045
```

→ **Kết luận**: PID có thể approximate được ~91% behavior của RL policy!

## References

- Distillation: [Policy Distillation (Rusu et al., 2015)](https://arxiv.org/abs/1511.06295)
- Balance control: [Inverted Pendulum PID Tuning](https://en.wikipedia.org/wiki/PID_controller)
