# PID Extraction from RL Policy

## 🎯 Mục đích

Chuyển đổi trained RL policy (neural network) thành classical PID controller với các tham số Kp, Ki, Kd.

## ⚡ Quick Start

```bash
# 1. Extract PID từ checkpoint
./isaaclab.sh -p source/isaaclab_assets/isaaclab_assets/evobot_v1/navigation/balance/extract_pid_from_rl.py \
    --checkpoint logs/rsl_rl/evobot_v1_velocity/YOUR_RUN/model_XXX.pt \
    --num_trajectories 200 \
    --visualize

# 2. Test extracted PID gains (copy command from output)
./isaaclab.sh -p source/isaaclab_assets/isaaclab_assets/evobot_v1/navigation/balance/test_pid_manual_tune.py \
    --kp_roll X.XX --ki_roll X.XX --kd_roll X.XX \
    --kp_pitch X.XX --ki_pitch X.XX --kd_pitch X.XX
```

## 📁 Files

| File | Mô tả |
|------|-------|
| `extract_pid_from_rl.py` | Main extraction tool |
| `test_pid_manual_tune.py` | PID testing & validation |
| `verify_extraction_logic.py` | Unit test cho extraction logic |
| `extract_pid_example.sh` | Example bash script |
| `PID_EXTRACTION_GUIDE.md` | Hướng dẫn chi tiết |

## 🔬 Phương pháp

### 1. Collect Trajectories
- Chạy trained RL policy với nhiều initial conditions
- Record: states, actions, errors

### 2. Linear Regression
```python
# Feature matrix: [roll_error, roll_integral, roll_vel, pitch_error, pitch_integral, pitch_vel]
X = [e_r, ∫e_r, ė_r, e_p, ∫e_p, ė_p]

# Fit linear model
left_wheel = X @ coef_left
right_wheel = X @ coef_right
```

### 3. Decompose Gains
```python
# Differential drive mapping
pitch_gains = (coef_left[3:6] + coef_right[3:6]) / 2
roll_gains = (coef_left[0:3] - coef_right[0:3]) / 2
```

## 📊 Output

### Console
```
Extracted PID gains:
  Roll:  Kp=1.234, Ki=0.023, Kd=0.456
  Pitch: Kp=1.567, Ki=0.045, Kd=0.678

Regression quality:
  Left wheel:  R²=0.923
  Right wheel: R²=0.915
```

### Files
- `logs/pid_extraction/pid_gains_TIMESTAMP.json` - Extracted gains
- `logs/pid_extraction/pid_extraction_comparison.png` - Visualization (if --visualize)

## 🎓 Interpretation

### R² Score (Coefficient of Determination)
- **R² > 0.9**: Excellent - PID có thể thay thế RL
- **R² > 0.8**: Good - PID xấp xỉ tốt RL policy
- **R² > 0.7**: Acceptable - Cần fine-tune thêm
- **R² < 0.7**: Poor - RL policy quá phức tạp cho PID

### MSE (Mean Squared Error)
- **MSE < 0.01**: Excellent match
- **MSE < 0.05**: Good match
- **MSE > 0.1**: Poor match

## 🎯 Ưu điểm

✅ **Interpretable**: Dễ hiểu hơn neural network
✅ **Real-time**: CPU inference, không cần GPU
✅ **Deployable**: Deploy dễ dàng lên embedded systems
✅ **Tunable**: Có thể fine-tune manually
✅ **Robust**: Không cần observation normalization

## ⚠️ Hạn chế

❌ **Approximation**: PID = linear approximation of RL
❌ **Performance**: Có thể kém hơn RL ở edge cases
❌ **Assumption**: Giả định RL policy gần tuyến tính

## 🔧 Troubleshooting

### R² thấp (< 0.7)

**Nguyên nhân**: RL policy phi tuyến hoặc không đủ data

**Giải pháp**:
```bash
# Collect more data
--num_trajectories 500 --trajectory_length 500

# Reduce regularization
--alpha 0.01
```

### PID không stable

**Giải pháp**:
```bash
# Scale down gains
Kp_new = Kp_extracted * 0.5
Ki_new = Ki_extracted * 0.5
Kd_new = Kd_extracted * 0.5
```

## 📚 Tài liệu

Xem chi tiết trong [PID_EXTRACTION_GUIDE.md](./PID_EXTRACTION_GUIDE.md)

## 🧪 Example

```bash
# Full example
cd /path/to/IsaacLabUTE

# Extract
./isaaclab.sh -p source/isaaclab_assets/isaaclab_assets/evobot_v1/navigation/balance/extract_pid_from_rl.py \
    --checkpoint logs/rsl_rl/evobot_v1_velocity/2026-01-21_08-24-47/model_1455.pt \
    --num_trajectories 200 \
    --trajectory_length 300 \
    --visualize

# Test (example output)
./isaaclab.sh -p source/isaaclab_assets/isaaclab_assets/evobot_v1/navigation/balance/test_pid_manual_tune.py \
    --kp_roll 0.823 --ki_roll 0.011 --kd_roll 0.346 \
    --kp_pitch 1.123 --ki_pitch 0.023 --kd_pitch 0.457
```

## 🔬 Theory Background

RL policy với observations: `[imu_acc, imu_vel, joint_pos, joint_vel, ...]`

Nếu policy học được linear mapping gần optimal:
```
action ≈ K_obs @ observations
```

Thì ta có thể decompose ra PID structure:
```
action ≈ Kp*error + Ki*∫error + Kd*derror/dt
```

Bằng cách identify error terms từ observations và fit linear regression.

## 💡 Use Cases

1. **Deploy to real robot**: PID dễ deploy hơn NN
2. **Understanding policy**: Analyze what RL learned
3. **Baseline comparison**: So sánh RL vs classical control
4. **Hybrid control**: Combine RL + PID
5. **Safety**: Fallback to PID nếu NN fails

---

**Created**: 2026-01-21
**Author**: Claude Code
**Status**: Tested & Verified ✅
