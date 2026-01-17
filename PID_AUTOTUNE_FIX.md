# PID Auto-Tune Fix - Twiddle Algorithm

## Vấn đề ban đầu
Thuật toán Twiddle **không hội tụ** do các lỗi sau:

### 1. **Step size scaling không hợp lý**
- **Cũ**: `step_scale=1.1`, `step_shrink=0.9`
- **Vấn đề**: Step tăng quá nhanh (10%) khi có improvement, giảm quá chậm (10%) khi không có improvement
- **Hậu quả**: Thuật toán "nhảy" qua local minimum và không bao giờ hội tụ

### 2. **Step size khởi tạo không tỷ lệ**
- **Cũ**: `init_step=0.1` cho TẤT CẢ parameters
- **Vấn đề**:
  - Với `Kp=0.5`: step = 0.1 (20% giá trị) - OK
  - Với `Ki=0.01`: step = 0.1 (1000% giá trị!) - RẤT TỒI
- **Hậu quả**: Ki bị thay đổi quá mạnh, phá vỡ quá trình tìm kiếm

### 3. **Logic tìm kiếm sai**
- **Cũ**: Thay đổi `self.gains[i]` trực tiếp, sau đó cộng/trừ
- **Vấn đề**:
  - Dòng 375: `self.gains[i] -= 2 * self.steps[i]` có thể tạo giá trị âm
  - Không luôn test từ `best_gains`, mà test từ gain vừa thử
- **Hậu quả**: Search space bị lệch, không systematic

### 4. **Evaluation không đủ**
- **Cũ**: `eval_steps=300` (5 giây)
- **Vấn đề**: Quá ngắn để PID controller ổn định, đặc biệt với integral term
- **Hậu quả**: Cost function có noise cao, không phản ánh chất lượng thực

### 5. **Cost function weight không hợp lý**
- **Cũ**: `cost_weight_action=0.1`
- **Vấn đề**: Penalty cho action quá cao, algorithm tìm gains nhỏ thay vì gains tốt
- **Hậu quả**: Converge về gains gần 0 (low action, nhưng high error)

### 6. **Không có early stopping**
- **Vấn đề**: Algorithm chạy hết `max_iterations` dù không còn improvement
- **Hậu quả**: Lãng phí thời gian

## Giải pháp đã áp dụng

### 1. **Step size scaling hợp lý hơn**
```python
# MỚI
step_scale=1.05      # Tăng chậm hơn (5% thay vì 10%)
step_shrink=0.95     # Giảm nhanh hơn (5% thay vì 10%)
```
→ Balance tốt hơn giữa exploration và exploitation

### 2. **Step size tỷ lệ với giá trị parameter**
```python
# MỚI
self.steps = [
    max(init_step * abs(g), init_step * 0.01) if g != 0 else init_step * 0.01
    for g in init_gains
]
```
→ Step size nhỏ cho Ki (0.01), lớn hơn cho Kp (0.5)

### 3. **Logic tìm kiếm chính xác**
```python
# MỚI - Luôn test từ best_gains
for i in range(len(self.gains)):
    original_gain = self.best_gains[i]

    # Test tăng
    test_gains = self.best_gains.copy()
    test_gains[i] = original_gain + self.steps[i]
    test_gains[i] = max(0.0, test_gains[i])

    if cost < best:
        self.best_gains = test_gains.copy()
    else:
        # Test giảm
        test_gains[i] = original_gain - self.steps[i]
        test_gains[i] = max(0.0, test_gains[i])
        ...
```
→ Systematic search từ best point

### 4. **Evaluation dài hơn**
```python
# MỚI
eval_steps=600  # 10 giây thay vì 5 giây
```
→ Đủ thời gian để PID ổn định

### 5. **Cost weight thấp hơn**
```python
# MỚI
cost_weight_action=0.01  # Thấp hơn 10x
```
→ Focus vào minimizing error, không quá penalty action

### 6. **Early stopping**
```python
# MỚI
if not iteration_improved:
    self.no_improvement_count += 1
    if self.no_improvement_count >= 10:
        break
```
→ Dừng sớm khi không còn progress

### 7. **Tolerance hợp lý**
```python
# MỚI
tolerance=0.01  # Thay vì 0.001
```
→ Dễ hội tụ hơn với step sizes đã shrink

## Kết quả mong đợi

### Trước (không hội tụ):
```
[Iteration 1] cost: 5.234
[Iteration 2] cost: 4.891
[Iteration 3] cost: 6.123  ← nhảy lên
[Iteration 4] cost: 5.456
...
[Iteration 30] cost: 5.001  ← vẫn dao động
```

### Sau (hội tụ):
```
[Iteration 1] cost: 5.234
[Iteration 2] cost: 4.891
[Iteration 3] cost: 4.567
[Iteration 4] cost: 4.321
...
[Iteration 15] cost: 3.123  ← ổn định
[Early stopping: No improvement for 10 iterations]
```

## Cách sử dụng

```bash
# Chạy auto-tune với tham số mặc định (đã fix)
./isaaclab.sh -p source/isaaclab_assets/isaaclab_assets/evobot_v1/balance/test_pid_autotune.py

# Hoặc tùy chỉnh
./isaaclab.sh -p source/isaaclab_assets/isaaclab_assets/evobot_v1/balance/test_pid_autotune.py \
    --init_kp 0.5 --init_ki 0.01 --init_kd 0.2 \
    --max_iterations 50 \
    --tolerance 0.01 \
    --eval_steps 600
```

## Lưu ý

1. **Nếu vẫn không hội tụ**: Thử giảm `init_step` xuống 0.01 hoặc 0.005
2. **Nếu hội tụ quá chậm**: Tăng `step_scale` lên 1.1 (nhưng có thể overshooting)
3. **Nếu cost cao**: Kiểm tra robot physics (damping, friction) và initial gains
4. **Monitor step sizes**: Nếu step sizes giảm quá nhanh (< 0.0001), tăng `step_shrink` lên 0.98

## Tóm tắt thay đổi

| Parameter | Cũ | Mới | Lý do |
|-----------|-----|-----|-------|
| `step_scale` | 1.1 | 1.05 | Tăng chậm hơn |
| `step_shrink` | 0.9 | 0.95 | Giảm nhanh hơn |
| `init_step` | Fixed 0.1 | Proportional | Tỷ lệ với gain value |
| `tolerance` | 0.001 | 0.01 | Dễ đạt được hơn |
| `eval_steps` | 300 | 600 | Evaluation ổn định hơn |
| `cost_weight_action` | 0.1 | 0.01 | Focus vào error |
| Search logic | Modify gains | Test from best | Systematic search |
| Early stopping | Không | 10 iterations | Tiết kiệm thời gian |
