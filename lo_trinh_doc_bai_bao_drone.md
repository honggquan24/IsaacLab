# Lộ Trình Đọc Bài Báo: Drone từ Cơ Bản đến Nâng Cao

> **Tác giả bài báo:** TS. Hà Lê Như Ngọc Thanh (HCMUTE)
> **Mục tiêu:** Nắm vững lý thuyết điều khiển quadcopter từ PID đến Backstepping nâng cao

---

## Giai đoạn 1 — Mô hình Quadcopter & PID Cơ Bản

> **Mục tiêu:** Hiểu drone hoạt động thế nào, mô hình toán học, điều khiển PID

| # | Bài báo | Tạp chí | Năm |
|---|---------|---------|-----|
| 1 | Simple nonlinear control of quadcopter for collision avoidance based on geometric approach | International Journal of Advanced Robotic Systems | 2018 |
| 2 | **Quadcopter robust adaptive second order sliding mode control based on PID sliding surface** ⭐ | IEEE Access | 2018 |

**Ghi chú:**
- Bài #1: Nhẹ nhất, giới thiệu mô hình quadcopter + điều khiển đơn giản
- Bài #2: **175 trích dẫn** — bài kinh điển nhất, nền tảng cho mọi bài sau, đọc kỹ

---

## Giai đoạn 2 — Sliding Mode Control (SMC) Cơ Bản

> **Mục tiêu:** Nắm vững kỹ thuật điều khiển chủ lực của thầy

| # | Bài báo | Tạp chí | Năm |
|---|---------|---------|-----|
| 3 | Robust Dynamic Sliding Mode Control-Based PID–Super Twisting Algorithm and Disturbance Observer for Second-Order Nonlinear Systems: Application to UAVs | Electronics | 2019 |
| 4 | An extended multi-surface sliding control for matched/mismatched uncertain nonlinear systems through a lumped disturbance estimator | IEEE Access | 2020 |
| 5 | Perturbation Observer-Based Robust Control Using a Multiple Sliding Surfaces for Nonlinear Systems with Influences of Matched and Unmatched Uncertainties | Mathematics | 2020 |

**Ghi chú:**
- Bài #3: Super Twisting giảm chattering — đọc ngay sau bài #2
- Bài #4: Mở rộng SMC cho hệ có nhiễu không khớp (mismatched uncertainty)
- Bài #5: Kết hợp observer ước lượng nhiễu + SMC đa mặt trượt

---

## Giai đoạn 3 — Ứng Dụng: Trajectory Tracking, Collision Avoidance & Landing

> **Mục tiêu:** Drone tự điều hướng, tránh vật cản, đáp xuống mục tiêu động

| # | Bài báo | Tạp chí | Năm |
|---|---------|---------|-----|
| 6 | Nonlinear control for autonomous trajectory tracking while considering collision avoidance of UAVs based on geometric relations | Energies | 2019 |
| 7 | Completion of collision avoidance control algorithm for multicopters based on geometrical constraints | IEEE Access | 2018 |
| 8 | Autonomous quadcopter precision landing onto a heaving platform: New method and experiment | IEEE Access | 2020 |
| 9 | An effective dynamic sliding mode control based nonlinear disturbance observer for a quadrotor UAV | IEEE ICCA 2020 | 2020 |

---

## Giai đoạn 4 — Nâng Cao: Neural Network + Adaptive Control

> **Mục tiêu:** Điều khiển thích nghi khi mô hình không chính xác hoặc có lỗi cơ cấu

| # | Bài báo | Tạp chí | Năm | Trích dẫn |
|---|---------|---------|-----|-----------|
| 10 | **Adaptive Sliding Mode Control for Attitude and Altitude system of a Quadcopter UAV via Neural Network** ⭐ | IEEE Access | 2021 | 127 |
| 11 | Finite-Time Attitude Fault Tolerant Control of Quadcopter System via Neural Networks | Mathematics | 2020 | 34 |
| 12 | Fuzzy Hybrid Neural Network Control for Uncertainty Nonlinear Systems Based on Enhancement Search Algorithm | International Journal of Fuzzy Systems | 2022 | 11 |

**Ghi chú:**
- Bài #10: **127 trích dẫn** — bài quan trọng về dùng Neural Network hỗ trợ SMC
- Bài #11: Điều khiển chịu lỗi (fault-tolerant control) khi motor hỏng

---

## Giai đoạn 5 — Đỉnh Cao: Backstepping + Observer Nâng Cao

> **Mục tiêu:** Nắm các phương pháp điều khiển hiện đại và tổng hợp nhất

| # | Bài báo | Tạp chí | Năm |
|---|---------|---------|-----|
| 13 | Quadcopter UAVs Extended States/Disturbance Observer-Based Nonlinear Robust Backstepping Control | Sensors | 2022 |
| 14 | Finite-Time Stability of MIMO Nonlinear Systems Based on Robust Adaptive Sliding Control: Methodology and Application to Stabilize Chaotic Motions | IEEE Access | 2021 |
| 15 | **Finite-Time Robust Composite Backstepping Control Based on a High-Order Sliding Mode Observer for Quadcopter AAV Trajectory Tracking** ⭐ | IEEE Access | 2025 |

**Ghi chú:**
- Bài #15: Bài mới nhất (2025) — tổng hợp tất cả kỹ thuật từ giai đoạn 1–4

---

## Tóm Tắt Lộ Trình

```
Giai đoạn 1  →  Mô hình quadcopter + PID                     (Bài 1–2)
Giai đoạn 2  →  Sliding Mode Control (SMC)                   (Bài 3–5)
Giai đoạn 3  →  Ứng dụng: Tracking, Collision, Landing       (Bài 6–9)
Giai đoạn 4  →  Neural Network + Adaptive Control            (Bài 10–12)
Giai đoạn 5  →  Backstepping + Composite Control             (Bài 13–15)
```

---

## Lời Khuyên Thực Tế

- **Bắt đầu bằng bài #2** (2018, 175 trích dẫn) — bài nền tảng quan trọng nhất
- Song song đọc lý thuyết SMC từ sách: *Slotine & Li* hoặc *Edwards & Spurgeon*
- Mỗi giai đoạn nên tự **code mô phỏng** trên MATLAB/Python trước khi đọc bài tiếp theo
- Giai đoạn 3 có thể đọc song song với giai đoạn 2 để thấy ứng dụng thực tế
- Liên hệ thầy để xin PDF nếu không tải được qua Sci-Hub: **thanh.hlnn@hcmute.edu.vn**

---

*Tổng hợp từ Google Scholar của TS. Hà Lê Như Ngọc Thanh — HCMUTE*
