# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""PID Controller cho Crazyflie quadcopter.

Cấu trúc:
  PIDController   — single-axis PID (base class)
  QuadcopterPID   — outer loops: position error → thrust (N) + desired attitude (rad)
                    Không dùng feedforward trọng lực — integral tự bù theo thời gian.
                    Không cần biết robot_mass hay gravity.
"""

from __future__ import annotations

import math
import torch

# ── Thông số Crazyflie 2.x ────────────────────────────────────────────────────
_ARM_M = 0.046                      # chiều dài cánh [m]
_D     = _ARM_M / math.sqrt(2)      # khoảng cách vuông góc tâm → prop ≈ 0.0325 m
_KM    = 0.005                      # tỉ lệ drag-torque / thrust [m] (đo từ thực nghiệm)


def make_alloc_inv(device: str = "cpu") -> torch.Tensor:
    """Tính ma trận phân bổ nghịch đảo cho Crazyflie cf2x.

    Chuyển đổi wrench [Fz, Tx, Ty, Tz] → lực từng prop [F1, F2, F3, F4] (Newton).

    Sơ đồ bố trí motor (nhìn từ trên xuống):

        m1(CCW)  m2(CW)       ← phía trước
            \\      /
             [body]
            /      \\
        m4(CW)  m3(CCW)       ← phía sau

    Ma trận phân bổ A (4×4):
        [Fz]   [ 1    1    1    1  ] [F1]
        [Tx] = [ D   -D   -D    D  ] [F2]
        [Ty]   [-D   -D    D    D  ] [F3]
        [Tz]   [-KM  KM  -KM   KM ] [F4]

    Trong đó:
        Fz  = tổng lực nâng (N)
        Tx  = moment roll  (Nm) — quay quanh trục X (tiến/lùi)
        Ty  = moment pitch (Nm) — quay quanh trục Y (trái/phải)
        Tz  = moment yaw   (Nm) — quay quanh trục Z, tạo bởi drag khác chiều
        D   = khoảng cách tâm → prop theo phương vuông góc
        KM  = drag-torque / thrust ratio

    Returns:
        A_inv: tensor (4, 4) trên device chỉ định.
    """
    A = torch.tensor([
        [ 1.0,   1.0,   1.0,   1.0],   # Fz  = F1+F2+F3+F4
        [ _D,   -_D,   -_D,    _D ],   # Tx  = D*(F1-F2-F3+F4)
        [-_D,   -_D,    _D,    _D ],   # Ty  = D*(-F1-F2+F3+F4)
        [-_KM,  _KM,  -_KM,   _KM],   # Tz  = KM*(-F1+F2-F3+F4)
    ], device=device)
    return torch.linalg.inv(A)


class PIDController:
    """Single-axis PID controller.

    Args:
        kp: Hệ số tỉ lệ.
        ki: Hệ số tích phân.
        kd: Hệ số vi phân.
        integral_limit: Giới hạn chống wind-up. None = không giới hạn.
    """

    def __init__(
        self,
        kp: float,
        ki: float,
        kd: float,
        integral_limit: float | None = None,
    ):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integral_limit = integral_limit

        self._integral   = 0.0
        self._prev_error = 0.0

    def update(self, error: float, dt: float) -> float:
        self._integral += error * dt
        if self.integral_limit is not None:
            self._integral = max(-self.integral_limit, min(self.integral_limit, self._integral))

        derivative = (error - self._prev_error) / dt if dt > 1e-6 else 0.0
        self._prev_error = error

        return self.kp * error + self.ki * self._integral + self.kd * derivative

    def reset(self):
        self._integral   = 0.0
        self._prev_error = 0.0


class QuadcopterPID:
    """Outer-loop PID cho quadcopter: position error → thrust + desired attitude.

    Không dùng feedforward trọng lực — integral tích lũy bù trọng lực tự nhiên.
    Không cần robot_mass hay gravity.

    Vòng trong (attitude → moments) KHÔNG nằm ở đây.
    Từng script tự tính moment từ desired_roll/pitch.
    """

    def __init__(self):
        # Altitude PID: z_error → thrust (N)
        self.pid_z = PIDController(kp=0.3, ki=0.08, kd=0.15, integral_limit=1.0)

        # Position PID: x/y_error → desired_pitch / desired_roll (rad)
        self.pid_x = PIDController(kp=0.4, ki=0.05, kd=0.3, integral_limit=0.3)
        self.pid_y = PIDController(kp=0.4, ki=0.05, kd=0.3, integral_limit=0.3)

        self._max_tilt_rad = math.radians(20.0)

    def compute(
        self,
        pos: torch.Tensor,
        vel: torch.Tensor,
        target_pos: torch.Tensor,
        dt: float,
    ) -> tuple[float, float, float]:
        """Tính outer-loop PID.

        Returns:
            (thrust_N, desired_roll, desired_pitch) — đơn vị N và rad.
        """
        z_err    = target_pos[2].item() - pos[2].item()
        thrust_N = max(0.0, self.pid_z.update(z_err, dt))

        x_err = target_pos[0].item() - pos[0].item()
        y_err = target_pos[1].item() - pos[1].item()

        desired_pitch = max(-self._max_tilt_rad, min(self._max_tilt_rad, -self.pid_x.update(x_err, dt)))
        desired_roll  = max(-self._max_tilt_rad, min(self._max_tilt_rad,  self.pid_y.update(y_err, dt)))

        return thrust_N, desired_roll, desired_pitch

    def reset(self):
        for ctrl in (self.pid_z, self.pid_x, self.pid_y):
            ctrl.reset()
