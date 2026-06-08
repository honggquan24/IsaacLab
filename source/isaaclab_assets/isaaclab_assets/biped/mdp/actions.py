"""Inner tilt PID action term cho wheeled inverted pendulum.

RL output: 7 × 3 = 21 gains, thứ tự:
    [0] hip_A1_L  : position PID
    [1] hip_A1_R  : position PID
    [2] hip_A2_L  : position PID
    [3] hip_A2_R  : position PID
    [4] wheel_L   : velocity PID  → tau_balance_L
    [5] wheel_R   : velocity PID  → tau_balance_R
    [6] yaw       : yaw PID       → tau_yaw

tau_left  = tau_balance_L + tau_yaw
tau_right = tau_balance_R - tau_yaw

Gain mapping: raw ∈ [-1,1] → gain = raw × scale + bias
              → action=0 cho reference gains (ổn định từ bước đầu).
"""
from __future__ import annotations

from dataclasses import MISSING
from typing import TYPE_CHECKING, Any

import torch

from isaaclab.managers.action_manager import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv

from isaaclab_assets.biped.biped_cfg import (
    HIP_ALL_JOINT_NAMES, WHEEL_JOINT_NAMES,
)

N_HIP   = len(HIP_ALL_JOINT_NAMES)   # 4
N_WHEEL = len(WHEEL_JOINT_NAMES)      # 2
N_PID   = N_HIP + N_WHEEL + 1        # 7  (4 hip + 2 wheel + 1 yaw)
ACTION_DIM     = N_PID * 3           # 21 — TiltPIDAction (kp, ki, kd)
VEL_ACTION_DIM = N_PID * 4           # 28 — VelDirectPIDAction (kp, ki, kd, sp)


class TiltPIDAction(ActionTerm):
    """RL tunes 7 × 3 PID gains để điều khiển tilt angle và bánh xe.

    Obs gợi ý (40-dim):
        tilt_error(3) + ang_vel_b(3) + projected_gravity(3)
        + hip_pos_error(4) + hip_vel(4) + wheel_vel(2) + last_action(21)
    """

    cfg: "TiltPIDActionCfg"

    def __init__(self, cfg: "TiltPIDActionCfg", env: "ManagerBasedEnv"):
        super().__init__(cfg, env)
        self._robot     = env.scene[cfg.asset_name]
        self._hip_ids   = self._robot.find_joints(HIP_ALL_JOINT_NAMES)[0]
        self._wheel_ids = self._robot.find_joints(WHEEL_JOINT_NAMES)[0]

        # Allocation vector: roll_err → e_hip per joint (L_A1, R_A1, L_A2, R_A2)
        self._roll_alloc = torch.tensor(cfg.hip_roll_alloc, dtype=torch.float32, device=self.device)

        # Scale/bias per PID group — shape (7, 3)
        def _sb(lo, hi):
            lo = torch.tensor(lo, dtype=torch.float32, device=self.device)
            hi = torch.tensor(hi, dtype=torch.float32, device=self.device)
            return (hi - lo) / 2.0, (hi + lo) / 2.0

        hip_sc,   hip_bi   = _sb(cfg.hip_gains_min,   cfg.hip_gains_max)
        wheel_sc, wheel_bi = _sb(cfg.wheel_gains_min, cfg.wheel_gains_max)
        yaw_sc,   yaw_bi   = _sb(cfg.yaw_gains_min,   cfg.yaw_gains_max)

        # Repeat hip gains for 4 joints, wheel for 2, yaw for 1 → (7, 3)
        self._scale = torch.stack([hip_sc]*4 + [wheel_sc]*2 + [yaw_sc])
        self._bias  = torch.stack([hip_bi]*4 + [wheel_bi]*2 + [yaw_bi])

        N = self.num_envs
        self._hip_integral    = torch.zeros(N, N_HIP,   device=self.device)
        self._hip_prev_err    = torch.zeros(N, N_HIP,   device=self.device)
        self._wheel_integral  = torch.zeros(N, N_WHEEL, device=self.device)
        self._yaw_integral    = torch.zeros(N,           device=self.device)
        self._yaw_prev_err    = torch.zeros(N,           device=self.device)

        # Torque rate limiting buffers
        self._tau_hip_prev   = torch.zeros(N, N_HIP,   device=self.device)
        self._tau_wheel_prev = torch.zeros(N, N_WHEEL, device=self.device)

        self._raw_actions = torch.zeros(N, ACTION_DIM, device=self.device)
        self._gains       = self._bias.unsqueeze(0).expand(N, -1, -1).clone()  # (N, 7, 3)

    # ── Properties ─────────────────────────────────────────────────────────────

    @property
    def action_dim(self) -> int:
        return ACTION_DIM

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._gains.view(self.num_envs, ACTION_DIM)

    # ── Lifecycle ───────────────────────────────────────────────────────────────

    def reset(self, env_ids: torch.Tensor):
        self._raw_actions[env_ids]    = 0.0
        self._gains[env_ids]          = self._bias
        self._hip_integral[env_ids]   = 0.0
        self._hip_prev_err[env_ids]   = 0.0
        self._wheel_integral[env_ids] = 0.0
        self._yaw_integral[env_ids]   = 0.0
        self._yaw_prev_err[env_ids]   = 0.0
        self._tau_hip_prev[env_ids]   = 0.0
        self._tau_wheel_prev[env_ids] = 0.0

    def process_actions(self, actions: torch.Tensor):
        self._raw_actions[:] = actions.clamp(-1.0, 1.0)
        # (N, 21) → (N, 7, 3)
        # action_scale amplifies raw actions before mapping; gains clamped to ≥ 0
        raw_3d = self._raw_actions.view(self.num_envs, N_PID, 3)
        self._gains[:] = (raw_3d * self._scale * self.cfg.action_scale + self._bias).clamp_min(0.0)

    # ── Core ────────────────────────────────────────────────────────────────────

    def apply_actions(self):
        from isaaclab.utils.math import euler_xyz_from_quat

        dt     = self._env.physics_dt
        dq_hip = self._robot.data.joint_vel[:, self._hip_ids]    # (N, 4)
        ang_vel = self._robot.data.root_ang_vel_b                  # (N, 3)
        quat    = self._robot.data.root_quat_w                     # (N, 4)

        roll, pitch, yaw = euler_xyz_from_quat(quat)               # (N,) each

        cmd = self._env.command_manager.get_command(self.cfg.command_name)  # (N, 3)
        roll_des  = cmd[:, 0]  # (N,)
        pitch_des = cmd[:, 1]
        yaw_des   = cmd[:, 2]

        # ── 1. Hip tilt PID — error = body roll error distributed to joints ───
        roll_err_body = (roll_des - roll).clamp(-0.8, 0.8)          # (N,) clamp trước khi nhân
        hip_err = roll_err_body.unsqueeze(1) * self._roll_alloc     # (N, 4)
        self._hip_integral = (self._hip_integral + hip_err * dt).clamp(
            -self.cfg.hip_int_lim, self.cfg.hip_int_lim)
        self._hip_prev_err = hip_err.clone()

        kp_hip = self._gains[:, :4, 0]   # (N, 4)
        ki_hip = self._gains[:, :4, 1]
        kd_hip = self._gains[:, :4, 2]

        tau_hip = (kp_hip * hip_err + ki_hip * self._hip_integral + kd_hip * (-dq_hip))
        tau_hip = tau_hip.clamp(-self.cfg.max_hip_torque, self.cfg.max_hip_torque)

        # ── 2. Wheel balance PID: body tilt → wheel torque (Segway mechanism) ─
        # Wheel joint axis = world X → wheels roll in world Y (robot forward).
        # Tipping direction for Y-forward robot = rotation around X = world ROLL.
        # Primary balance uses roll_err (X rotation), NOT pitch_err.
        # pitch_err is secondary for differential correction.
        roll_err  = roll_des  - roll    # (N,) — main balance axis
        pitch_err = pitch_des - pitch   # (N,) — differential correction

        # Left/right allocation: roll acts on both, pitch acts differentially
        err_L = roll_err + self.cfg.roll_wheel_alloc * pitch_err   # (N,)
        err_R = roll_err - self.cfg.roll_wheel_alloc * pitch_err   # (N,)
        wheel_err = torch.stack([err_L, err_R], dim=-1)            # (N, 2)

        # Derivative: roll rate is primary (X angular velocity = forward tipping rate)
        roll_rate  = ang_vel[:, 0]  # (N,) — primary
        pitch_rate = ang_vel[:, 1]  # (N,) — differential
        wheel_rate = torch.stack([
            roll_rate + self.cfg.roll_wheel_alloc * pitch_rate,
            roll_rate - self.cfg.roll_wheel_alloc * pitch_rate,
        ], dim=-1)                                                  # (N, 2)

        self._wheel_integral = (self._wheel_integral + wheel_err * dt).clamp(
            -self.cfg.wheel_int_lim, self.cfg.wheel_int_lim)

        kp_whl = self._gains[:, 4:6, 0]  # (N, 2)
        ki_whl = self._gains[:, 4:6, 1]
        kd_whl = self._gains[:, 4:6, 2]

        # Sign NEGATED: positive torque around +X axis → wheel center moves -Y (backward).
        # For forward lean (negative roll), need wheel going +Y → need NEGATIVE torque.
        # Correct Segway law: tau = -(kp*err + ki*int + kd*(-rate)) = kp*(roll-des) + kd*rate
        tau_balance = -(kp_whl * wheel_err + ki_whl * self._wheel_integral + kd_whl * (-wheel_rate))

        # ── 3. Yaw PID → tau_yaw ────────────────────────────────────────────

        _, _, yaw = euler_xyz_from_quat(quat)
        yaw_err = torch.atan2(torch.sin(yaw_des - yaw), torch.cos(yaw_des - yaw))
        self._yaw_integral = (self._yaw_integral + yaw_err * dt).clamp(
            -self.cfg.yaw_int_lim, self.cfg.yaw_int_lim)
        yaw_rate  = ang_vel[:, 2]
        yaw_deriv = ((yaw_err - self._yaw_prev_err) / dt).clamp(
            -self.cfg.yaw_deriv_lim, self.cfg.yaw_deriv_lim)
        self._yaw_prev_err = yaw_err.clone()

        kp_yaw = self._gains[:, 6, 0]  # (N,)
        ki_yaw = self._gains[:, 6, 1]
        kd_yaw = self._gains[:, 6, 2]

        tau_yaw = (kp_yaw * yaw_err + ki_yaw * self._yaw_integral + kd_yaw * (-yaw_rate))
        tau_yaw = tau_yaw.clamp(-self.cfg.max_yaw_torque, self.cfg.max_yaw_torque)

        # ── 4. Combine wheel torques ─────────────────────────────────────────
        tau_wheel = torch.stack([
            tau_balance[:, 0] + tau_yaw,
            tau_balance[:, 1] - tau_yaw,
        ], dim=-1).clamp(-self.cfg.max_wheel_torque, self.cfg.max_wheel_torque)

        # ── 5. Torque rate limiting — ngăn spike đột biến ────────────────────
        rl = self.cfg.tau_rate_lim
        tau_hip   = (self._tau_hip_prev
                     + (tau_hip   - self._tau_hip_prev).clamp(-rl, rl))
        tau_wheel = (self._tau_wheel_prev
                     + (tau_wheel - self._tau_wheel_prev).clamp(-rl, rl))
        self._tau_hip_prev[:]   = tau_hip
        self._tau_wheel_prev[:] = tau_wheel

        # ── 6. Apply ─────────────────────────────────────────────────────────
        self._robot.set_joint_effort_target(tau_hip,   joint_ids=self._hip_ids)
        self._robot.set_joint_effort_target(tau_wheel, joint_ids=self._wheel_ids)


@configclass
class TiltPIDActionCfg(ActionTermCfg):
    class_type: type[ActionTerm] = TiltPIDAction

    asset_name:   str = "robot"
    command_name: str = MISSING   # tên TargetTiltCommand

    # Hip allocation vector: roll → delta_q per joint (L_A1, R_A1, L_A2, R_A2)
    hip_roll_alloc: tuple = (1.0, -1.0, 1.0, -1.0)

    # Scale applied to raw RL actions before gain mapping (default 1.0 = full range)
    action_scale: float = 1.0

    # Roll allocation for left/right wheel differential balance
    roll_wheel_alloc: float = 0.5

    # Gain bounds — đối xứng quanh reference
    # Hip position PID: ref kp=50, ki=1.5, kd=1.5 (vật lý: ~20 Nm/rad cần thiết)
    hip_gains_min: tuple[float, float, float] = (0.0,   0.0, 0.0)
    hip_gains_max: tuple[float, float, float] = (100.0, 3.0, 3.0)

    # Wheel tilt-balance PID (error = tilt angle rad, output = Nm)
    # kp bias=350, kd bias=30 (giảm để tránh spike khi roll_rate đột biến)
    wheel_gains_min: tuple[float, float, float] = (0.01, 0.0,  5.0)
    wheel_gains_max: tuple[float, float, float] = (500.0, 20.0, 55.0)

    # Yaw PID
    yaw_gains_min: tuple[float, float, float] = (0.01, 0.0, 0.0)
    yaw_gains_max: tuple[float, float, float] = (50.0, 0.6, 2.0)

    # Limits
    hip_int_lim:    float =  0.5
    hip_deriv_lim:  float = 100.0
    wheel_int_lim:  float =  1.0   # rad·s
    yaw_int_lim:    float =  2.0
    yaw_deriv_lim:  float = 20.0
    max_hip_torque:   float = 150.0   # Nm
    max_wheel_torque: float = 200.0   # Nm
    max_yaw_torque:   float =  20.0   # Nm
    tau_rate_lim:     float =  20.0   # Nm/step — giới hạn thay đổi torque mỗi bước 5ms


# ══════════════════════════════════════════════════════════════════════════════
# Outer loop actions
# Cả 2 class nhúng TiltPIDAction bên trong (inner chạy với reference gains cố định).
# Outer RL chỉ điều khiển setpoint tilt; inner PID lo torque thực tế.
# ══════════════════════════════════════════════════════════════════════════════

class OuterVelDirectAction(ActionTerm):
    """Outer direct: RL → (roll_des, pitch_des, yaw_rate) trực tiếp → inner TiltPID.

    RL action (3-dim): [roll_raw, pitch_raw, yaw_rate_raw] ∈ [-1, 1]
    Inner TiltPIDAction chạy embedded với reference gains (action=0 → bias).
    Outer ghi đè target_tilt command mỗi bước trước khi inner đọc.
    """

    cfg: "OuterVelDirectActionCfg"

    def __init__(self, cfg: "OuterVelDirectActionCfg", env: "ManagerBasedRLEnv"):
        super().__init__(cfg, env)
        self._inner = TiltPIDAction(cfg.inner_cfg, env)
        N = self.num_envs
        self._raw     = torch.zeros(N, 3,  device=self.device)
        self._yaw_des = torch.zeros(N,     device=self.device)

    @property
    def action_dim(self) -> int:
        return 3

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._raw

    def reset(self, env_ids: torch.Tensor):
        self._raw[env_ids] = 0.0
        self._inner.reset(env_ids)
        from isaaclab.utils.math import euler_xyz_from_quat
        robot = self._env.scene[self.cfg.inner_cfg.asset_name]
        _, _, yaw = euler_xyz_from_quat(robot.data.root_quat_w[env_ids])
        self._yaw_des[env_ids] = yaw

    def process_actions(self, actions: torch.Tensor):
        self._raw[:] = actions.clamp(-1.0, 1.0)
        self._inner.process_actions(
            torch.zeros(self.num_envs, ACTION_DIM, device=self.device)
        )

    def apply_actions(self):
        dt = self._env.physics_dt
        roll_des  = self._raw[:, 0] * self.cfg.roll_scale
        pitch_des = self._raw[:, 1] * self.cfg.pitch_scale
        self._yaw_des = self._yaw_des + self._raw[:, 2] * self.cfg.yaw_rate_scale * dt

        tilt = self._env.command_manager.get_command(self.cfg.tilt_command_name)
        tilt[:, 0] = roll_des
        tilt[:, 1] = pitch_des
        tilt[:, 2] = self._yaw_des

        self._inner.apply_actions()


@configclass
class OuterVelDirectActionCfg(ActionTermCfg):
    class_type: type[ActionTerm] = OuterVelDirectAction

    inner_cfg: Any = MISSING
    tilt_command_name: str = "target_tilt"

    roll_scale:     float = 0.30   # rad max lean ±17°
    pitch_scale:    float = 0.20   # rad
    yaw_rate_scale: float = 1.57   # rad/s


class OuterVelPIDAction(ActionTerm):
    """Outer PID: RL → 9 velocity PID gains → vel_error → tilt_cmd → inner TiltPID.

    RL action (9-dim): [[kp,ki,kd], [kp,ki,kd], [kp,ki,kd]] ∈ [-1, 1]
      PID 0: vy_error  → roll_des   (forward speed via lean)
      PID 1: vx_error  → pitch_des  (lateral)
      PID 2: yaw_rate_error → yaw_rate_cmd → integrate → yaw_des
    Inner TiltPIDAction chạy embedded với reference gains cố định.
    """

    cfg: "OuterVelPIDActionCfg"

    def __init__(self, cfg: "OuterVelPIDActionCfg", env: "ManagerBasedRLEnv"):
        super().__init__(cfg, env)
        self._inner = TiltPIDAction(cfg.inner_cfg, env)
        N = self.num_envs

        def _sb(lo, hi):
            lo = torch.tensor(lo, dtype=torch.float32, device=self.device)
            hi = torch.tensor(hi, dtype=torch.float32, device=self.device)
            return (hi - lo) / 2.0, (hi + lo) / 2.0

        vy_sc, vy_bi = _sb(cfg.vy_gains_min, cfg.vy_gains_max)
        vx_sc, vx_bi = _sb(cfg.vx_gains_min, cfg.vx_gains_max)
        yr_sc, yr_bi = _sb(cfg.yr_gains_min, cfg.yr_gains_max)

        self._scale = torch.stack([vy_sc, vx_sc, yr_sc])          # (3, 3)
        self._bias  = torch.stack([vy_bi, vx_bi, yr_bi])          # (3, 3)

        self._raw   = torch.zeros(N, 9, device=self.device)
        self._gains = self._bias.unsqueeze(0).expand(N, -1, -1).clone()  # (N, 3, 3)

        self._vy_int  = torch.zeros(N, device=self.device)
        self._vx_int  = torch.zeros(N, device=self.device)
        self._yr_int  = torch.zeros(N, device=self.device)
        self._yaw_des = torch.zeros(N, device=self.device)

    @property
    def action_dim(self) -> int:
        return 9

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._gains.view(self.num_envs, 9)

    def reset(self, env_ids: torch.Tensor):
        self._raw[env_ids]   = 0.0
        self._gains[env_ids] = self._bias
        self._vy_int[env_ids]  = 0.0
        self._vx_int[env_ids]  = 0.0
        self._yr_int[env_ids]  = 0.0
        self._inner.reset(env_ids)
        from isaaclab.utils.math import euler_xyz_from_quat
        robot = self._env.scene[self.cfg.inner_cfg.asset_name]
        _, _, yaw = euler_xyz_from_quat(robot.data.root_quat_w[env_ids])
        self._yaw_des[env_ids] = yaw

    def process_actions(self, actions: torch.Tensor):
        self._raw[:] = actions.clamp(-1.0, 1.0)
        raw_3d = self._raw.view(self.num_envs, 3, 3)
        self._gains[:] = (raw_3d * self._scale * self.cfg.action_scale + self._bias).clamp_min(0.0)
        self._inner.process_actions(
            torch.zeros(self.num_envs, ACTION_DIM, device=self.device)
        )

    def apply_actions(self):
        dt = self._env.physics_dt
        robot = self._env.scene[self.cfg.inner_cfg.asset_name]
        lin_vel_b = robot.data.root_lin_vel_b   # (N, 3)
        ang_vel_b = robot.data.root_ang_vel_b   # (N, 3)

        vel_cmd      = self._env.command_manager.get_command(self.cfg.vel_command_name)
        vx_des       = vel_cmd[:, 0]
        vy_des       = vel_cmd[:, 1]
        yaw_rate_des = vel_cmd[:, 2]

        vx       = lin_vel_b[:, 0]
        vy       = lin_vel_b[:, 1]
        yaw_rate = ang_vel_b[:, 2]

        vy_err = vy_des  - vy
        vx_err = vx_des  - vx
        yr_err = yaw_rate_des - yaw_rate

        lim = self.cfg.int_lim
        self._vy_int = (self._vy_int + vy_err * dt).clamp(-lim, lim)
        self._vx_int = (self._vx_int + vx_err * dt).clamp(-lim, lim)
        self._yr_int = (self._yr_int + yr_err * dt).clamp(-lim, lim)

        kp_vy, ki_vy, kd_vy = self._gains[:, 0, 0], self._gains[:, 0, 1], self._gains[:, 0, 2]
        kp_vx, ki_vx, kd_vx = self._gains[:, 1, 0], self._gains[:, 1, 1], self._gains[:, 1, 2]
        kp_yr, ki_yr, kd_yr = self._gains[:, 2, 0], self._gains[:, 2, 1], self._gains[:, 2, 2]

        roll_des = (kp_vy * vy_err + ki_vy * self._vy_int + kd_vy * (-vy)).clamp(
            -self.cfg.roll_lim, self.cfg.roll_lim)
        pitch_des = (kp_vx * vx_err + ki_vx * self._vx_int + kd_vx * (-vx)).clamp(
            -self.cfg.pitch_lim, self.cfg.pitch_lim)
        yaw_rate_cmd = (kp_yr * yr_err + ki_yr * self._yr_int + kd_yr * (-yaw_rate)).clamp(
            -self.cfg.max_yaw_rate, self.cfg.max_yaw_rate)
        self._yaw_des = self._yaw_des + yaw_rate_cmd * dt

        tilt = self._env.command_manager.get_command(self.cfg.tilt_command_name)
        tilt[:, 0] = roll_des
        tilt[:, 1] = pitch_des
        tilt[:, 2] = self._yaw_des

        self._inner.apply_actions()


@configclass
class OuterVelPIDActionCfg(ActionTermCfg):
    class_type: type[ActionTerm] = OuterVelPIDAction

    inner_cfg: Any = MISSING
    vel_command_name:  str = "velocity_cmd"
    tilt_command_name: str = "target_tilt"

    action_scale: float = 1.0

    # Gain bounds: [kp, ki, kd] cho mỗi velocity PID
    # vy_error (m/s) → roll_des (rad): kp đơn vị rad/(m/s)
    vy_gains_min: tuple[float, float, float] = (0.0, 0.0, 0.0)
    vy_gains_max: tuple[float, float, float] = (1.0, 0.3, 0.5)

    # vx_error (m/s) → pitch_des (rad)
    vx_gains_min: tuple[float, float, float] = (0.0, 0.0, 0.0)
    vx_gains_max: tuple[float, float, float] = (0.5, 0.1, 0.3)

    # yaw_rate_error (rad/s) → yaw_rate_cmd (rad/s): kp dimensionless
    yr_gains_min: tuple[float, float, float] = (0.0, 0.0, 0.0)
    yr_gains_max: tuple[float, float, float] = (2.0, 0.5, 1.0)

    roll_lim:    float = 0.30   # rad — clamp roll_des
    pitch_lim:   float = 0.20   # rad
    max_yaw_rate: float = 2.0   # rad/s — clamp yaw_rate_cmd
    int_lim:     float = 2.0    # s·(m/s) — integral wind-up limit


# ══════════════════════════════════════════════════════════════════════════════
# Unified: vel_cmd → 7 PIDs → torque (không qua tilt setpoint trung gian)
# ══════════════════════════════════════════════════════════════════════════════

class VelDirectPIDAction(ActionTerm):
    """Unified single-stage: vel_cmd → 7 PIDs (kp, ki, kd, sp) → torque.

    Action output: 7 × 4 = 28 dim.
    Mỗi PID i có output [kp_i, ki_i, kd_i, sp_i] tại các chỉ số 4i, 4i+1, 4i+2, 4i+3.
    Setpoints (sp) là tự học — RL quyết định cả gains lẫn setpoint.

    PIDs:
      hip  [0-3] : error = sp_hip_i - q_hip_i   (rad), D = -dq_hip
      wheel[4-5] : error = sp_roll  - roll       (rad), D = -roll_rate
      yaw  [6]   : error = sp_yr    - yaw_rate   (rad/s), D = -yr
    """

    cfg: "VelDirectPIDActionCfg"

    def __init__(self, cfg: "VelDirectPIDActionCfg", env: "ManagerBasedRLEnv"):
        super().__init__(cfg, env)
        self._robot     = env.scene[cfg.asset_name]
        self._hip_ids   = self._robot.find_joints(HIP_ALL_JOINT_NAMES)[0]
        self._wheel_ids = self._robot.find_joints(WHEEL_JOINT_NAMES)[0]

        def _sb(lo, hi):
            lo = torch.tensor(lo, dtype=torch.float32, device=self.device)
            hi = torch.tensor(hi, dtype=torch.float32, device=self.device)
            return (hi - lo) / 2.0, (hi + lo) / 2.0

        # Gain scale/bias: (7, 3)
        hip_sc,   hip_bi   = _sb(cfg.hip_gains_min,   cfg.hip_gains_max)
        wheel_sc, wheel_bi = _sb(cfg.wheel_gains_min, cfg.wheel_gains_max)
        yaw_sc,   yaw_bi   = _sb(cfg.yaw_gains_min,   cfg.yaw_gains_max)
        self._gain_scale = torch.stack([hip_sc]*4 + [wheel_sc]*2 + [yaw_sc])   # (7, 3)
        self._gain_bias  = torch.stack([hip_bi]*4 + [wheel_bi]*2 + [yaw_bi])   # (7, 3)

        # Setpoint scale/bias: (7,) — sp = raw * sp_scale + sp_bias
        hip_sp_sc,   hip_sp_bi   = _sb(cfg.hip_sp_min,   cfg.hip_sp_max)    # scalar
        wheel_sp_sc, wheel_sp_bi = _sb(cfg.wheel_sp_min, cfg.wheel_sp_max)  # scalar
        yaw_sp_sc,   yaw_sp_bi   = _sb(cfg.yaw_sp_min,   cfg.yaw_sp_max)    # scalar
        self._sp_scale = torch.stack([hip_sp_sc]*4 + [wheel_sp_sc]*2 + [yaw_sp_sc])  # (7,)
        self._sp_bias  = torch.stack([hip_sp_bi]*4 + [wheel_sp_bi]*2 + [yaw_sp_bi])  # (7,)

        N = self.num_envs
        self._hip_integral   = torch.zeros(N, N_HIP,   device=self.device)
        self._wheel_integral = torch.zeros(N, N_WHEEL, device=self.device)
        self._yaw_integral   = torch.zeros(N,           device=self.device)
        self._raw_actions    = torch.zeros(N, VEL_ACTION_DIM, device=self.device)
        # _gains: (N, 7, 3), _sp: (N, 7)
        self._gains = self._gain_bias.unsqueeze(0).expand(N, -1, -1).clone()
        self._sp    = self._sp_bias.unsqueeze(0).expand(N, -1).clone()
        self._tau_hip_prev   = torch.zeros(N, N_HIP,   device=self.device)
        self._tau_wheel_prev = torch.zeros(N, N_WHEEL, device=self.device)

    @property
    def action_dim(self) -> int:
        return VEL_ACTION_DIM

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        # Concatenate gains(N,7,3) và sp(N,7,1) → (N,7,4) → (N,28)
        return torch.cat([self._gains, self._sp.unsqueeze(-1)], dim=-1).view(self.num_envs, VEL_ACTION_DIM)

    def reset(self, env_ids: torch.Tensor):
        self._raw_actions[env_ids]    = 0.0
        self._gains[env_ids]          = self._gain_bias
        self._sp[env_ids]             = self._sp_bias
        self._hip_integral[env_ids]   = 0.0
        self._wheel_integral[env_ids] = 0.0
        self._yaw_integral[env_ids]   = 0.0
        self._tau_hip_prev[env_ids]   = 0.0
        self._tau_wheel_prev[env_ids] = 0.0

    def process_actions(self, actions: torch.Tensor):
        self._raw_actions[:] = actions.clamp(-1.0, 1.0)
        raw_4d = self._raw_actions.view(self.num_envs, N_PID, 4)   # (N, 7, 4)
        # kp/ki/kd từ cột 0-2, sp từ cột 3
        self._gains[:] = (raw_4d[:, :, :3] * self._gain_scale * self.cfg.action_scale
                          + self._gain_bias).clamp_min(0.0)
        self._sp[:]    = raw_4d[:, :, 3] * self._sp_scale + self._sp_bias

    def apply_actions(self):
        from isaaclab.utils.math import euler_xyz_from_quat
        dt      = self._env.physics_dt
        dq_hip  = self._robot.data.joint_vel[:, self._hip_ids]   # (N, 4)
        ang_vel = self._robot.data.root_ang_vel_b                  # (N, 3)
        roll, _, _ = euler_xyz_from_quat(self._robot.data.root_quat_w)
        roll_rate = ang_vel[:, 0]   # (N,)
        yr        = ang_vel[:, 2]   # (N,)

        # ── 1. Hip PIDs: sp_hip (learned) ─────────────────────────────────
        q_hip   = self._robot.data.joint_pos[:, self._hip_ids]   # (N, 4)
        sp_hip  = self._sp[:, :4]                                  # (N, 4)
        hip_err = sp_hip - q_hip
        self._hip_integral = (self._hip_integral + hip_err * dt).clamp(
            -self.cfg.hip_int_lim, self.cfg.hip_int_lim)
        kp_hip = self._gains[:, :4, 0]
        ki_hip = self._gains[:, :4, 1]
        kd_hip = self._gains[:, :4, 2]
        tau_hip = (kp_hip * hip_err + ki_hip * self._hip_integral + kd_hip * (-dq_hip))
        tau_hip = tau_hip.clamp(-self.cfg.max_hip_torque, self.cfg.max_hip_torque)

        # ── 2. Wheel PIDs: roll error → Segway balance ────────────────────
        # Robot 2 bánh: không có pitch DOF → 2 bánh dùng cùng error
        sp_roll     = self._sp[:, 4]                               # (N,) roll setpoint (learned)
        wheel_err   = (sp_roll - roll).clamp(-0.8, 0.8)           # (N,)
        wheel_err_2 = wheel_err.unsqueeze(1).expand(-1, N_WHEEL)   # (N, 2)
        self._wheel_integral = (self._wheel_integral + wheel_err_2 * dt).clamp(
            -self.cfg.wheel_int_lim, self.cfg.wheel_int_lim)
        kp_whl = self._gains[:, 4:6, 0]
        ki_whl = self._gains[:, 4:6, 1]
        kd_whl = self._gains[:, 4:6, 2]
        roll_rate_2 = roll_rate.unsqueeze(1).expand(-1, N_WHEEL)
        tau_balance = -(kp_whl * wheel_err_2 + ki_whl * self._wheel_integral
                        + kd_whl * (-roll_rate_2))

        # ── 3. Yaw PID: sp_yr (learned) ───────────────────────────────────
        sp_yr   = self._sp[:, 6]                                   # (N,)
        yr_err  = sp_yr - yr
        self._yaw_integral = (self._yaw_integral + yr_err * dt).clamp(
            -self.cfg.yaw_int_lim, self.cfg.yaw_int_lim)
        kp_yaw = self._gains[:, 6, 0]
        ki_yaw = self._gains[:, 6, 1]
        kd_yaw = self._gains[:, 6, 2]
        tau_yaw = (kp_yaw * yr_err + ki_yaw * self._yaw_integral + kd_yaw * (-yr))
        tau_yaw = tau_yaw.clamp(-self.cfg.max_yaw_torque, self.cfg.max_yaw_torque)

        # ── 4. Combine ─────────────────────────────────────────────────────
        tau_wheel = torch.stack([
            tau_balance[:, 0] + tau_yaw,
            tau_balance[:, 1] - tau_yaw,
        ], dim=-1).clamp(-self.cfg.max_wheel_torque, self.cfg.max_wheel_torque)

        # ── 5. Rate limiting ───────────────────────────────────────────────
        rl = self.cfg.tau_rate_lim
        tau_hip   = self._tau_hip_prev   + (tau_hip   - self._tau_hip_prev).clamp(-rl, rl)
        tau_wheel = self._tau_wheel_prev + (tau_wheel - self._tau_wheel_prev).clamp(-rl, rl)
        self._tau_hip_prev[:]   = tau_hip
        self._tau_wheel_prev[:] = tau_wheel

        self._robot.set_joint_effort_target(tau_hip,   joint_ids=self._hip_ids)
        self._robot.set_joint_effort_target(tau_wheel, joint_ids=self._wheel_ids)


@configclass
class VelDirectPIDActionCfg(ActionTermCfg):
    class_type: type[ActionTerm] = VelDirectPIDAction

    asset_name:         str   = "robot"
    vel_command_name:   str   = MISSING
    action_scale:       float = 1.0

    # Gain bounds — (kp, ki, kd)
    hip_gains_min:   tuple[float, float, float] = (0.01,  0.0,  0.0)
    hip_gains_max:   tuple[float, float, float] = (200.0, 10.0, 20.0)
    wheel_gains_min: tuple[float, float, float] = (200.0,  0.0, 10.0)
    wheel_gains_max: tuple[float, float, float] = (500.0, 20.0, 80.0)
    yaw_gains_min:   tuple[float, float, float] = (0.01,  0.0,  0.0)
    yaw_gains_max:   tuple[float, float, float] = (20.0,  2.0,  5.0)

    # Setpoint bounds — RL học setpoint trong range này
    hip_sp_min:   float = -0.3    # rad — hip joint position
    hip_sp_max:   float =  0.3
    wheel_sp_min: float = -0.3    # rad — body roll angle (sp_roll)
    wheel_sp_max: float =  0.3
    yaw_sp_min:   float = -1.5    # rad/s — yaw rate
    yaw_sp_max:   float =  1.5

    hip_int_lim:      float =  0.5
    wheel_int_lim:    float =  1.0
    yaw_int_lim:      float =  2.0
    max_hip_torque:   float = 150.0
    max_wheel_torque: float = 200.0
    max_yaw_torque:   float =  20.0
    tau_rate_lim:     float =  20.0
