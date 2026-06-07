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
from typing import TYPE_CHECKING

import torch

from isaaclab.managers.action_manager import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

from isaaclab_assets.biped.biped_cfg import (
    HIP_ALL_JOINT_NAMES, HIP_DEFAULT_Q, WHEEL_JOINT_NAMES,
)

N_HIP   = len(HIP_ALL_JOINT_NAMES)   # 4
N_WHEEL = len(WHEEL_JOINT_NAMES)      # 2
N_PID   = N_HIP + N_WHEEL + 1        # 7  (4 hip + 2 wheel + 1 yaw)
ACTION_DIM = N_PID * 3               # 21


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

        # Default hip positions: (4,)
        self._q_hip_default = torch.tensor(HIP_DEFAULT_Q, dtype=torch.float32, device=self.device)
        # Allocation vectors: roll/pitch → delta_q for each hip (4,)
        self._roll_alloc  = torch.tensor(cfg.hip_roll_alloc,  dtype=torch.float32, device=self.device)
        self._pitch_alloc = torch.tensor(cfg.hip_pitch_alloc, dtype=torch.float32, device=self.device)

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

    def process_actions(self, actions: torch.Tensor):
        self._raw_actions[:] = actions.clamp(-1.0, 1.0)
        # (N, 21) → (N, 7, 3)
        # action_scale amplifies raw actions before mapping; gains clamped to ≥ 0
        raw_3d = self._raw_actions.view(self.num_envs, N_PID, 3)
        self._gains[:] = (raw_3d * self._scale * self.cfg.action_scale + self._bias).clamp_min(0.0)

    # ── Core ────────────────────────────────────────────────────────────────────

    def apply_actions(self):
        from isaaclab.utils.math import euler_xyz_from_quat

        dt      = self._env.physics_dt
        q_hip   = self._robot.data.joint_pos[:, self._hip_ids]    # (N, 4)
        dq_hip  = self._robot.data.joint_vel[:, self._hip_ids]    # (N, 4)
        ang_vel = self._robot.data.root_ang_vel_b                  # (N, 3)
        quat    = self._robot.data.root_quat_w                     # (N, 4)

        roll, pitch, yaw = euler_xyz_from_quat(quat)               # (N,) each

        cmd = self._env.command_manager.get_command(self.cfg.command_name)  # (N, 3)
        roll_des  = cmd[:, 0]  # (N,)
        pitch_des = cmd[:, 1]
        yaw_des   = cmd[:, 2]

        # ── 1. Hip position PID ─────────────────────────────────────────────
        q_des_hip = (self._q_hip_default
                     + roll_des.unsqueeze(1)  * self._roll_alloc
                     + pitch_des.unsqueeze(1) * self._pitch_alloc)   # (N, 4)
        q_des_hip = q_des_hip.clamp(self.cfg.hip_q_lower, self.cfg.hip_q_upper)

        hip_err = q_des_hip - q_hip
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

        # ── 5. Apply ─────────────────────────────────────────────────────────
        self._robot.set_joint_effort_target(tau_hip,   joint_ids=self._hip_ids)
        self._robot.set_joint_effort_target(tau_wheel, joint_ids=self._wheel_ids)


@configclass
class TiltPIDActionCfg(ActionTermCfg):
    class_type: type[ActionTerm] = TiltPIDAction

    asset_name:   str = "robot"
    command_name: str = MISSING   # tên TargetTiltCommand

    # Hip allocation vectors: roll/pitch → delta_q per joint (L_A1, R_A1, L_A2, R_A2)
    hip_roll_alloc:  tuple = ( 0.5, -0.5,  0.5, -0.5)
    hip_pitch_alloc: tuple = ( 0.3,  0.3,  0.3,  0.3)

    # Scale applied to raw RL actions before gain mapping (default 1.0 = full range)
    action_scale: float = 1.0

    # Roll allocation for left/right wheel differential balance
    roll_wheel_alloc: float = 0.5

    # Gain bounds — đối xứng quanh reference
    # Hip position PID: ref kp=40, ki=1, kd=1
    hip_gains_min: tuple[float, float, float] = (10.0,  0.0, 0.0)
    hip_gains_max: tuple[float, float, float] = (160.0, 6.0, 6.0)

    # Wheel tilt-balance PID (error = tilt angle rad, output = Nm)
    wheel_gains_min: tuple[float, float, float] = (20.0,  0.0,  2.0)
    wheel_gains_max: tuple[float, float, float] = (200.0, 10.0, 40.0)

    # Yaw PID
    yaw_gains_min: tuple[float, float, float] = (1.0, 0.0, 0.0)
    yaw_gains_max: tuple[float, float, float] = (16.0, 0.6, 2.0)

    # Limits
    hip_q_lower:    float = -0.8
    hip_q_upper:    float =  0.8
    hip_int_lim:    float =  1.0
    hip_deriv_lim:  float = 100.0
    wheel_int_lim:  float =  1.0   # rad·s
    yaw_int_lim:    float =  2.0
    yaw_deriv_lim:  float = 20.0
    max_hip_torque:   float = 160.0   # Nm
    max_wheel_torque: float = 120.0   # Nm
    max_yaw_torque:   float =  20.0   # Nm
