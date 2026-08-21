# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Custom actions cho V5: hip với auto-mimic.

HipMimicPositionAction:
  - Input (policy): 2 giá trị cho right_hip_joint và left_hip_joint
  - Output (sim): 4 targets — hip_active giữ nguyên, hip_mimic = -hip_active
  - Không dùng PhysxMimicJointAPI (không reliable qua session layer)
"""

from __future__ import annotations

from dataclasses import MISSING
from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class HipMimicPositionAction(ActionTerm):
    """Position action cho hip_active, tự mirror sang hip_mimic (gearing=-1)."""

    cfg: HipMimicPositionActionCfg
    _asset: Articulation

    def __init__(self, cfg: HipMimicPositionActionCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self._asset: Articulation = env.scene[cfg.asset_name]

        # Tìm joint indices
        all_joints = self._asset.data.joint_names
        self._active_ids = [all_joints.index(n) for n in cfg.active_joint_names]
        self._mimic_ids = [all_joints.index(n) for n in cfg.mimic_joint_names]

        if len(self._active_ids) != len(self._mimic_ids):
            raise ValueError("active_joint_names và mimic_joint_names phải có cùng số lượng.")

        self._num_active = len(self._active_ids)
        self._raw_actions = torch.zeros(env.num_envs, self._num_active, device=env.device)
        self._processed = torch.zeros(env.num_envs, self._num_active, device=env.device)

    @property
    def action_dim(self) -> int:
        return self._num_active

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed

    def process_actions(self, actions: torch.Tensor):
        self._raw_actions[:] = actions
        self._processed = actions * self.cfg.scale

    def apply_actions(self):
        targets = self._processed
        self._asset.set_joint_position_target(targets, joint_ids=self._active_ids)
        self._asset.set_joint_position_target(-targets, joint_ids=self._mimic_ids)

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        # reset_joints_by_offset đã cập nhật data.joint_pos cho env_ids.
        # Sync mimic = -active để 5-bar luôn đúng ràng buộc trước step đầu.
        if env_ids is None:
            active_pos = self._asset.data.joint_pos[:, self._active_ids]
            self._asset.write_joint_position_to_sim(-active_pos, joint_ids=self._mimic_ids)
            self._asset.set_joint_position_target(active_pos, joint_ids=self._active_ids)
            self._asset.set_joint_position_target(-active_pos, joint_ids=self._mimic_ids)
            self._raw_actions.zero_()
            self._processed.zero_()
        else:
            active_pos = self._asset.data.joint_pos[env_ids][:, self._active_ids]
            self._asset.write_joint_position_to_sim(-active_pos, joint_ids=self._mimic_ids, env_ids=env_ids)
            self._asset.set_joint_position_target(active_pos, joint_ids=self._active_ids, env_ids=env_ids)
            self._asset.set_joint_position_target(-active_pos, joint_ids=self._mimic_ids, env_ids=env_ids)
            self._raw_actions[env_ids] = 0.0
            self._processed[env_ids] = 0.0


@configclass
class HipMimicPositionActionCfg(ActionTermCfg):
    """Config cho HipMimicPositionAction."""

    class_type: type = HipMimicPositionAction

    asset_name: str = MISSING
    active_joint_names: list[str] = MISSING
    mimic_joint_names: list[str] = MISSING
    scale: float = 1.0


# ──────────────────────────────────────────────────────────────────────────────
# PI-ANN cho HIP: mạng xuất hệ số PID, PID tính góc hip từ sai số cao độ + nghiêng ngang
# ──────────────────────────────────────────────────────────────────────────────


class HipMimicPIDAction(ActionTerm):
    """PI-ANN cho hip (cascade): mạng xuất SETPOINT cao độ + hệ số PID, PID tính GÓC hip.

    Khác bản đầu: mạng KHÔNG chỉ xuất gains mà còn xuất luôn ĐIỂM ĐẶT cao độ ``r_h``
    (vòng ngoài), PID kéo cao độ đo về ``r_h`` (vòng trong). Mạng thấy height_cmd trong
    obs → tự chọn r_h bám lệnh, và có thể chủ động "giữ chân duỗi" → chống sụm.

    Gains qua ``tanh`` (CÓ DẤU) → RL tự tìm dấu ổn định, KHÔNG cần công tắc height_sign.

    Luồng mỗi bước::

        raw (policy)  = [rh_raw, kp_raw, ki_raw, kd_raw, kp_lat_raw]   # action_dim = 5
        r_h           = height_center + tanh(rh_raw) * height_range    # ∈ [center±range] m
        Kp,Ki,Kd      = tanh(raw) * [max_kp, max_ki, max_kd]           # CÓ DẤU
        Kp_lat        = tanh(raw) * max_kp_lat                         # CÓ DẤU
        e_h           = r_h - (z_base - z_axle)        (sai số bám cao độ học được)
        e_lat         = projected_gravity_b[:, 0]      (nghiêng ngang, setpoint = 0)
        hip_common    = Kp·e_h + Ki·∫e_h + Kd·de_h/dt          # chung 2 chân
        hip_diff      = Kp_lat * e_lat                         # vi sai 2 chân
        right_active  = clamp(hip_common + hip_diff, ±output_limit)
        left_active   = clamp(hip_common - hip_diff, ±output_limit)
        mimic         = -active  (ràng buộc 5-bar trong mỗi chân)
    """

    cfg: HipMimicPIDActionCfg
    _asset: Articulation

    def __init__(self, cfg: HipMimicPIDActionCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self._asset = env.scene[cfg.asset_name]

        jn = self._asset.data.joint_names
        self._active_ids = [jn.index(n) for n in cfg.active_joint_names]
        self._mimic_ids = [jn.index(n) for n in cfg.mimic_joint_names]
        if len(self._active_ids) != 2 or len(self._mimic_ids) != 2:
            raise ValueError("HipMimicPIDAction cần đúng 2 active + 2 mimic (phải, trái).")

        bn = self._asset.data.body_names
        self._wr_idx = bn.index(cfg.wheel_right_body)
        self._wl_idx = bn.index(cfg.wheel_left_body)

        self._dt = env.physics_dt  # apply chạy mỗi physics substep

        # raw/log = [r_h, Kp, Ki, Kd, Kp_lat]
        self._raw_actions = torch.zeros(env.num_envs, 5, device=env.device)
        self._gains = torch.zeros(env.num_envs, 5, device=env.device)
        self._processed = torch.zeros(env.num_envs, 2, device=env.device)  # [right, left] active

        self._integral = torch.zeros(env.num_envs, device=env.device)
        self._prev_error = torch.zeros(env.num_envs, device=env.device)

        self._max_gains = torch.tensor([cfg.max_kp, cfg.max_ki, cfg.max_kd], device=env.device).unsqueeze(0)

    @property
    def action_dim(self) -> int:
        return 5

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._gains  # [r_h, Kp, Ki, Kd, Kp_lat] đã map (log/debug)

    def process_actions(self, actions: torch.Tensor):
        cfg = self.cfg
        self._raw_actions[:] = actions
        # setpoint cao độ ∈ [center-range, center+range]
        self._gains[:, 0] = cfg.height_center + torch.tanh(actions[:, 0]) * cfg.height_range
        # gains CÓ DẤU (RL tự dò chiều) — bỏ height_sign
        self._gains[:, 1:4] = torch.tanh(actions[:, 1:4]) * self._max_gains
        self._gains[:, 4] = torch.tanh(actions[:, 4]) * cfg.max_kp_lat

    def apply_actions(self):
        cfg = self.cfg
        data = self._asset.data

        # cao độ base so với trục bánh (terrain-independent, khớp reward track_height)
        z_axle = 0.5 * (data.body_pos_w[:, self._wr_idx, 2] + data.body_pos_w[:, self._wl_idx, 2])
        height = data.root_pos_w[:, 2] - z_axle

        r_h, kp, ki, kd, kp_lat = (
            self._gains[:, 0],
            self._gains[:, 1],
            self._gains[:, 2],
            self._gains[:, 3],
            self._gains[:, 4],
        )
        e_h = r_h - height  # bám SETPOINT do mạng xuất (không phải height_cmd cứng)

        # PID common (anti-windup) → góc hip chung 2 chân (đặt cao độ)
        self._integral = torch.clamp(self._integral + e_h * self._dt, -cfg.integral_limit, cfg.integral_limit)
        derivative = (e_h - self._prev_error) / self._dt
        self._prev_error = e_h
        hip_common = kp * e_h + ki * self._integral + kd * derivative

        # vi sai 2 chân → sửa nghiêng ngang (lateral, axis 0; setpoint = 0)
        e_lat = data.projected_gravity_b[:, cfg.lateral_axis]
        hip_diff = kp_lat * e_lat

        right = torch.clamp(hip_common + hip_diff, -cfg.output_limit, cfg.output_limit)
        left = torch.clamp(hip_common - hip_diff, -cfg.output_limit, cfg.output_limit)
        self._processed[:] = torch.stack([right, left], dim=-1)

        self._asset.set_joint_position_target(self._processed, joint_ids=self._active_ids)
        self._asset.set_joint_position_target(-self._processed, joint_ids=self._mimic_ids)

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        # đồng bộ mimic = -active từ pose hiện tại + reset trạng thái PID
        if env_ids is None:
            active_pos = self._asset.data.joint_pos[:, self._active_ids]
            self._asset.write_joint_position_to_sim(-active_pos, joint_ids=self._mimic_ids)
            self._integral.zero_()
            self._prev_error.zero_()
            self._raw_actions.zero_()
            self._gains.zero_()
            self._processed.zero_()
        else:
            active_pos = self._asset.data.joint_pos[env_ids][:, self._active_ids]
            self._asset.write_joint_position_to_sim(-active_pos, joint_ids=self._mimic_ids, env_ids=env_ids)
            self._integral[env_ids] = 0.0
            self._prev_error[env_ids] = 0.0
            self._raw_actions[env_ids] = 0.0
            self._gains[env_ids] = 0.0
            self._processed[env_ids] = 0.0


@configclass
class HipMimicPIDActionCfg(ActionTermCfg):
    """Config cho HipMimicPIDAction (PI-ANN hip cascade — mạng xuất setpoint + gains)."""

    class_type: type = HipMimicPIDAction

    asset_name: str = MISSING
    active_joint_names: list[str] = MISSING
    mimic_joint_names: list[str] = MISSING
    wheel_right_body: str = "wheel"
    wheel_left_body: str = "wheel_01"

    # Trần |gain| (tanh CÓ DẤU → ± trị này; RL tự tìm dấu, không cần height_sign).
    max_kp: float = 25.0
    max_ki: float = 10.0
    max_kd: float = 2.5
    max_kp_lat: float = 10.0

    # Setpoint cao độ mạng xuất: r_h ∈ [center - range, center + range] (m, so với trục bánh).
    height_center: float = 0.30
    height_range: float = 0.10  # → r_h ∈ [0.20, 0.40]

    output_limit: float = 0.5  # rad — dải góc hip
    integral_limit: float = 5.0

    lateral_axis: int = 0  # projected_gravity_b nghiêng ngang


# ──────────────────────────────────────────────────────────────────────────────
# PI-ANN: tầng cuối mạng XUẤT RA hệ số PID, khối PID là "tầng cuối" tính lệnh bánh
# ──────────────────────────────────────────────────────────────────────────────


class WheelPIDBalanceAction(ActionTerm):
    """PI-ANN cho bánh (cascade): mạng xuất SETPOINT độ nghiêng + hệ số PID → vận tốc bánh.

    Cấu trúc cân bằng chuẩn của xe con-lắc-ngược: mạng (vòng ngoài) xuất ĐỘ NGHIÊNG
    MỤC TIÊU ``r_lean`` để tạo gia tốc tiến; PID (vòng trong) kéo độ nghiêng đo về
    ``r_lean`` bằng vận tốc bánh. Muốn tiến → mạng đặt r_lean ngả tới → PID giữ → bánh
    lăn theo. Mạng thấy velocity_cmd trong obs nên tự chọn r_lean để bám tốc.

    Gains qua ``tanh`` (CÓ DẤU) → RL tự tìm dấu ổn định, KHÔNG cần công tắc control_sign.

    Luồng mỗi bước::

        raw (policy)  = [rl_raw, kp_raw, ki_raw, kd_raw, kp_yaw_raw]   # action_dim = 5
        r_lean        = tanh(rl_raw) * lean_limit               # ± rad (độ nghiêng mục tiêu)
        Kp,Ki,Kd      = tanh(raw) * [max_kp, max_ki, max_kd]    # CÓ DẤU
        Kp_yaw        = tanh(raw) * max_kp_yaw                  # CÓ DẤU
        e_bal         = r_lean - pitch_lean    (bám SETPOINT nghiêng học được)
        u_common      = Kp·e_bal + Ki·∫e_bal + Kd·de_bal/dt    # chung 2 bánh: balance + tiến
        u_yaw         = Kp_yaw * (wz_cmd - wz_meas)            # vi sai 2 bánh: xoay
        wheel_R       = clamp(u_common + u_yaw) * dir_R
        wheel_L       = clamp(u_common - u_yaw) * dir_L

    ⚠️ QUY ƯỚC:
      - ``balance_axis`` = 2: ``projected_gravity_b[:, 2]`` = nghiêng dọc fore-aft.
      - wz_meas = ``root_ang_vel_w[:, 2]`` (world Z), khớp lệnh ang_vel_z.
      - ``wheel_dir``: [phải, trái], đổi [1,-1] nếu robot xoay tại chỗ thay vì tiến
        (mirror joint USD — hình học, KHÁC với dấu điều khiển đã gộp vào gains).
    """

    cfg: WheelPIDBalanceActionCfg
    _asset: Articulation

    def __init__(self, cfg: WheelPIDBalanceActionCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self._asset = env.scene[cfg.asset_name]

        all_joints = self._asset.data.joint_names
        self._wheel_ids = [all_joints.index(n) for n in cfg.wheel_joint_names]

        # apply_actions() chạy MỖI physics substep → dt = physics_dt (PID ở 200 Hz).
        self._dt = env.physics_dt

        # raw/log = [r_lean, Kp, Ki, Kd, Kp_yaw]
        self._raw_actions = torch.zeros(env.num_envs, 5, device=env.device)
        self._gains = torch.zeros(env.num_envs, 5, device=env.device)
        self._wheel_target = torch.zeros(env.num_envs, len(self._wheel_ids), device=env.device)

        self._integral = torch.zeros(env.num_envs, device=env.device)
        self._prev_error = torch.zeros(env.num_envs, device=env.device)

        self._max_gains = torch.tensor([cfg.max_kp, cfg.max_ki, cfg.max_kd], device=env.device).unsqueeze(0)
        self._wheel_dir = torch.tensor(cfg.wheel_dir, device=env.device).unsqueeze(0)

    @property
    def action_dim(self) -> int:
        return 5

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._gains  # [r_lean, Kp, Ki, Kd, Kp_yaw] đã map (log/debug)

    def process_actions(self, actions: torch.Tensor):
        cfg = self.cfg
        self._raw_actions[:] = actions
        # setpoint độ nghiêng ∈ ±lean_limit
        self._gains[:, 0] = torch.tanh(actions[:, 0]) * cfg.lean_limit
        # gains CÓ DẤU (RL tự dò chiều) — bỏ control_sign
        self._gains[:, 1:4] = torch.tanh(actions[:, 1:4]) * self._max_gains
        self._gains[:, 4] = torch.tanh(actions[:, 4]) * cfg.max_kp_yaw

    def apply_actions(self):
        cfg = self.cfg
        data = self._asset.data

        r_lean, kp, ki, kd, kp_yaw = (
            self._gains[:, 0],
            self._gains[:, 1],
            self._gains[:, 2],
            self._gains[:, 3],
            self._gains[:, 4],
        )

        # sai số bám SETPOINT độ nghiêng (mạng xuất r_lean để tạo gia tốc tiến)
        e_bal = r_lean - data.projected_gravity_b[:, cfg.balance_axis]

        self._integral = torch.clamp(self._integral + e_bal * self._dt, -cfg.integral_limit, cfg.integral_limit)
        derivative = (e_bal - self._prev_error) / self._dt
        self._prev_error = e_bal
        u_common = kp * e_bal + ki * self._integral + kd * derivative

        # vi sai 2 bánh → track yaw (wz_cmd - wz_meas), wz_meas = world-Z ang vel
        cmd = self._env.command_manager.get_command(cfg.velocity_command_name)
        e_yaw = cmd[:, cfg.ang_vel_cmd_index] - data.root_ang_vel_w[:, 2]
        u_yaw = kp_yaw * e_yaw

        wheel_r = torch.clamp(u_common + u_yaw, -cfg.output_limit, cfg.output_limit)
        wheel_l = torch.clamp(u_common - u_yaw, -cfg.output_limit, cfg.output_limit)
        self._wheel_target[:] = torch.stack([wheel_r, wheel_l], dim=-1) * self._wheel_dir
        self._asset.set_joint_velocity_target(self._wheel_target, joint_ids=self._wheel_ids)

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        if env_ids is None:
            self._integral.zero_()
            self._prev_error.zero_()
            self._raw_actions.zero_()
            self._gains.zero_()
        else:
            self._integral[env_ids] = 0.0
            self._prev_error[env_ids] = 0.0
            self._raw_actions[env_ids] = 0.0
            self._gains[env_ids] = 0.0


@configclass
class WheelPIDBalanceActionCfg(ActionTermCfg):
    """Config cho WheelPIDBalanceAction (PI-ANN bánh cascade — mạng xuất setpoint + gains)."""

    class_type: type = WheelPIDBalanceAction

    asset_name: str = MISSING
    wheel_joint_names: list[str] = MISSING

    # Trần |gain| (tanh CÓ DẤU → ± trị này; RL tự tìm dấu, không cần control_sign).
    max_kp: float = 200.0
    max_ki: float = 50.0
    max_kd: float = 25.0
    max_kp_yaw: float = 150.0

    lean_limit: float = 0.3  # ± rad — dải độ nghiêng mục tiêu mạng được phép xuất

    wheel_dir: list[float] = (1.0, 1.0)  # [phải, trái]; đổi [1,-1] nếu robot xoay tại chỗ
    output_limit: float = 60.0  # rad/s — khớp velocity_limit bánh
    integral_limit: float = 20.0  # anti-windup

    balance_axis: int = 2  # projected_gravity_b fore-aft
    velocity_command_name: str = "velocity_command"
    ang_vel_cmd_index: int = 2  # ang_vel_z trong command
