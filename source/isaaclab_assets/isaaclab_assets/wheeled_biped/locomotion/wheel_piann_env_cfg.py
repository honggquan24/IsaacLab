# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""PI-ANN variant của legged_v5 wheeled locomotion.

Khác bản gốc: CẢ hip VÀ bánh đều do PI-ANN điều khiển — tầng cuối mạng XUẤT RA
hệ số PID, khối PID tính lệnh từ sai số đo (đề tài "PI-ANN trên ESP32").

  - bánh = WheelPIDBalanceAction: [Kp,Ki,Kd,Kp_yaw] → PID common (balance+tiến)
    + P differential (yaw). action_dim = 4.
  - hip  = HipMimicPIDAction:     [Kp,Ki,Kd,Kp_lat] → PID common (bám cao độ)
    + P differential (nghiêng ngang). action_dim = 4. mimic = -active.

Tổng policy action_dim = 8 (4 hip + 4 bánh). Baseline end-to-end là 4 (2 hip + 2 bánh).
Scene, obs, reward, termination, command kế thừa nguyên bản.

Train:
    ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \\
        --task Isaac-Wheeled-Biped-Wheel-PIANN --num_envs 4096 --headless
"""

from isaaclab.utils import configclass

from .. import mdp
from .wheel_env_cfg import (
    ActionCfg,
    WheeledBipedWheelEnvCfg,
)


@configclass
class PIANNActionCfg(ActionCfg):
    """CẢ hip và bánh đều PI-ANN (mạng xuất hệ số PID)."""

    # Gỡ action end-to-end của baseline (hip position trực tiếp + wheel velocity).
    leg_pos = None
    wheel_vel = None

    # Hip: mạng xuất [setpoint cao độ, Kp, Ki, Kd, Kp_lat] → PID tính góc hip.
    leg_pid = mdp.actions.HipMimicPIDActionCfg(
        asset_name="robot",
        active_joint_names=["right_hip_joint", "left_hip_joint"],
        mimic_joint_names=["right_hip_joint_mimic", "left_hip_joint_mimic"],
        wheel_right_body="wheel",
        wheel_left_body="wheel_01",
        max_kp=25.0,
        max_ki=10.0,
        max_kd=2.5,
        max_kp_lat=10.0,
        height_center=0.30,
        height_range=0.10,  # setpoint cao độ ∈ [0.20, 0.40] m
        output_limit=0.5,
        integral_limit=5.0,
    )

    # Bánh: mạng xuất [setpoint độ nghiêng, Kp, Ki, Kd, Kp_yaw] → PID tính vận tốc bánh.
    wheel_pid = mdp.actions.WheelPIDBalanceActionCfg(
        asset_name="robot",
        wheel_joint_names=["right_wheel_joint", "left_wheel_joint"],
        max_kp=200.0,
        max_ki=50.0,
        max_kd=25.0,
        max_kp_yaw=150.0,
        lean_limit=0.3,  # độ nghiêng mục tiêu mạng được phép xuất ±0.3 rad
        output_limit=60.0,
        integral_limit=20.0,
    )


@configclass
class WheeledBipedWheelPIANNEnvCfg(WheeledBipedWheelEnvCfg):
    """Env PI-ANN — chỉ đổi tầng action bánh, còn lại kế thừa baseline."""

    actions: PIANNActionCfg = PIANNActionCfg()
