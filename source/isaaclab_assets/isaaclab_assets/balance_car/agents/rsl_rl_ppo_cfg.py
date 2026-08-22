# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Cấu hình PPO cho xe cân bằng, theo ``AnymalCFlatPPORunnerCfg`` của Isaac Lab.

Bản cũ lệch khỏi mọi mặc định của Isaac Lab theo hướng học chậm hơn hẳn, và không có ghi chú
nào giải thích vì sao:

=========================  ==========  ==========  =====================================
tham số                    bản cũ      bản này     vì sao
=========================  ==========  ==========  =====================================
``num_learning_epochs``    1           5           mỗi rollout chỉ được dùng 1 lần → tốn
                                                   gấp 5 lần số mẫu cho cùng tiến bộ
``num_mini_batches``       64          4           batch quá nhỏ → gradient nhiễu
``learning_rate``          1e-4        1e-3        chậm gấp 10 lần
``schedule``               "adam"      "adaptive"  "adaptive" tự chỉnh lr theo KL, đây là
                                                   mặc định của mọi task locomotion
``init_noise_std``         0.2         1.0         0.2 gần như tắt thăm dò ngay từ đầu
``num_steps_per_env``      100         24          rollout dài không cần thiết khi
                                                   episode chỉ 20 s
=========================  ==========  ==========  =====================================
"""

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class BalanceCarPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 1000
    save_interval = 50
    # ĐỪNG đổi tên này: tầng navigation dò policy đã export theo
    # ``logs/rsl_rl/carbalance_ppo/*/exported/policy.pt``.
    experiment_name = "carbalance_ppo"
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        # Bật chuẩn hoá quan sát — chỗ này KHÁC mẫu anymal (để False) và là có lý do: quan sát
        # ở đây trộn hai thang rất lệch nhau, tốc độ bánh tới ±40 rad/s bên cạnh
        # projected_gravity trong [-1, 1]. Không chuẩn hoá thì lớp đầu bị tốc độ bánh át.
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        actor_hidden_dims=[128, 128, 128],
        critic_hidden_dims=[128, 128, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
