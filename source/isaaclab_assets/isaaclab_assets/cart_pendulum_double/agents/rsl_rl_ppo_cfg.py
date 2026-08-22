# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class CartPendulumDoublePPORunnerCfg(RslRlOnPolicyRunnerCfg):
    # 200 bước ở 120 Hz = 1.67 s. Bình luận cũ đòi phủ trọn một chu kỳ lắc (1.09 s) —
    # lập luận đó thuộc về bài SWING-UP, mà `hanging_prob = 0.0` nên mọi env đã khởi động ở tư
    # thế đứng và bài giờ là giữ thăng bằng thuần. Thứ rollout cần phủ là tầm nhìn của GAE:
    # 1/(1-gamma*lam) = 18 bước, nhỏ hơn 200 rất nhiều. Giữ 200 để chi phí mỗi vòng
    # không đổi so với lúc chạy 60 Hz.
    num_steps_per_env = 200
    max_iterations = 2000
    save_interval = 100
    experiment_name = "cartpole_v2_ppo"
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.5,
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        actor_hidden_dims=[512, 512],
        critic_hidden_dims=[512, 512],
        activation="relu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        # 1 epoch + 64 mini-batch + lr cố định 1e-4 là bộ tham số cũ, học chậm hơn mặc định
        # Isaac Lab khoảng 5-10 lần: mỗi rollout chỉ được dùng đúng một lần, mini-batch nhỏ nên
        # gradient nhiễu, và `schedule="adam"` KHÔNG phải adaptive — nó để lr đứng yên, tức
        # `desired_kl` bên dưới là dòng config chết.
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        # 120 Hz thay vì 60 Hz nên gamma phải co theo, nếu không tầm nhìn tính bằng GIÂY bị
        # cắt ngắn đúng 2 lần: 1/(1-0.995) = 200 bước = 1.67 s, bằng 0.99 ở 60 Hz.
        gamma=0.995,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=0.5,
    )


@configclass
class CartPendulumDoublePositionPPORunnerCfg(CartPendulumDoublePPORunnerCfg):
    """Bám vị trí: khởi động sẵn ở tư thế đứng nên không cần nhiều vòng như swing-up."""

    max_iterations = 1500
    experiment_name = "cartpole_v2_position_ppo"
