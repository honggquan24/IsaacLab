# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class CartPendulumDoublePPORunnerCfg(RslRlOnPolicyRunnerCfg):
    # chuỗi này có chu kỳ lắc 1.09 s; 100 bước ở 60 Hz mới được 1.67 s, chưa đủ một nhịp
    # lắc nên rollout cắt ngang giữa chừng và credit assignment của swing-up bị hỏng.
    # 200 bước = 3.33 s, phủ 3.1 chu kỳ.
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
        num_learning_epochs=1,
        num_mini_batches=64,
        learning_rate=1.0e-4,
        schedule="adam",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=0.5,
    )


@configclass
class CartPendulumDoublePositionPPORunnerCfg(CartPendulumDoublePPORunnerCfg):
    """Bám vị trí: khởi động sẵn ở tư thế đứng nên không cần nhiều vòng như swing-up."""

    max_iterations = 1500
    experiment_name = "cartpole_v2_position_ppo"
