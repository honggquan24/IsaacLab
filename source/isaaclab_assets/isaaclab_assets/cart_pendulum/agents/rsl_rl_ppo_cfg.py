# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class CartPendulumPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 200
    max_iterations = 200
    save_interval = 50
    experiment_name = "cartpole_v1_ppo"
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.2,
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        actor_hidden_dims=[256, 256],
        critic_hidden_dims=[256, 512, 256],
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
class CartPendulumPositionPPORunnerCfg(CartPendulumPPORunnerCfg):
    """Task bám vị trí khó hơn task cân bằng: cần thêm vòng lặp và nhiễu khám phá lớn hơn."""

    max_iterations = 600
    save_interval = 100
    experiment_name = "cartpole_v1_position_ppo"
    policy = RslRlPpoActorCriticCfg(
        # nhiễu ban đầu lớn hơn 0.2 để xe dám chạy hết ray đi tìm mốc ở xa
        init_noise_std=0.5,
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        actor_hidden_dims=[256, 256],
        critic_hidden_dims=[256, 512, 256],
        activation="relu",
    )
