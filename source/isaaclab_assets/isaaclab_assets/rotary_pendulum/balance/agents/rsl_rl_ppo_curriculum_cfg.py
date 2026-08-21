# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""PPO configurations for curriculum learning stages."""

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class RotaryPendulumStage1PPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """Stage 1: Swing-up only - More exploration, longer training."""

    num_steps_per_env = 20 * 60  # 1200 steps (20s episodes)
    max_iterations = 1000  # More iterations for swing-up discovery
    save_interval = 100
    experiment_name = "rotary_pendulum_v2_stage1_swingup"
    empirical_normalization = False

    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,  # Higher noise for exploration
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        actor_hidden_dims=[64, 128, 64],
        critic_hidden_dims=[64, 128, 64],
        activation="elu",
    )

    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.02,  # Higher entropy for exploration
        num_learning_epochs=5,
        num_mini_batches=16,
        learning_rate=1e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class RotaryPendulumStage2PPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """Stage 2: Balance + heading tracking - Fine-tuning from Stage 1."""

    num_steps_per_env = 15 * 60  # 900 steps (15s episodes)
    max_iterations = 500  # Fewer iterations (fine-tuning)
    save_interval = 50
    experiment_name = "rotary_pendulum_v2_stage2_tracking"
    empirical_normalization = False

    # Resume from Stage 1 checkpoint
    resume = True  # Set via command line --resume
    load_run = ""  # Set via command line --load_run
    load_checkpoint = ""  # Set via command line --checkpoint

    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.3,  # Lower noise for fine-tuning
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        actor_hidden_dims=[64, 128, 64],
        critic_hidden_dims=[64, 128, 64],
        activation="elu",
    )

    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,  # Lower entropy for exploitation
        num_learning_epochs=5,
        num_mini_batches=16,
        learning_rate=5e-4,  # Lower learning rate for fine-tuning
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
