"""RSL-RL PPO configuration for Legged Robot V3."""
from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)


@configclass
class LeggedV3WheelPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """PPO runner config for legged_v3 wheeled locomotion."""

    # episode_length_s=10s, decimation=2, sim.dt=1/60 → 10×30 = 300 steps/episode
    # 24  = H1/AnymalC standard (nhiều update/episode, cần num_envs ≥ 1024)
    # 300 = 1 full episode (ổn định hơn khi debug với ít env)
    num_steps_per_env = 24
    max_iterations = 5000
    save_interval = 100
    experiment_name = "legged_v3_wheel"

    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.5,
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        actor_hidden_dims=[256, 256, 256],
        critic_hidden_dims=[256, 256, 256],
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


@configclass
class LeggedV3LegPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """PPO runner config for legged_v3 leg-based locomotion."""

    num_steps_per_env = 24
    max_iterations = 5000
    save_interval = 100
    experiment_name = "legged_v3_leg"

    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.5,
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        actor_hidden_dims=[256, 256, 256],
        critic_hidden_dims=[256, 256, 256],
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
