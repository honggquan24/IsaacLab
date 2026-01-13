"""PPO configuration for Evobot V1 navigation task."""

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg
)

@configclass
class EvobotVelocityPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """PPO configuration for navigation task.

    This config is optimized for the navigation task where the robot
    needs to balance while moving to target positions.
    """

    # steps_per_episode = episode_length_s / (sim.dt × decimation)
    # Navigation episodes are 30s (vs 20s for balance)
    num_steps_per_env = 10 * 60  # 30s episodes at 60Hz
    max_iterations = 1000
    save_interval = 20
    experiment_name = "evobot_v1_velocity"

    # Network architecture (slightly smaller than balance for faster training)
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.05,
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        actor_hidden_dims=[128, 256, 128],
        critic_hidden_dims=[128, 256, 128],
        activation="elu",
    )

    # PPO algorithm parameters
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,  # Lower entropy for more focused exploration
        num_learning_epochs=4,
        num_mini_batches=32,
        learning_rate=1e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class EvobotNavigationPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """PPO configuration for navigation task.

    This config is optimized for the navigation task where the robot
    needs to balance while moving to target positions.
    """

    # steps_per_episode = episode_length_s / (sim.dt × decimation)
    # Navigation episodes are 30s (vs 20s for balance)
    num_steps_per_env = 10 * 60  # 30s episodes at 60Hz
    max_iterations = 1000
    save_interval = 20
    experiment_name = "evobot_v1_velocity"

    # Network architecture (slightly smaller than balance for faster training)
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.05,
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        actor_hidden_dims=[128, 256, 128],
        critic_hidden_dims=[128, 256, 128],
        activation="elu",
    )

    # PPO algorithm parameters
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,  # Lower entropy for more focused exploration
        num_learning_epochs=4,
        num_mini_batches=32,
        learning_rate=1e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
