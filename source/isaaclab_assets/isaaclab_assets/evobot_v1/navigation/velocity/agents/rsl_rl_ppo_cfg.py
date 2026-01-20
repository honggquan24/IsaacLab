"""PPO configuration for Evobot V1 navigation task."""

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg
)

@configclass
class EvobotVelocityPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """PPO configuration for low-level velocity tracking task.

    This config is optimized for the navigation task where the robot
    needs to balance while moving to target positions.
    Low-level policy learns both balance AND velocity tracking.
    """

    # steps_per_episode = episode_length_s / (sim.dt × decimation)
    # Navigation episodes are 30s (vs 20s for balance)
    num_steps_per_env = 10 * 60  # 30s episodes at 60Hz
    max_iterations = 1000
    save_interval = 5
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
        entropy_coef=0.01,
        num_learning_epochs=4,
        num_mini_batches=32,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.02,
        max_grad_norm=1.0,
    )


@configclass
class EvobotVelocityPretrainPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """PPO configuration for high-level hierarchical velocity command policy.

    This config is for training the high-level policy that uses a pre-trained
    low-level balance policy. The high-level policy only learns to generate
    velocity commands (vx, vy, omega) while the low-level handles balance.

    Key differences from low-level config:
    - Smaller network (simpler task: only velocity commands)
    - Higher learning rate (faster convergence)
    - Higher entropy (more exploration of velocity commands)
    - Shorter episodes (faster feedback)
    """

    # Episode length from hierarchical_vel_env_cfg.py
    # episode_length_s = commands.base_velocity.resampling_time_range[1] = 5.0s
    # decimation = LOW_LEVEL_ENV_CFG.decimation * 4 = 1 * 4 = 4
    # sim.dt = 1/60 = 0.0167s
    # num_steps = 5.0 / (0.0167 * 4) = 75 steps
    num_steps_per_env = 5 * 60  # 5s episodes with 4x decimation
    max_iterations = 1000  # More iterations for high-level learning
    save_interval = 20
    experiment_name = "evobot_v1_velocity_hierarchical"

    # Smaller network for high-level policy (3D action space: vx, vy, wz)
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.1,  # Low noise to keep actions in [-1, 1] range
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        actor_hidden_dims=[64, 128, 64],  # Smaller than low-level
        critic_hidden_dims=[64, 128, 64],
        activation="elu",
    )

    # PPO algorithm parameters optimized for hierarchical learning
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.02,  # Higher entropy for velocity exploration
        num_learning_epochs=5,  # More epochs (smaller batches)
        num_mini_batches=16,  # Fewer mini-batches (less data per episode)
        learning_rate=3e-4,  # Higher LR for faster convergence
        schedule="adaptive",
        gamma=0.98,  # Slightly lower discount (shorter episodes)
        lam=0.95,
        desired_kl=0.015,  # Slightly higher KL tolerance
        max_grad_norm=1.0,
    )


@configclass
class EvobotVelocityPIDPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """PPO configuration for PID-based velocity control.

    This config is for training a high-level policy that outputs velocity
    commands (vx, wz), which are then converted to wheel torques by a PID
    controller. This is an alternative to learning end-to-end control.

    Key characteristics:
    - 2D action space (vx, wz) instead of 5D joint control
    - PID handles low-level control (deterministic)
    - Policy focuses on high-level navigation strategy
    """

    num_steps_per_env = 10 * 60  # 10s episodes at 60Hz
    max_iterations = 1000
    save_interval = 10
    experiment_name = "evobot_v1_velocity_pid"

    # Smaller network for high-level policy (2D action space: vx, wz)
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.1,
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        actor_hidden_dims=[64, 128, 64],
        critic_hidden_dims=[64, 128, 64],
        activation="elu",
    )

    # PPO algorithm parameters
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=32,
        learning_rate=1e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.02,
        max_grad_norm=1.0,
    )


@configclass
class EvobotGripperFineTunePPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """PPO configuration for gripper fine-tuning.

    Fine-tuning from pretrained velocity policy with focus on gripper control.
    Uses lower learning rate to avoid catastrophic forgetting of base skills.

    Key characteristics:
    - LOW learning rate (preserve pretrained weights)
    - Same network size as base velocity policy
    - Longer episodes (30s for gripper practice)
    - Higher entropy (explore gripper movements)
    """

    num_steps_per_env = 10 * 60  # 30s episodes at 60Hz (from velocity_env_cfg_gripper_finetune.py)
    max_iterations = 500  # Fewer iterations for fine-tuning
    save_interval = 5
    experiment_name = "evobot_v1_gripper_finetune"

    # Same network as base velocity policy
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.05,
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        actor_hidden_dims=[128, 256, 128],
        critic_hidden_dims=[128, 256, 128],
        activation="elu",
    )

    # PPO algorithm with LOWER learning rate for fine-tuning
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.02,  # HIGHER than base (0.01) - explore gripper movements
        num_learning_epochs=5,
        num_mini_batches=32,
        learning_rate=3e-5,  # MUCH LOWER than base (1e-3) - preserve pretrained skills
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,  # LOWER KL limit - careful updates
        max_grad_norm=0.5,  # LOWER gradient clipping - stable fine-tuning
    )
