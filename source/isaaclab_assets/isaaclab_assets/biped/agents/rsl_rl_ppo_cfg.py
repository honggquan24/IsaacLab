"""RSL-RL PPO configs cho tất cả Biped tasks.

Inner (Biped-Inner-Tilt):
  Obs: 74-dim   Action: 21-dim (7 × [kp,ki,kd])
Outer-Vel-PID (Biped-Outer-Vel-PID):
  Obs: 26-dim   Action: 9-dim  (3 × [kp,ki,kd])
Outer-Vel-Direct (Biped-Outer-Vel-Direct):
  Obs: 20-dim   Action: 3-dim  (roll_des, pitch_des, yaw_rate)
Unified-Vel (Biped-Unified-Vel):
  Obs: 70-dim   Action: 28-dim (7 × [kp,ki,kd,sp]), vel→torque trực tiếp
"""
from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)


@configclass
class BipedInnerTiltRunnerCfg(RslRlOnPolicyRunnerCfg):
    """Train inner tilt PID — 7 × 3 = 21 gains."""

    num_steps_per_env = 48
    max_iterations    = 5000
    save_interval     = 200
    experiment_name   = "biped_inner_tilt"
    obs_groups        = {"policy": ["policy"], "critic": ["policy"]}

    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.5,
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        actor_hidden_dims=[256, 256],
        critic_hidden_dims=[256, 256],
        activation="elu",
    )

    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=3.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class BipedOuterVelPIDRunnerCfg(RslRlOnPolicyRunnerCfg):
    """Train outer velocity PID — 3 × 3 = 9 gains."""

    num_steps_per_env = 48
    max_iterations    = 3000
    save_interval     = 200
    experiment_name   = "biped_outer_vel_pid"
    obs_groups        = {"policy": ["policy"], "critic": ["policy"]}

    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.3,
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        actor_hidden_dims=[256, 256],
        critic_hidden_dims=[256, 256],
        activation="elu",
    )

    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=3.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class BipedOuterVelDirectRunnerCfg(RslRlOnPolicyRunnerCfg):
    """Train outer direct — 3-dim tilt setpoint."""

    num_steps_per_env = 48
    max_iterations    = 3000
    save_interval     = 200
    experiment_name   = "biped_outer_vel_direct"
    obs_groups        = {"policy": ["policy"], "critic": ["policy"]}

    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.5,
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        actor_hidden_dims=[256, 256],
        critic_hidden_dims=[256, 256],
        activation="elu",
    )

    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=3.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class BipedUnifiedVelRunnerCfg(RslRlOnPolicyRunnerCfg):
    """Train unified vel→torque — 7 × 3 = 21 gains, single stage."""

    num_steps_per_env = 48
    max_iterations    = 5000
    save_interval     = 200
    experiment_name   = "biped_unified_vel"
    obs_groups        = {"policy": ["policy"], "critic": ["policy"]}

    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.5,
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        actor_hidden_dims=[256, 256],
        critic_hidden_dims=[256, 256],
        activation="elu",
    )

    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=3.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
