"""RSL-RL PPO config cho Biped-Inner-Tilt.

Obs: 74-dim = tilt_error(3)+imu_quat(4)+imu_lin_acc(3)+ang_vel(3)+gravity(3)
              +hip_pos_err(4)+hip_vel(4)+wheel_vel(2)
              +all_joint_pos(10)+all_joint_vel(10)+all_joint_acc(10)+last_act(21)
Action: 21-dim = 7 × [kp, ki, kd]
Network: [128, 128] hidden (output dim tự động = action_dim)
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
        init_noise_std=0.1,
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        actor_hidden_dims=[128, 128],
        critic_hidden_dims=[128, 128],
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
