from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg
)

@configclass
class LegV2PPORunnerCfgBalance(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 100
    max_iterations = 1000
    save_interval = 10
    experiment_name = "leg_v2_ppo_balance"
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.2,
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        actor_hidden_dims=[1024, 1024, 1024, 1024],
        critic_hidden_dims=[1024, 1024, 1024, 1024],
        activation="relu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.05,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=128,
        learning_rate=1e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.02,
        max_grad_norm=0.5,
    )