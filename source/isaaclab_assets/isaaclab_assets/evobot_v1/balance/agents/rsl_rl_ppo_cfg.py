from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg
)

@configclass
class EvobotPPORunnerCfgBalance(RslRlOnPolicyRunnerCfg):
    # SPEED OPTIMIZATION: Giảm từ 300 xuống 24 steps
    # - Mỗi iteration chỉ cần chờ 24 steps thay vì 300 (12.5x faster!)
    # - Với 9999 envs: 24×9999 = ~240k samples mỗi iteration vẫn đủ lớn
    num_steps_per_env = 10 * 60
    max_iterations = 1000  # Tăng vì mỗi iteration ít steps hơn
    save_interval = 100
    experiment_name = "evobot_v1_ppo_balance"
    empirical_normalization = False  # Tắt normalization để nhanh hơn

    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.05,
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        # Network size vừa phải cho tốc độ tốt
        actor_hidden_dims=[128, 256, 128],
        critic_hidden_dims=[128, 256, 128],
        activation="elu",  # ELU nhanh hơn ReLU một chút
    )
    
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=4,  # Giảm từ 5 xuống 4
        num_mini_batches=32,
        learning_rate=1e-4,  # Tăng learning rate vì batch nhỏ hơn
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
    
