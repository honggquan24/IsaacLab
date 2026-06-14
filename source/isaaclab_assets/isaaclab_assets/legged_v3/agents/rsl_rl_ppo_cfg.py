"""RSL-RL PPO configuration cho Legged Robot V5."""
from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)


@configclass
class LeggedV5WheelPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """PPO runner config cho robot V5 — bipedal wheeled locomotion (bản MIMIC)."""

    num_steps_per_env = 24
    max_iterations = 5000
    save_interval = 100
    experiment_name = "legged_v5_wheel_mimic"

    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.5,
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        actor_hidden_dims=[1024, 1024],
        critic_hidden_dims=[1024, 1024],
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
class LeggedV5WheelNoMimicPPORunnerCfg(LeggedV5WheelPPORunnerCfg):
    """Bản KHÔNG mimic — policy điều khiển trực tiếp cả 4 khớp hip. Log riêng."""

    experiment_name = "legged_v5_wheel_nomimic"


@configclass
class LeggedV5NavPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """PPO runner cho tầng cao navigation (low-level đóng băng)."""

    num_steps_per_env = 24
    max_iterations = 3000
    save_interval = 100
    experiment_name = "legged_v5_navigation"

    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
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
class LeggedV5WarehouseNavPPORunnerCfg(LeggedV5NavPPORunnerCfg):
    """PPO runner cho navigation trong kho có LiDAR né vật cản (log riêng).

    Cùng siêu tham số với nav phẳng; obs tầng cao lớn hơn (thêm 90 tia LiDAR),
    mạng tự co giãn theo input nên không cần đổi kiến trúc.
    """

    experiment_name = "legged_v5_warehouse_nav"
