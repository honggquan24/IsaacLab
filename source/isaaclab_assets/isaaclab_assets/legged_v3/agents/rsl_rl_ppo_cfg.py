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


# ─────────────────────────── Curriculum PPO Runners ──────────────────────────

@configclass
class LeggedV3CurrStage1PPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """Stage 1: Wheels only — policy học velocity tracking + balance đơn giản."""

    num_steps_per_env = 24
    max_iterations = 2000
    save_interval = 100
    experiment_name = "legged_v3_curr_stage1"

    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,          # noise cao hơn để khám phá rộng hơn
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
        entropy_coef=0.01,           # entropy cao hơn để tránh hội tụ sớm
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
class LeggedV3CurrStage2PPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """Stage 2: Wheels + Knees — policy học thêm height control."""

    num_steps_per_env = 24
    max_iterations = 2000
    save_interval = 100
    experiment_name = "legged_v3_curr_stage2"

    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.7,
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
        entropy_coef=0.008,
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
class LeggedV3CurrStage3PPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """Stage 3: Wheels + Knees + Thighs — policy học phối hợp thigh-knee."""

    num_steps_per_env = 24
    max_iterations = 2000
    save_interval = 100
    experiment_name = "legged_v3_curr_stage3"

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
class LeggedV3CurrStage4PPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """Stage 4: Full 8-DOF — tất cả khớp, fine-tune từ Stage 3."""

    num_steps_per_env = 24
    max_iterations = 3000
    save_interval = 100
    experiment_name = "legged_v3_curr_stage4"

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
class LeggedV3CurriculumPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """Single-run auto-curriculum (wheel → knee → thigh → hip).

    Runs for 5000 iterations total:
      Phase 0 (iter    0–999 ): wheels only
      Phase 1 (iter 1000–1999): + knee
      Phase 2 (iter 2000–2999): + thigh
      Phase 3 (iter 3000–4999): + hip (full 8-DOF)

    Higher entropy coef and noise at the start → encourages exploration during
    the wheel-only phase. The network naturally adapts as joints unlock because
    the action space shape is fixed throughout.
    """

    num_steps_per_env = 24      # must match _STEPS_PER_ITER in curriculums.py
    max_iterations = 5000
    save_interval = 100
    experiment_name = "legged_v3_curriculum"

    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,          # high initial exploration for wheel-only phase
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
        entropy_coef=0.01,           # higher entropy → avoid premature convergence
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
