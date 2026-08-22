# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""PPO cho hai task navigation, theo ``NavigationEnvPPORunnerCfg`` của Isaac Lab."""

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class BalanceCarNavigationPretrainedPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """Cascade: tầng cao xuất lệnh vận tốc cho policy thăng bằng đã đóng băng.

    Bài này nhẹ — mạng chỉ phải học ánh xạ "lệch quỹ đạo → lệnh vận tốc", phần vật lý khó đã
    nằm ở tầng thấp. Vì vậy mạng nhỏ và rollout ngắn giống mẫu navigation của Isaac Lab, không
    cần cỡ của một policy locomotion.
    """

    # tầng cao chạy 10 Hz, episode 20 s = 200 bước
    num_steps_per_env = 24
    max_iterations = 1000
    save_interval = 50
    # Thư mục log RIÊNG với task nav học-từ-đầu: hai task có obs/action khác hẳn nhau (bản
    # từ-đầu xuất 2 mô-men bánh, bản này xuất 3 số vận tốc). Dùng chung ``experiment_name`` thì
    # ``--resume`` và ``play.py`` đều lấy run mới nhất bất kể nó thuộc task nào, và nạp nhầm
    # là lệch shape.
    experiment_name = "balance_car_nav_pretrained"
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.5,
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
class BalanceCarNavigationPPORunnerCfg(BalanceCarNavigationPretrainedPPORunnerCfg):
    """Học từ đầu: một mạng vừa cân bằng vừa chạy tới đích, xuất thẳng mô-men bánh.

    Mạng to hơn và train lâu hơn bản cascade vì nó phải học lại toàn bộ phần cân bằng.
    """

    experiment_name = "balance_car_nav_scratch"
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        actor_hidden_dims=[128, 128, 128],
        critic_hidden_dims=[128, 128, 128],
        activation="elu",
    )

    def __post_init__(self) -> None:
        self.max_iterations = 3000
