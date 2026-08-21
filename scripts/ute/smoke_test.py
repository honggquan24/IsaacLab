# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Chạy thử nhanh MỘT task: tạo env, bước vài lần rồi thoát.

Dùng để kiểm tra task còn dựng được sau khi đổi cấu trúc/đổi tên, KHÔNG dùng để
đánh giá chất lượng policy. Mỗi lần chạy chỉ dựng một env vì Kit không dựng lại
được env thứ hai trong cùng một phiên.

Chạy:
    ./isaaclab.sh -p scripts/ute/smoke_test.py --task Isaac-Wheeled-Biped-Wheel
    bash scripts/ute/smoke_test_all.sh          # chạy lần lượt toàn bộ task
"""

import argparse
import contextlib

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", type=str, default=None, help="Task cần chạy thử.")
parser.add_argument("--list", action="store_true", help="Chỉ in danh sách task của dự án rồi thoát.")
parser.add_argument("--num_envs", type=int, default=2, help="Số môi trường song song.")
parser.add_argument("--steps", type=int, default=10, help="Số bước mô phỏng mỗi task.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import traceback  # noqa: E402

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import isaaclab_tasks  # noqa: F401, E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

import isaaclab_assets  # noqa: F401, E402  (đăng ký task của dự án)

PROJECT_TASKS = [
    "Isaac-Wheeled-Biped-Wheel",
    "Isaac-Wheeled-Biped-Wheel-Play",
    "Isaac-Wheeled-Biped-Wheel-NoMimic",
    "Isaac-Wheeled-Biped-Wheel-PIANN",
    "Isaac-Wheeled-Biped-Navigation",
    "Isaac-Wheeled-Biped-Warehouse-Nav",
    "Isaac-Wheeled-Biped-Obstacle-Nav",
    "Isaac-Rotary-Pendulum-Balance",
    "Isaac-Rotary-Pendulum-Balance-Stage1",
    "Isaac-Rotary-Pendulum-Balance-Stage2",
    "Isaac-Cart-Pendulum",
    "Isaac-Cart-Pendulum-Double",
    "Isaac-Balance-Car",
    "Isaac-Balance-Car-Navigation",
    "Isaac-Balance-Car-Navigation-Play",
    "Isaac-Balance-Car-Navigation-Pretrained",
    "Isaac-Balance-Car-Navigation-Pretrained-Play",
    "Isaac-Evobot-Balance",
    "Isaac-Evobot-Velocity",
    "Isaac-Evobot-Velocity-Play",
    "Isaac-Evobot-Navigation",
    "Isaac-Evobot-Navigation-Play",
    "Isaac-Evobot-Manipulation",
    "Isaac-Evobot-Arm-FineTune",
    "Isaac-Evobot-Gripper-FineTune",
]


def run_task(task: str, num_envs: int, steps: int) -> str:
    """Dựng ``task``, bước ``steps`` lần với hành động ngẫu nhiên, trả về dòng kết quả."""
    env = None
    try:
        env_cfg = parse_env_cfg(task, device=args_cli.device, num_envs=num_envs)
        env = gym.make(task, cfg=env_cfg)
        env.reset()
        action_space = env.unwrapped.action_space  # type: ignore[attr-defined]
        for _ in range(steps):
            actions = torch.zeros((num_envs, action_space.shape[1]), device=args_cli.device)
            env.step(actions)
        obs_dim = env.unwrapped.observation_manager.group_obs_dim  # type: ignore[attr-defined]
        return f"PASS  {task:<48} act={action_space.shape[1]:<3} obs={obs_dim}"
    except Exception as exc:  # noqa: BLE001 - báo cáo mọi lỗi dựng env
        traceback.print_exc()
        return f"FAIL  {task:<48} {type(exc).__name__}: {exc}"
    finally:
        if env is not None:
            # Env dựng dở (ví dụ thiếu file checkpoint) có thể treo khi đóng — bỏ qua lỗi ở đây.
            with contextlib.suppress(Exception):
                env.close()


def main() -> None:
    if args_cli.list:
        print("\n".join(PROJECT_TASKS))
        return
    if args_cli.task is None:
        raise SystemExit("Cần --task <ID> (hoặc --list để xem danh sách)")
    result = run_task(args_cli.task, args_cli.num_envs, args_cli.steps)
    print("\n" + "=" * 100)
    print("KẾT QUẢ SMOKE TEST: " + result)
    print("=" * 100 + "\n")


main()
simulation_app.close()
