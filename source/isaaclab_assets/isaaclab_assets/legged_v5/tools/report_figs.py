"""Xuất 6 hình kết quả RL (đúng tên file experiment_results.tex tham chiếu) từ log
2000ep, cắt step <= max_step, lưu vào figure/results/. In thống kê theo mốc iter
để viết phân tích.

    python source/isaaclab_assets/isaaclab_assets/legged_v5/tools/report_figs.py
"""
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator as ea

LOGDIR = "source/isaaclab_assets/isaaclab_assets/legged_v5/logs/legged_v5_wheel_mimic_2000ep"
OUTDIR = "source/isaaclab_assets/isaaclab_assets/legged_v5/NCKH/figure/results"
MAX_STEP = 2000

os.makedirs(OUTDIR, exist_ok=True)
f = sorted(glob.glob(os.path.join(LOGDIR, "events.out.tfevents.*")))[0]
acc = ea.EventAccumulator(f, size_guidance={ea.SCALARS: 0})
acc.Reload()


def series(tag):
    return [(s.step, s.value) for s in acc.Scalars(tag) if s.step <= MAX_STEP]


def plot(fname, title, ylabel, tags, labels=None, logy=False):
    plt.figure(figsize=(9, 5))
    labels = labels or [t.split("/")[-1] for t in tags]
    for t, lb in zip(tags, labels):
        if t not in acc.Tags()["scalars"]:
            continue
        xs, ys = zip(*series(t))
        plt.plot(xs, ys, label=lb, linewidth=1.3)
    if logy:
        plt.yscale("symlog")
    plt.xlabel("Iteration"); plt.ylabel(ylabel); plt.title(title)
    plt.legend(fontsize=8, ncol=2); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, fname), dpi=150)
    plt.close()
    print(f"  saved {fname}")


# 01 — tổng quan: reward + episode length (twin axis)
def plot_overview():
    fig, ax1 = plt.subplots(figsize=(9, 5))
    r = series("Train/mean_reward"); el = series("Train/mean_episode_length")
    ax1.plot(*zip(*r), color="tab:blue", linewidth=1.4, label="mean_reward")
    ax1.set_xlabel("Iteration"); ax1.set_ylabel("Mean reward", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue"); ax1.grid(alpha=0.3)
    ax2 = ax1.twinx()
    ax2.plot(*zip(*el), color="tab:red", linewidth=1.4, label="mean_episode_length")
    ax2.set_ylabel("Mean episode length (steps)", color="tab:red")
    ax2.tick_params(axis="y", labelcolor="tab:red")
    plt.title("Tong quan huan luyen: reward & episode length")
    fig.tight_layout(); fig.savefig(os.path.join(OUTDIR, "01_train_overview.png"), dpi=150)
    plt.close(); print("  saved 01_train_overview.png")


plot_overview()
plot("02_task_rewards.png", "Phan thuong nhiem vu", "Episode reward",
     ["Episode_Reward/track_height", "Episode_Reward/track_lin_vel_y_exp",
      "Episode_Reward/track_lin_vel_x_exp", "Episode_Reward/track_ang_vel_z_exp"])
plot("03_penalty_rewards.png", "Phan thuong phat", "Episode reward",
     ["Episode_Reward/lateral_tilt", "Episode_Reward/com_wheel_plane",
      "Episode_Reward/hip_symmetry", "Episode_Reward/stand_still",
      "Episode_Reward/action_rate", "Episode_Reward/fore_aft_tilt",
      "Episode_Reward/velocity_overshoot", "Episode_Reward/leg_joint_acc",
      "Episode_Reward/termination_penalty"], logy=True)
plot("04_tracking_metrics.png", "Chi so bam lenh", "Error",
     ["Metrics/velocity_command/error_vel_xy", "Metrics/velocity_command/error_vel_yaw",
      "Metrics/height_command/position_error", "Metrics/height_command/orientation_error"])
plot("05_terminations.png", "Ti le ket thuc episode", "Ti le",
     ["Episode_Termination/time_out", "Episode_Termination/bad_orientation",
      "Episode_Termination/base_height", "Episode_Termination/illegal_contact_hip",
      "Episode_Termination/illegal_contact_knee", "Episode_Termination/joint_vel_limit"])
plot("06_loss.png", "PPO losses", "Loss",
     ["Loss/value_function", "Loss/surrogate", "Loss/entropy"])

# ── Thống kê theo mốc iter ────────────────────────────────────────────────────
print("\n=== Giá trị theo mốc iter ===")
tags = ["Train/mean_reward", "Train/mean_episode_length",
        "Episode_Reward/track_height", "Episode_Reward/track_lin_vel_y_exp",
        "Episode_Reward/track_lin_vel_x_exp", "Episode_Reward/track_ang_vel_z_exp",
        "Episode_Termination/time_out", "Episode_Termination/bad_orientation",
        "Episode_Termination/base_height", "Episode_Termination/illegal_contact_hip",
        "Metrics/velocity_command/error_vel_xy", "Metrics/velocity_command/error_vel_yaw",
        "Loss/value_function", "Loss/surrogate", "Loss/entropy",
        "Policy/mean_noise_std"]
marks = [0, 200, 500, 800, 1100, 1400, 1700, 2000]
lut = {t: dict(series(t)) for t in tags}
hdr = "iter".ljust(34) + "".join(str(m).rjust(9) for m in marks)
print(hdr)
for t in tags:
    row = t.ljust(34)
    for m in marks:
        v = lut[t].get(m)
        row += (f"{v:9.3f}" if v is not None else " " * 9)
    print(row)
