"""Trích scalar TensorBoard của một run rsl_rl, lọc tới max_step, xuất CSV + hình.

Dùng để viết báo cáo với kết quả tới N iteration (vd 2000) dù run dài hơn —
KHÔNG sửa log gốc, chỉ đọc file events.* và ghi ra thư mục con report_<N>/.

Chạy:
    python source/isaaclab_assets/isaaclab_assets/legged_v5/tools/report_curves.py \
        --logdir logs/rsl_rl/legged_v5_wheel_mimic/2026-06-15_19-24-41 \
        --max_step 2000
"""
import argparse
import csv
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator as ea


def load_scalars(logdir):
    files = sorted(glob.glob(os.path.join(logdir, "events.out.tfevents.*")))
    if not files:
        raise FileNotFoundError(f"Không thấy file events.* trong {logdir}")
    acc = ea.EventAccumulator(files[0], size_guidance={ea.SCALARS: 0})
    acc.Reload()
    data = {}
    for tag in acc.Tags()["scalars"]:
        data[tag] = [(s.step, s.value) for s in acc.Scalars(tag)]
    return data


def clip(series, max_step):
    return [(st, v) for st, v in series if st <= max_step]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logdir", required=True)
    ap.add_argument("--max_step", type=int, default=2000)
    args = ap.parse_args()

    data = load_scalars(args.logdir)
    outdir = os.path.join(args.logdir, f"report_{args.max_step}")
    os.makedirs(outdir, exist_ok=True)

    # ── 1) CSV rộng: mỗi tag một cột, index theo step (đã lọc) ────────────────
    steps = sorted({st for tag in data for st, _ in clip(data[tag], args.max_step)})
    lut = {tag: {st: v for st, v in clip(s, args.max_step)} for tag, s in data.items()}
    tags = sorted(data.keys())
    csv_path = os.path.join(outdir, "scalars.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["step"] + tags)
        for st in steps:
            w.writerow([st] + [lut[t].get(st, "") for t in tags])

    # ── 2) Giá trị tại iter cuối (≤ max_step) → bảng kết quả báo cáo ──────────
    print(f"\n=== Giá trị tại iter ≈ {args.max_step} (logdir lọc ≤ {args.max_step}) ===")
    summary_path = os.path.join(outdir, "summary_at_max.txt")
    with open(summary_path, "w") as f:
        for t in tags:
            s = clip(data[t], args.max_step)
            if not s:
                continue
            line = f"{t:45s} = {s[-1][1]:.4f}   (step {s[-1][0]})"
            print(line)
            f.write(line + "\n")

    # ── 3) Hình gộp ───────────────────────────────────────────────────────────
    def plot_group(fname, title, tags_sel):
        plt.figure(figsize=(8, 5))
        for t in tags_sel:
            if t not in data:
                continue
            s = clip(data[t], args.max_step)
            if not s:
                continue
            xs, ys = zip(*s)
            plt.plot(xs, ys, label=t.split("/")[-1], linewidth=1.2)
        plt.xlabel("Iteration"); plt.title(title); plt.legend(fontsize=8)
        plt.grid(alpha=0.3); plt.tight_layout()
        plt.savefig(os.path.join(outdir, fname), dpi=150)
        plt.close()

    plot_group("fig_progress.png", "Reward & Episode length",
               ["Train/mean_reward", "Train/mean_episode_length"])
    plot_group("fig_terminations.png", "Episode terminations",
               [t for t in tags if t.startswith("Episode_Termination/")])
    plot_group("fig_tracking.png", "Tracking errors",
               ["Metrics/velocity_command/error_vel_xy",
                "Metrics/velocity_command/error_vel_yaw",
                "Metrics/height_command/position_error"])
    plot_group("fig_reward_terms.png", "Reward terms",
               [t for t in tags if t.startswith("Episode_Reward/")])
    plot_group("fig_losses.png", "PPO losses",
               ["Loss/value_function", "Loss/surrogate", "Loss/entropy"])

    print(f"\nĐã ghi: {csv_path}\n        {summary_path}\n        5 hình PNG trong {outdir}")


if __name__ == "__main__":
    main()
