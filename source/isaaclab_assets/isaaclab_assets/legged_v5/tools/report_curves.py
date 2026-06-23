"""Trích scalar TensorBoard của một run rsl_rl, lọc tới max_step, xuất CSV + hình.

Dùng để viết báo cáo với kết quả tới N iteration (vd 2000) dù run dài hơn —
KHÔNG sửa log gốc, chỉ đọc file events.* và ghi ra thư mục con report_<N>/.

Chạy:
    python source/isaaclab_assets/isaaclab_assets/legged_v5/tools/report_curves.py \
        --logdir logs/rsl_rl/legged_v5_wheel_mimic/2026-06-18_05-53-51 \
        --max_step 600
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

    # ── 3) Hình: MỖI metric MỘT biểu đồ riêng ─────────────────────────────────
    figdir = os.path.join(outdir, "figs")
    os.makedirs(figdir, exist_ok=True)

    def safe_name(tag):
        # "Episode_Reward/track_lin_vel_y_exp" -> "Episode_Reward__track_lin_vel_y_exp"
        return tag.replace("/", "__").replace(" ", "_")

    n = 0
    for t in tags:
        s = clip(data[t], args.max_step)
        if not s:
            continue
        xs, ys = zip(*s)
        plt.figure(figsize=(7, 4))
        plt.plot(xs, ys, linewidth=1.3, color="#1f77b4")
        plt.xlabel("Iteration")
        plt.ylabel(t.split("/")[-1])
        plt.title(t)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(figdir, safe_name(t) + ".png"), dpi=150)
        plt.close()
        n += 1

    print(f"\nĐã ghi: {csv_path}\n        {summary_path}\n        {n} hình PNG (mỗi metric 1 file) trong {figdir}")


if __name__ == "__main__":
    main()
