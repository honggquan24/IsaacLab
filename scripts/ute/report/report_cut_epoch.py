# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tạo bản copy của một run rsl_rl đã CẮT tới max_iter.

Giữ nguyên bản gốc; ghi ra thư mục đích:
  - events.* : viết lại file TensorBoard chỉ chứa scalar có step <= max_iter
               (giữ đúng step + wall_time để đồ thị "/time" vẫn đúng).
  - model_*.pt : copy các checkpoint có số iter <= max_iter.
  - params/, git/ : copy nguyên (cấu hình + diff).
  - videos/ : copy video có env-step <= max_iter * num_steps_per_env.

Chạy:
    python scripts/ute/report/report_cut_epoch.py \
        --src logs/rsl_rl/legged_v5_wheel_mimic/2026-06-15_19-24-41 \
        --dst source/isaaclab_assets/isaaclab_assets/wheeled_biped/logs/legged_v5_wheel_mimic_2000ep \
        --max_iter 2000
"""

import argparse
import glob
import os
import re
import shutil

import yaml
from tensorboard.backend.event_processing import event_accumulator as ea
from torch.utils.tensorboard import SummaryWriter


def truncate_events(src, dst, max_iter):
    files = sorted(glob.glob(os.path.join(src, "events.out.tfevents.*")))
    if not files:
        print("  [events] không thấy file events.* — bỏ qua")
        return
    acc = ea.EventAccumulator(files[0], size_guidance={ea.SCALARS: 0})
    acc.Reload()
    w = SummaryWriter(log_dir=dst)
    n = 0
    for tag in acc.Tags()["scalars"]:
        for s in acc.Scalars(tag):
            if s.step <= max_iter:
                w.add_scalar(tag, s.value, global_step=s.step, walltime=s.wall_time)
                n += 1
    w.flush()
    w.close()
    print(f"  [events] ghi {n} điểm scalar (step <= {max_iter})")


def copy_checkpoints(src, dst, max_iter):
    n = 0
    for p in glob.glob(os.path.join(src, "model_*.pt")):
        m = re.match(r"model_(\d+)\.pt$", os.path.basename(p))
        if m and int(m.group(1)) <= max_iter:
            shutil.copy2(p, os.path.join(dst, os.path.basename(p)))
            n += 1
    print(f"  [ckpt] copy {n} checkpoint (iter <= {max_iter})")


def copy_dirs(src, dst):
    for d in ("params", "git"):
        s = os.path.join(src, d)
        if os.path.isdir(s):
            shutil.copytree(s, os.path.join(dst, d), dirs_exist_ok=True)
            print(f"  [{d}] copy xong")


def copy_videos(src, dst, max_step):
    sdir = os.path.join(src, "videos")
    if not os.path.isdir(sdir):
        return
    n = kept = 0
    for p in glob.glob(os.path.join(sdir, "**", "*.mp4"), recursive=True):
        n += 1
        m = re.search(r"rl-video-step-(\d+)", os.path.basename(p))
        if m and int(m.group(1)) > max_step:
            continue
        rel = os.path.relpath(p, sdir)
        out = os.path.join(dst, "videos", rel)
        os.makedirs(os.path.dirname(out), exist_ok=True)
        shutil.copy2(p, out)
        kept += 1
    print(f"  [videos] copy {kept}/{n} video (env-step <= {max_step})")


def get_steps_per_env(src):
    y = os.path.join(src, "params", "agent.yaml")
    if os.path.isfile(y):
        with open(y) as f:
            cfg = yaml.safe_load(f)
        for k in ("num_steps_per_env",):
            if isinstance(cfg, dict) and k in cfg:
                return int(cfg[k])
    return 24  # mặc định rsl_rl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--max_iter", type=int, default=2000)
    args = ap.parse_args()

    os.makedirs(args.dst, exist_ok=True)
    spe = get_steps_per_env(args.src)
    max_step = args.max_iter * spe
    print(f"num_steps_per_env={spe} -> ngưỡng env-step video = {max_step}")
    print(f"CẮT tới iter {args.max_iter}: {args.src} -> {args.dst}")

    truncate_events(args.src, args.dst, args.max_iter)
    copy_checkpoints(args.src, args.dst, args.max_iter)
    copy_dirs(args.src, args.dst)
    copy_videos(args.src, args.dst, max_step)
    print("Xong.")


if __name__ == "__main__":
    main()
