# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tìm epoch (checkpoint) tốt nhất từ log 2000ep — KHÔNG mặc định iter 2000.

Dùng mean filter (trung bình trượt) làm đường XU HƯỚNG để chọn ổn định, rồi xếp
hạng các checkpoint đã lưu (mỗi 100 iter) theo điểm tổng hợp:
  - reward (xu hướng)         : càng cao càng tốt
  - episode_length (xu hướng) : càng cao càng tốt (sống lâu)
  - time_out                  : càng cao càng tốt (chạy hết 60s)
  - bad_orientation           : càng THẤP càng tốt (ít lật)
"""

import glob

import numpy as np
from tensorboard.backend.event_processing import event_accumulator as ea

LOG = "source/isaaclab_assets/isaaclab_assets/wheeled_biped/logs/2026-06-18_02-19-07"
MAXSTEP = 3500
WIN = 11  # cửa sổ trung bình trượt (số iter)

acc = ea.EventAccumulator(sorted(glob.glob(f"{LOG}/events.out.tfevents.*"))[0], size_guidance={ea.SCALARS: 0})
acc.Reload()


def arr(tag):
    d = {s.step: s.value for s in acc.Scalars(tag) if s.step <= MAXSTEP}
    steps = np.array(sorted(d))
    return steps, np.array([d[s] for s in steps])


def smooth(y, w=WIN):
    """Trung bình trượt trung tâm (mean filter)."""
    if len(y) < w:
        return y
    k = np.ones(w) / w
    return np.convolve(y, k, mode="same")


steps, reward = arr("Train/mean_reward")
_, eplen = arr("Train/mean_episode_length")
_, timeout = arr("Episode_Termination/time_out")
_, badori = arr("Episode_Termination/bad_orientation")

reward_s = smooth(reward)
eplen_s = smooth(eplen)
timeout_s = smooth(timeout)
badori_s = smooth(badori)


def at(step, steps, y):
    i = int(np.argmin(np.abs(steps - step)))
    return y[i]


# Checkpoint đã lưu: 0,100,...,2000
ckpts = list(range(0, MAXSTEP + 1, 100))


def norm(x, lo, hi):
    return np.clip((x - lo) / (hi - lo + 1e-9), 0, 1)


# Biên để chuẩn hoá (theo xu hướng)
r_lo, r_hi = reward_s.min(), reward_s.max()
e_lo, e_hi = eplen_s.min(), eplen_s.max()

rows = []
for c in ckpts:
    r = at(c, steps, reward_s)
    e = at(c, steps, eplen_s)
    t = at(c, steps, timeout_s)
    b = at(c, steps, badori_s)
    # điểm tổng hợp: reward 0.4, eplen 0.3, timeout 0.15, (1-bad) 0.15
    score = 0.40 * norm(r, r_lo, r_hi) + 0.30 * norm(e, e_lo, e_hi) + 0.15 * t + 0.15 * (1 - b)
    rows.append((c, r, e, t, b, score))

rows_sorted = sorted(rows, key=lambda x: -x[5])

print(f"Mean filter window = {WIN} iter\n")
print(f"{'ckpt':>5} {'reward~':>9} {'eplen~':>9} {'timeout':>8} {'bad_ori':>8} {'SCORE':>7}")
for c, r, e, t, b, s in rows:
    star = "  <<< BEST" if c == rows_sorted[0][0] else ""
    print(f"{c:>5} {r:9.1f} {e:9.0f} {t:8.3f} {b:8.3f} {s:7.3f}{star}")

print("\n=== TOP 5 checkpoint (theo điểm tổng hợp, đường xu hướng) ===")
for c, r, e, t, b, s in rows_sorted[:5]:
    print(
        f"  iter {c:4d}  | score {s:.3f} | reward~{r:6.1f} | eplen~{e:5.0f}"
        f" ({100 * e / 6000:.0f}%) | timeout {t:.0%} | lat {b:.0%}"
    )

best = rows_sorted[0][0]
print(f"\n>>> ĐỀ XUẤT: model_{best}.pt  (file: {LOG}/model_{best}.pt)")

# Đỉnh thô (không lọc) để đối chiếu
ipk = int(np.argmax(reward))
print(
    f"    (đỉnh reward THÔ tại iter {steps[ipk]}: reward={reward[ipk]:.1f}, "
    f"eplen={eplen[ipk]:.0f}) — dễ là đỉnh nhiễu nên ưu tiên đường xu hướng)"
)
