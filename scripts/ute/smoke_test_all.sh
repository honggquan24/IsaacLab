#!/usr/bin/env bash
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Chạy smoke test lần lượt cho từng task của dự án, mỗi task một tiến trình riêng
# (Kit không dựng lại được env thứ hai trong cùng một phiên).
#
#   bash scripts/ute/smoke_test_all.sh [--num_envs 2] [--steps 5]

set -u
cd "$(dirname "${BASH_SOURCE[0]}")/../.." || exit 1

TIMEOUT_S=${SMOKE_TIMEOUT_S:-420}
TASKS=$(grep -oE '"Isaac-[A-Za-z0-9-]+"' scripts/ute/smoke_test.py | tr -d '"')
SUMMARY=""

for task in ${TASKS}; do
    line=$(PYTHONUNBUFFERED=1 timeout "${TIMEOUT_S}" ./isaaclab.sh -p scripts/ute/smoke_test.py --task "${task}" "$@" 2>&1 \
        | grep -m1 'KẾT QUẢ SMOKE TEST: ')
    line=${line#KẾT QUẢ SMOKE TEST: }
    [ -z "${line}" ] && line="FAIL  ${task}  (không dựng được env / quá ${TIMEOUT_S}s)"
    echo "${line}"
    SUMMARY="${SUMMARY}${line}"$'\n'
done

echo
echo "===================== TỔNG KẾT ====================="
printf '%s' "${SUMMARY}"
echo "===================================================="
echo "$(printf '%s' "${SUMMARY}" | grep -c '^PASS')/$(printf '%s' "${SUMMARY}" | grep -c .) task dựng được"
