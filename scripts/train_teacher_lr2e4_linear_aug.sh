#!/usr/bin/env bash
set -u -o pipefail
cd "$(dirname "$0")/.." || exit 1
run_dir="experiments/teacher_lr2e4_linear_aug_50k"
mkdir -p "$run_dir"
printf 'started_at_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$run_dir/console.log"
resume_args=()
if [[ -f "$run_dir/latest.pt" ]]; then
  resume_args=(--resume "$run_dir/latest.pt")
  printf 'resuming_from=%s\n' "$run_dir/latest.pt" | tee -a "$run_dir/console.log"
fi
.venv/bin/python -u -m teacher.train \
  --config configs/teacher/teacher_lr2e4_linear_aug_50k.json \
  "${resume_args[@]}" 2>&1 | tee -a "$run_dir/console.log"
train_rc=${PIPESTATUS[0]}
printf 'finished_at_utc=%s exit_code=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$train_rc" | tee -a "$run_dir/console.log"
printf '%s\n' "$train_rc" > "$run_dir/exit_code"
exit "$train_rc"
