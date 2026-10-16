#!/usr/bin/env bash
set -u -o pipefail
cd "$(dirname "$0")/.." || exit 1
run_dir="experiments/teacher_b128_100k"
mkdir -p "$run_dir"
printf 'started_at_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$run_dir/console.log"
resume_args=()
if [[ -f "$run_dir/latest.pt" ]]; then
  resume_args=(--resume "$run_dir/latest.pt")
  printf 'resuming_from=%s\n' "$run_dir/latest.pt" | tee -a "$run_dir/console.log"
fi
if [[ ! -f "$run_dir/final.pt" ]]; then
  .venv/bin/python -u -m teacher.train --config configs/teacher/teacher_b128_100k.json \
    "${resume_args[@]}" 2>&1 | tee -a "$run_dir/console.log"
  train_rc=${PIPESTATUS[0]}
  printf 'training_finished_at_utc=%s exit_code=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$train_rc" | tee -a "$run_dir/console.log"
  printf '%s\n' "$train_rc" > "$run_dir/exit_code"
  if [[ "$train_rc" != 0 ]]; then exit "$train_rc"; fi
fi
.venv/bin/python -u scripts/review_teacher_run.py --run-dir "$run_dir" \
  2>&1 | tee -a "$run_dir/development_review_console.log"
review_rc=${PIPESTATUS[0]}
printf '%s\n' "$review_rc" > "$run_dir/review_exit_code"
exit "$review_rc"
