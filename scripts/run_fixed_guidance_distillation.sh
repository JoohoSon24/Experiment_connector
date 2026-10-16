#!/usr/bin/env bash
set -u -o pipefail
cd "$(dirname "$0")/.." || exit 1
mkdir -p experiments/fixed_cfg2_progressive
# Correct the adjacent-seed overlap after the current teacher training finishes.
.venv/bin/python -u scripts/compare_teacher_steps_seeds.py 2>&1 | tee -a experiments/teacher_disjoint_seeds_console.log
seed_rc=${PIPESTATUS[0]}
printf '%s\n' "$seed_rc" > experiments/teacher_disjoint_seeds_exit_code
if [[ "$seed_rc" != 0 ]]; then exit "$seed_rc"; fi
.venv/bin/python -u -m distillation.pipeline 2>&1 | tee -a experiments/fixed_cfg2_progressive/console.log
rc=${PIPESTATUS[0]}
printf '%s\n' "$rc" > experiments/fixed_cfg2_progressive/exit_code
exit "$rc"
