#!/usr/bin/env bash
# Restart entry point: completed evaluations are reused and training resumes.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p experiments/teacher_sampling_steps_seeds
.venv/bin/python -u scripts/compare_teacher_steps_seeds.py 2>&1 | tee -a experiments/teacher_steps_seed_console.log
printf '0\n' > experiments/teacher_steps_seed_exit_code
bash scripts/train_teacher_b64_noaug_constant.sh
# This marker is written only after implementation and actual smoke validation.
if [[ ! -f experiments/distillation_implementation_ready ]]; then
  printf 'Distillation implementation has not passed its smoke gate.\n' >&2
  exit 1
fi
bash scripts/run_fixed_guidance_distillation.sh
