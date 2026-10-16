#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
for run in teacher_b64_constant_100k teacher_b64_fliponly_constant_100k; do
  .venv/bin/python -u scripts/supplement_ddim_diagnostics.py --run-dir "experiments/$run"
done
for run in teacher_b64_constant_100k teacher_b64_fliponly_constant_100k; do
  .venv/bin/python -u scripts/review_teacher_run.py --run-dir "experiments/$run"
done
