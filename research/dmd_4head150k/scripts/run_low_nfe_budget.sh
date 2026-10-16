#!/usr/bin/env bash
set -u -o pipefail
cd "$(dirname "$0")/.." || exit 1
mkdir -p experiments/low_nfe_budget_100k
printf 'RUNNING\n' > experiments/low_nfe_budget_100k/exit_code
.venv/bin/python -u -m scripts.run_low_nfe_budget 2>&1 | tee -a experiments/low_nfe_budget_100k/console.log
rc=${PIPESTATUS[0]}
printf '%s\n' "$rc" > experiments/low_nfe_budget_100k/exit_code
exit "$rc"
