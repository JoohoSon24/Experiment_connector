#!/usr/bin/env bash
set -euo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"
session=teacher_lr_comparison_200k
if tmux has-session -t "$session" 2>/dev/null; then
  echo 'Existing experiment preserved'
  exit 0
fi
mkdir -p experiments/teacher_lr_comparison_200k
printf -v command 'cd %q && exec %q -u %q >> %q 2>&1' "$project_root" "$project_root/.venv/bin/python" "$project_root/scripts/run_teacher_lr_comparison.py" "$project_root/experiments/teacher_lr_comparison_200k/orchestration.log"
tmux new-session -d -s "$session" "$command"
echo "Started $session"
