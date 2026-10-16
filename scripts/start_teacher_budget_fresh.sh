#!/usr/bin/env bash
set -euo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"
if tmux has-session -t fastgen_noaug_budget_200k 2>/dev/null; then
  tmux send-keys -t fastgen_noaug_budget_200k C-c
fi
session_name=fastgen_noaug_fresh_200k
if tmux has-session -t "$session_name" 2>/dev/null; then
  echo "Existing fresh training session preserved"
  exit 0
fi
run_root="$project_root/experiments/teacher_budget_noaug_fresh_200k"
mkdir -p "$run_root"
printf -v run_command 'cd %q && exec %q -u %q >> %q 2>&1' "$project_root" "$project_root/.venv/bin/python" "$project_root/scripts/run_teacher_budget_fresh.py" "$run_root/orchestration.log"
tmux new-session -d -s "$session_name" "$run_command"
echo "Started: $session_name"
