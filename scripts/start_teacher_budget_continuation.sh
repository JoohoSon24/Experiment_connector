#!/usr/bin/env bash
# Online run explicitly waits when the full baseline teacher is unavailable.
set -euo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"
session_name="fastgen_noaug_budget_200k"
checkpoint_path="${1:-$project_root/experiments/teacher_b64_noaug_constant_100k/final.pt}"
run_root="$project_root/experiments/teacher_budget_noaug_100k_200k"
mkdir -p "$run_root"
if tmux has-session -t "$session_name" 2>/dev/null; then
  echo "Existing session preserved: $session_name"
  exit 0
fi
printf -v run_command 'cd %q && exec %q -u %q --checkpoint %q --wait-for-checkpoint >> %q 2>&1' \
  "$project_root" "$project_root/.venv/bin/python" \
  "$project_root/scripts/run_teacher_budget_continuation.py" \
  "$checkpoint_path" "$run_root/orchestration.log"
tmux new-session -d -s "$session_name" "$run_command"
echo "Started: $session_name"
echo "Status: $run_root/status.json"
echo "Log: $run_root/orchestration.log"
