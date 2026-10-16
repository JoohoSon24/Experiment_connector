#!/usr/bin/env bash
set -euo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"
session=teacher_4head150k_followup
if tmux has-session -t "$session" 2>/dev/null; then
 echo 'Existing follow-up preserved';exit 0
fi
mkdir -p experiments/teacher_4head150k_followup
printf -v command 'cd %q && exec %q -u %q >> %q 2>&1' "$project_root" "$project_root/.venv/bin/python" "$project_root/scripts/run_teacher_4head150k_followup.py" "$project_root/experiments/teacher_4head150k_followup/orchestration.log"
tmux new-session -d -s "$session" "$command"
echo "Started $session"
