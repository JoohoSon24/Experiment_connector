#!/usr/bin/env bash
set -euo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"
session=teacher_variant_sequence
if tmux has-session -t "$session" 2>/dev/null; then
  echo 'Existing sequence preserved'
  exit 0
fi
mkdir -p experiments/teacher_variant_sequence
printf -v command 'cd %q && exec %q -u %q >> %q 2>&1' "$project_root" "$project_root/.venv/bin/python" "$project_root/scripts/run_teacher_variant_sequence.py" "$project_root/experiments/teacher_variant_sequence/orchestration.log"
tmux new-session -d -s "$session" "$command"
echo "Started $session"
