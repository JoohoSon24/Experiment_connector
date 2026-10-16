#!/usr/bin/env bash
set -euo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"
session=dmd_4head150k_two_arms
if tmux has-session -t "$session" 2>/dev/null; then
 echo 'Existing DMD run preserved';exit 0
fi
mkdir -p experiments/dmd_4head150k
printf -v command 'cd %q && OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 exec %q -u %q >> %q 2>&1' "$project_root" '/root/Diffusion/workspace_distillation/.venv/bin/python' "$project_root/scripts/run_dmd_selected_teacher.py" "$project_root/experiments/dmd_4head150k/orchestration.log"
tmux new-session -d -s "$session" "$command"
echo "Started $session"
