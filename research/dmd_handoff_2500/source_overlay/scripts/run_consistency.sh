#!/usr/bin/env bash
# Start INSIDE a remote GPU server's tmux session. No prior experiments rerun.
set -euo pipefail
cd "$(dirname "$0")/.."
fastgen_python="${FASTGEN_PYTHON:-.venv/bin/python}"
teacher_path="${1:-experiments/teacher_b64_noaug_constant_100k/final.pt}"
config_path="${2:-configs/consistency/fixed_cfg2.json}"
run_path="${3:-experiments/consistency_cfg2}"
if [[ $# -gt 3 ]]; then
  echo "Usage: bash scripts/run_consistency.sh [teacher.pt] [config.json] [run-directory]" >&2
  exit 2
fi
if [[ ! -x "$fastgen_python" || ! -f "$teacher_path" || ! -f "$config_path" ]]; then
  echo "Missing Python, teacher or config. Set FASTGEN_PYTHON to your environment's Python." >&2
  exit 2
fi
mkdir -p "$run_path"
# Each launch has a separate log; the Python runner locks the output directory.
launch_log="$run_path/console_$(date -u +%Y%m%dT%H%M%SZ)_$$.log"
set +e
"$fastgen_python" -u -m consistency.train --config "$config_path" \
  --teacher-checkpoint "$teacher_path" --output-dir "$run_path" 2>&1 | tee "$launch_log"
rc=${PIPESTATUS[0]}
set -e
printf '%s\n' "$rc" > "$launch_log.exit_code"
exit "$rc"
