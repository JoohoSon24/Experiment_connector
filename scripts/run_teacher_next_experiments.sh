#!/usr/bin/env bash
set -u -o pipefail
cd "$(dirname "$0")/.." || exit 1
while tmux has-session -t fastgen_b64_official_cfg2 2>/dev/null; do sleep 3; done
bash scripts/train_teacher_memorization64.sh
memo_rc=$?
if [[ "$memo_rc" != 0 ]]; then exit "$memo_rc"; fi
# Only the LR policy changes in the full-dataset comparison. The diagnostic
# small-subset run is a separate short experiment, not a reduced full run.
bash scripts/train_teacher_b64_constant.sh
exit $?
