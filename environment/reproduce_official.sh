#!/usr/bin/env bash
# Re-run frozen reports, never overwrite previous evaluation directories.
set -euo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"
run_name="${1:-reproduction}"
case "$run_name" in
  *[!a-zA-Z0-9_-]*|'') echo 'Use letters, digits, underscore or dash for run name.' >&2; exit 1 ;;
esac
for mode in one_nfe few_nfe; do
  run_dir="results/${run_name}_${mode}"
  if [[ -e "$run_dir" ]]; then
    echo "Existing output preserved: $run_dir; select a new run name." >&2
    exit 1
  fi
done
if [[ -f environment/inception-2015-12-05.pt && ! -f /tmp/inception-2015-12-05.pt ]]; then
  cp environment/inception-2015-12-05.pt /tmp/inception-2015-12-05.pt
fi
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
for mode in one_nfe few_nfe; do
  .venv/bin/python evaluate.py \
    --model_checkpoint "checkpoints/${mode}.ckpt" \
    --evaluate_mode "$mode" --reference_dir results/reference \
    --output_dir "results/${run_name}_${mode}" --batch_size 16 --num_workers 4 \
    2>&1 | tee "environment/${run_name}_${mode}.log"
done
