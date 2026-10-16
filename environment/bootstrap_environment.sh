#!/usr/bin/env bash
# Rebuild dependencies on another machine; keep existing working environments.
set -euo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"
python310="${DIFFUSION_PYTHON310:-python3.10}"
if [[ -e .venv ]]; then
  echo 'Existing .venv preserved. Activate it or choose a clean checkout.' >&2
  exit 1
fi
"$python310" -m venv .venv
.venv/bin/python -m pip install -r environment/requirements.lock.txt --extra-index-url https://download.pytorch.org/whl/cu124
.venv/bin/python -m pip check
echo 'Ready: source .venv/bin/activate'
