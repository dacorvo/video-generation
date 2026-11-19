#!/usr/bin/env bash
# Runs python with ARGS in the environment of the demo: with the variables of
# .env, which setup.sh writes, in the venv of their WORK_DIR.
#
# Usage: ./run.sh ARGS...  (for example: ./run.sh generate.py --cpu)
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
set -a
source "$REPO_DIR/.env"
set +a
if [[ ! -f "$WORK_DIR/.venv/bin/activate" ]]; then
    echo "missing $WORK_DIR/.venv: ./setup.sh $WORK_DIR creates it" >&2
    exit 1
fi
source "$WORK_DIR/.venv/bin/activate"
exec python "$@"
