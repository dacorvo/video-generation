#!/usr/bin/env bash
# Creates .venv on the Python of the AWS Neuron image and installs
# requirements.txt in it, without replacing any of the image's packages.
#
# .venv sees the image's packages (--system-site-packages). Constraining pip to
# their installed versions makes it add packages without ever replacing one of
# the image's: a dependency that needs another torch fails instead of
# shadowing the image's Neuron build. uv pip ignores system site-packages, so
# it must never touch this venv.
#
# Usage: ./setup.sh
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

SYSTEM_PYTHON=/usr/local/bin/python3
export PIP_DISABLE_PIP_VERSION_CHECK=1

constraints="$(mktemp)"
trap 'rm -f "$constraints"' EXIT
"$SYSTEM_PYTHON" -m pip list --format=freeze > "$constraints"
if [[ ! -x .venv/bin/python ]]; then
    "$SYSTEM_PYTHON" -m venv --system-site-packages .venv
fi
.venv/bin/python -m pip install -c "$constraints" -r requirements.txt
