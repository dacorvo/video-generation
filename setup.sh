#!/usr/bin/env bash
# Creates WORK_DIR, the working directory of the demo, with a venv in
# WORK_DIR/.venv on the Python of the AWS Neuron image, and installs
# requirements.txt in it, without replacing any of the image's packages. Then
# downloads the weights to WORK_DIR/models, and writes the path of WORK_DIR to
# .env.
#
# .venv sees the image's packages (--system-site-packages). Constraining pip to
# their installed versions makes it add packages without ever replacing one of
# the image's: a dependency that needs another torch fails instead of
# shadowing the image's Neuron build. uv pip ignores system site-packages, so
# it must never touch this venv.
#
# Usage: ./setup.sh WORK_DIR
set -euo pipefail

if (( $# != 1 )); then
    echo "usage: ./setup.sh WORK_DIR" >&2
    exit 1
fi
WORK_DIR=$1
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SYSTEM_PYTHON=/usr/local/bin/python3
export PIP_DISABLE_PIP_VERSION_CHECK=1

# On a fresh pod, only root can create a top-level directory.
if [[ ! -d "$WORK_DIR" ]] && ! mkdir -p "$WORK_DIR" 2>/dev/null; then
    sudo mkdir -p "$WORK_DIR"
    sudo chown "$(id -u):$(id -g)" "$WORK_DIR"
fi
WORK_DIR="$(cd "$WORK_DIR" && pwd)"
constraints="$(mktemp)"
trap 'rm -f "$constraints"' EXIT
"$SYSTEM_PYTHON" -m pip list --format=freeze > "$constraints"
if [[ ! -x "$WORK_DIR/.venv/bin/python" ]]; then
    "$SYSTEM_PYTHON" -m venv --system-site-packages "$WORK_DIR/.venv"
fi
"$WORK_DIR/.venv/bin/python" -m pip install -c "$constraints" -r "$REPO_DIR/requirements.txt"
# The excludes skip what the pipeline does not load (82 GB are left).
hf download Lightricks/LTX-2.5-Diffusers --local-dir "$WORK_DIR/models/LTX-2.5-Diffusers" \
    --exclude "transformer_full/*" \
    --exclude "ltx-2.5-22b-distilled-lora-450-bf16.safetensors" \
    --exclude "transformer/*-of-00008.safetensors" \
    --exclude "connectors/diffusion_pytorch_model.safetensors"
echo "WORK_DIR=$WORK_DIR" > "$REPO_DIR/.env"
