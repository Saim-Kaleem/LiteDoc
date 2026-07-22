#!/usr/bin/env bash
# slurm/setup_env.sh -- one-time Conda environment creation (LOGIN NODE).
# Usage:  source slurm/env.sh && bash slurm/setup_env.sh
set -euo pipefail

ENV_NAME="${LITEDOC_ENV:-litedoc}"

module purge
module load miniforge3/latest
# shellcheck disable=SC1091
source "$ANACONDA_HOME/etc/profile.d/conda.sh"

if ! conda env list | grep -qE "^\s*${ENV_NAME}\s"; then
    conda create -y -n "$ENV_NAME" python=3.11
fi
conda activate "$ENV_NAME"

# torch comes from PyPI CUDA wheels -> no CUDA module needed (the V100 node's
# driver provides libcuda). Pinned to 2.2.x so it stays off the Blackwell dgx
# nodes (those need CUDA 12.8+/torch 2.7+).
pip install --upgrade pip
pip install -r "${LITEDOC_REPO:-.}/requirements-hpc.txt"

echo "[setup] Conda env '$ENV_NAME' ready. Next: python slurm/prefetch_models.py"
