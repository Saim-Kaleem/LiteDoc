#!/usr/bin/env bash
#SBATCH --job-name=litedoc-prune
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=%x-%j.out
# Stage 1b: expert pruning + router repair.
# Args:  $1 = keep_ratio (0.20|0.30|0.40|0.60, default 0.20)
set -euo pipefail
module purge && module load miniforge3/latest
source "$ANACONDA_HOME/etc/profile.d/conda.sh" && conda activate "${LITEDOC_ENV:-litedoc}"
source "$(dirname "$0")/env.sh"
export TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1

KEEP="${1:-0.20}"
python -m litedoc.prune \
    --model_path "$LITEDOC_MODEL" \
    --cka_clusters_path "$LITEDOC_CKA_CLUSTERS" \
    --manual_pruning_path "$LITEDOC_MANUAL_PRUNING" \
    --keep_ratio "$KEEP" \
    --output_path "$LITEDOC_CKPT/litedoc-pruned-$KEEP"
