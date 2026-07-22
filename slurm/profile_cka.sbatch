#!/usr/bin/env bash
#SBATCH --job-name=litedoc-cka
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=%x-%j.out
# Stage 1a: CKA profiling. Optional -- data/cka_clusters.json ships precomputed.
# Args (optional):  $1 = annotations_dir   $2 = images_root   $3 = tau (default 0.7)
set -euo pipefail
module purge && module load miniforge3/latest
source "$ANACONDA_HOME/etc/profile.d/conda.sh" && conda activate "${LITEDOC_ENV:-litedoc}"
source "$(dirname "$0")/env.sh"
export TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1

ANN="${1:-$LITEDOC_DATA/funsd/train/annotations}"
IMG="${2:-$LITEDOC_DATA/funsd/train/images}"
TAU="${3:-0.7}"
python -m litedoc.profile_cka \
    --model_path "$LITEDOC_MODEL" \
    --annotations_dir "$ANN" --images_root "$IMG" \
    --num_prompts_for_cka 200 --threshold "$TAU" \
    --output_dir "$LITEDOC_ARTIFACTS/cka"
