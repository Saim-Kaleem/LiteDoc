#!/usr/bin/env bash
#SBATCH --job-name=litedoc-distill
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=%x-%j.out
# Stage 2: Sparse-to-Sparse knowledge distillation.
# Args:  $1 = keep_ratio (default 0.20)   $2 = task/dataset name (default docvqa)
set -euo pipefail
module purge && module load miniforge3/latest
source "$ANACONDA_HOME/etc/profile.d/conda.sh" && conda activate "${LITEDOC_ENV:-litedoc}"
source "$(dirname "$0")/env.sh"
export TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1

KEEP="${1:-0.20}"; TASK="${2:-docvqa}"
python -m litedoc.distill \
    --teacher_path "$LITEDOC_MODEL" \
    --student_path "$LITEDOC_CKPT/litedoc-pruned-$KEEP" \
    --train_jsonl  "$LITEDOC_DATA/$TASK/train.jsonl" \
    --image_root   "$LITEDOC_DATA/$TASK/images" \
    --output_dir   "$LITEDOC_CKPT/litedoc-$KEEP-distilled-$TASK" \
    --epochs 1 --batch_size 1 --grad_accum 8 --lr 1e-5 --alpha 1.0 --beta 1.0
