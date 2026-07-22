#!/usr/bin/env bash
#SBATCH --job-name=litedoc-eval
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=%x-%j.out
# Evaluation + verification vs the paper.
# Args:  $1 = model checkpoint dir   $2 = task/dataset   $3 = metric (anls|f1|em)
#        $4 = expected score (optional, enables PASS/WARN record)
set -euo pipefail
module purge && module load miniforge3/latest
source "$ANACONDA_HOME/etc/profile.d/conda.sh" && conda activate "${LITEDOC_ENV:-litedoc}"
source "$(dirname "$0")/env.sh"
export TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1

MODEL="${1:?pass a checkpoint dir}"; TASK="${2:-docvqa}"; METRIC="${3:-anls}"
EXPECTED="${4:-}"
EXTRA=(); [ -n "$EXPECTED" ] && EXTRA=(--verify_name "eval_${TASK}" --expected "$EXPECTED")
python -m litedoc.evaluate \
    --model_path "$MODEL" \
    --eval_jsonl "$LITEDOC_DATA/$TASK/val.jsonl" \
    --image_root "$LITEDOC_DATA/$TASK/images" \
    --metric "$METRIC" --max_new_tokens 128 \
    --output_json "$LITEDOC_RESULTS/${TASK}_$(basename "$MODEL").json" \
    "${EXTRA[@]}"
