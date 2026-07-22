#!/usr/bin/env bash
# slurm/submit_all.sh -- chain prune -> distill -> evaluate with SLURM afterok
# dependencies for one variant/task. CKA profiling is skipped (data shipped).
# Usage:  source slurm/env.sh && bash slurm/submit_all.sh [keep_ratio] [task] [metric] [expected]
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
KEEP="${1:-0.20}"; TASK="${2:-docvqa}"; METRIC="${3:-anls}"; EXPECTED="${4:-82.13}"

j1=$(sbatch --parsable "$HERE/prune.sbatch" "$KEEP")
echo "prune    -> job $j1"
j2=$(sbatch --parsable --dependency=afterok:$j1 "$HERE/distill.sbatch" "$KEEP" "$TASK")
echo "distill  -> job $j2 (afterok:$j1)"
CKPT="$LITEDOC_CKPT/litedoc-$KEEP-distilled-$TASK"
j3=$(sbatch --parsable --dependency=afterok:$j2 "$HERE/evaluate.sbatch" "$CKPT" "$TASK" "$METRIC" "$EXPECTED")
echo "evaluate -> job $j3 (afterok:$j2)"
echo "Submitted chain for LiteDoc keep=$KEEP task=$TASK."
