#!/usr/bin/env bash
# Stage 1b: prune redundant experts. Set KEEP=0.20/0.30/0.40/0.60 for the four variants.
set -euo pipefail
KEEP="${KEEP:-0.20}"
python -m litedoc.prune \
    --model_path "${MODEL:-deepseek-ai/deepseek-vl2-tiny}" \
    --cka_clusters_path "${CKA:-data/cka_clusters.json}" \
    --manual_pruning_path "${MANUAL:-data/manual_pruning.json}" \
    --keep_ratio "${KEEP}" \
    --output_path "${OUT:-checkpoints/litedoc-pruned-${KEEP}}"
