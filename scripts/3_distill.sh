#!/usr/bin/env bash
# Stage 2: Sparse-to-Sparse knowledge distillation to recover accuracy.
set -euo pipefail
python -m litedoc.distill \
    --teacher_path "${TEACHER:-deepseek-ai/deepseek-vl2-tiny}" \
    --student_path "${STUDENT:-checkpoints/litedoc-pruned-0.20}" \
    --train_jsonl "${TRAIN:-data/docvqa/train.jsonl}" \
    --image_root "${IMG:-data/docvqa/images}" \
    --output_dir "${OUT:-checkpoints/litedoc-20-distilled}" \
    --epochs "${EPOCHS:-1}" --batch_size 1 --grad_accum "${ACCUM:-8}" --lr "${LR:-1e-5}"
