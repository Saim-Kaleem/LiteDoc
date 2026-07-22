#!/usr/bin/env bash
# Stage 1a: profile expert redundancy with CKA on a DU training split.
set -euo pipefail
python -m litedoc.profile_cka \
    --model_path "${MODEL:-deepseek-ai/deepseek-vl2-tiny}" \
    --annotations_dir "${ANN:-data/funsd/train/annotations}" \
    --images_root "${IMG:-data/funsd/train/images}" \
    --num_prompts_for_cka "${NPROMPT:-200}" \
    --threshold "${TAU:-0.7}" \
    --output_dir "${OUT:-artifacts/cka_tiny}"
