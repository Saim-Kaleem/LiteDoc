#!/usr/bin/env bash
# Evaluate a model on a benchmark. METRIC in {anls,f1,em}.
set -euo pipefail
python -m litedoc.evaluate \
    --model_path "${MODEL:?set MODEL to a checkpoint}" \
    --eval_jsonl "${EVAL:-data/docvqa/val.jsonl}" \
    --image_root "${IMG:-data/docvqa/images}" \
    --metric "${METRIC:-anls}" \
    --max_new_tokens "${MAXTOK:-128}" \
    --output_json "${OUT:-results/eval.json}"
