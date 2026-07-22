#!/usr/bin/env bash
# scripts/run_all.sh -- local (non-cluster) runner for small tests / smoke checks.
# Paths default to ./data, ./checkpoints, ./results via litedoc/config.py when the
# LITEDOC_* env vars are unset. A real prune/distill/eval run needs a GPU + the
# teacher weights + benchmark JSONL; set MODEL, TASK and DATA to enable them.
#
#   bash scripts/run_all.sh              # smoke test only (no weights/GPU needed)
#   MODEL=deepseek-ai/deepseek-vl2-tiny KEEP=0.20 TASK=docvqa \
#       RUN_PIPELINE=1 bash scripts/run_all.sh
set -euo pipefail
cd "$(dirname "$0")/.."

echo "== smoke test =="
python -m tests.test_model_smoke

if [ "${RUN_PIPELINE:-0}" = "1" ]; then
  KEEP="${KEEP:-0.20}"; TASK="${TASK:-docvqa}"; METRIC="${METRIC:-anls}"
  MODEL="${MODEL:-deepseek-ai/deepseek-vl2-tiny}"
  echo "== prune (keep=$KEEP) =="
  python -m litedoc.prune --model_path "$MODEL" --keep_ratio "$KEEP" \
      --output_path "checkpoints/litedoc-pruned-$KEEP"
  echo "== distill ($TASK) =="
  python -m litedoc.distill --teacher_path "$MODEL" \
      --student_path "checkpoints/litedoc-pruned-$KEEP" \
      --train_jsonl "data/$TASK/train.jsonl" --image_root "data/$TASK/images" \
      --output_dir "checkpoints/litedoc-$KEEP-distilled-$TASK"
  echo "== evaluate ($METRIC) =="
  python -m litedoc.evaluate --model_path "checkpoints/litedoc-$KEEP-distilled-$TASK" \
      --eval_jsonl "data/$TASK/val.jsonl" --image_root "data/$TASK/images" \
      --metric "$METRIC" --output_json "results/${TASK}.json"
fi
echo "run_all.sh done."
