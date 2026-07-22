# Reproducing the LiteDoc results

This guide maps each number in the paper to a concrete command in this repo.
It assumes a single CUDA GPU (the paper used an RTX 6000 Ada; the pruned
LiteDoc-20 fits in 8 GB).

> **Honesty note.** The pruning, profiling, and evaluation code is the authors'
> own, cleaned up. The Stage-2 distillation *trainer* was not present in any
> source archive and is reconstructed from the paper's equations
> (`litedoc/distill.py`) — expect to validate/tune it. See README "Known gaps".

---

## 0. Environment

```bash
conda create -n litedoc python=3.10 -y && conda activate litedoc
pip install -r requirements.txt
# models load with trust_remote_code=True in bf16 on CUDA
```

Pins that matter: `transformers==4.38.2`, `torch` 2.2.x, `xformers`. Newer
`transformers` will break the vendored modeling code.

## 1. Get the teacher

```bash
huggingface-cli download deepseek-ai/deepseek-vl2-tiny --local-dir models/deepseek-vl2-tiny
```

Architecture recap (paper Sec. 4.1): 12 transformer blocks, 64 routed + 2 shared
experts each, top-6 routing, hidden dim 768. Pruning operates on the first 11
MoE blocks; block 12 is reserved for generative decoding.

## 2. Prepare benchmark data

Convert each benchmark's split to a JSONL where every line is:

```json
{"image": "doc_0001.png", "prompt": "<task-specific instruction>", "answers": ["gold", "..."]}
```

Task/metric/prompt conventions (paper Table 2), and the metric flag to pass to
`litedoc.evaluate`:

| Benchmark | Task | Split size | Metric (`--metric`) | Prompt style |
|-----------|------|-----------|---------------------|--------------|
| CORD      | Receipt KIE | 100  | `f1`   | JSON schema (see below) |
| SROIE     | Receipt KIE | 347  | `f1`   | "Extract the company name, date, address, and total amount from this receipt." |
| FUNSD     | Form KIE    | 50   | `f1`   | `<image>\n<|grounding|>Extract all form fields and their labels from this document` |
| DocVQA    | Doc VQA     | 5,188| `anls` | `<image>\n` + question |
| InfoVQA   | Infographic VQA | ~5k | `anls` | `<image>\n` + question |
| WTQ       | Table QA    | 4,344| `em`   | table rendered to an image (Selenium; Arial, alternating rows) |
| RVL-CDIP  | Classification | 40,000 | `em` | enumerated 16-class prompt (see below) |

CORD JSON-schema prompt:

```
Extract receipt fields as JSON with keys:
{"menu": [{"nm":"...","cnt":"...","unitprice":"...","price":"..."}],
 "sub_total": {"subtotal_price":"...","tax_price":"..."},
 "total": {"total_price":"...","cashprice":"...","changeprice":"..."}}
```

RVL-CDIP prompt (use the first page of each TIFF):

```
Classify this document into exactly one category from: letter, form, email,
handwritten, advertisement, scientific report, scientific publication,
specification, file folder, news article, budget, invoice, presentation,
questionnaire, resume, memo. Respond with only the category name.
```

**No test leakage:** use *training* splits for CKA profiling and distillation;
reserve test splits for evaluation only.

## 3. Baseline — teacher numbers (Table 3, first column)

Run each benchmark against the unpruned teacher, e.g. DocVQA:

```bash
python -m litedoc.evaluate --model_path models/deepseek-vl2-tiny \
    --eval_jsonl data/docvqa/val.jsonl --image_root data/docvqa/images \
    --metric anls --max_new_tokens 128 --output_json results/teacher_docvqa.json
```

Expected ≈ DocVQA 88.76, CORD 96.76, … (full column in README).

## 4. Stage 1a — CKA profiling (optional; data shipped)

`data/cka_clusters.json` is provided (τ=0.7). To regenerate or sweep τ:

```bash
python -m litedoc.profile_cka \
    --model_path models/deepseek-vl2-tiny \
    --annotations_dir data/funsd/train/annotations \
    --images_root data/funsd/train/images \
    --num_prompts_for_cka 200 --threshold 0.7 \
    --output_dir artifacts/cka_tiny
# -> artifacts/cka_tiny/cka_clusters.json, cka_matrices.pt, cka_heatmaps/
```

τ=0.7 is moderate; τ=0.5 is aggressive (larger cliques → more pruning). The
heatmaps reproduce paper Fig. 3; the layer-wise retention curve (Fig. 7) can be
read off the resulting per-layer prune counts.

## 5. Stage 1b — pruning (Table 3, "Pruning only" columns)

Produce all four variants by varying `keep_ratio`:

```bash
for KEEP in 0.60 0.40 0.30 0.20; do
  python -m litedoc.prune --model_path models/deepseek-vl2-tiny \
      --cka_clusters_path data/cka_clusters.json \
      --manual_pruning_path data/manual_pruning.json \
      --keep_ratio $KEEP --output_path checkpoints/litedoc-pruned-$KEEP
done
```

Evaluate a pruned (undistilled) checkpoint to get the "Pruning only" numbers,
e.g. LiteDoc-20 @80% on DocVQA should land near 51.14 ANLS.

## 6. Stage 2 — distillation (Table 3, "Pruning + KD" columns)

Distill each pruned student against the frozen teacher on that benchmark's
training split:

```bash
python -m litedoc.distill \
    --teacher_path models/deepseek-vl2-tiny \
    --student_path checkpoints/litedoc-pruned-0.20 \
    --train_jsonl data/docvqa/train.jsonl --image_root data/docvqa/images \
    --output_dir checkpoints/litedoc-20-distilled \
    --epochs 1 --batch_size 1 --grad_accum 8 --lr 1e-5 --alpha 1.0 --beta 1.0
```

`alpha` weights `L_KL`, `beta` weights `L_hidden`. Ablation (paper, DocVQA):
KL-only 81.7, hidden-only 82.9, both 84.2 — so keep both on. After distillation,
LiteDoc-20 on DocVQA should recover to ≈ 82.13 ANLS.

## 7. Final evaluation (Table 3, "Pruning + KD")

```bash
# DocVQA / InfoVQA -> anls ; CORD / SROIE / FUNSD -> f1 ; WTQ / RVL-CDIP -> em
for M in litedoc-60 litedoc-40 litedoc-30 litedoc-20; do
  python -m litedoc.evaluate --model_path checkpoints/$M-distilled \
     --eval_jsonl data/docvqa/val.jsonl --image_root data/docvqa/images \
     --metric anls --output_json results/${M}_docvqa.json
done
```

The distilled LiteDoc-20 should average ≈ **76.70** across the seven benchmarks
(≈ 89.6% of the teacher's average).

## 8. Efficiency numbers (Sec. 5.2 / 5.4)

Measure peak VRAM and throughput on your GPU while running `litedoc.evaluate`
(e.g. wrap generation with `torch.cuda.max_memory_allocated()` and time the
tokens/s). Paper reference on RTX 6000 Ada: teacher 12.3 GB / 47.2 tok/s →
LiteDoc-20 7.4 GB / 71.6 tok/s (−39.8% VRAM, +51.7% throughput).

## 9. Sanity test (no GPU/weights needed)

```bash
python -m tests.test_model_smoke
```

Builds a tiny `DeepseekV2MoE` from a mock config and checks a synthetic
forward/backward, the `valid_expert_mask` router and `cka_tracking` hook, the
prune-and-repair surgery, CKA invariances, and the ANLS/F1/EM metrics. Exit 0 =
all passed.

---

### Troubleshooting

- **`ignore_mismatched_sizes` / gate shape errors when loading a pruned model** —
  expected; the router is resized during pruning. `litedoc.prune` already repairs
  `gate.weight` and `valid_expert_mask`.
- **OOM during distillation** — keep `batch_size 1`, raise `grad_accum`, and use
  the teacher in `no_grad` (already the default).
- **Wrong `transformers` version** — must be `4.38.2`; the vendored modeling code
  relies on its internals.
