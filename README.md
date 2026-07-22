# LiteDoc

**Distilling large document VLMs into efficient, task-specific encoders via
CKA-based expert pruning + Sparse-to-Sparse knowledge distillation.**

LiteDoc compresses **DeepSeek-VL2 Tiny** (a ~3.37B-parameter Mixture-of-Experts
vision-language model) into a compact document-understanding encoder. At **80%
expert pruning** it shrinks the model **3.37B → ~1.4B parameters (2.4×
compression)** while retaining **89.6%** of the teacher's accuracy across seven
document-understanding benchmarks, cutting VRAM from 12.3 GB to 7.4 GB and
raising throughput by 51.7% — enough to run on a consumer 8 GB GPU.

This repository is a cleaned, consolidated version of the original research code
(previously spread across three overlapping archives). See
[`REPRODUCE.md`](REPRODUCE.md) for the full step-by-step reproduction guide.

---

## Method in one picture

```
                 ┌──────────────────────────── Stage 1: Task-aware pruning ───────────────────────────┐
 DeepSeek-VL2    │  profile_cka.py            prune.py                                                 │   distill.py            evaluate.py
 Tiny (teacher)  │  ─────────────            ────────                                                 │   ──────────            ──────────
 64 experts/layer│  run teacher on DU  →  CKA  build redundancy graph (edge if CKA>τ),  →  compact     │   S2S KD vs frozen  →   F1 / ANLS / EM
 frozen          │  data, hook expert   matrix keep most-utilised expert per clique,     student       │   teacher:              per benchmark
                 │  activations               redirect routing of pruned experts         (fewer experts)│   L_KL + L_hidden
                 └───────────────────────────────────────────────────────────────────────────────────┘
```

- **Stage 1 — CKA expert pruning.** Routing frequency is near-uniform (load
  balancing) and weight similarity is unreliable (neuron permutation
  invariance), so LiteDoc measures *functional* redundancy with **Centered
  Kernel Alignment** on expert output activations. Experts whose activations are
  highly aligned (CKA > τ) form redundancy cliques; the most-utilised expert in
  each clique is kept, the rest pruned, and tokens routed to a pruned expert are
  redirected to their CKA-nearest retained neighbour (routing coherence).
- **Stage 2 — Sparse-to-Sparse distillation.** The pruned student keeps the
  teacher's depth and hidden width, so it is healed by aligning to the frozen
  teacher with two objectives at once: layer-wise hidden-state **MSE**
  (`L_hidden`) and output-logit **KL divergence** (`L_KL`).

## Model variants (paper Table 1)

| Model      | `keep_ratio` | Pruning | Experts/layer | Params | Compression |
|------------|-------------|---------|---------------|--------|-------------|
| Teacher    | 1.00        | 0%      | 64            | 3.37B  | 1.0×        |
| LiteDoc-60 | 0.60        | 40%     | ~38           | 2.4B   | 1.4×        |
| LiteDoc-40 | 0.40        | 60%     | ~26           | 1.9B   | 1.8×        |
| LiteDoc-30 | 0.30        | 70%     | ~19           | 1.7B   | 2.0×        |
| LiteDoc-20 | 0.20        | 80%     | ~13           | 1.4B   | 2.4×        |

`LiteDoc-X` = X% of experts **retained** (LiteDoc-20 = 80% pruned).

## Headline results (paper Table 3, teacher → LiteDoc-20 with KD)

| Benchmark | Task | Metric | Teacher | LiteDoc-20 (+KD) |
|-----------|------|--------|---------|------------------|
| CORD      | KIE  | F1     | 96.76   | 85.32 |
| SROIE     | KIE  | F1     | 87.03   | 76.61 |
| FUNSD     | KIE  | F1     | 87.59   | 80.44 |
| DocVQA    | VQA  | ANLS   | 88.76   | 82.13 |
| InfoVQA   | VQA  | ANLS   | 62.29   | 53.80 |
| WTQ       | TableQA | EM  | 84.81   | 75.95 |
| RVL-CDIP  | Class | EM    | 91.15   | 82.67 |

Distilled 80% model averages **76.70**, recovering **71.6%** of the pruning loss
and **~89.6%** of teacher performance overall.

## Repository layout

```
LiteDoc/
├── README.md                  # this file (incl. Elwetritsch HPC guide below)
├── REPRODUCE.md               # step-by-step reproduction guide
├── requirements.txt           # local / generic
├── requirements-hpc.txt       # cluster: torch 2.2.x CUDA wheels, no gradio
├── NOTICE                     # DeepSeek-VL2 attribution + list of LiteDoc changes
├── configs/
│   └── pruning_variants.yaml  # keep_ratios, τ, distillation weights
├── deepseek_vl2/              # vendored DeepSeek-VL2 model (LiteDoc-modified)
│   ├── models/                #   modeling_deepseek.py has the 2 LiteDoc hooks
│   └── utils/                 #   cka_utils.py, expert_prune_merge.py  (LiteDoc)
├── litedoc/                   # the LiteDoc pipeline (clean, importable)
│   ├── config.py              #   central path config (LITEDOC_* env or ./ defaults)
│   ├── profile_cka.py         #   Stage 1a  — CKA profiling → cka_clusters.json
│   ├── prune.py               #   Stage 1b  — expert pruning + router repair
│   ├── distill.py             #   Stage 2   — S2S knowledge distillation
│   └── evaluate.py            #   evaluation — F1 / ANLS / EM (+ paper verification)
├── slurm/                     # Elwetritsch (RPTU HPC) SLURM + Conda infrastructure
│   ├── env.sh                 #   single source of truth for paths + caches
│   ├── setup_env.sh           #   one-time Conda env creation
│   ├── prefetch_models.py     #   login-node weight download (nodes run offline)
│   ├── {profile_cka,prune,distill,evaluate}.sbatch
│   └── submit_all.sh          #   prune → distill → evaluate (afterok chain)
├── scripts/                   # thin local wrappers (1_… 2_… 3_… 4_…) + run_all.sh
├── data/
│   ├── cka_clusters.json      # precomputed redundant expert pairs per layer
│   └── manual_pruning.json    # hand-curated expert indices per layer
├── tests/
│   └── test_model_smoke.py    # CPU smoke test: forward/backward + invariants
└── experiments/               # original Kaggle driver notebook (preserved)
```

## Running on Elwetritsch (RPTU HPC, SLURM + Conda) — primary path

Tuned to this cluster's conventions (native Conda, no containers; storage on
`$SCRATCH`; GPUs via `--gres=gpu:v100:1` with no `--partition`/`--account`;
internet only on the login node so batch jobs run offline). `slurm/env.sh` is the
single source of truth for paths and caches (`HF_HOME`/`TORCH_HOME`/`XDG_CACHE_HOME`
all live under `$SCRATCH/LiteDoc`, never in the 40 GB `$HOME`).

### 0. One-time setup (LOGIN NODE)
```bash
git clone <your-repo-url> $SCRATCH/LiteDoc/repo && cd $SCRATCH/LiteDoc/repo

module purge && module load miniforge3/latest
source $ANACONDA_HOME/etc/profile.d/conda.sh
conda create -n litedoc python=3.11 && conda activate litedoc
pip install -r requirements-hpc.txt
# or simply: source slurm/env.sh && bash slurm/setup_env.sh

source slurm/env.sh
python slurm/prefetch_models.py      # caches deepseek-vl2-tiny for the offline nodes
```

### 1. Stage the benchmark data under `$LITEDOC_DATA`
Convert each benchmark split to JSONL (`{"image","prompt","answers":[...]}`) as
described in [`REPRODUCE.md`](REPRODUCE.md), placed at
`$LITEDOC_DATA/<task>/{train,val}.jsonl` + `$LITEDOC_DATA/<task>/images/`.

### 2. Run the pipeline (BATCH — 1× V100 each, offline)
```bash
source slurm/env.sh
# whole chain for one variant/task (prune → distill → evaluate via afterok):
bash slurm/submit_all.sh 0.20 docvqa anls 82.13

# or individual jobs:
sbatch slurm/prune.sbatch 0.20                          # LiteDoc-20 (80% pruned)
sbatch slurm/distill.sbatch 0.20 docvqa
sbatch slurm/evaluate.sbatch $LITEDOC_CKPT/litedoc-0.20-distilled-docvqa docvqa anls 82.13
# CKA profiling is optional (data/cka_clusters.json ships precomputed):
sbatch slurm/profile_cka.sbatch
```

Every SLURM job follows the same template: load Miniforge, `conda activate
litedoc`, `source slurm/env.sh`, export `TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1`,
then run `python -m litedoc.…` natively. A single V100 is ample; avoid the
Blackwell `dgx` nodes (they need CUDA 12.8+/torch 2.7+, incompatible with the
pinned torch 2.2.x).

## Running locally (small tests / no cluster)

Paths default to `./data/...`, `./checkpoints`, `./results` when the `LITEDOC_*`
env vars are unset (see `litedoc/config.py`).

```bash
pip install -r requirements.txt
bash scripts/run_all.sh                    # smoke test only (no weights/GPU)
RUN_PIPELINE=1 KEEP=0.20 TASK=docvqa bash scripts/run_all.sh   # full local run (needs GPU)
```

Every stage is also a plain module: `python -m litedoc.prune --help`, etc. The
thin `scripts/{1,2,3,4}_*.sh` wrappers accept the same env vars for ad-hoc runs.

## Verify the code (no data / GPU needed)

```bash
conda activate litedoc          # or any env with the requirements installed
python -m tests.test_model_smoke
```

Synthetic MoE forward/backward, the `valid_expert_mask` router and `cka_tracking`
hook, prune-and-repair, CKA invariances, and the ANLS/F1/EM metrics. Exit 0 = all
passed. Evaluation jobs also append a PASS/WARN record (vs the paper target, ±2
pts by default) to `$LITEDOC_WORK/verification_report.json` when given `--expected`.

## What changed vs. the original archives

This repo consolidates `LiteDoc-main.zip`, `LiteDoc_-main.zip`, and
`Major_download-main.zip` (nested `sara code yha he.zip`), which heavily
overlapped and each contained a partial view of the project. Key clean-ups:

- **Single coherent package.** One `deepseek_vl2/` (the version whose
  `modeling_deepseek.py` carries *both* LiteDoc modifications — the
  `valid_expert_mask` router and the `cka_tracking` activation hook), plus one
  clean `litedoc/` pipeline package.
- **Recovered Stage-1 profiler.** The CKA-collection code existed only as
  commented-out history inside a 4,500-line `inference.py`; it is now the
  standalone, documented `litedoc/profile_cka.py`.
- **Fixed the broken entry point.** The original `prune_and_infer.py` imported
  three modules (`expert_pruner`, `inference_utils`, `cka_collector`) that were
  absent from every archive. The working pruning logic (`prune_model.py`) is now
  the documented `litedoc/prune.py`.
- **De-duplicated `cka_utils.py`.** ~19 pasted/duplicated function definitions
  were replaced with one canonical HSIC/CKA implementation.
- **Reconstructed the distillation trainer.** See the note below.
- **Adapted for the RPTU Elwetritsch cluster.** Added `slurm/` (env.sh + Conda
  setup + login-node prefetch + per-stage `.sbatch` + an afterok chain),
  `requirements-hpc.txt`, `litedoc/config.py` (env-driven paths), and a
  `python -m tests.test_model_smoke` check — matching the cluster's SLURM/Conda,
  `$SCRATCH`-storage, and offline-compute-node conventions.
- **Removed redundant material** not needed for batch reproduction: the
  interactive Gradio demo (`deepseek_vl2/serve/`, incl. a 15 MB font), demo
  images, the DeepSeek grounding example, and broken/duplicate test scripts that
  imported the missing `expert_pruner` module. Repo shrank from ~29 MB to <1 MB.

## Known gaps / honesty notes

- **Distillation trainer is a reference reconstruction.** None of the archives
  contained the Stage-2 training loop (only a helper that saved a processor into
  `distilled_model80/checkpoint-7000`, implying training happened in a Kaggle
  notebook that was not included). `litedoc/distill.py` faithfully implements the
  paper's objective (Eqs. 3–6) and is runnable in principle, but it is **not**
  the authors' exact script; validate on a small subset and tune hyper-parameters.
- **Weights and datasets are not included.** You need the public
  `deepseek-ai/deepseek-vl2-tiny` checkpoint and the seven benchmark datasets
  (converted to the simple JSONL format described in `REPRODUCE.md`).
- **`data/cka_clusters.json`** ships precomputed for τ=0.7; re-run
  `litedoc.profile_cka` to regenerate or sweep τ.

## License

The `deepseek_vl2/` model code is derived from DeepSeek-VL2 (MIT, © DeepSeek).
See [`NOTICE`](NOTICE) for attribution and the exact list of modifications.
