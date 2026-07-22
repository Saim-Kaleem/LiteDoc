"""
Central path / environment configuration for LiteDoc.

Single source of truth for where models, data, checkpoints and results live, so
the same code runs unchanged on a laptop and on the RPTU Elwetritsch cluster.

Resolution order for every path:
  1. an explicit CLI argument (always wins),
  2. the matching ``LITEDOC_*`` environment variable (set by ``slurm/env.sh``),
  3. a local ``./…`` default (used when the env vars are unset).

On the cluster, ``slurm/env.sh`` points all of these under ``$SCRATCH/LiteDoc``
(BeeGFS, no quota) and redirects the HF / Torch caches there too, so nothing
lands in the 40 GB ``$HOME``.
"""
from __future__ import annotations

import os

# Root everything under LITEDOC_ROOT if set, else the current working directory.
ROOT = os.environ.get("LITEDOC_ROOT", os.getcwd())


def _p(env_key: str, default_rel: str) -> str:
    """Env override or a path relative to ROOT."""
    val = os.environ.get(env_key)
    return val if val else os.path.join(ROOT, default_rel)


# Teacher checkpoint: an HF hub id by default (prefetched into HF_HOME on the
# login node), or a local path via LITEDOC_MODEL.
MODEL = os.environ.get("LITEDOC_MODEL", "deepseek-ai/deepseek-vl2-tiny")

# Directories
DATA_DIR = _p("LITEDOC_DATA", "data")            # benchmark splits + pruning json
WORK_DIR = _p("LITEDOC_WORK", "data/work")       # caches, artifacts, reports
CKPT_DIR = _p("LITEDOC_CKPT", "checkpoints")     # pruned / distilled students
ARTIFACTS_DIR = _p("LITEDOC_ARTIFACTS", "artifacts")   # CKA matrices, heatmaps
RESULTS_DIR = _p("LITEDOC_RESULTS", "results")   # eval jsons

# Shipped pruning inputs (live in the repo's data/ folder by default)
CKA_CLUSTERS = os.environ.get("LITEDOC_CKA_CLUSTERS", os.path.join(DATA_DIR, "cka_clusters.json"))
MANUAL_PRUNING = os.environ.get("LITEDOC_MANUAL_PRUNING", os.path.join(DATA_DIR, "manual_pruning.json"))

# Per-stage verification report (mirrors the cluster's convention of appending a
# PASS/WARN/FAIL record per stage).
VERIFICATION_REPORT = os.path.join(WORK_DIR, "verification_report.json")

# Paper reference scores for --expected verification (Table 3, teacher then
# LiteDoc-20 + KD). metric name -> {benchmark: (teacher, litedoc20_kd)}.
PAPER_TARGETS = {
    "f1":   {"cord": (96.76, 85.32), "sroie": (87.03, 76.61), "funsd": (87.59, 80.44)},
    "anls": {"docvqa": (88.76, 82.13), "infovqa": (62.29, 53.80)},
    "em":   {"wtq": (84.81, 75.95), "rvlcdip": (91.15, 82.67)},
}


def ensure_dirs() -> None:
    """Create the writable output dirs (safe to call repeatedly)."""
    for d in (WORK_DIR, CKPT_DIR, ARTIFACTS_DIR, RESULTS_DIR):
        os.makedirs(d, exist_ok=True)
