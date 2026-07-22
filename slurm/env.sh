#!/usr/bin/env bash
# slurm/env.sh -- single source of truth for LiteDoc paths + caches on Elwetritsch
# (RPTU HPC). `source` this in every login-node command and every SLURM job.
#
# Everything lives under $SCRATCH (BeeGFS, no quota); nothing touches the 40 GB
# $HOME. All HF / Torch / generic caches are redirected under $LITEDOC_WORK so the
# offline compute nodes read prefetched weights instead of hitting the network.

# --- root: $SCRATCH/LiteDoc, or override with LITEDOC_ROOT --------------------
: "${SCRATCH:=$HOME/scratch}"                 # fallback if $SCRATCH is unset
export LITEDOC_ROOT="${LITEDOC_ROOT:-$SCRATCH/LiteDoc}"

export LITEDOC_REPO="${LITEDOC_REPO:-$LITEDOC_ROOT/repo}"
export LITEDOC_DATA="${LITEDOC_DATA:-$LITEDOC_ROOT/data}"
export LITEDOC_WORK="${LITEDOC_WORK:-$LITEDOC_ROOT/work}"
export LITEDOC_CKPT="${LITEDOC_CKPT:-$LITEDOC_ROOT/checkpoints}"
export LITEDOC_ARTIFACTS="${LITEDOC_ARTIFACTS:-$LITEDOC_ROOT/artifacts}"
export LITEDOC_RESULTS="${LITEDOC_RESULTS:-$LITEDOC_ROOT/results}"

# Teacher checkpoint (HF hub id; prefetched into HF_HOME on the login node).
export LITEDOC_MODEL="${LITEDOC_MODEL:-deepseek-ai/deepseek-vl2-tiny}"

# Shipped pruning inputs (fall back to the repo copies if not staged in $DATA).
export LITEDOC_CKA_CLUSTERS="${LITEDOC_CKA_CLUSTERS:-$LITEDOC_REPO/data/cka_clusters.json}"
export LITEDOC_MANUAL_PRUNING="${LITEDOC_MANUAL_PRUNING:-$LITEDOC_REPO/data/manual_pruning.json}"

# --- caches: keep them off $HOME --------------------------------------------
export HF_HOME="${HF_HOME:-$LITEDOC_WORK/hf}"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export TORCH_HOME="${TORCH_HOME:-$LITEDOC_WORK/torch}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$LITEDOC_WORK/cache}"

# --- create writable dirs ----------------------------------------------------
mkdir -p "$LITEDOC_DATA" "$LITEDOC_WORK" "$LITEDOC_CKPT" "$LITEDOC_ARTIFACTS" \
         "$LITEDOC_RESULTS" "$HF_HOME" "$TORCH_HOME" "$XDG_CACHE_HOME"

# So `python -m litedoc.*` resolves from anywhere.
export PYTHONPATH="${LITEDOC_REPO}:${PYTHONPATH:-}"

echo "[env] LITEDOC_ROOT=$LITEDOC_ROOT  MODEL=$LITEDOC_MODEL"
