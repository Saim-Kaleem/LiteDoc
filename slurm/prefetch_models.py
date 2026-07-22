#!/usr/bin/env python3
"""
slurm/prefetch_models.py -- download the teacher weights on the LOGIN NODE.

Compute nodes run offline (TRANSFORMERS_OFFLINE=1 / HF_HUB_OFFLINE=1), so the
DeepSeek-VL2 Tiny checkpoint must be cached into $HF_HOME here, where the
internet is available. Run after `source slurm/env.sh`:

    conda activate litedoc
    source slurm/env.sh
    python slurm/prefetch_models.py
"""
import os
import sys

MODEL = os.environ.get("LITEDOC_MODEL", "deepseek-ai/deepseek-vl2-tiny")


def main() -> int:
    hf_home = os.environ.get("HF_HOME", "(default HF cache)")
    print(f"Prefetching '{MODEL}' into HF_HOME={hf_home} ...")
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("huggingface_hub not installed; run slurm/setup_env.sh first.")
        return 1

    path = snapshot_download(
        repo_id=MODEL,
        allow_patterns=["*.json", "*.model", "*.safetensors", "*.bin",
                        "*.txt", "*.py", "tokenizer*"],
    )
    print(f"Cached teacher at: {path}")

    # Sanity: make sure it loads offline exactly as the batch jobs will.
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"
    try:
        from transformers import AutoConfig
        AutoConfig.from_pretrained(MODEL, trust_remote_code=True)
        print("Offline load check: OK")
    except Exception as e:  # pragma: no cover
        print(f"[warn] offline load check failed: {e}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
