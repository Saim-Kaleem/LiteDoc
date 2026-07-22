#!/usr/bin/env python3
"""
LiteDoc Stage 1b -- Structured expert pruning + router repair.

Consumes the redundancy information produced by ``litedoc.profile_cka``
(``cka_clusters.json``) plus an optional hand-curated ``manual_pruning.json``,
removes redundant routed experts from every MoE layer of DeepSeek-VL2 Tiny, and
rebuilds the gate so the top-k routing budget stays full (paper Sec. 3.2,
"Graph-Based Expert Selection" + "Routing Coherence Preservation").

Per layer, the set of experts to prune is chosen with this priority:
  1. explicit indices in ``manual_pruning.json`` (if given);
  2. CKA cluster members from ``cka_clusters.json`` -- within a cluster the first
     expert is the retained representative and the rest are pruned, their weights
     merged into the representative (mean) and their gate rows folded in so the
     routing mass is preserved;
  3. fill from the highest expert indices until the target prune count is met.

The number pruned per layer is ``round(N * (1 - keep_ratio))``. LiteDoc variants:
  keep_ratio 0.60 -> LiteDoc-60 (40% pruned)   keep_ratio 0.30 -> LiteDoc-30
  keep_ratio 0.40 -> LiteDoc-40 (60% pruned)   keep_ratio 0.20 -> LiteDoc-20 (80%)

The pruned checkpoint is written with ``save_pretrained`` and can be evaluated
directly (Pruning-only column of paper Table 3) or handed to ``litedoc.distill``
for performance recovery (Pruning+KD column).

Example:
    python -m litedoc.prune \
        --model_path deepseek-ai/deepseek-vl2-tiny \
        --cka_clusters_path data/cka_clusters.json \
        --manual_pruning_path data/manual_pruning.json \
        --keep_ratio 0.20 \
        --output_path checkpoints/litedoc-20-pruned
"""
from __future__ import annotations

import argparse
import json
import os
import re

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM

from deepseek_vl2.models.modeling_deepseek import DeepseekV2Config as LangConfig
from deepseek_vl2.models.modeling_deepseek import DeepseekV2MoE  # noqa: F401

from litedoc import config


def _load_json_lenient(path: str) -> dict:
    """Parse JSON that may contain // comments or trailing commas (the historical
    ``manual_pruning.json`` uses both)."""
    if not path or not os.path.exists(path):
        return {}
    with open(path, "r") as f:
        content = f.read()
    content = re.sub(r"//.*", "", content)            # strip // comments
    content = re.sub(r",\s*([\]}])", r"\1", content)  # strip trailing commas
    try:
        return json.loads(content)
    except json.JSONDecodeError as e:
        print(f"[warn] could not parse {path}: {e}")
        return {}


def _restore_gate_params(model, model_path: str) -> None:
    """If a raw checkpoint is available, copy its gate weight / valid_expert_mask
    (padding or truncating on shape mismatch) so a freshly-instantiated gate does
    not start from random values."""
    ckpt_path = os.path.join(model_path, "pytorch_model.bin")
    if not os.path.exists(ckpt_path):
        return
    try:
        ckpt = torch.load(ckpt_path, map_location="cpu")
    except Exception as e:  # pragma: no cover
        print(f"[warn] could not read checkpoint gate params: {e}")
        return

    def pad_or_truncate(t, shape):
        out = torch.zeros(shape, dtype=t.dtype)
        sl = tuple(slice(0, min(t.shape[i], shape[i])) for i in range(len(shape)))
        out[sl] = t[sl]
        return out

    for i in range(len(model.language.model.layers)):
        layer = model.language.model.layers[i]
        if not hasattr(layer.mlp, "gate"):
            continue
        gate = layer.mlp.gate
        kw = f"language.model.layers.{i}.mlp.gate.weight"
        km = f"language.model.layers.{i}.mlp.gate.valid_expert_mask"
        if kw in ckpt and isinstance(ckpt[kw], torch.Tensor):
            w = ckpt[kw]
            if tuple(w.shape) != tuple(gate.weight.shape):
                w = pad_or_truncate(w, tuple(gate.weight.shape))
            gate.weight.data = w.to(gate.weight.device, gate.weight.dtype)
        if km in ckpt and isinstance(ckpt[km], torch.Tensor) and hasattr(gate, "valid_expert_mask"):
            m = ckpt[km]
            tgt = gate.valid_expert_mask.shape[0]
            if m.shape[0] != tgt:
                nm = torch.ones(tgt, dtype=torch.bool)
                nm[: min(m.shape[0], tgt)] = m[: min(m.shape[0], tgt)].bool()
                m = nm
            gate.valid_expert_mask = m.to(gate.weight.device)


def prune_experts(args) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading model on {device}: {args.model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, trust_remote_code=True,
        torch_dtype=torch.bfloat16, device_map=device,
    )
    _restore_gate_params(model, args.model_path)

    config = model.config
    lang_config = config.language_config
    if not isinstance(lang_config, LangConfig):
        lang_config = LangConfig(**lang_config)

    n_experts = lang_config.n_routed_experts
    target = int(n_experts * args.keep_ratio)
    prune_count = n_experts - target
    print(f"Target experts/layer: {n_experts} -> {target} "
          f"(pruning {prune_count}, keep_ratio={args.keep_ratio})")

    cka_data = _load_json_lenient(args.cka_clusters_path).get("DeepseekV2MoE", {})
    manual_data = _load_json_lenient(args.manual_pruning_path)

    for layer_idx in tqdm(range(lang_config.num_hidden_layers), desc="Pruning layers"):
        layer = model.language.model.layers[layer_idx]
        if not hasattr(layer.mlp, "experts"):
            continue
        moe = layer.mlp

        to_prune: set[int] = set()
        merge_map: dict[int, int] = {}  # pruned source -> retained target

        # Priority 1: manual list
        for idx in manual_data.get(str(layer_idx), []):
            if 0 <= idx < n_experts:
                to_prune.add(idx)

        # Priority 2: CKA clusters (1-indexed layer keys in the JSON)
        for cluster in cka_data.get(f"layer{layer_idx + 1}", []):
            if len(cluster) >= 2:
                target_e = cluster[0]
                for src in cluster[1:]:
                    if len(to_prune) < prune_count and src not in to_prune:
                        to_prune.add(src)
                        merge_map[src] = target_e

        # Priority 3: fill from the top indices down
        for idx in range(n_experts - 1, -1, -1):
            if len(to_prune) >= prune_count:
                break
            to_prune.add(idx)

        keeping = [i for i in range(n_experts) if i not in to_prune]
        remap = {old: new for new, old in enumerate(keeping)}

        # Merge pruned experts' weights into their retained representative.
        for src, tgt in merge_map.items():
            if tgt in keeping:
                s_sd = moe.experts[src].state_dict()
                t_sd = moe.experts[tgt].state_dict()
                moe.experts[tgt].load_state_dict(
                    {k: (t_sd[k] + s_sd[k]) / 2.0 for k in t_sd}
                )

        # Rebuild the expert ModuleList and the gate weight matrix.
        moe.experts = torch.nn.ModuleList([moe.experts[i] for i in keeping])
        old_gate = moe.gate.weight.data
        new_gate = old_gate[keeping].clone()
        for src, tgt in merge_map.items():          # fold routing mass forward
            if tgt in keeping:
                new_gate[remap[tgt]] += old_gate[src]

        moe.gate.n_routed_experts = len(keeping)
        moe.gate.weight = torch.nn.Parameter(new_gate.to(old_gate.device))

        # Keep valid_expert_mask consistent with the new expert ordering.
        if hasattr(moe.gate, "valid_expert_mask"):
            old_mask = moe.gate.valid_expert_mask
            if old_mask.shape[0] == n_experts:
                moe.gate.valid_expert_mask = old_mask[keeping].to(new_gate.device)
            else:
                moe.gate.valid_expert_mask = torch.ones(
                    len(keeping), dtype=torch.bool, device=new_gate.device)

    lang_config.n_routed_experts = target
    config.language_config = lang_config

    os.makedirs(args.output_path, exist_ok=True)
    print(f"Saving pruned model to {args.output_path}")
    model.save_pretrained(args.output_path)
    try:
        from deepseek_vl2.models import DeepseekVLV2Processor
        DeepseekVLV2Processor.from_pretrained(args.model_path).save_pretrained(args.output_path)
    except Exception as e:  # pragma: no cover
        print(f"[note] could not save processor: {e}")
    print("Pruning complete.")


def main() -> None:
    ap = argparse.ArgumentParser(description="LiteDoc Stage 1b: expert pruning")
    ap.add_argument("--model_path", default=config.MODEL, help="teacher model path or hub id")
    ap.add_argument("--cka_clusters_path", default=config.CKA_CLUSTERS)
    ap.add_argument("--manual_pruning_path", default=config.MANUAL_PRUNING)
    ap.add_argument("--output_path", required=True, help="where to save the student")
    ap.add_argument("--keep_ratio", type=float, default=0.20,
                    help="fraction of experts to KEEP (0.20 = LiteDoc-20 / 80%% pruned)")
    prune_experts(ap.parse_args())


if __name__ == "__main__":
    main()
