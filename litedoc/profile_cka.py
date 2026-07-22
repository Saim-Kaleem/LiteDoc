#!/usr/bin/env python3
"""
LiteDoc Stage 1a -- Task-aware CKA profiling.

Runs the frozen teacher (DeepSeek-VL2 Tiny) over document-understanding data,
captures every MoE expert's output activations via the `cka_tracking` hook built
into `deepseek_vl2.models.modeling_deepseek.DeepseekV2MoE`, then computes a
per-layer [E, E] CKA redundancy matrix (paper Sec. 3.2, Eqs. 1-2).

Outputs:
  * ``cka_clusters.json`` -- redundant expert pairs per layer at the chosen
    threshold, in the schema consumed by ``litedoc.prune``:
        {"DeepseekV2MoE": {"layer1": [[i, j], ...], "layer2": [...], ...}}
    (layers are 1-indexed here to match the historical pruning data files).
  * ``cka_matrices.pt`` -- the raw torch tensors, so thresholds can be re-swept
    offline without re-running the model.
  * ``cka_heatmaps/``   -- the per-layer heatmaps used in the paper figures.

The profiling set should come from **training splits only** to avoid test
leakage. FUNSD annotation JSONs + page images are the default (the paper profiles
on DocVQA + FUNSD validation-of-training data); any folder of {image, prompt}
pairs works.

Example:
    python -m litedoc.profile_cka \
        --model_path deepseek-ai/deepseek-vl2-tiny \
        --annotations_dir data/funsd/train/annotations \
        --images_root      data/funsd/train/images \
        --num_prompts_for_cka 200 \
        --threshold 0.7 \
        --output_dir artifacts/cka_tiny
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List

import torch
from transformers import AutoModelForCausalLM

from deepseek_vl2.models import DeepseekVLV2ForCausalLM, DeepseekVLV2Processor
from deepseek_vl2.utils.cka_utils import (
    compute_layer_cka,
    find_similar_experts,
    save_cka_heatmaps,
)

try:  # PIL is only needed at runtime
    from PIL import Image
except ImportError:  # pragma: no cover
    Image = None


def _load_images(paths: List[str]):
    imgs = []
    for p in paths:
        imgs.append(Image.open(p).convert("RGB"))
    return imgs


def _enable_cka_tracking(model) -> Dict[int, List[List[torch.Tensor]]]:
    """Turn on the per-expert activation hook and allocate activation buffers."""
    store: Dict[int, List[List[torch.Tensor]]] = {}
    for idx, layer in enumerate(model.language.model.layers):
        mlp = getattr(layer, "mlp", None)
        if mlp is not None and hasattr(mlp, "experts"):
            mlp.cka_tracking = True
            mlp._expert_activations = [None] * len(mlp.experts)
            store[idx] = [[] for _ in mlp.experts]
    return store


def collect_cka_matrices(
    model,
    processor,
    annotations_dir: str,
    images_root: str,
    num_prompts_for_cka: int = 200,
    prompt: str = "Extract all form fields and their labels from this document.",
    grounding: bool = True,
    max_new_tokens: int = 512,
    use_rbf: bool = True,
) -> Dict[int, torch.Tensor]:
    """Profile the teacher and return dict[layer_idx] -> [E, E] CKA matrix."""
    dtype = next(model.parameters()).dtype
    all_acts = _enable_cka_tracking(model)

    processed = 0
    files = sorted(f for f in os.listdir(annotations_dir) if f.endswith(".json"))
    for ann_file in files:
        if processed >= num_prompts_for_cka:
            break

        image_name = os.path.splitext(ann_file)[0] + ".png"
        img_path = os.path.join(images_root, image_name)
        if not os.path.exists(img_path):
            print(f"[skip] image not found: {img_path}")
            continue

        content = f"<image>\n<|grounding|>{prompt}" if grounding else f"<image>\n{prompt}"
        convo = [
            {"role": "<|User|>", "content": content, "images": [img_path]},
            {"role": "<|Assistant|>", "content": ""},
        ]
        inputs = processor(
            conversations=convo,
            images=_load_images([img_path]),
            force_batchify=True,
            system_prompt="",
        ).to(model.device, dtype=dtype)

        with torch.no_grad():
            inputs_embeds = model.prepare_inputs_embeds(**inputs)
            model.generate(
                inputs_embeds=inputs_embeds,
                input_ids=inputs.input_ids,
                images=inputs.images,
                images_seq_mask=inputs.images_seq_mask,
                images_spatial_crop=inputs.images_spatial_crop,
                attention_mask=inputs.attention_mask,
                pad_token_id=processor.tokenizer.eos_token_id,
                bos_token_id=processor.tokenizer.bos_token_id,
                eos_token_id=processor.tokenizer.eos_token_id,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=True,
            )

        # Harvest the activations the hook stashed on each MoE layer.
        for idx, layer in enumerate(model.language.model.layers):
            mlp = getattr(layer, "mlp", None)
            if mlp is None or not hasattr(mlp, "_expert_activations"):
                continue
            for e, act in enumerate(mlp._expert_activations):
                if act is not None and act.numel() > 0:
                    all_acts[idx][e].append(act.detach().cpu().float())
        processed += 1
        print(f"[cka] profiled {processed}/{num_prompts_for_cka}: {ann_file}")

    # Stack per-expert activations and compute the layer CKA matrices.
    hidden = model.config.hidden_size if hasattr(model.config, "hidden_size") else None
    cka: Dict[int, torch.Tensor] = {}
    for idx, act_lists in all_acts.items():
        mats = [torch.cat(a, dim=0) if a else torch.empty((0, hidden or 0))
                for a in act_lists]
        counts = [m.shape[0] for m in mats]
        n_min = min(counts) if counts else 0
        if n_min == 0:
            print(f"[cka] layer {idx}: no activations captured, skipping")
            continue
        batch = torch.stack([m[:n_min] for m in mats], dim=1)  # [n_min, E, D]
        cka[idx] = compute_layer_cka(batch, use_rbf=use_rbf)
        print(f"[cka] layer {idx}: CKA matrix {tuple(cka[idx].shape)} "
              f"from {n_min} tokens/expert")
    return cka


def to_clusters_json(similar: Dict[int, list]) -> dict:
    """Convert {layer_idx: [(i, j, sim), ...]} to the cka_clusters.json schema."""
    out = {}
    for layer_idx, pairs in similar.items():
        out[f"layer{layer_idx + 1}"] = [[int(i), int(j)] for i, j, _ in pairs]
    return {"DeepseekV2MoE": out}


def main() -> None:
    from litedoc import config
    ap = argparse.ArgumentParser(description="LiteDoc Stage 1a: CKA profiling")
    ap.add_argument("--model_path", default=config.MODEL,
                    help="teacher model name or local path")
    ap.add_argument("--annotations_dir", required=True,
                    help="folder of per-page annotation JSONs (train split)")
    ap.add_argument("--images_root", required=True,
                    help="folder of page images matching the annotations")
    ap.add_argument("--num_prompts_for_cka", type=int, default=200)
    ap.add_argument("--threshold", type=float, default=0.7,
                    help="CKA edge threshold tau (0.7 moderate, 0.5 aggressive)")
    ap.add_argument("--linear_cka", action="store_true",
                    help="use linear CKA instead of the RBF-kernel default")
    ap.add_argument("--dtype", default="bfloat16",
                    choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--output_dir", default=os.path.join(config.ARTIFACTS_DIR, "cka"))
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
             "float32": torch.float32}[args.dtype]

    print(f"Loading teacher: {args.model_path}")
    processor = DeepseekVLV2Processor.from_pretrained(args.model_path)
    model: DeepseekVLV2ForCausalLM = AutoModelForCausalLM.from_pretrained(
        args.model_path, trust_remote_code=True, torch_dtype=dtype,
    ).cuda().eval()

    cka = collect_cka_matrices(
        model, processor,
        annotations_dir=args.annotations_dir,
        images_root=args.images_root,
        num_prompts_for_cka=args.num_prompts_for_cka,
        use_rbf=not args.linear_cka,
    )

    torch.save(cka, os.path.join(args.output_dir, "cka_matrices.pt"))
    save_cka_heatmaps(cka, threshold=args.threshold,
                      output_dir=os.path.join(args.output_dir, "cka_heatmaps"))

    similar = find_similar_experts(cka, threshold=args.threshold)
    clusters = to_clusters_json(similar)
    out_json = os.path.join(args.output_dir, "cka_clusters.json")
    with open(out_json, "w") as f:
        json.dump(clusters, f, indent=2)

    n_pairs = sum(len(v) for v in similar.values())
    print(f"\nWrote {out_json} ({n_pairs} redundant pairs at tau={args.threshold})")
    print(f"Raw matrices:  {os.path.join(args.output_dir, 'cka_matrices.pt')}")


if __name__ == "__main__":
    main()
