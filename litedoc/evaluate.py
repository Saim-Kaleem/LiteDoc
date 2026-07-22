#!/usr/bin/env python3
"""
LiteDoc evaluation -- unified benchmark inference + metrics.

Runs a (teacher, pruned, or distilled) model over a benchmark JSONL and reports
the metric that matches the task, reproducing the columns of paper Table 3:

    metric   task category                 benchmarks
    ------   ---------------------------   ------------------------------------
    anls     VQA                           DocVQA, InfoVQA           (threshold 0.5)
    f1       Key Information Extraction     CORD, SROIE, FUNSD        (token-level)
    em       Table QA / Classification      WTQ, RVL-CDIP            (case-insensitive)

Input JSONL, one record per line:
    {"image": "xxx.png", "prompt": "<question or extraction instruction>",
     "answers": ["gold1", "gold2", ...]}      # "answer": "gold" also accepted

Example:
    python -m litedoc.evaluate \
        --model_path checkpoints/litedoc-20-distilled \
        --eval_jsonl data/docvqa/val.jsonl \
        --image_root data/docvqa/images \
        --metric anls \
        --max_new_tokens 128 \
        --output_json results/litedoc20_docvqa.json
"""
from __future__ import annotations

import argparse
import json
import os
import re
import string
from typing import List

import torch
from transformers import AutoModelForCausalLM

from deepseek_vl2.models import DeepseekVLV2ForCausalLM, DeepseekVLV2Processor
from litedoc import config

try:
    from PIL import Image
except ImportError:  # pragma: no cover
    Image = None


def _write_verification(name: str, metric: str, score: float,
                        expected: float | None, tol: float) -> None:
    """Append a PASS/WARN/FAIL record to the shared verification report, mirroring
    the cluster convention. WARN if off but finite (paper under-specifies filters),
    FAIL only on a hard error upstream."""
    status = "PASS"
    if expected is not None:
        status = "PASS" if abs(score - expected) <= tol else "WARN"
    report = {}
    if os.path.exists(config.VERIFICATION_REPORT):
        try:
            report = json.load(open(config.VERIFICATION_REPORT))
        except Exception:
            report = {}
    report[name] = {"status": status, "metric": metric, "score": round(score, 2),
                    "expected": expected, "tolerance": tol}
    os.makedirs(os.path.dirname(config.VERIFICATION_REPORT), exist_ok=True)
    with open(config.VERIFICATION_REPORT, "w") as f:
        json.dump(report, f, indent=2)
    print(f"[verify] {name}: {status} (score={score:.2f}"
          + (f", expected={expected}±{tol})" if expected is not None else ")"))


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def _normalize(s: str) -> str:
    s = s.lower()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    return " ".join(s.split())


def _levenshtein(a: str, b: str) -> int:
    m, n = len(a), len(b)
    if m == 0 or n == 0:
        return max(m, n)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, n + 1):
            cur = dp[j]
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1,
                        prev + (0 if a[i - 1] == b[j - 1] else 1))
            prev = cur
    return dp[n]


def anls(pred: str, golds: List[str], threshold: float = 0.5) -> float:
    """Average Normalized Levenshtein Similarity for one question (max over golds,
    zeroed below `threshold`), per Biten et al. and the paper's VQA protocol."""
    pred = pred.strip().lower()
    best = 0.0
    for g in golds:
        g = g.strip().lower()
        d = _levenshtein(pred, g)
        nl = 1 - d / max(len(pred), len(g), 1)
        best = max(best, nl)
    return best if best >= threshold else 0.0


def token_f1(pred: str, golds: List[str]) -> float:
    """Token-level F1 (max over golds), used for KIE benchmarks."""
    p_tokens = _normalize(pred).split()
    best = 0.0
    for g in golds:
        g_tokens = _normalize(g).split()
        if not p_tokens and not g_tokens:
            best = max(best, 1.0); continue
        if not p_tokens or not g_tokens:
            continue
        overlap = 0
        gtmp = list(g_tokens)
        for t in p_tokens:
            if t in gtmp:
                overlap += 1
                gtmp.remove(t)
        if overlap == 0:
            continue
        prec = overlap / len(p_tokens)
        rec = overlap / len(g_tokens)
        best = max(best, 2 * prec * rec / (prec + rec))
    return best


def exact_match(pred: str, golds: List[str]) -> float:
    """Case-insensitive exact match (max over golds), for Table QA / classification."""
    p = _normalize(pred)
    return 1.0 if any(p == _normalize(g) for g in golds) else 0.0


METRICS = {"anls": anls, "f1": token_f1, "em": exact_match}


# --------------------------------------------------------------------------- #
# Inference
# --------------------------------------------------------------------------- #
def _find_image(image_root: str, name: str) -> str | None:
    base = os.path.splitext(name)[0]
    for ext in (os.path.splitext(name)[1], ".png", ".jpg", ".jpeg", ".tif", ".tiff"):
        if not ext:
            continue
        p = os.path.join(image_root, base + ext)
        if os.path.exists(p):
            return p
    direct = os.path.join(image_root, name)
    return direct if os.path.exists(direct) else None


def evaluate(args) -> None:
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
             "float32": torch.float32}[args.dtype]
    metric_fn = METRICS[args.metric]

    print(f"Loading model: {args.model_path}")
    processor = DeepseekVLV2Processor.from_pretrained(args.model_path)
    tokenizer = processor.tokenizer
    model: DeepseekVLV2ForCausalLM = AutoModelForCausalLM.from_pretrained(
        args.model_path, trust_remote_code=True, torch_dtype=dtype,
        device_map="auto").eval()

    with open(args.eval_jsonl) as f:
        records = [json.loads(l) for l in f if l.strip()]

    results, scores = [], []
    for i, r in enumerate(records):
        if args.max_questions and i >= args.max_questions:
            break
        prompt = r.get("prompt", r.get("question", ""))
        golds = r.get("answers") or ([r["answer"]] if "answer" in r else [])
        img_path = _find_image(args.image_root, r.get("image", ""))
        if img_path is None:
            print(f"[skip] missing image: {r.get('image')}")
            continue

        convo = [
            {"role": "<|User|>", "content": f"<image>\n{prompt}", "images": [img_path]},
            {"role": "<|Assistant|>", "content": ""},
        ]
        inputs = processor(conversations=convo, images=[Image.open(img_path).convert("RGB")],
                           force_batchify=True, system_prompt="").to(model.device, dtype=dtype)

        with torch.no_grad():
            inputs_embeds = model.prepare_inputs_embeds(**inputs)
            out = model.generate(
                inputs_embeds=inputs_embeds,
                input_ids=inputs.input_ids,
                images=inputs.images,
                images_seq_mask=inputs.images_seq_mask,
                images_spatial_crop=inputs.images_spatial_crop,
                attention_mask=inputs.attention_mask,
                pad_token_id=tokenizer.eos_token_id,
                bos_token_id=tokenizer.bos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                max_new_tokens=args.max_new_tokens,
                do_sample=args.do_sample,
                temperature=args.temperature,
                top_p=args.top_p,
                repetition_penalty=args.repetition_penalty,
                use_cache=True,
            )
        answer = tokenizer.decode(
            out[0][inputs.input_ids.shape[1]:].cpu().tolist(),
            skip_special_tokens=True).strip()
        if args.strip_regex:
            answer = re.sub(args.strip_regex, "", answer).strip()

        score = metric_fn(answer, golds) if golds else None
        if score is not None:
            scores.append(score)
        results.append({"prompt": prompt, "image": r.get("image"),
                        "prediction": answer, "answers": golds, args.metric: score})
        if args.verbose:
            print(f"[{i}] {args.metric}={score}  pred={answer!r}  gold={golds}")

    mean = sum(scores) / len(scores) if scores else 0.0
    scale = 100.0 if args.metric in ("anls", "f1", "em") else 1.0
    final = mean * scale
    print(f"\n=== {args.metric.upper()} over {len(scores)} scored / "
          f"{len(results)} total: {final:.2f} ===")

    if args.verify_name:
        _write_verification(args.verify_name, args.metric, final,
                            args.expected, args.tolerance)

    if args.output_json:
        os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
        with open(args.output_json, "w") as f:
            json.dump({"metric": args.metric, "score": mean * scale,
                       "n": len(scores), "results": results}, f, indent=2)
        print(f"Wrote {args.output_json}")


def main() -> None:
    ap = argparse.ArgumentParser(description="LiteDoc benchmark evaluation")
    ap.add_argument("--model_path", default=config.MODEL)
    ap.add_argument("--eval_jsonl", required=True)
    ap.add_argument("--image_root", required=True)
    ap.add_argument("--metric", choices=list(METRICS), required=True)
    ap.add_argument("--dtype", default="bfloat16",
                    choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--max_new_tokens", type=int, default=128)
    ap.add_argument("--do_sample", action="store_true")
    ap.add_argument("--temperature", type=float, default=0.4)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--repetition_penalty", type=float, default=1.1)
    ap.add_argument("--strip_regex", default="",
                    help="optional regex of artifacts to remove from outputs")
    ap.add_argument("--max_questions", type=int, default=None)
    ap.add_argument("--output_json", default="")
    ap.add_argument("--verbose", action="store_true")
    # optional per-stage verification (writes PASS/WARN to the shared report)
    ap.add_argument("--verify_name", default="",
                    help="e.g. 'eval_docvqa_litedoc20'; enables the verification record")
    ap.add_argument("--expected", type=float, default=None,
                    help="expected score (paper target) for PASS/WARN")
    ap.add_argument("--tolerance", type=float, default=2.0,
                    help="absolute tolerance on --expected (paper compares within ~2 pts)")
    evaluate(ap.parse_args())


if __name__ == "__main__":
    main()
