#!/usr/bin/env python3
"""
LiteDoc Stage 2 -- Sparse-to-Sparse (S2S) knowledge distillation.

Recovers the accuracy lost during Stage-1 pruning by aligning the pruned student
with the frozen, unpruned DeepSeek-VL2 Tiny teacher on a target DU dataset,
using the two complementary objectives from the paper (Sec. 3.3):

    L_hidden = (1/M) * sum_i MSE( H_student^i , H_teacher^i )      (Eqs. 3-5)
    L_KL     = KL( p_student || p_teacher )   over next-token dists (Eq. 6)
    L        = alpha * L_KL + beta * L_hidden

The student keeps the teacher's *depth* and hidden width (d = 768), so hidden
states align one-to-one per transformer layer even though the student has fewer
experts. The teacher is never updated.

Ablation reference (paper, DocVQA): output-only 81.7, layer-only 82.9,
combined 84.2 -- hence both terms are on by default.

------------------------------------------------------------------------------
PROVENANCE / IMPORTANT
The three source archives contained the pruning + eval code and a helper that
merely saved a processor into ``distilled_model80/checkpoint-7000``, but NOT the
distillation training loop itself (it was run in a Kaggle notebook that is not in
the archives). This module is a faithful REFERENCE reconstruction of the paper's
objective, written to be correct and runnable in principle. Validate on a small
subset before committing to a full run, and treat the hyper-parameters as
starting points rather than the authors' exact settings.
------------------------------------------------------------------------------

Example:
    python -m litedoc.distill \
        --teacher_path deepseek-ai/deepseek-vl2-tiny \
        --student_path checkpoints/litedoc-20-pruned \
        --train_jsonl  data/docvqa/train.jsonl \
        --image_root   data/docvqa/images \
        --output_dir   checkpoints/litedoc-20-distilled \
        --epochs 1 --batch_size 1 --grad_accum 8 --lr 1e-5
"""
from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import List

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM

from deepseek_vl2.models import DeepseekVLV2ForCausalLM, DeepseekVLV2Processor
from litedoc import config

try:
    from PIL import Image
except ImportError:  # pragma: no cover
    Image = None


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
@dataclass
class Sample:
    image: str
    prompt: str
    answer: str


class DocDistillDataset(Dataset):
    """Reads a JSONL of {"image", "prompt"/"question", "answer"} records and
    turns each into processor-ready multimodal inputs. Only the training split
    of a benchmark should be used here (no test leakage)."""

    def __init__(self, jsonl_path: str, image_root: str,
                 processor: DeepseekVLV2Processor, max_len: int = 2048):
        self.samples: List[Sample] = []
        with open(jsonl_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                self.samples.append(Sample(
                    image=r["image"],
                    prompt=r.get("prompt", r.get("question", "")),
                    answer=r.get("answer", (r.get("answers") or [""])[0]),
                ))
        self.image_root = image_root
        self.processor = processor
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int):
        s = self.samples[i]
        img_path = os.path.join(self.image_root, s.image)
        pil = Image.open(img_path).convert("RGB")
        convo = [
            {"role": "<|User|>", "content": f"<image>\n{s.prompt}", "images": [img_path]},
            {"role": "<|Assistant|>", "content": s.answer},
        ]
        inputs = self.processor(
            conversations=convo, images=[pil],
            force_batchify=True, system_prompt="",
        )
        return inputs


def _collate(batch):
    # force_batchify already returns a batched, model-ready object per item;
    # with batch_size 1 (recommended for these variable-resolution inputs) we
    # simply unwrap it. For larger batches, pad upstream in the processor.
    return batch[0]


# --------------------------------------------------------------------------- #
# Losses
# --------------------------------------------------------------------------- #
def hidden_state_loss(student_hs, teacher_hs) -> torch.Tensor:
    """L_hidden: mean per-layer MSE between aligned hidden states (Eqs. 4-5).
    `*_hs` are tuples of [B, T, D] tensors, one per layer (+ embeddings)."""
    n = min(len(student_hs), len(teacher_hs))
    loss = student_hs[0].new_zeros(())
    for i in range(n):
        loss = loss + F.mse_loss(student_hs[i].float(), teacher_hs[i].float())
    return loss / n


def kl_loss(student_logits, teacher_logits, temperature: float = 1.0) -> torch.Tensor:
    """L_KL: KL(student || teacher) over the vocabulary (Eq. 6), temperature-scaled."""
    s = F.log_softmax(student_logits.float() / temperature, dim=-1)
    t = F.softmax(teacher_logits.float() / temperature, dim=-1)
    return F.kl_div(s, t, reduction="batchmean") * (temperature ** 2)


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
def _forward(model, inputs, dtype):
    inputs = inputs.to(model.device, dtype=dtype)
    inputs_embeds = model.prepare_inputs_embeds(**inputs)
    out = model.language.model(
        inputs_embeds=inputs_embeds,
        attention_mask=inputs.attention_mask,
        output_hidden_states=True,
        use_cache=False,
        return_dict=True,
    )
    logits = model.language.lm_head(out.last_hidden_state)
    return logits, out.hidden_states


def train(args) -> None:
    dtype = torch.bfloat16
    print(f"Loading teacher (frozen): {args.teacher_path}")
    processor = DeepseekVLV2Processor.from_pretrained(args.student_path)
    teacher: DeepseekVLV2ForCausalLM = AutoModelForCausalLM.from_pretrained(
        args.teacher_path, trust_remote_code=True, torch_dtype=dtype).cuda().eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    print(f"Loading student (trainable): {args.student_path}")
    student: DeepseekVLV2ForCausalLM = AutoModelForCausalLM.from_pretrained(
        args.student_path, trust_remote_code=True, torch_dtype=dtype).cuda().train()

    ds = DocDistillDataset(args.train_jsonl, args.image_root, processor)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                    collate_fn=_collate, num_workers=args.num_workers)

    optim = torch.optim.AdamW(
        [p for p in student.parameters() if p.requires_grad], lr=args.lr,
        weight_decay=args.weight_decay)
    total_steps = (len(dl) // args.grad_accum) * args.epochs
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=max(total_steps, 1))

    os.makedirs(args.output_dir, exist_ok=True)
    step = 0
    for epoch in range(args.epochs):
        for it, inputs in enumerate(dl):
            with torch.no_grad():
                t_logits, t_hs = _forward(teacher, inputs, dtype)
            s_logits, s_hs = _forward(student, inputs, dtype)

            l_kl = kl_loss(s_logits, t_logits, temperature=args.temperature)
            l_hidden = hidden_state_loss(s_hs, t_hs)
            loss = (args.alpha * l_kl + args.beta * l_hidden) / args.grad_accum
            loss.backward()

            if (it + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(student.parameters(), args.max_grad_norm)
                optim.step()
                sched.step()
                optim.zero_grad(set_to_none=True)
                step += 1
                if step % args.log_every == 0:
                    print(f"epoch {epoch} step {step}/{total_steps} "
                          f"L_KL={l_kl.item():.4f} L_hidden={l_hidden.item():.4f}")
                if args.save_every and step % args.save_every == 0:
                    ckpt = os.path.join(args.output_dir, f"checkpoint-{step}")
                    student.save_pretrained(ckpt)
                    processor.save_pretrained(ckpt)
                    print(f"[ckpt] saved {ckpt}")

    student.save_pretrained(args.output_dir)
    processor.save_pretrained(args.output_dir)
    print(f"Distillation complete -> {args.output_dir}")


def main() -> None:
    ap = argparse.ArgumentParser(description="LiteDoc Stage 2: S2S distillation")
    ap.add_argument("--teacher_path", default=config.MODEL)
    ap.add_argument("--student_path", required=True, help="pruned student from Stage 1b")
    ap.add_argument("--train_jsonl", required=True)
    ap.add_argument("--image_root", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--grad_accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--max_grad_norm", type=float, default=1.0)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--alpha", type=float, default=1.0, help="weight on L_KL")
    ap.add_argument("--beta", type=float, default=1.0, help="weight on L_hidden")
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--log_every", type=int, default=10)
    ap.add_argument("--save_every", type=int, default=1000)
    train(ap.parse_args())


if __name__ == "__main__":
    main()
