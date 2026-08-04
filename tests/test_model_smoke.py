#!/usr/bin/env python3
"""
LiteDoc smoke test -- synthetic forward/backward + core invariants, no weights
or GPU required. Mirrors the cluster convention `python -m tests.test_model_smoke`.

Covers:
  1. MoE forward on a tiny mock config (shape-correct).
  2. The two LiteDoc modifications to the model: `valid_expert_mask` router and
     the `cka_tracking` activation hook.
  3. Prune-and-repair of the expert list + gate (shapes, forward, backward grads).
  4. CKA-util invariances the method relies on (identity=1, permutation- and
     scale-invariant, independent~small).
  5. The evaluation metrics (ANLS / F1 / EM) on hand-checked cases.

Run:  python -m tests.test_model_smoke   (exit 0 = all passed)
"""
from __future__ import annotations

import sys

import torch

from deepseek_vl2.models.modeling_deepseek import DeepseekV2MoE


class MockConfig:
    """Tiny MoE config -- small enough to build/run on CPU in a second."""
    def __init__(self):
        self.vocab_size = 100
        self.hidden_size = 128
        self.intermediate_size = 256
        self.moe_intermediate_size = 64
        self.num_hidden_layers = 1
        self.num_attention_heads = 4
        self.n_routed_experts = 8
        self.num_experts_per_tok = 2
        self.routed_scaling_factor = 1.0
        self.norm_topk_prob = False
        self.scoring_func = "softmax"
        self.aux_loss_alpha = 0.001
        self.seq_aux = True
        self.topk_method = "greedy"
        self.n_group = None
        self.topk_group = None
        self.hidden_act = "silu"
        self.n_shared_experts = None
        self.ep_size = 1


def test_moe_forward():
    moe = DeepseekV2MoE(MockConfig())
    x = torch.randn(1, 4, 128)
    y = moe(x)
    assert y.shape == (1, 4, 128), f"bad output shape {y.shape}"
    assert torch.isfinite(y).all(), "non-finite MoE output"


def test_valid_expert_mask():
    """The LiteDoc pruning-aware router exposes valid_expert_mask and stays finite
    when an expert is masked out."""
    moe = DeepseekV2MoE(MockConfig())
    assert hasattr(moe.gate, "valid_expert_mask"), "gate missing valid_expert_mask"
    assert moe.gate.valid_expert_mask.shape[0] == 8
    moe.gate.valid_expert_mask[3] = False          # disable one expert
    y = moe(torch.randn(1, 6, 128))
    assert torch.isfinite(y).all(), "masking an expert produced non-finite output"


def test_cka_tracking_hook():
    """The Stage-1 profiler relies on cka_tracking populating _expert_activations.

    Capture happens inside DeepseekV2MoE.moe_infer, which forward() only calls in
    EVAL mode (the training branch uses a separate per-expert loop with no hook).
    So we profile with the module in eval + no_grad -- exactly how
    litedoc.profile_cka runs the teacher during generation.
    """
    moe = DeepseekV2MoE(MockConfig()).eval()
    assert hasattr(moe, "cka_tracking"), "MoE missing cka_tracking flag"
    moe.cka_tracking = True
    moe._expert_activations = [None] * len(moe.experts)
    with torch.no_grad():
        moe(torch.randn(1, 8, 128))
    captured = [a for a in moe._expert_activations if a is not None and a.numel() > 0]
    assert captured, "cka_tracking did not capture any expert activations"


def test_prune_and_repair():
    """Reproduce the litedoc.prune router surgery on a mock layer and check a
    forward + backward pass works afterward."""
    moe = DeepseekV2MoE(MockConfig())
    n_old = 8
    keeping = [0, 1, 2, 3, 4, 5]
    merge_map = {6: 0, 7: 1}
    remap = {old: new for new, old in enumerate(keeping)}

    assert len(moe.experts) == n_old and moe.gate.weight.shape[0] == n_old

    new_gate = moe.gate.weight.data[keeping].clone()
    for src, tgt in merge_map.items():
        if tgt in keeping:
            new_gate[remap[tgt]] += moe.gate.weight.data[src]
    moe.experts = torch.nn.ModuleList([moe.experts[i] for i in keeping])
    moe.gate.n_routed_experts = len(keeping)
    moe.gate.weight = torch.nn.Parameter(new_gate)
    if hasattr(moe.gate, "valid_expert_mask"):
        moe.gate.valid_expert_mask = moe.gate.valid_expert_mask[keeping]

    assert len(moe.experts) == 6 and moe.gate.weight.shape[0] == 6

    x = torch.randn(1, 4, 128, requires_grad=True)
    y = moe(x)
    assert y.shape == (1, 4, 128)
    y.sum().backward()                              # gradients must flow
    assert x.grad is not None and torch.isfinite(x.grad).all(), "backward failed"


def test_cka_invariances():
    from deepseek_vl2.utils.cka_utils import linear_cka
    torch.manual_seed(0)
    X = torch.randn(200, 32)
    Q, _ = torch.linalg.qr(torch.randn(32, 32))     # orthogonal (permutation-like)
    assert abs(linear_cka(X, X) - 1.0) < 1e-4, "CKA(X,X) != 1"
    assert abs(linear_cka(X, X @ Q) - 1.0) < 1e-4, "CKA not orthogonal-invariant"
    assert abs(linear_cka(X, 3 * X) - 1.0) < 1e-4, "CKA not scale-invariant"
    assert linear_cka(X, torch.randn(200, 32)) < 0.5, "CKA(indep) too high"


def test_metrics():
    from litedoc.evaluate import anls, token_f1, exact_match
    assert anls("new york", ["New York"]) == 1.0
    assert anls("cat", ["elephant"]) == 0.0          # below 0.5 threshold -> 0
    assert token_f1("total 42", ["total 42"]) == 1.0
    assert 0.0 < token_f1("the total 42", ["total 42 usd"]) < 1.0
    assert exact_match("Invoice", ["invoice"]) == 1.0
    assert exact_match("letter", ["memo"]) == 0.0


TESTS = [
    test_moe_forward,
    test_valid_expert_mask,
    test_cka_tracking_hook,
    test_prune_and_repair,
    test_cka_invariances,
    test_metrics,
]


def main() -> int:
    failed = 0
    for t in TESTS:
        try:
            t()
            print(f"PASS  {t.__name__}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"FAIL  {t.__name__}: {e}")
    print(f"\n{len(TESTS) - failed}/{len(TESTS)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
