"""
LiteDoc — a two-stage compression pipeline for MoE vision-language models.

Stage 1 (task-aware expert pruning):
    litedoc.profile_cka  -> collect expert activations on DU data, compute
                            per-layer CKA redundancy matrices, emit cka_clusters.json
    litedoc.prune        -> remove/merge redundant experts, rebuild the router,
                            save a compact student checkpoint

Stage 2 (performance recovery):
    litedoc.distill      -> Sparse-to-Sparse knowledge distillation against the
                            frozen teacher (L_hidden MSE + L_KL)

Evaluation:
    litedoc.evaluate     -> unified benchmark inference + metrics (F1/ANLS/EM)

See README.md and REPRODUCE.md for the end-to-end workflow.
"""

__version__ = "1.0.0"
