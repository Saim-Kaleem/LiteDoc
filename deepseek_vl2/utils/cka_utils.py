# deepseek_vl2/utils/cka_utils.py
"""
Centered Kernel Alignment (CKA) utilities for LiteDoc Stage-1 expert profiling.

CKA measures the functional similarity between two sets of activations in a way
that is invariant to (a) orthogonal transforms / neuron permutations and
(b) isotropic scaling. This is exactly what we need to detect *functionally
redundant* MoE experts, since weight-space metrics are unreliable under neuron
permutation invariance and routing-frequency is near-uniform due to the
load-balancing objective (see LiteDoc paper, Sec. 3.2).

Given expert output activations X, Y in R^{n x d} (n tokens, d = hidden dim):

    HSIC(K, L) = tr(K H L H) / (n - 1)^2                         (paper Eq. 1)
    CKA(K, L)  = HSIC(K, L) / sqrt(HSIC(K, K) * HSIC(L, L))      (paper Eq. 2)

where K, L are Gram matrices and H = I_n - (1/n) J_n is the centering matrix.

Two kernels are supported:
  * linear kernel  K = X X^T           -> `linear_cka`  (fast, feature-space form)
  * RBF kernel     K = exp(-gamma ||.||)-> `rbf_cka`    (paper default, median sigma)

`compute_layer_cka` builds the full [E, E] expert-similarity matrix for a layer,
`find_similar_experts` thresholds it into redundant pairs, and
`save_cka_heatmaps` renders the matrices used in the paper figures.
"""

from __future__ import annotations

import os
from typing import Dict, List, Tuple

import numpy as np
import torch


# --------------------------------------------------------------------------- #
# Linear (feature-space) CKA
# --------------------------------------------------------------------------- #
def linear_cka(X: torch.Tensor, Y: torch.Tensor, eps: float = 1e-8) -> float:
    """Linear CKA via the equivalent feature-space (Frobenius) formula::

        CKA = ||X_c^T Y_c||_F^2 / (||X_c^T X_c||_F * ||Y_c^T Y_c||_F)

    X, Y are [n_samples, feature_dim], mean-centred along the sample dim.
    Computed in double precision for numerical stability. Returns a float.
    """
    Xc = (X - X.mean(dim=0, keepdim=True)).double()
    Yc = (Y - Y.mean(dim=0, keepdim=True)).double()

    Kxy = Xc.T @ Yc
    Kxx = Xc.T @ Xc
    Kyy = Yc.T @ Yc

    num = torch.norm(Kxy, p="fro") ** 2
    den = torch.norm(Kxx, p="fro") * torch.norm(Kyy, p="fro")
    if den < eps:
        return 0.0
    return float(num / den)


# --------------------------------------------------------------------------- #
# RBF (Gaussian) kernel CKA  -- paper default
# --------------------------------------------------------------------------- #
def rbf_kernel(X: torch.Tensor, sigma: float | None = None) -> torch.Tensor:
    """RBF Gram matrix for X = [n_samples, feature_dim].

    If `sigma` is None it is chosen by the median heuristic over pairwise
    squared distances, which makes the kernel scale-adaptive per expert.
    """
    X = X.double()
    X_norm = (X ** 2).sum(dim=1, keepdim=True)
    dists = X_norm + X_norm.T - 2 * (X @ X.T)
    dists = dists.clamp_min(0)

    if sigma is None:
        d = dists.detach().cpu().numpy()
        pos = d[d > 0]
        sigma = float(np.sqrt(0.5 * np.median(pos))) if pos.size else 1.0
        sigma = sigma if sigma > 0 else 1.0

    gamma = 1.0 / (2 * sigma ** 2)
    return torch.exp(-gamma * dists)


def _hsic(K: torch.Tensor, L: torch.Tensor) -> torch.Tensor:
    """Hilbert-Schmidt Independence Criterion, HSIC(K, L) = tr(K H L H)/(n-1)^2."""
    n = K.size(0)
    H = torch.eye(n, dtype=K.dtype, device=K.device) - (1.0 / n)
    Kc = H @ K @ H
    Lc = H @ L @ H
    return torch.trace(Kc @ Lc) / ((n - 1) ** 2)


def rbf_cka(X: torch.Tensor, Y: torch.Tensor, sigma: float | None = None,
            eps: float = 1e-8) -> float:
    """Kernel CKA with RBF kernels (paper Eqs. 1-2). Returns a float."""
    K = rbf_kernel(X, sigma)
    L = rbf_kernel(Y, sigma)
    hsic_xy = _hsic(K, L)
    hsic_xx = _hsic(K, K)
    hsic_yy = _hsic(L, L)
    denom = torch.sqrt(hsic_xx * hsic_yy)
    if denom < eps:
        return 0.0
    return float(hsic_xy / denom)


# --------------------------------------------------------------------------- #
# Layer-level similarity matrix
# --------------------------------------------------------------------------- #
def compute_layer_cka(expert_acts: torch.Tensor, use_rbf: bool = True,
                      sigma: float | None = None) -> torch.Tensor:
    """Build the [E, E] pairwise CKA matrix for one MoE layer.

    Args:
        expert_acts: [n_tokens, n_experts, hidden_dim] tensor of expert outputs.
        use_rbf:     use RBF kernel CKA (paper default) if True, else linear CKA.
        sigma:       optional fixed RBF bandwidth; None -> median heuristic.

    Returns:
        [n_experts, n_experts] float tensor, symmetric, diag == 1.
    """
    n_tokens, n_experts, _ = expert_acts.shape
    sim = torch.zeros((n_experts, n_experts), dtype=torch.float32,
                      device=expert_acts.device)
    for i in range(n_experts):
        Xi = expert_acts[:, i, :].float()
        for j in range(i, n_experts):
            Yj = expert_acts[:, j, :].float()
            c = rbf_cka(Xi, Yj, sigma=sigma) if use_rbf else linear_cka(Xi, Yj)
            sim[i, j] = sim[j, i] = c
    return sim


def find_similar_experts(cka_matrices: Dict[int, torch.Tensor],
                         threshold: float = 0.7
                         ) -> Dict[int, List[Tuple[int, int, float]]]:
    """Threshold each layer's CKA matrix into redundant expert pairs.

    Returns dict[layer_idx] -> list of (i, j, similarity) with i < j and
    similarity >= threshold. These pairs seed the redundancy graph used by
    graph-based expert selection (paper Sec. 3.2).
    """
    similar: Dict[int, List[Tuple[int, int, float]]] = {}
    for layer_idx, mat in cka_matrices.items():
        if not isinstance(mat, torch.Tensor):
            mat = torch.as_tensor(mat)
        E = mat.size(0)
        pairs = [
            (i, j, float(mat[i, j]))
            for i in range(E)
            for j in range(i + 1, E)
            if float(mat[i, j]) >= threshold
        ]
        similar[layer_idx] = pairs
    return similar


def save_cka_heatmaps(cka_matrices: Dict[int, torch.Tensor],
                      threshold: float = 0.7,
                      output_dir: str = "cka_heatmaps",
                      cmap: str = "viridis",
                      figsize: Tuple[int, int] = (8, 6)) -> None:
    """Render one CKA heatmap per layer (paper Fig. 3), marking redundant pairs
    above `threshold` with red crosses."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(output_dir, exist_ok=True)
    for layer_idx, mat in cka_matrices.items():
        if isinstance(mat, torch.Tensor):
            mat = mat.detach().cpu().numpy()

        fig, ax = plt.subplots(figsize=figsize)
        cax = ax.imshow(mat, interpolation="nearest", aspect="equal", cmap=cmap)
        ax.set_title(f"Layer {layer_idx} Expert CKA")
        ax.set_xlabel("Expert index")
        ax.set_ylabel("Expert index")

        E = mat.shape[0]
        for i in range(E):
            for j in range(E):
                if i < j and mat[i, j] >= threshold:
                    ax.plot(j, i, "x", markersize=8, markeredgewidth=2,
                            color="red")

        fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)
        plt.tight_layout()
        fig.savefig(os.path.join(output_dir, f"cka_layer_{layer_idx}.png"))
        plt.close(fig)
    print(f"Saved CKA heatmaps to '{output_dir}' (pairs >= {threshold} marked).")
