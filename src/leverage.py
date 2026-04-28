"""Leverage scores: random projection + truncated left singular vectors (float32 SVD)."""

from __future__ import annotations

import torch


def row_leverage_scores(
    X: torch.Tensor,
    proj_dim: int,
    k: int,
    seed: int,
) -> torch.Tensor:
    """
    X: (N, hidden_dim) — one row per branch.
    Returns (N,) nonnegative scores (higher = more "leveraged" in the top-k subspace).
    """
    device = X.device
    N, H = X.shape
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    P = torch.randn(H, proj_dim, generator=g, dtype=torch.float32)
    P = P.to(device)
    Xf = X.to(torch.float32) @ P
    U, S, Vh = torch.linalg.svd(Xf, full_matrices=False)
    kk = min(k, U.shape[1])
    Uk = U[:, :kk]
    return (Uk**2).sum(dim=1)


def exact_row_leverage_scores(X: torch.Tensor, k: int) -> torch.Tensor:
    """
    Exact leverage from full-dimension SVD (no random projection).

    X: (N, hidden_dim) — one row per branch.
    Returns (N,) nonnegative scores: row norms of top-k left singular vectors.
    """
    Xf = X.to(torch.float32)
    U, _S, _Vh = torch.linalg.svd(Xf, full_matrices=False)
    rank = U.shape[1]
    kk = min(k, rank)
    Uk = U[:, :kk]
    return (Uk**2).sum(dim=1)
