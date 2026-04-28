"""Metrics for Step 2.2: Kendall-τ, relative error, randomized leverage Oracle Pass@m."""

from __future__ import annotations

import torch
from scipy.stats import kendalltau

from src.leverage import row_leverage_scores


def omniscient_retention_pass(alive: list[int], correct: list[bool]) -> float:
    """
    Omniscient ceiling for the keep-m task (policy-free): success iff at least one *alive*
    branch is correct under final graded labels.

    Indexing: ``correct[i]`` is the boolean for **global** branch index ``i`` (same convention
    as ``oracle_pass_one`` / ``oracle_pass_randomized_top_m``). ``alive`` lists global branch
    indices that are still decoding at the checkpoint.

    Returns 1.0 if a correct branch could be placed in any retained set of size m_eff >= 1
    (i.e. ∃ correct among alive), else 0.0. NaN if no alive branches.
    """
    if not alive:
        return float("nan")
    return 1.0 if any(correct[b] for b in alive if 0 <= b < len(correct)) else 0.0


def kendall_tau_scores(a: torch.Tensor, b: torch.Tensor) -> float:
    """
    Kendall-τ correlation between two score vectors (same length, paired branches).
    Returns NaN if undefined (e.g. constant input).
    """
    x = a.detach().cpu().numpy().astype(float).ravel()
    y = b.detach().cpu().numpy().astype(float).ravel()
    if x.size == 0 or y.size == 0 or x.size != y.size:
        return float("nan")
    res = kendalltau(x, y)
    cor = res.correlation
    return float(cor) if cor == cor else float("nan")


def relative_l2_error(rand: torch.Tensor, exact: torch.Tensor, eps: float = 1e-12) -> float:
    """||rand - exact||_2 / ||exact||_2; NaN if exact norm ~0 and rand non-zero."""
    num = float(torch.norm(rand - exact, p=2).item())
    den = float(torch.norm(exact, p=2).item())
    if den < eps:
        return 0.0 if num < eps else float("nan")
    return num / den


def oracle_pass_randomized_top_m(
    *,
    alive: list[int],
    correct: list[bool],
    m: int,
    H_full: torch.Tensor,
    leverage_k: int,
    proj_dim: int,
    seed: int,
) -> float:
    """
    Policy retention Pass@m: top-m branches by randomized row_leverage_scores among alive;
    1 if any retained branch is correct.

    (Historically called "Oracle Pass@m" in scripts; the omniscient ceiling is
    ``omniscient_retention_pass`` — independent of leverage.)

    One random projection per call is high-variance for Pass@4. Table 3 in step2_2 averages
    many seeds per (problem, checkpoint, d); see --num_pass_draws in step2_2_approximation.py.
    """
    n_alive = len(alive)
    if n_alive == 0:
        return float("nan")
    m_eff = min(m, n_alive)
    rows = torch.stack([H_full[b] for b in alive], dim=0)
    lev = row_leverage_scores(rows, proj_dim, leverage_k, seed)
    order = torch.argsort(lev, descending=True).tolist()
    top = [alive[j] for j in order[:m_eff]]
    return 1.0 if any(correct[j] for j in top if j < len(correct)) else 0.0
