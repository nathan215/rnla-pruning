"""Shared Step 2.3 evaluation utilities for sequential pruning."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.leverage import exact_row_leverage_scores, row_leverage_scores


def load_step1_oracle_pass16(project_root: Path) -> float | None:
    """Optional Step 1 aggregate Pass@16 (best-of-16, no pruning)."""
    p = project_root / "results" / "step_1_oracle_200_summary.json"
    if not p.is_file():
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        merged = data.get("merged_pass_at_k_from_json_files") or {}
        v = merged.get("oracle_pass_at_16_reference_weighted")
        return float(v) if v is not None else None
    except Exception:
        return None


def default_batch_roots(project_root: Path) -> list[Path]:
    return [project_root / "results" / f"step_1_oracle_b{i}" for i in range(8)]


def load_oracle(pt_path: Path) -> dict[str, Any]:
    return torch.load(pt_path, map_location="cpu", weights_only=False)


def alive_branch_indices(oracle: dict[str, Any], cp: int) -> list[int]:
    am = oracle.get("alive_mask", {})
    if cp not in am:
        return []
    t = am[cp]
    if not isinstance(t, torch.Tensor):
        return []
    m = t.bool()
    return [i for i in range(m.numel()) if bool(m[i].item())]


def top_k_global(branch_ids: list[int], scores: list[float], k: int) -> list[int]:
    pairs = sorted(zip(branch_ids, scores), key=lambda x: x[1], reverse=True)
    return [b for b, _ in pairs[: min(k, len(pairs))]]


def sequential_rand_leverage(
    oracle: dict[str, Any],
    t1: int,
    t2: int,
    k_rank: int,
    proj_dim: int,
    seed1: int,
    seed2: int,
    correct: list[bool],
    *,
    n_ensemble: int = 1,
) -> float:
    alive1 = alive_branch_indices(oracle, t1)
    alive2 = set(alive_branch_indices(oracle, t2))
    if len(alive1) < 8 or not alive2:
        return float("nan")
    H1 = oracle["checkpoint_hidden_states"][t1]
    rows1 = torch.stack([H1[b] for b in alive1], dim=0)
    lev1 = torch.zeros(rows1.shape[0], dtype=torch.float32, device=rows1.device)
    for e in range(n_ensemble):
        lev1 = lev1 + row_leverage_scores(rows1, proj_dim, k_rank, seed1 + e * 50_021)
    lev1 = lev1 / float(n_ensemble)
    order1 = torch.argsort(lev1, descending=True).tolist()
    top8 = [alive1[j] for j in order1[:8]]
    subset = [b for b in top8 if b in alive2]
    if not subset:
        return 0.0
    H2 = oracle["checkpoint_hidden_states"][t2]
    rows2 = torch.stack([H2[b] for b in subset], dim=0)
    lev2 = torch.zeros(rows2.shape[0], dtype=torch.float32, device=rows2.device)
    for e in range(n_ensemble):
        lev2 = lev2 + row_leverage_scores(rows2, proj_dim, k_rank, seed2 + e * 50_021)
    lev2 = lev2 / float(n_ensemble)
    order2 = torch.argsort(lev2, descending=True).tolist()
    m_eff = min(4, len(subset))
    top4 = [subset[j] for j in order2[:m_eff]]
    return 1.0 if any(correct[b] for b in top4) else 0.0


def sequential_exact_leverage(
    oracle: dict[str, Any],
    t1: int,
    t2: int,
    k_rank: int,
    correct: list[bool],
) -> float:
    alive1 = alive_branch_indices(oracle, t1)
    alive2 = set(alive_branch_indices(oracle, t2))
    if len(alive1) < 8 or not alive2:
        return float("nan")
    H1 = oracle["checkpoint_hidden_states"][t1]
    rows1 = torch.stack([H1[b] for b in alive1], dim=0)
    lev1 = exact_row_leverage_scores(rows1, k_rank)
    order1 = torch.argsort(lev1, descending=True).tolist()
    top8 = [alive1[j] for j in order1[:8]]
    subset = [b for b in top8 if b in alive2]
    if not subset:
        return 0.0
    H2 = oracle["checkpoint_hidden_states"][t2]
    rows2 = torch.stack([H2[b] for b in subset], dim=0)
    lev2 = exact_row_leverage_scores(rows2, k_rank)
    order2 = torch.argsort(lev2, descending=True).tolist()
    m_eff = min(4, len(subset))
    top4 = [subset[j] for j in order2[:m_eff]]
    return 1.0 if any(correct[b] for b in top4) else 0.0


def sequential_logprob(
    oracle: dict[str, Any],
    t1: int,
    t2: int,
    correct: list[bool],
) -> float:
    clp1 = oracle.get("checkpoint_cumulative_logprob", {}).get(t1)
    clp2 = oracle.get("checkpoint_cumulative_logprob", {}).get(t2)
    if clp1 is None or clp2 is None:
        return float("nan")
    alive1 = alive_branch_indices(oracle, t1)
    alive2 = set(alive_branch_indices(oracle, t2))
    if len(alive1) < 8 or not alive2:
        return float("nan")
    scores1 = [float(clp1[b].item()) for b in alive1]
    top8 = top_k_global(alive1, scores1, 8)
    subset = [b for b in top8 if b in alive2]
    if not subset:
        return 0.0
    scores2 = [float(clp2[b].item()) for b in subset]
    top4 = top_k_global(subset, scores2, 4)
    return 1.0 if any(correct[b] for b in top4) else 0.0


def sequential_random(
    oracle: dict[str, Any],
    t1: int,
    t2: int,
    rng: random.Random,
    correct: list[bool],
) -> float:
    alive1 = alive_branch_indices(oracle, t1)
    alive2 = set(alive_branch_indices(oracle, t2))
    if len(alive1) < 8 or not alive2:
        return float("nan")
    top8 = rng.sample(alive1, 8)
    subset = [b for b in top8 if b in alive2]
    if not subset:
        return 0.0
    m_take = min(4, len(subset))
    top4 = rng.sample(subset, m_take)
    return 1.0 if any(correct[b] for b in top4) else 0.0


def sequential_random_expectation(
    oracle: dict[str, Any],
    t1: int,
    t2: int,
    base_seed: int,
    correct: list[bool],
    n_trials: int,
) -> float:
    xs: list[float] = []
    for t in range(n_trials):
        rng = random.Random(base_seed + t * 100_003)
        v = sequential_random(oracle, t1, t2, rng, correct)
        if v == v:
            xs.append(v)
    return float(np.mean(xs)) if xs else float("nan")


def oracle_pass4_exact_all_branches_at_t2(
    oracle: dict[str, Any],
    t2: int,
    k_rank: int,
    correct: list[bool],
) -> float:
    alive = alive_branch_indices(oracle, t2)
    if not alive:
        return float("nan")
    H = oracle["checkpoint_hidden_states"][t2]
    rows = torch.stack([H[b] for b in alive], dim=0)
    lev = exact_row_leverage_scores(rows, k_rank)
    order = torch.argsort(lev, descending=True).tolist()
    m_eff = min(4, len(alive))
    top = [alive[j] for j in order[:m_eff]]
    return 1.0 if any(correct[b] for b in top) else 0.0
