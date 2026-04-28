"""Shared Step 2 I/O helpers for oracle artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def default_batch_roots(project_root: Path, n_batches: int = 8) -> list[Path]:
    """Default step-1 batch directories: results/step_1_oracle_b0..b{n_batches-1}."""
    return [project_root / "results" / f"step_1_oracle_b{i}" for i in range(n_batches)]


def load_oracle(pt_path: Path) -> dict[str, Any]:
    """Load one per-problem oracle_data.pt file."""
    return torch.load(pt_path, map_location="cpu", weights_only=False)


def alive_branch_indices(oracle: dict[str, Any], checkpoint: int) -> list[int]:
    """Return alive branch ids at a checkpoint from oracle alive_mask."""
    alive_map = oracle.get("alive_mask", {})
    if checkpoint not in alive_map:
        return []
    mask = alive_map[checkpoint]
    if not isinstance(mask, torch.Tensor):
        return []
    alive = mask.bool()
    return [i for i in range(alive.numel()) if bool(alive[i].item())]
