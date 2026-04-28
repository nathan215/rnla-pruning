"""Load or build per-branch correctness flags for Step 2 post-hoc analysis."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

from src.answer_matching import answers_equal, get_reference_answer, safe_problem_id
from src.data_loader import load_jsonl

MatchMode = Literal["boxed_only", "legacy_last_line"]


def batch_dir_for_global_index(global_idx: int, batch_size: int = 25) -> tuple[int, int]:
    """Return (batch_id, offset_within_batch) for step_1_oracle_b{batch_id}."""
    batch_id = global_idx // batch_size
    offset = global_idx % batch_size
    return batch_id, offset


def per_problem_dir(
    batch_root: Path,
    row: dict[str, Any],
    global_idx: int,
) -> Path:
    pid = str(row.get("unique_id", f"problem_{global_idx:04d}"))
    pid_safe = safe_problem_id(pid)
    return batch_root / "per_problem" / pid_safe


def load_branch_correctness_cache(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def build_branch_correctness(
    *,
    data_path: Path,
    batch_roots: list[Path],
    match_mode: MatchMode,
    limit: int,
    batch_size: int,
    n_branches: int,
) -> dict[int, list[bool]]:
    """
    Returns mapping global_index -> list of n_branches booleans.
    Missing branch files are treated as incorrect.
    """
    rows = load_jsonl(data_path, limit=limit)
    out: dict[int, list[bool]] = {}
    for global_idx, row in enumerate(rows):
        batch_id, _ = batch_dir_for_global_index(global_idx, batch_size)
        if batch_id >= len(batch_roots):
            break
        root = batch_roots[batch_id]
        ref = get_reference_answer(row)
        pdir = per_problem_dir(root, row, global_idx)
        sol_dir = pdir / "solution_texts"
        flags: list[bool] = []
        if ref is None:
            flags = [False] * n_branches
            out[global_idx] = flags
            continue
        for b in range(n_branches):
            fpath = sol_dir / f"branch_{b}.txt"
            if not fpath.is_file():
                flags.append(False)
                continue
            text = fpath.read_text(encoding="utf-8")
            flags.append(answers_equal(text, ref, match_mode=match_mode))
        out[global_idx] = flags
    return out


def save_branch_correctness_cache(
    path: Path,
    *,
    match_mode: MatchMode,
    branch_correct: dict[int, list[bool]],
    meta: dict[str, Any] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "match_mode": match_mode,
        "num_problems": len(branch_correct),
        "branch_correct": {str(k): v for k, v in sorted(branch_correct.items())},
    }
    if meta:
        payload["meta"] = meta
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_branch_correctness_from_cache(path: Path, match_mode: MatchMode) -> dict[int, list[bool]] | None:
    raw = load_branch_correctness_cache(path)
    if raw is None:
        return None
    if raw.get("match_mode") != match_mode:
        return None
    bc = raw.get("branch_correct")
    if not isinstance(bc, dict):
        return None
    out: dict[int, list[bool]] = {}
    for k, v in bc.items():
        try:
            idx = int(k)
        except ValueError:
            continue
        if isinstance(v, list) and all(isinstance(x, bool) for x in v):
            out[idx] = list(v)
    return out if out else None


def get_or_build_branch_correctness(
    *,
    cache_path: Path | None,
    data_path: Path,
    batch_roots: list[Path],
    match_mode: MatchMode,
    limit: int,
    batch_size: int,
    n_branches: int,
    force_rebuild: bool,
) -> dict[int, list[bool]]:
    if cache_path and not force_rebuild:
        loaded = load_branch_correctness_from_cache(cache_path, match_mode)
        if loaded is not None and all(i in loaded for i in range(limit)):
            return {i: loaded[i] for i in range(limit)}
    bc = build_branch_correctness(
        data_path=data_path,
        batch_roots=batch_roots,
        match_mode=match_mode,
        limit=limit,
        batch_size=batch_size,
        n_branches=n_branches,
    )
    if cache_path:
        save_branch_correctness_cache(
            cache_path,
            match_mode=match_mode,
            branch_correct=bc,
            meta={"data_path": str(data_path), "batch_roots": [str(p) for p in batch_roots]},
        )
    return bc
