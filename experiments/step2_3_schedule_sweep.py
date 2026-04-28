"""
Part 3 schedule sweep: sequential pruning Pass@4 rates across (T1, T2) pairs.

Outputs rates only (relative VRAM vs full-16 varies with schedule; omit per row).
Always includes pool-16 exact leverage @ T2 (reference) and sequential policies.

Example:
  python experiments/step2_3_schedule_sweep.py --limit 200
  python experiments/step2_3_schedule_sweep.py --t1-list 50 100 150 --t2-list 50 100 200 300 400 500
  python experiments/step2_3_schedule_sweep.py --limit 10 --no-progress   # CI / no tqdm
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import defaults as D  # noqa: E402
from src.data_loader import load_jsonl  # noqa: E402
from src.step2_branch_correctness import (  # noqa: E402
    batch_dir_for_global_index,
    get_or_build_branch_correctness,
    per_problem_dir,
)

from src.step2_3_eval import (  # noqa: E402
    default_batch_roots,
    load_oracle,
    load_step1_oracle_pass16,
    oracle_pass4_exact_all_branches_at_t2,
    sequential_exact_leverage,
    sequential_logprob,
    sequential_rand_leverage,
    sequential_random_expectation,
)


def valid_pairs(t1_list: list[int], t2_list: list[int]) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    for t1 in sorted(set(t1_list)):
        for t2 in sorted(set(t2_list)):
            if t1 < t2:
                pairs.append((t1, t2))
    return pairs


def run_one_schedule(
    *,
    rows: list[dict[str, Any]],
    branch_correct: dict[int, list[bool]],
    batch_roots: list[Path],
    limit: int,
    t1: int,
    t2: int,
    k: int,
    d: int,
    ensemble: int,
    random_trials: int,
    seed_base: int,
    horizon: int,
    progress: bool = True,
) -> dict[str, Any]:
    exact_p: list[float] = []
    rand_l_p: list[float] = []
    logprob_p: list[float] = []
    random_p: list[float] = []
    oracle_ub: list[float] = []
    pass16_flags: list[float] = []
    n_skip = 0

    n_max = min(limit, len(rows))
    idx_iter: range | tqdm = range(n_max)
    if progress:
        idx_iter = tqdm(
            range(n_max),
            desc=f"T1={t1} T2={t2}",
            unit="problem",
            leave=False,
        )
    for global_idx in idx_iter:
        row = rows[global_idx]
        bc = branch_correct.get(global_idx)
        if bc is None or len(bc) != D.N_BRANCHES:
            n_skip += 1
            continue
        pass16_flags.append(1.0 if any(bc) else 0.0)
        bid, _ = batch_dir_for_global_index(global_idx, 25)
        if bid >= len(batch_roots):
            n_skip += 1
            continue
        pdir = per_problem_dir(batch_roots[bid], row, global_idx)
        pt_path = pdir / "oracle_data.pt"
        if not pt_path.is_file():
            n_skip += 1
            continue
        oracle = load_oracle(pt_path)
        chs = oracle.get("checkpoint_hidden_states", {})
        if t1 not in chs or t2 not in chs:
            n_skip += 1
            continue

        s1 = seed_base + global_idx * 1_003
        s2 = seed_base + global_idx * 1_003 + 77_777

        o = oracle_pass4_exact_all_branches_at_t2(oracle, t2, k, bc)
        if o == o:
            oracle_ub.append(o)

        e = sequential_exact_leverage(oracle, t1, t2, k, bc)
        if e == e:
            exact_p.append(e)

        rl = sequential_rand_leverage(
            oracle, t1, t2, k, d, s1, s2, bc, n_ensemble=max(1, ensemble)
        )
        if rl == rl:
            rand_l_p.append(rl)

        lp = sequential_logprob(oracle, t1, t2, bc)
        if lp == lp:
            logprob_p.append(lp)

        rr = sequential_random_expectation(
            oracle, t1, t2, seed_base + global_idx * 17, bc, max(1, random_trials)
        )
        if rr == rr:
            random_p.append(rr)

    def mean(xs: list[float]) -> float | None:
        return float(np.mean(xs)) if xs else None

    return {
        "t1": t1,
        "t2": t2,
        "horizon": horizon,
        "oracle_pass_at_16_no_pruning_mean": mean(pass16_flags),
        "exact_leverage_pool16_at_t2_mean": mean(oracle_ub),
        "sequential_exact_leverage_mean": mean(exact_p),
        "sequential_rand_leverage_mean": mean(rand_l_p),
        "sequential_logprob_mean": mean(logprob_p),
        "sequential_random_mean": mean(random_p),
        "n_counted": len(exact_p),
        "n_skip_missing_checkpoint_or_oracle": n_skip,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=PROJECT_ROOT / D.DATA_REL_PATH)
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--batch_roots", type=Path, nargs="*", default=None)
    parser.add_argument(
        "--t1-list",
        type=int,
        nargs="*",
        default=[50, 100, 150],
        help="First prune depths (16->8)",
    )
    parser.add_argument(
        "--t2-list",
        type=int,
        nargs="*",
        default=[50, 100, 200, 300, 400, 500],
        help="Second prune depths (8->4); only pairs with T2>T1 are run",
    )
    parser.add_argument("--k", type=int, default=D.LEVERAGE_SVD_RANK)
    parser.add_argument("--d", type=int, default=D.LEVERAGE_PROJ_DIM)
    parser.add_argument("--ensemble", type=int, default=24)
    parser.add_argument("--random-trials", type=int, default=48)
    parser.add_argument("--seed-base", type=int, default=42_001)
    parser.add_argument("--horizon", type=int, default=D.STEP1_MAX_NEW_TOKENS)
    parser.add_argument(
        "--cache",
        type=Path,
        default=PROJECT_ROOT / "results" / "step_2_1_branch_correct.json",
    )
    parser.add_argument(
        "--out-json",
        type=Path,
        default=PROJECT_ROOT / "results" / "step_2_3" / "schedule_sweep.json",
    )
    parser.add_argument(
        "--out-md",
        type=Path,
        default=PROJECT_ROOT / "results" / "step_2_3" / "schedule_sweep.md",
    )
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=PROJECT_ROOT / "results" / "step_2_3" / "schedule_sweep.csv",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable tqdm bars (logs / CI)",
    )
    args = parser.parse_args()

    batch_roots = list(args.batch_roots) if args.batch_roots else default_batch_roots(PROJECT_ROOT)
    rows = load_jsonl(args.data, limit=args.limit)
    branch_correct = get_or_build_branch_correctness(
        cache_path=args.cache,
        data_path=args.data,
        batch_roots=batch_roots,
        match_mode="boxed_only",
        limit=args.limit,
        batch_size=25,
        n_branches=D.N_BRANCHES,
        force_rebuild=False,
    )

    pairs = valid_pairs(list(args.t1_list), list(args.t2_list))
    p16_step1 = load_step1_oracle_pass16(PROJECT_ROOT)

    sweep_rows: list[dict[str, Any]] = []
    pair_loop: list[tuple[int, int]] | tqdm = pairs
    if not args.no_progress:
        pair_loop = tqdm(pairs, desc="Schedule sweep", unit="schedule")
    for t1, t2 in pair_loop:
        if t2 > args.horizon:
            continue
        row = run_one_schedule(
            rows=rows,
            branch_correct=branch_correct,
            batch_roots=batch_roots,
            limit=args.limit,
            t1=t1,
            t2=t2,
            k=args.k,
            d=args.d,
            ensemble=args.ensemble,
            random_trials=args.random_trials,
            seed_base=args.seed_base,
            horizon=args.horizon,
            progress=not args.no_progress,
        )
        row["oracle_pass_at_16_from_step1_summary"] = p16_step1
        sweep_rows.append(row)

    out = {
        "t1_list": list(args.t1_list),
        "t2_list": list(args.t2_list),
        "k": args.k,
        "d_random": args.d,
        "ensemble_runs": args.ensemble,
        "random_trials_per_problem": args.random_trials,
        "limit": args.limit,
        "note": (
            "Rates only. exact_leverage_pool16_at_t2_mean = top-4 by exact leverage among all alive at T2 "
            "(not sequential). Relative VRAM vs full-16 depends on (T1,T2); omitted here."
        ),
        "rows": sweep_rows,
    }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(out, indent=2), encoding="utf-8")

    # Markdown: rates only
    lines = [
        "# Part 3 — Schedule sweep (Pass@4 rates)",
        "",
        f"Settings: $k={args.k}$, randomized $d={args.d}$, ensemble={args.ensemble}, random MC trials={args.random_trials}, $N={args.limit}$.",
        "",
        "| T1 | T2 | Oracle@16 | Pool-16 exact @T2 | Seq exact | Seq rand lev | Seq logprob | Seq random | n |",
        "|----|----|------------|-------------------|-----------|--------------|-------------|------------|---|",
    ]
    for r in sweep_rows:
        def pct(x: Any) -> str:
            if x is None:
                return "—"
            return f"{100.0 * float(x):.1f}%"

        lines.append(
            f"| {r['t1']} | {r['t2']} | {pct(r.get('oracle_pass_at_16_no_pruning_mean'))} | "
            f"{pct(r.get('exact_leverage_pool16_at_t2_mean'))} | {pct(r.get('sequential_exact_leverage_mean'))} | "
            f"{pct(r.get('sequential_rand_leverage_mean'))} | {pct(r.get('sequential_logprob_mean'))} | "
            f"{pct(r.get('sequential_random_mean'))} | {r.get('n_counted', '')} |"
        )
    lines.append("")
    args.out_md.write_text("\n".join(lines), encoding="utf-8")

    fieldnames = [
        "t1",
        "t2",
        "oracle_pass_at_16_no_pruning_mean",
        "exact_leverage_pool16_at_t2_mean",
        "sequential_exact_leverage_mean",
        "sequential_rand_leverage_mean",
        "sequential_logprob_mean",
        "sequential_random_mean",
        "n_counted",
    ]
    with args.out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in sweep_rows:
            w.writerow({k: r.get(k) for k in fieldnames})

    print(json.dumps({"n_schedules": len(sweep_rows), "json": str(args.out_json), "md": str(args.out_md), "csv": str(args.out_csv)}, indent=2))


if __name__ == "__main__":
    main()
