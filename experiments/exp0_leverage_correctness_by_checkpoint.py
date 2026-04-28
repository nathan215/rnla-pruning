"""
Experiment 0 (phenomenon probe): leverage vs correctness across token checkpoints.

Uses existing Step 1 oracle cache (hidden states + alive_mask), exact leverage
(same k as Step 2), and branch correctness labels.

Per checkpoint, reports:
  - Pooled mean leverage (correct vs wrong branches).
  - Mean within-problem percentile rank of correct branches (50 = median leverage among alive).
  - When ≥1 correct branch alive: fraction where the single top-leverage alive branch is correct.
  - Same conditioning: **top-1 cumulative logprob** branch is correct (apples-to-apples vs top-1 leverage).
  - **Random baseline**: mean over problems of (#correct alive / #alive) = P(random uniform branch is correct).

Run (all Step~1 checkpoints from config):
  python experiments/exp0_leverage_correctness_by_checkpoint.py --limit 200 --all-step1-checkpoints

Outputs:
  results/step_2_2/exp0_leverage_vs_correctness_by_cp.json
  results/step_2_2/exp0_leverage_vs_correctness_by_cp.md
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.stats import rankdata

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import defaults as D  # noqa: E402
from src.data_loader import load_jsonl  # noqa: E402
from src.leverage import exact_row_leverage_scores  # noqa: E402
from src.step2_branch_correctness import (  # noqa: E402
    batch_dir_for_global_index,
    get_or_build_branch_correctness,
    per_problem_dir,
)
from src.step2_io import alive_branch_indices, load_oracle  # noqa: E402


def percentile_ranks_same_length(lev: np.ndarray) -> np.ndarray:
    """Per-row percentile in [0,100] among alive branches (tie-aware)."""
    n = len(lev)
    if n <= 1:
        return np.full_like(lev, 50.0, dtype=np.float64)
    r = rankdata(lev, method="average")
    return 100.0 * (r - 1.0) / (n - 1.0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=PROJECT_ROOT / D.DATA_REL_PATH)
    parser.add_argument("--results-root", type=Path, default=PROJECT_ROOT / "results")
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--k", type=int, default=D.LEVERAGE_SVD_RANK)
    parser.add_argument(
        "--checkpoints",
        type=int,
        nargs="+",
        default=None,
        help="Explicit list of token checkpoints. Default: 50,100,...,500 (8 cp).",
    )
    parser.add_argument(
        "--all-step1-checkpoints",
        action="store_true",
        help=f"Use every checkpoint from config (STEP1_CHECKPOINTS: {D.STEP1_CHECKPOINTS}).",
    )
    parser.add_argument(
        "--match_mode",
        choices=("boxed_only", "legacy_last_line"),
        default="boxed_only",
    )
    parser.add_argument(
        "--cache",
        type=Path,
        default=PROJECT_ROOT / "results" / "step_2_1_branch_correct.json",
    )
    parser.add_argument("--force-rebuild-cache", action="store_true")
    args = parser.parse_args()

    if args.all_step1_checkpoints:
        cps = list(D.STEP1_CHECKPOINTS)
    elif args.checkpoints is not None:
        cps = list(args.checkpoints)
    else:
        cps = [50, 100, 150, 200, 250, 300, 400, 500]

    batch_roots = [args.results_root / f"step_1_oracle_b{i}" for i in range(8)]
    rows = load_jsonl(args.data, limit=args.limit)
    bc_map = get_or_build_branch_correctness(
        cache_path=args.cache,
        data_path=args.data,
        batch_roots=batch_roots,
        match_mode=args.match_mode,
        limit=args.limit,
        batch_size=25,
        n_branches=D.N_BRANCHES,
        force_rebuild=args.force_rebuild_cache,
    )

    out_by_cp: dict[int, dict[str, Any]] = {}

    for cp in cps:
        lev_correct: list[float] = []
        lev_wrong: list[float] = []
        pct_correct: list[float] = []
        n_problems_with_correct = 0
        n_top1_lev_correct = 0
        n_top1_lp_correct = 0
        n_problems_with_lp = 0
        random_hit_sum = 0.0
        n_pairs = 0

        for global_idx, row in enumerate(rows):
            if global_idx >= args.limit:
                break
            bct = bc_map.get(global_idx)
            if bct is None or len(bct) != D.N_BRANCHES:
                continue
            bid, _ = batch_dir_for_global_index(global_idx, 25)
            if bid >= len(batch_roots):
                continue
            pdir = per_problem_dir(batch_roots[bid], row, global_idx)
            pt_path = pdir / "oracle_data.pt"
            if not pt_path.is_file():
                continue
            oracle = load_oracle(pt_path)
            if cp not in oracle.get("checkpoint_hidden_states", {}):
                continue
            alive = alive_branch_indices(oracle, cp)
            if len(alive) < 2:
                continue
            clp = oracle.get("checkpoint_cumulative_logprob", {}).get(cp)
            H_full = oracle["checkpoint_hidden_states"][cp]
            rows_m = torch.stack([H_full[b] for b in alive], dim=0).to(torch.float32)
            lev_t = exact_row_leverage_scores(rows_m, args.k)
            lev = lev_t.detach().cpu().numpy().astype(np.float64)
            labels = np.array([1 if bct[b] else 0 for b in alive], dtype=np.int64)
            pr = percentile_ranks_same_length(lev)

            for i, lab in enumerate(labels):
                n_pairs += 1
                if lab:
                    lev_correct.append(float(lev[i]))
                    pct_correct.append(float(pr[i]))
                else:
                    lev_wrong.append(float(lev[i]))

            if labels.sum() >= 1:
                n_problems_with_correct += 1
                n_c = int(labels.sum())
                n_a = len(alive)
                random_hit_sum += float(n_c) / float(n_a)

                j_lev = int(np.argmax(lev))
                if labels[j_lev] == 1:
                    n_top1_lev_correct += 1

                if clp is not None and isinstance(clp, torch.Tensor) and clp.numel() > max(alive):
                    lp_scores = np.array([float(clp[b].item()) for b in alive], dtype=np.float64)
                    j_lp = int(np.argmax(lp_scores))
                    n_problems_with_lp += 1
                    if labels[j_lp] == 1:
                        n_top1_lp_correct += 1

        mc = float(np.mean(lev_correct)) if lev_correct else float("nan")
        mw = float(np.mean(lev_wrong)) if lev_wrong else float("nan")
        mpct = float(np.mean(pct_correct)) if pct_correct else float("nan")
        top1_lev = (
            float(n_top1_lev_correct) / float(n_problems_with_correct)
            if n_problems_with_correct
            else float("nan")
        )
        top1_lp = (
            float(n_top1_lp_correct) / float(n_problems_with_lp)
            if n_problems_with_lp
            else float("nan")
        )
        random_baseline = (
            float(random_hit_sum) / float(n_problems_with_correct)
            if n_problems_with_correct
            else float("nan")
        )

        out_by_cp[cp] = {
            "checkpoint": cp,
            "k": args.k,
            "n_branch_slots": n_pairs,
            "mean_leverage_correct": mc,
            "mean_leverage_wrong": mw,
            "mean_leverage_correct_minus_wrong": mc - mw if mc == mc and mw == mw else float("nan"),
            "mean_percentile_rank_of_correct_branches": mpct,
            "fraction_problems_top1_leverage_is_correct": top1_lev,
            "fraction_problems_top1_logprob_is_correct": top1_lp,
            "n_problems_with_ge_one_correct_alive": n_problems_with_correct,
            "n_problems_used_for_top1_logprob": n_problems_with_lp,
            "mean_random_single_branch_hit_rate": random_baseline,
        }

    doc = {
        "model_setup": (
            "Step 1 oracle: exact_row_leverage_scores on alive rows; "
            "logprob = checkpoint_cumulative_logprob[cp][b] (higher better). "
            "Top-1 metrics: among problems with ≥1 correct alive branch."
        ),
        "limit_problems": args.limit,
        "checkpoints": cps,
        "by_checkpoint": {str(k): v for k, v in sorted(out_by_cp.items())},
    }

    out_dir = args.results_root / "step_2_2"
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "exp0_leverage_vs_correctness_by_cp.json"
    md_path = out_dir / "exp0_leverage_vs_correctness_by_cp.md"
    json_path.write_text(json.dumps(doc, indent=2), encoding="utf-8")

    lines = [
        "# Exp0: Exact leverage vs correctness (by checkpoint)",
        "",
        f"Problems: **{args.limit}**, `k={args.k}`, exact SVD leverage on **alive** rows at each $t$.",
        "",
        "**Top-1 columns:** among problems with **≥1 correct** alive branch — is the **single** top branch correct? "
        "(Compare leverage vs cumulative logprob; **random** = mean$_i$(\\#correct alive / \\#alive) on the same problems.)",
        "",
        "| cp | Δ lev | pct-rank(corr) | top-1 **lev** | top-1 **logprob** | random baseline |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for cp in sorted(out_by_cp.keys()):
        r = out_by_cp[cp]
        tlp = r["fraction_problems_top1_logprob_is_correct"]
        tlp_s = f"{tlp * 100:.1f}\\%" if tlp == tlp else "---"
        lines.append(
            f"| {cp} | {r['mean_leverage_correct_minus_wrong']:+.4f} | "
            f"{r['mean_percentile_rank_of_correct_branches']:.1f} | "
            f"{r['fraction_problems_top1_leverage_is_correct']*100:.1f}\\% | "
            f"{tlp_s} | {r['mean_random_single_branch_hit_rate']*100:.1f}\\% |"
        )
    lines.append("")
    lines.append(
        "**Read:** **Pass@2** in the paper is *not* top-1; it asks whether **either** of the top-2 (by logprob) is correct. "
        "Here **top-1 logprob** is the fair apples-to-apples comparison to **top-1 leverage**."
    )
    lines.append("")
    md_path.write_text("\n".join(lines), encoding="utf-8")

    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")
    for cp in sorted(out_by_cp.keys()):
        r = out_by_cp[cp]
        print(
            f"cp={cp}: top1_lev={r['fraction_problems_top1_leverage_is_correct']*100:.1f}% "
            f"top1_lp={r['fraction_problems_top1_logprob_is_correct']*100:.1f}% "
            f"random={r['mean_random_single_branch_hit_rate']*100:.1f}%"
        )


if __name__ == "__main__":
    main()
