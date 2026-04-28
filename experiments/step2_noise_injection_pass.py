"""
Small noise-injection sanity check for Step 2 (reviewer / advisor concern).

Question: Does randomized leverage act as a "geometric filter" that beats exact
under additive noise? If not, we should only claim approximation + scalability.

Protocol (per problem, fixed checkpoint):
  - Take alive-branch hidden rows H ∈ R^{N_a × D}.
  - Add H' = H + η · s · Ξ with Ξ_ij ~ N(0,1), s = mean row L2 norm (η dimensionless).
  - Policy Pass@4: top-4 among alive by exact vs by randomized leverage on H'
    (random: mean over ``num_pass_draws`` projection seeds per trial).

Variance reduction:
  - ``--noise-trials`` (default 30): independent noise draws (η>0). Each trial
    yields one dataset-level Pass@4; we report mean ± SE across trials.
  - For η=0, noise is fixed; random Pass@4 still varies across trials because
    we shift the projection-seed block per trial (Monte Carlo over R).

Outputs:
  results/step_2_2/noise_injection_pass.json
  results/step_2_2/noise_injection_pass.md

Run (recommended):
  python experiments/step2_noise_injection_pass.py --checkpoint 300 --limit 200 \\
    --noise-trials 40 --num-pass-draws 64
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import defaults as D  # noqa: E402
from src.data_loader import load_jsonl  # noqa: E402
from src.leverage import exact_row_leverage_scores, row_leverage_scores  # noqa: E402
from src.step2_branch_correctness import (  # noqa: E402
    batch_dir_for_global_index,
    get_or_build_branch_correctness,
    per_problem_dir,
)
from src.step2_io import alive_branch_indices, load_oracle  # noqa: E402


def pass_top_m_exact(
    rows: torch.Tensor,
    alive: list[int],
    correct: list[bool],
    m: int,
    k: int,
) -> float:
    n = rows.shape[0]
    m_eff = min(m, n)
    if m_eff == 0:
        return float("nan")
    lev = exact_row_leverage_scores(rows, k)
    order = torch.argsort(lev, descending=True).tolist()
    top = [alive[j] for j in order[:m_eff]]
    return 1.0 if any(correct[j] for j in top if j < len(correct)) else 0.0


def pass_top_m_random_mean(
    rows: torch.Tensor,
    alive: list[int],
    correct: list[bool],
    m: int,
    k: int,
    proj_dim: int,
    num_draws: int,
    seed0: int,
) -> float:
    n = rows.shape[0]
    m_eff = min(m, n)
    if m_eff == 0:
        return float("nan")
    hits = 0
    for r in range(num_draws):
        lev = row_leverage_scores(rows, proj_dim, k, seed0 + r * 1_000_003)
        order = torch.argsort(lev, descending=True).tolist()
        top = [alive[j] for j in order[:m_eff]]
        if any(correct[j] for j in top if j < len(correct)):
            hits += 1
    return hits / float(num_draws)


def add_noise(
    rows: torch.Tensor,
    eta: float,
    rng: torch.Generator,
) -> torch.Tensor:
    """H' = H + (eta * s) * Xi, s = mean_i ||h_i||_2."""
    rows = rows.to(torch.float32)
    s = float(rows.norm(dim=1).mean().clamp_min(1e-8))
    sigma = eta * s
    noise = torch.randn(rows.shape, generator=rng, dtype=torch.float32)
    return rows + sigma * noise


@dataclass
class _Case:
    global_idx: int
    rows0: torch.Tensor
    alive: list[int]
    correct: list[bool]


def _mean_std_se(xs: list[float]) -> tuple[float, float, float]:
    if not xs:
        return float("nan"), float("nan"), float("nan")
    mu = float(statistics.fmean(xs))
    if len(xs) < 2:
        return mu, 0.0, 0.0
    sd = float(statistics.pstdev(xs))
    se = sd / math.sqrt(len(xs))
    return mu, sd, se


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=PROJECT_ROOT / D.DATA_REL_PATH)
    parser.add_argument("--results-root", type=Path, default=PROJECT_ROOT / "results")
    parser.add_argument("--checkpoint", type=int, default=300)
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--k", type=int, default=D.LEVERAGE_SVD_RANK)
    parser.add_argument("--proj-dim", type=int, default=D.LEVERAGE_PROJ_DIM)
    parser.add_argument(
        "--num-pass-draws",
        type=int,
        default=64,
        help="Projection seeds averaged per problem per trial (higher = lower MC noise).",
    )
    parser.add_argument(
        "--noise-trials",
        type=int,
        default=30,
        help="Independent noise replicates (η>0); for η=0, shifts projection-seed block per trial.",
    )
    parser.add_argument(
        "--etas",
        type=str,
        default="0,0.02,0.05,0.1,0.2",
        help="Comma-separated noise levels η in H' = H + η·(mean row L2)·N(0,1).",
    )
    parser.add_argument("--noise-seed", type=int, default=20260407)
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
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable tqdm progress bars.",
    )
    args = parser.parse_args()

    etas = [float(x.strip()) for x in args.etas.split(",") if x.strip()]
    batch_roots = [args.results_root / f"step_1_oracle_b{i}" for i in range(8)]
    rows_jsonl = load_jsonl(args.data, limit=args.limit)
    branch_correct = get_or_build_branch_correctness(
        cache_path=args.cache,
        data_path=args.data,
        batch_roots=batch_roots,
        match_mode=args.match_mode,
        limit=args.limit,
        batch_size=25,
        n_branches=D.N_BRANCHES,
        force_rebuild=args.force_rebuild_cache,
    )

    cases: list[_Case] = []
    for global_idx, row in enumerate(rows_jsonl):
        if global_idx >= args.limit:
            break
        bc = branch_correct.get(global_idx)
        if bc is None or len(bc) != D.N_BRANCHES:
            continue
        bid, _ = batch_dir_for_global_index(global_idx, 25)
        if bid >= len(batch_roots):
            continue
        pdir = per_problem_dir(batch_roots[bid], row, global_idx)
        pt_path = pdir / "oracle_data.pt"
        if not pt_path.is_file():
            continue
        oracle = load_oracle(pt_path)
        cp = args.checkpoint
        if cp not in oracle.get("checkpoint_hidden_states", {}):
            continue
        alive = alive_branch_indices(oracle, cp)
        if not alive:
            continue
        H_full = oracle["checkpoint_hidden_states"][cp]
        rows0 = torch.stack([H_full[b] for b in alive], dim=0).to(torch.float32)
        cases.append(_Case(global_idx=global_idx, rows0=rows0, alive=alive, correct=bc))

    out_dir = args.results_root / "step_2_2"
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "noise_injection_pass.json"
    md_path = out_dir / "noise_injection_pass.md"

    m = 4
    n_cases = len(cases)
    summary_rows: list[dict[str, Any]] = []

    eta_iter = tqdm(etas, desc="noise η", disable=args.no_progress)
    for eta in eta_iter:
        trial_exact: list[float] = []
        trial_rand: list[float] = []
        trial_delta_pp: list[float] = []

        if eta == 0:
            # Exact ranking is deterministic; only projection MC needs repeats.
            ev_one: list[float] = []
            for c in cases:
                pe = pass_top_m_exact(c.rows0, c.alive, c.correct, m, args.k)
                if pe == pe:
                    ev_one.append(pe)
            if not ev_one:
                continue
            exact_once = float(np.mean(ev_one))
            trial_iter = tqdm(
                range(args.noise_trials),
                desc=f"  trials η=0",
                leave=False,
                disable=args.no_progress,
            )
            for trial in trial_iter:
                rv: list[float] = []
                for c in cases:
                    seed0 = 900_000 + c.global_idx + args.checkpoint + trial * 499_979
                    pr = pass_top_m_random_mean(
                        c.rows0,
                        c.alive,
                        c.correct,
                        m,
                        args.k,
                        args.proj_dim,
                        args.num_pass_draws,
                        seed0=seed0,
                    )
                    if pr == pr:
                        rv.append(pr)
                if not rv:
                    continue
                mr = float(np.mean(rv))
                trial_exact.append(exact_once)
                trial_rand.append(mr)
                trial_delta_pp.append((mr - exact_once) * 100.0)
        else:
            trial_iter = tqdm(
                range(args.noise_trials),
                desc=f"  trials η={eta}",
                leave=False,
                disable=args.no_progress,
            )
            for trial in trial_iter:
                ev: list[float] = []
                rv: list[float] = []
                for c in cases:
                    g = torch.Generator(device="cpu")
                    g.manual_seed(
                        args.noise_seed
                        + trial * 834_437
                        + int(eta * 1_000_000) % 2_000_000_000
                        + c.global_idx * 17
                    )
                    rows_noisy = add_noise(c.rows0, eta, g)

                    pe = pass_top_m_exact(rows_noisy, c.alive, c.correct, m, args.k)
                    seed0 = (
                        900_000
                        + c.global_idx
                        + args.checkpoint
                        + trial * 499_979
                    )
                    pr = pass_top_m_random_mean(
                        rows_noisy,
                        c.alive,
                        c.correct,
                        m,
                        args.k,
                        args.proj_dim,
                        args.num_pass_draws,
                        seed0=seed0,
                    )
                    if pe == pe and pr == pr:
                        ev.append(pe)
                        rv.append(pr)

                if not ev:
                    continue
                me = float(np.mean(ev))
                mr = float(np.mean(rv))
                trial_exact.append(me)
                trial_rand.append(mr)
                trial_delta_pp.append((mr - me) * 100.0)

        e_mean, e_std, e_se = _mean_std_se(trial_exact)
        r_mean, r_std, r_se = _mean_std_se(trial_rand)
        d_mean, d_std, d_se = _mean_std_se(trial_delta_pp)

        summary_rows.append(
            {
                "eta_noise_scale": eta,
                "definition": "H' = H + η · (mean_i ||h_i||_2) · Ξ, Ξ_ij ~ N(0,1)",
                "n_problem_checkpoints": n_cases,
                "noise_trials": args.noise_trials,
                "num_pass_draws_per_trial": args.num_pass_draws,
                "pass_at_4_exact_mean": e_mean,
                "pass_at_4_exact_std_across_trials": e_std,
                "pass_at_4_exact_se": e_se,
                "pass_at_4_random_mean": r_mean,
                "pass_at_4_random_std_across_trials": r_std,
                "pass_at_4_random_se": r_se,
                "random_minus_exact_pp_mean": d_mean,
                "random_minus_exact_pp_std_across_trials": d_std,
                "random_minus_exact_pp_se": d_se,
            }
        )

    doc = {
        "checkpoint": args.checkpoint,
        "limit_requested": args.limit,
        "n_cases_loaded": n_cases,
        "k": args.k,
        "proj_dim": args.proj_dim,
        "num_pass_draws": args.num_pass_draws,
        "noise_trials": args.noise_trials,
        "etas": etas,
        "by_eta": summary_rows,
        "interpretation": (
            "If randomized were a superior noise filter, random Pass@4 would degrade "
            "more slowly than exact as η increases. Typically both drop; if random drops "
            "faster or tracks exact, do not claim denoising—only JL approximation + cost."
        ),
    }
    json_path.write_text(json.dumps(doc, indent=2), encoding="utf-8")

    lines = [
        "# Noise injection — Pass@4 (exact vs randomized leverage)",
        "",
        f"Checkpoint **{args.checkpoint}**, **{n_cases}** problems, top-{m} among alive, "
        f"`proj_dim={args.proj_dim}`, `k={args.k}`.",
        "",
        f"- **Projection MC:** {args.num_pass_draws} seeds averaged per problem per trial.",
        f"- **Trials:** {args.noise_trials} independent runs (new noise for η>0; η=0 varies projection seeds only).",
        f"- **SE:** standard error of the *trial-level* mean Pass@4 (std / sqrt(trials)).",
        "",
        "| η | Pass@4 exact | Pass@4 random | random−exact (pp) |",
        "|---|---:|---:|---:|",
    ]
    for r in summary_rows:
        lines.append(
            f"| {r['eta_noise_scale']} | "
            f"{r['pass_at_4_exact_mean']*100:.2f} ± {r['pass_at_4_exact_se']*100:.2f}\\% | "
            f"{r['pass_at_4_random_mean']*100:.2f} ± {r['pass_at_4_random_se']*100:.2f}\\% | "
            f"{r['random_minus_exact_pp_mean']:+.2f} ± {r['random_minus_exact_pp_se']:.2f} |"
        )
    lines += ["", doc["interpretation"], ""]
    md_path.write_text("\n".join(lines), encoding="utf-8")

    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")
    print(
        f"cases={n_cases}, noise_trials={args.noise_trials}, num_pass_draws={args.num_pass_draws}"
    )
    for r in summary_rows:
        print(
            f"eta={r['eta_noise_scale']}: exact={r['pass_at_4_exact_mean']*100:.2f}"
            f"±{r['pass_at_4_exact_se']*100:.2f}% "
            f"random={r['pass_at_4_random_mean']*100:.2f}±{r['pass_at_4_random_se']*100:.2f}% "
            f"delta={r['random_minus_exact_pp_mean']:+.2f}±{r['random_minus_exact_pp_se']:.2f}pp"
        )


if __name__ == "__main__":
    main()
