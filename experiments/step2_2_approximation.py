"""
Step 2.2: Randomized leverage vs exact — Kendall-τ, relative error, d ablation (Table 2–3).

Alive-branch policy matches Step 2.1 (see results/step_2_1/oracle_pass_m_policy.md).

Outputs:
  results/step_2_2/table2_stability.json / .md
  results/step_2_2/table3_ablation_d.json / .md
  results/step_2_2/run_config.json
  results/step_2_2/latex_table_snippets.md

Canonical final run:
  python experiments/step2_2_approximation.py --checkpoint 300 --limit 200 --mode all --num_pass_draws 50

Optional exploratory sweep (not needed for final report):
  python experiments/step2_2_approximation.py --step1-checkpoints --limit 200
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from collections import defaultdict
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
from src.step2_io import alive_branch_indices, default_batch_roots, load_oracle  # noqa: E402
from src.step2_metrics import (  # noqa: E402
    kendall_tau_scores,
    omniscient_retention_pass,
    oracle_pass_randomized_top_m,
    relative_l2_error,
)


def read_hero_checkpoint(project_root: Path) -> int | None:
    p = project_root / "results" / "step_2_1" / "hero_checkpoint_selection.json"
    if not p.is_file():
        return None
    try:
        return int(json.loads(p.read_text(encoding="utf-8"))["selected_checkpoint"])
    except Exception:
        return None


def run_table2_stability(
    *,
    rows: list[dict[str, Any]],
    branch_correct: dict[int, list[bool]],
    batch_roots: list[Path],
    checkpoints: list[int],
    limit: int,
    k: int,
    d_fixed: int,
    num_draws: int,
    seed_offset: int,
    run2_seed_delta: int,
    progress: bool = True,
) -> tuple[dict[str, Any], list[float], list[float], list[float]]:
    """
    Per (problem, cp, draw): rel error, τ(rand,exact), τ(rand1,rand2).
    Returns summary dict and raw lists for diagnostics.
    """
    rel_errs: list[float] = []
    tau_re: list[float] = []
    tau_rr: list[float] = []
    rel_by_cp: dict[int, list[float]] = defaultdict(list)
    tau_re_by_cp: dict[int, list[float]] = defaultdict(list)
    tau_rr_by_cp: dict[int, list[float]] = defaultdict(list)

    n_problems = min(limit, len(rows))
    for global_idx in tqdm(
        range(n_problems),
        desc="Table 2 stability",
        unit="problem",
        disable=not progress,
    ):
        row = rows[global_idx]
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

        for cp in checkpoints:
            if cp not in oracle.get("checkpoint_hidden_states", {}):
                continue
            alive = alive_branch_indices(oracle, cp)
            if not alive:
                continue
            H_full = oracle["checkpoint_hidden_states"][cp]
            rows_alive = torch.stack([H_full[b] for b in alive], dim=0)
            ell_exact = exact_row_leverage_scores(rows_alive, k)

            for r in range(num_draws):
                seed_r = seed_offset + global_idx * 100_000 + cp * 7 + r * 1_009
                ell_rand = row_leverage_scores(rows_alive, d_fixed, k, seed_r)
                re = relative_l2_error(ell_rand, ell_exact)
                if re == re:
                    rel_errs.append(re)
                    rel_by_cp[cp].append(re)
                tr = kendall_tau_scores(ell_rand, ell_exact)
                if tr == tr:
                    tau_re.append(tr)
                    tau_re_by_cp[cp].append(tr)

                seed_r2 = seed_r + run2_seed_delta
                ell_rand2 = row_leverage_scores(rows_alive, d_fixed, k, seed_r2)
                trr = kendall_tau_scores(ell_rand, ell_rand2)
                if trr == trr:
                    tau_rr.append(trr)
                    tau_rr_by_cp[cp].append(trr)

    # Use population std (ddof=0) for report tables
    st_rel = statistics.pstdev(rel_errs) if len(rel_errs) > 1 else 0.0
    st_tre = statistics.pstdev(tau_re) if len(tau_re) > 1 else 0.0
    st_trr = statistics.pstdev(tau_rr) if len(tau_rr) > 1 else 0.0

    def _mean_std(xs: list[float]) -> dict[str, float]:
        if not xs:
            return {"mean": float("nan"), "std": 0.0}
        st = statistics.pstdev(xs) if len(xs) > 1 else 0.0
        return {"mean": float(np.mean(xs)), "std": st}

    per_checkpoint: list[dict[str, Any]] = []
    for cp in sorted(set(rel_by_cp.keys()) | set(tau_re_by_cp.keys()) | set(tau_rr_by_cp.keys())):
        re_c = rel_by_cp.get(cp, [])
        tre_c = tau_re_by_cp.get(cp, [])
        trr_c = tau_rr_by_cp.get(cp, [])
        per_checkpoint.append(
            {
                "checkpoint": cp,
                "n_samples_rel_err": len(re_c),
                "n_samples_tau_re": len(tre_c),
                "n_samples_tau_rr": len(trr_c),
                "relative_l2_error": _mean_std(re_c),
                "kendall_tau_random_vs_exact": _mean_std(tre_c),
                "kendall_tau_random_vs_random": _mean_std(trr_c),
            }
        )

    summary = {
        "d": d_fixed,
        "k": k,
        "num_random_draws": num_draws,
        "checkpoints": checkpoints,
        "n_samples_rel_err": len(rel_errs),
        "n_samples_tau_re": len(tau_re),
        "n_samples_tau_rr": len(tau_rr),
        "relative_l2_error": {
            "mean": float(np.mean(rel_errs)) if rel_errs else float("nan"),
            "std": st_rel,
        },
        "kendall_tau_random_vs_exact": {
            "mean": float(np.mean(tau_re)) if tau_re else float("nan"),
            "std": st_tre,
        },
        "kendall_tau_random_vs_random": {
            "mean": float(np.mean(tau_rr)) if tau_rr else float("nan"),
            "std": st_trr,
        },
        "per_checkpoint": per_checkpoint,
    }
    return summary, rel_errs, tau_re, tau_rr


def benchmark_leverage_ms(
    rows_alive: torch.Tensor,
    proj_dim: int,
    k: int,
    seed: int,
    n_repeat: int,
    warmup: int,
) -> float:
    """Mean wall time per row_leverage_scores call (ms)."""
    for _ in range(warmup):
        row_leverage_scores(rows_alive, proj_dim, k, seed)
    t0 = time.perf_counter()
    for i in range(n_repeat):
        row_leverage_scores(rows_alive, proj_dim, k, seed + i)
    elapsed = time.perf_counter() - t0
    return (elapsed / max(n_repeat, 1)) * 1000.0


def run_table3_ablation(
    *,
    rows: list[dict[str, Any]],
    branch_correct: dict[int, list[bool]],
    batch_roots: list[Path],
    checkpoints: list[int],
    limit: int,
    k: int,
    d_values: list[int],
    num_draws_tau: int,
    num_pass_draws: int,
    seed_pass_base: int,
    seed_offset: int,
    timing_repeats: int,
    timing_warmup: int,
    progress: bool = True,
) -> dict[str, Any]:
    """Per d: mean Kendall vs exact, Oracle Pass@4 (averaged over num_pass_draws seeds), mean timing ms."""
    rows_out: list[dict[str, Any]] = []
    tau_by_cp_d: dict[tuple[int, int], list[float]] = defaultdict(list)
    pass_by_cp_d: dict[tuple[int, int], list[float]] = defaultdict(list)
    time_by_cp_d: dict[tuple[int, int], list[float]] = defaultdict(list)

    n_problems = min(limit, len(rows))
    omniscient_passes: list[float] = []
    for global_idx in range(n_problems):
        row = rows[global_idx]
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
        for cp in checkpoints:
            if cp not in oracle.get("checkpoint_hidden_states", {}):
                continue
            alive = alive_branch_indices(oracle, cp)
            if not alive:
                continue
            omni = omniscient_retention_pass(alive, bc)
            if omni == omni:
                omniscient_passes.append(omni)
    omniscient_mean = float(np.mean(omniscient_passes)) if omniscient_passes else float("nan")
    omniscient_n = len(omniscient_passes)
    for d in tqdm(d_values, desc="Table 3 ablation (d)", disable=not progress):
        taus: list[float] = []
        passes: list[float] = []
        times: list[float] = []

        for global_idx in tqdm(
            range(n_problems),
            desc=f"  problems d={d}",
            unit="problem",
            leave=False,
            disable=not progress,
        ):
            row = rows[global_idx]
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

            for cp in checkpoints:
                if cp not in oracle.get("checkpoint_hidden_states", {}):
                    continue
                alive = alive_branch_indices(oracle, cp)
                if not alive:
                    continue
                H_full = oracle["checkpoint_hidden_states"][cp]
                rows_alive = torch.stack([H_full[b] for b in alive], dim=0)
                ell_exact = exact_row_leverage_scores(rows_alive, k)

                for r in range(num_draws_tau):
                    seed_r = seed_offset + global_idx * 50_000 + cp * 3 + r * 503 + d
                    ell_rand = row_leverage_scores(rows_alive, d, k, seed_r)
                    tr = kendall_tau_scores(ell_rand, ell_exact)
                    if tr == tr:
                        taus.append(tr)
                        tau_by_cp_d[(cp, d)].append(tr)

                pass_samples: list[float] = []
                for r in range(num_pass_draws):
                    seed_p = seed_pass_base + d * 17 + global_idx + cp + r * 99_017
                    op = oracle_pass_randomized_top_m(
                        alive=alive,
                        correct=bc,
                        m=4,
                        H_full=H_full,
                        leverage_k=k,
                        proj_dim=d,
                        seed=seed_p,
                    )
                    if op == op:
                        pass_samples.append(op)
                if pass_samples:
                    pass_mean = float(np.mean(pass_samples))
                    passes.append(pass_mean)
                    pass_by_cp_d[(cp, d)].append(pass_mean)

                t_ms = benchmark_leverage_ms(
                    rows_alive, d, k, seed_p, timing_repeats, timing_warmup
                )
                times.append(t_ms)
                time_by_cp_d[(cp, d)].append(t_ms)

        pass_mean = float(np.mean(passes)) if passes else float("nan")
        rows_out.append(
            {
                "d": d,
                "label": f"d={d}",
                "mean_kendall_tau_vs_exact": float(np.mean(taus)) if taus else float("nan"),
                "std_kendall_tau_vs_exact": float(np.std(taus)) if len(taus) > 1 else 0.0,
                "oracle_pass_at_4_mean": pass_mean,
                "policy_pass_at_4_mean": pass_mean,
                "oracle_pass_at_4_n": len(passes),
                "mean_ms_per_leverage_call": float(np.mean(times)) if times else float("nan"),
            }
        )

    # Exact SVD row (D = hidden dim)
    exact_passes: list[float] = []
    exact_pass_by_cp: dict[int, list[float]] = defaultdict(list)
    for global_idx in tqdm(
        range(n_problems),
        desc="Table 3 exact Pass@4",
        unit="problem",
        disable=not progress,
    ):
        row = rows[global_idx]
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
        for cp in checkpoints:
            if cp not in oracle.get("checkpoint_hidden_states", {}):
                continue
            alive = alive_branch_indices(oracle, cp)
            if not alive:
                continue
            H_full = oracle["checkpoint_hidden_states"][cp]
            rows_alive = torch.stack([H_full[b] for b in alive], dim=0)
            lev = exact_row_leverage_scores(rows_alive, k)
            order = torch.argsort(lev, descending=True).tolist()
            m_eff = min(4, len(alive))
            top = [alive[j] for j in order[:m_eff]]
            v = 1.0 if any(bc[j] for j in top) else 0.0
            exact_passes.append(v)
            exact_pass_by_cp[cp].append(v)

    exact_timing: list[float] = []
    exact_timing_by_cp: dict[int, list[float]] = defaultdict(list)
    for global_idx in tqdm(
        range(n_problems),
        desc="Table 3 exact timing",
        unit="problem",
        disable=not progress,
    ):
        row = rows[global_idx]
        bc = branch_correct.get(global_idx)
        if bc is None:
            continue
        bid, _ = batch_dir_for_global_index(global_idx, 25)
        if bid >= len(batch_roots):
            continue
        pdir = per_problem_dir(batch_roots[bid], row, global_idx)
        pt_path = pdir / "oracle_data.pt"
        if not pt_path.is_file():
            continue
        oracle = load_oracle(pt_path)
        for cp in checkpoints:
            if cp not in oracle.get("checkpoint_hidden_states", {}):
                continue
            alive = alive_branch_indices(oracle, cp)
            if not alive:
                continue
            H_full = oracle["checkpoint_hidden_states"][cp]
            rows_alive = torch.stack([H_full[b] for b in alive], dim=0)
            for _ in range(3):
                exact_row_leverage_scores(rows_alive, k)
            t0 = time.perf_counter()
            for _ in range(timing_repeats):
                exact_row_leverage_scores(rows_alive, k)
            elapsed = time.perf_counter() - t0
            ms = (elapsed / timing_repeats) * 1000.0
            exact_timing.append(ms)
            exact_timing_by_cp[cp].append(ms)

    exact_pass_mean = float(np.mean(exact_passes)) if exact_passes else float("nan")
    rows_out.append(
        {
            "d": None,
            "label": "Exact SVD (D=1536)",
            "mean_kendall_tau_vs_exact": 1.0,
            "std_kendall_tau_vs_exact": 0.0,
            "oracle_pass_at_4_mean": exact_pass_mean,
            "policy_pass_at_4_mean": exact_pass_mean,
            "oracle_pass_at_4_n": len(exact_passes),
            "mean_ms_per_leverage_call": float(np.mean(exact_timing)) if exact_timing else float("nan"),
        }
    )

    all_cp = sorted(
        set(c for (c, _) in tau_by_cp_d.keys())
        | set(c for (c, _) in pass_by_cp_d.keys())
        | set(exact_pass_by_cp.keys())
    )
    per_checkpoint: list[dict[str, Any]] = []
    for cp in all_cp:
        subrows: list[dict[str, Any]] = []
        for d in d_values:
            key = (cp, d)
            ts = tau_by_cp_d.get(key, [])
            ps = pass_by_cp_d.get(key, [])
            tms = time_by_cp_d.get(key, [])
            pm = float(np.mean(ps)) if ps else float("nan")
            subrows.append(
                {
                    "d": d,
                    "label": f"d={d}",
                    "mean_kendall_tau_vs_exact": float(np.mean(ts)) if ts else float("nan"),
                    "std_kendall_tau_vs_exact": float(np.std(ts)) if len(ts) > 1 else 0.0,
                    "oracle_pass_at_4_mean": pm,
                    "policy_pass_at_4_mean": pm,
                    "oracle_pass_at_4_n": len(ps),
                    "mean_ms_per_leverage_call": float(np.mean(tms)) if tms else float("nan"),
                }
            )
        ep = exact_pass_by_cp.get(cp, [])
        et = exact_timing_by_cp.get(cp, [])
        epm = float(np.mean(ep)) if ep else float("nan")
        subrows.append(
            {
                "d": None,
                "label": "Exact SVD (D=1536)",
                "mean_kendall_tau_vs_exact": 1.0,
                "std_kendall_tau_vs_exact": 0.0,
                "oracle_pass_at_4_mean": epm,
                "policy_pass_at_4_mean": epm,
                "oracle_pass_at_4_n": len(ep),
                "mean_ms_per_leverage_call": float(np.mean(et)) if et else float("nan"),
            }
        )
        per_checkpoint.append({"checkpoint": cp, "rows": subrows})

    return {
        "rows": rows_out,
        "checkpoints": checkpoints,
        "k": k,
        "num_pass_draws": num_pass_draws,
        "per_checkpoint": per_checkpoint,
        "omniscient_pass_at_4_mean": omniscient_mean,
        "omniscient_pass_at_4_n": omniscient_n,
        "note": (
            "omniscient_pass_at_4_mean: policy-free ceiling (any correct among alive at this checkpoint). "
            "oracle_pass_at_4_mean / policy_pass_at_4_mean: same value — randomized or exact leverage ranking "
            "(policy retention Pass@4)."
        ),
    }


def write_md_table2(path: Path, summary: dict[str, Any]) -> None:
    r = summary["relative_l2_error"]
    a = summary["kendall_tau_random_vs_exact"]
    b = summary["kendall_tau_random_vs_random"]
    lines = [
        "# Table 2 — Stability (tab:stability-cross-run)",
        "",
        f"Settings: d={summary['d']}, k={summary['k']}, draws per (problem,cp)={summary['num_random_draws']}",
        "",
        "| Metric | Mean | Std Dev |",
        "|--------|------|---------|",
        f"| Relative Error | {r['mean']:.6f} | {r['std']:.6f} |",
        f"| Kendall-τ (Random vs Exact) | {a['mean']:.6f} | {a['std']:.6f} |",
        f"| Kendall-τ (Random vs Random) | {b['mean']:.6f} | {b['std']:.6f} |",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def write_md_table3(path: Path, data: dict[str, Any]) -> None:
    npd = data.get("num_pass_draws", "?")
    om = data.get("omniscient_pass_at_4_mean")
    omn = data.get("omniscient_pass_at_4_n", "")
    om_cell = f"{om * 100:.2f}% (n={omn})" if isinstance(om, (int, float)) and om == om else "—"
    lines = [
        "# Table 3 — Ablation on d (tab:ablation-d)",
        "",
        f"**Omniscient Pass@4 (ceiling):** {om_cell} — any correct among alive branches; same `alive` policy as policies below.",
        "",
        f"**Policy Pass@4 (randomized leverage):** mean over {npd} independent projection seeds per (problem, checkpoint), then mean over problems. Column `oracle_pass_at_4_mean` duplicates `policy_pass_at_4_mean` in JSON for backward compatibility.",
        "",
        "| Dimension | Mean Kendall-τ | Policy Pass@4 | Time (ms) |",
        "|-----------|----------------|---------------|-----------|",
    ]
    for row in data["rows"]:
        lab = row["label"]
        kt = row["mean_kendall_tau_vs_exact"]
        op = row.get("policy_pass_at_4_mean", row["oracle_pass_at_4_mean"])
        ms = row["mean_ms_per_leverage_call"]
        op_cell = f"{op * 100:.2f}%" if op == op else "—"
        lines.append(f"| {lab} | {kt:.4f} | {op_cell} | {ms:.4f} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_checkpoint_variation_md(path: Path, t2: dict[str, Any], t3: dict[str, Any]) -> None:
    """Summarize spread of Table 2–3 metrics across training checkpoints (when per_checkpoint is populated)."""
    lines = [
        "# Step 2.2 — Cross-checkpoint variation",
        "",
        "When multiple `--checkpoints` are used (or `--step1-checkpoints`), pooled Table 2–3 still ",
        "aggregate all (problem, checkpoint) pairs; the JSON fields `per_checkpoint` break out ",
        "means **within each checkpoint** so you can see whether approximation quality drifts with depth.",
        "",
    ]
    pc2 = t2.get("per_checkpoint", [])
    pc3 = t3.get("per_checkpoint", [])
    if len(pc2) <= 1 and len(pc3) <= 1:
        lines.append("*No multi-checkpoint sweep in this run (need ≥2 checkpoints with data).*")
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return

    if pc2:
        taus = [
            x["kendall_tau_random_vs_exact"]["mean"]
            for x in pc2
            if math.isfinite(x["kendall_tau_random_vs_exact"]["mean"])
        ]
        rels = [
            x["relative_l2_error"]["mean"]
            for x in pc2
            if math.isfinite(x["relative_l2_error"]["mean"])
        ]
        lines.append("## Table 2 (pooled metrics are across all listed checkpoints)")
        lines.append("")
        lines.append("| Checkpoint | n (τ vs exact) | Mean τ (rand vs exact) | Mean rel ℓ₂ err |")
        lines.append("|------------|----------------|-------------------------|----------------|")
        for x in sorted(pc2, key=lambda r: r["checkpoint"]):
            cp = x["checkpoint"]
            nt = x["n_samples_tau_re"]
            mt = x["kendall_tau_random_vs_exact"]["mean"]
            mr = x["relative_l2_error"]["mean"]
            lines.append(
                f"| {cp} | {nt} | {mt:.4f} | {mr:.4f} |"
            )
        lines.append("")
        if len(taus) >= 2:
            lines.append(
                f"**τ (rand vs exact):** min {min(taus):.4f}, max {max(taus):.4f}, "
                f"range {max(taus) - min(taus):.4f}."
            )
        if len(rels) >= 2:
            lines.append(
                f"**Relative ℓ₂:** min {min(rels):.4f}, max {max(rels):.4f}, "
                f"range {max(rels) - min(rels):.4f}."
            )
        lines.append("")

    if pc3:
        lines.append("## Table 3 — Per-checkpoint (focus: d=128 vs Exact row)")
        lines.append("")
        lines.append("| cp | τ vs exact @ d=128 | Pass@4 @ d=128 | Pass@4 Exact SVD |")
        lines.append("|----|---------------------|----------------|------------------|")
        for block in sorted(pc3, key=lambda b: b["checkpoint"]):
            cp = block["checkpoint"]
            r128 = next((r for r in block["rows"] if r.get("d") == 128), None)
            rex = next((r for r in block["rows"] if r.get("d") is None), None)
            if r128 and rex:
                lines.append(
                    f"| {cp} | {r128['mean_kendall_tau_vs_exact']:.4f} | "
                    f"{r128['oracle_pass_at_4_mean'] * 100:.2f}% | {rex['oracle_pass_at_4_mean'] * 100:.2f}% |"
                )
        lines.append("")
        taus128 = []
        p128 = []
        pex = []
        for block in pc3:
            r128 = next((r for r in block["rows"] if r.get("d") == 128), None)
            rex = next((r for r in block["rows"] if r.get("d") is None), None)
            if r128 and math.isfinite(r128["mean_kendall_tau_vs_exact"]):
                taus128.append(r128["mean_kendall_tau_vs_exact"])
            if r128 and math.isfinite(r128["oracle_pass_at_4_mean"]):
                p128.append(r128["oracle_pass_at_4_mean"])
            if rex and math.isfinite(rex["oracle_pass_at_4_mean"]):
                pex.append(rex["oracle_pass_at_4_mean"])
        if len(taus128) >= 2:
            lines.append(
                f"**τ @ d=128:** min {min(taus128):.4f}, max {max(taus128):.4f}, "
                f"range {max(taus128) - min(taus128):.4f}."
            )
        if len(p128) >= 2:
            lines.append(
                f"**Pass@4 @ d=128:** min {min(p128) * 100:.2f}%, max {max(p128) * 100:.2f}%."
            )
        if len(pex) >= 2:
            lines.append(
                f"**Pass@4 Exact:** min {min(pex) * 100:.2f}%, max {max(pex) * 100:.2f}%."
            )

    lines.append("")
    lines.append(
        "*Interpretation:* Large range in τ or Pass@4 across checkpoints usually reflects **depth-dependent** "
        "geometry of hidden states (and varying alive-branch counts), not necessarily failure of the randomized "
        "leverage estimator at a fixed (d, k)."
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=PROJECT_ROOT / D.DATA_REL_PATH)
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--batch_roots", type=Path, nargs="*", default=None)
    parser.add_argument(
        "--checkpoint",
        type=int,
        default=None,
        help="Single checkpoint (default: hero from step_2_1/hero_checkpoint_selection.json or 300)",
    )
    parser.add_argument(
        "--checkpoints",
        type=int,
        nargs="*",
        default=None,
        help="If set, overrides --checkpoint",
    )
    parser.add_argument(
        "--step1-checkpoints",
        action="store_true",
        help="Use all checkpoints from config.STEP1_CHECKPOINTS (overrides --checkpoint / --checkpoints)",
    )
    parser.add_argument("--k", type=int, default=D.LEVERAGE_SVD_RANK)
    parser.add_argument("--d_fixed", type=int, default=D.LEVERAGE_PROJ_DIM, help="Table 2 projection dim")
    parser.add_argument("--d_list", type=int, nargs="*", default=[64, 128, 256])
    parser.add_argument("--num_random_draws", type=int, default=100)
    parser.add_argument("--num_draws_tau_ablation", type=int, default=20, help="Draws per problem for mean τ in Table 3")
    parser.add_argument(
        "--num_pass_draws",
        type=int,
        default=50,
        help="Monte Carlo seeds per (problem,cp,d) for Oracle Pass@4 in Table 3 (reduces seed noise vs single draw)",
    )
    parser.add_argument("--seed_offset", type=int, default=12_345)
    parser.add_argument("--run2_seed_delta", type=int, default=9_912_345)
    parser.add_argument("--seed_pass_base", type=int, default=99_001)
    parser.add_argument("--timing_repeats", type=int, default=50)
    parser.add_argument("--timing_warmup", type=int, default=3)
    parser.add_argument(
        "--mode",
        choices=("all", "stability", "ablation"),
        default="all",
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
    parser.add_argument("--force_rebuild_cache", action="store_true")
    parser.add_argument("--out_dir", type=Path, default=PROJECT_ROOT / "results" / "step_2_2")
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable tqdm progress bars (for logs / CI)",
    )
    args = parser.parse_args()

    batch_roots = list(args.batch_roots) if args.batch_roots else default_batch_roots(PROJECT_ROOT)
    hero = read_hero_checkpoint(PROJECT_ROOT)
    if args.step1_checkpoints:
        cps = list(D.STEP1_CHECKPOINTS)
    elif args.checkpoints is not None:
        cps = list(args.checkpoints)
    elif args.checkpoint is not None:
        cps = [args.checkpoint]
    else:
        cps = [hero if hero is not None else 300]

    rows = load_jsonl(args.data, limit=args.limit)
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

    args.out_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "checkpoint(s)": cps,
        "step1_checkpoints_flag": args.step1_checkpoints,
        "k": args.k,
        "d_fixed_table2": args.d_fixed,
        "d_list_table3": list(args.d_list),
        "num_random_draws_table2": args.num_random_draws,
        "num_draws_tau_ablation": args.num_draws_tau_ablation,
        "num_pass_draws_table3": args.num_pass_draws,
        "limit": args.limit,
        "mode": args.mode,
    }
    (args.out_dir / "run_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    if args.mode in ("all", "stability"):
        t2, _, _, _ = run_table2_stability(
            rows=rows,
            branch_correct=branch_correct,
            batch_roots=batch_roots,
            checkpoints=cps,
            limit=args.limit,
            k=args.k,
            d_fixed=args.d_fixed,
            num_draws=args.num_random_draws,
            seed_offset=args.seed_offset,
            run2_seed_delta=args.run2_seed_delta,
            progress=not args.no_progress,
        )
        (args.out_dir / "table2_stability.json").write_text(json.dumps(t2, indent=2), encoding="utf-8")
        write_md_table2(args.out_dir / "table2_stability.md", t2)

    if args.mode in ("all", "ablation"):
        t3 = run_table3_ablation(
            rows=rows,
            branch_correct=branch_correct,
            batch_roots=batch_roots,
            checkpoints=cps,
            limit=args.limit,
            k=args.k,
            d_values=list(args.d_list),
            num_draws_tau=args.num_draws_tau_ablation,
            num_pass_draws=args.num_pass_draws,
            seed_pass_base=args.seed_pass_base,
            seed_offset=args.seed_offset,
            timing_repeats=args.timing_repeats,
            timing_warmup=args.timing_warmup,
            progress=not args.no_progress,
        )
        (args.out_dir / "table3_ablation_d.json").write_text(json.dumps(t3, indent=2), encoding="utf-8")
        write_md_table3(args.out_dir / "table3_ablation_d.md", t3)

    # LaTeX snippets
    latex: list[str] = []
    t2p = args.out_dir / "table2_stability.json"
    t3p = args.out_dir / "table3_ablation_d.json"
    if args.mode in ("all", "stability") and t2p.is_file():
        t2 = json.loads(t2p.read_text(encoding="utf-8"))
        r = t2["relative_l2_error"]
        a = t2["kendall_tau_random_vs_exact"]
        b = t2["kendall_tau_random_vs_random"]
        latex.append(
            f"% Table 2: Relative err mean={r['mean']:.6f} std={r['std']:.6f}; "
            f"tau_RE mean={a['mean']:.6f} std={a['std']:.6f}; "
            f"tau_RR mean={b['mean']:.6f} std={b['std']:.6f}"
        )
    if args.mode in ("all", "ablation") and t3p.is_file():
        t3 = json.loads(t3p.read_text(encoding="utf-8"))
        for row in t3["rows"]:
            latex.append(
                f"% {row['label']}: tau={row['mean_kendall_tau_vs_exact']:.4f} "
                f"Pass@4={row['oracle_pass_at_4_mean']:.4f} ms={row['mean_ms_per_leverage_call']:.4f}"
            )
    (args.out_dir / "latex_table_snippets.md").write_text("\n".join(latex) + ("\n" if latex else ""), encoding="utf-8")

    t2_dict: dict[str, Any] = {}
    t3_dict: dict[str, Any] = {}
    if t2p.is_file():
        t2_dict = json.loads(t2p.read_text(encoding="utf-8"))
    if t3p.is_file():
        t3_dict = json.loads(t3p.read_text(encoding="utf-8"))
    if t2_dict or t3_dict:
        write_checkpoint_variation_md(args.out_dir / "checkpoint_variation.md", t2_dict, t3_dict)


if __name__ == "__main__":
    main()
