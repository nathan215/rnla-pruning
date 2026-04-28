"""
Step 2.1: Coverage validation — Policy retention Pass@m vs omniscient ceiling.

Alive / finished-branch policy (see results/step_2_1/oracle_pass_m_policy.md):
  - Rank and sample only among branches with alive_mask[checkpoint][b] == True.
  - Effective retention m_eff = min(m, num_alive). If num_alive == 0, the problem is skipped
    for that checkpoint (does not enter the mean denominator unless you use --include-no-alive).
  - Omniscient (method=omniscient): 1 iff any alive branch is correct (ceiling; no ranking).
  - Policy retention Pass@m: 1 iff at least one retained branch is correct under random / logprob / leverage.

Outputs:
  results/step_2_1/oracle_pass_m.csv
  results/step_2_1/oracle_pass_m_summary.md
  results/step_2_1/oracle_pass_m_policy.md
  results/step_2_1/hero_checkpoint_selection.json  # auto hero = argmax mean(leverage@4, leverage@8)
  results/step_2_1/cluster_mechanism.json + plots (if --cluster)

Canonical final run from project root:
  python experiments/step2_1_coverage.py --limit 200 --checkpoints 300 --m 1 2 4 8

All Step~1 checkpoints + Pass@1 (wide table in `oracle_pass_m_wide_by_checkpoint.md`):
  python experiments/step2_1_coverage.py --limit 200 --all-step1-checkpoints --m 1 2 4 8

Optional exploratory mode (not needed for final report):
  python experiments/step2_1_coverage.py --cluster --cluster_top_n 30
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import defaults as D  # noqa: E402
from src.data_loader import load_jsonl  # noqa: E402
from src.leverage import exact_row_leverage_scores  # noqa: E402
from src.step2_metrics import omniscient_retention_pass  # noqa: E402
from src.step2_branch_correctness import (  # noqa: E402
    batch_dir_for_global_index,
    get_or_build_branch_correctness,
    per_problem_dir,
)
from src.step2_io import alive_branch_indices, default_batch_roots, load_oracle  # noqa: E402

MatchMode = Literal["boxed_only", "legacy_last_line"]
Method = Literal["random", "logprob", "leverage"]


def checkpoints_50_750() -> list[int]:
    """Nine checkpoints in [50, 750] from config defaults."""
    cps = [c for c in D.STEP1_CHECKPOINTS if 50 <= c <= 750]
    return sorted(set(cps))


def all_step1_checkpoints() -> list[int]:
    """Every token checkpoint listed in config.defaults.STEP1_CHECKPOINTS."""
    return sorted(set(D.STEP1_CHECKPOINTS))


def oracle_pass_one(
    *,
    method: Method,
    alive: list[int],
    correct: list[bool],
    m: int,
    H_cp: torch.Tensor | None,
    cum_logprob: torch.Tensor | None,
    random_trials: int,
    rng: random.Random,
    leverage_k: int,
) -> float:
    """
    Return Oracle Pass rate in [0, 1]: for random, mean over trials; else 0/1.
    """
    n_alive = len(alive)
    if n_alive == 0:
        return float("nan")
    m_eff = min(m, n_alive)
    if m_eff == 0:
        return float("nan")

    alive_set = set(alive)

    def any_correct(idxs: list[int]) -> bool:
        return any(correct[j] for j in idxs if j < len(correct))

    if method == "logprob":
        if cum_logprob is None:
            return float("nan")
        scores = [(cum_logprob[b].item(), b) for b in alive]
        scores.sort(key=lambda x: -x[0])
        top = [b for _, b in scores[:m_eff]]
        return 1.0 if any_correct(top) else 0.0

    if method == "leverage":
        if H_cp is None:
            return float("nan")
        rows = torch.stack([H_cp[b] for b in alive], dim=0)
        lev = exact_row_leverage_scores(rows, leverage_k)
        order = torch.argsort(lev, descending=True).tolist()
        top = [alive[j] for j in order[:m_eff]]
        return 1.0 if any_correct(top) else 0.0

    # random
    hits = 0
    for _ in range(random_trials):
        sample = rng.sample(alive, m_eff) if m_eff < n_alive else list(alive)
        if any_correct(sample):
            hits += 1
    return hits / float(random_trials)


def top_m_branch_indices(
    *,
    method: Literal["logprob", "leverage"],
    alive: list[int],
    m: int,
    H_full: torch.Tensor,
    cum_logprob: torch.Tensor | None,
    leverage_k: int,
) -> list[int]:
    """Return global branch indices for top-m by logprob or exact leverage (alive only)."""
    n_alive = len(alive)
    if n_alive == 0:
        return []
    m_eff = min(m, n_alive)
    if method == "logprob" and cum_logprob is not None:
        scores = [(float(cum_logprob[b].item()), b) for b in alive]
        scores.sort(key=lambda x: -x[0])
        return [b for _, b in scores[:m_eff]]
    if method == "leverage":
        rows = torch.stack([H_full[b] for b in alive], dim=0)
        lev = exact_row_leverage_scores(rows, leverage_k)
        order = torch.argsort(lev, descending=True).tolist()
        return [alive[j] for j in order[:m_eff]]
    return []


def pick_hero_checkpoint(
    agg: list[dict[str, Any]],
    checkpoints: list[int],
) -> tuple[int, dict[str, Any]]:
    """
    Choose checkpoint where exact leverage is strongest on m=4 and m=8 (primary).

    score = mean(leverage@4, leverage@8). Tie-break: maximize
    (leverage@8 - logprob@8) + (leverage@4 - logprob@4).
    """
    def get(cp: int, m: int, method: str) -> float | None:
        for r in agg:
            if int(r["checkpoint"]) == cp and int(r["m"]) == m and r["method"] == method:
                return float(r["mean_oracle_pass"])
        return None

    candidates: list[dict[str, Any]] = []
    best_cp: int | None = None
    best_score = -1.0
    best_gap_sum = -1e9

    for cp in checkpoints:
        lv4 = get(cp, 4, "leverage")
        lv8 = get(cp, 8, "leverage")
        lp4 = get(cp, 4, "logprob")
        lp8 = get(cp, 8, "logprob")
        if lv4 is None or lv8 is None:
            continue
        score = (lv4 + lv8) / 2.0
        gap_sum = 0.0
        if lp4 is not None and lp8 is not None:
            gap_sum = (lv4 - lp4) + (lv8 - lp8)
        candidates.append(
            {
                "checkpoint": cp,
                "leverage_m4": lv4,
                "leverage_m8": lv8,
                "logprob_m4": lp4,
                "logprob_m8": lp8,
                "score_mean_leverage_m4_m8": score,
                "gap_sum_vs_logprob_m4_m8": gap_sum,
            }
        )
        if score > best_score + 1e-12:
            best_score = score
            best_gap_sum = gap_sum
            best_cp = cp
        elif abs(score - best_score) < 1e-12 and gap_sum > best_gap_sum + 1e-12:
            best_gap_sum = gap_sum
            best_cp = cp

    if best_cp is None:
        best_cp = int(checkpoints[0]) if checkpoints else 250

    meta = {
        "selected_checkpoint": best_cp,
        "rationale": (
            "Maximize mean Oracle Pass (exact leverage) at m=4 and m=8; "
            "tie-break: larger sum of (leverage−logprob) at m=4 and m=8."
        ),
        "candidates": sorted(candidates, key=lambda x: (-x["score_mean_leverage_m4_m8"], -x["gap_sum_vs_logprob_m4_m8"])),
    }
    return best_cp, meta


def run_coverage_grid(
    *,
    rows: list[dict[str, Any]],
    branch_correct: dict[int, list[bool]],
    batch_roots: list[Path],
    checkpoints: list[int],
    m_values: list[int],
    methods: list[Method],
    random_trials: int,
    leverage_k: int,
    rng: random.Random,
    include_no_alive: bool,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """
    Returns list of result rows and counts of skipped (no oracle file / no checkpoint).
    """
    results: list[dict[str, Any]] = []
    stats = {"missing_pt": 0, "missing_cp": 0, "no_alive": 0}

    for global_idx, row in enumerate(rows):
        bc = branch_correct.get(global_idx)
        if bc is None or len(bc) != D.N_BRANCHES:
            continue
        bid, _ = batch_dir_for_global_index(global_idx, 25)
        if bid >= len(batch_roots):
            continue
        root = batch_roots[bid]
        pdir = per_problem_dir(root, row, global_idx)
        pt_path = pdir / "oracle_data.pt"
        if not pt_path.is_file():
            stats["missing_pt"] += 1
            continue
        oracle = load_oracle(pt_path)

        for cp in checkpoints:
            if cp not in oracle.get("checkpoint_hidden_states", {}):
                stats["missing_cp"] += 1
                continue
            alive = alive_branch_indices(oracle, cp)
            if not alive and not include_no_alive:
                stats["no_alive"] += 1
                continue

            H_full = oracle["checkpoint_hidden_states"][cp]
            clp = oracle.get("checkpoint_cumulative_logprob", {}).get(cp)

            for m in m_values:
                omni = omniscient_retention_pass(alive, bc)
                if omni == omni:
                    results.append(
                        {
                            "global_idx": global_idx,
                            "checkpoint": cp,
                            "m": m,
                            "method": "omniscient",
                            "oracle_pass": omni,
                        }
                    )
                for method in methods:
                    val = oracle_pass_one(
                        method=method,
                        alive=alive,
                        correct=bc,
                        m=m,
                        H_cp=H_full,
                        cum_logprob=clp,
                        random_trials=random_trials,
                        rng=rng,
                        leverage_k=leverage_k,
                    )
                    if val != val:  # nan
                        continue
                    results.append(
                        {
                            "global_idx": global_idx,
                            "checkpoint": cp,
                            "m": m,
                            "method": method,
                            "oracle_pass": val,
                        }
                    )

    return results, stats


def aggregate_mean(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Group by (checkpoint, m, method) -> mean oracle_pass."""
    from collections import defaultdict

    acc: dict[tuple[int, int, str], list[float]] = defaultdict(list)
    for r in rows:
        acc[(r["checkpoint"], r["m"], r["method"])].append(float(r["oracle_pass"]))

    out: list[dict[str, Any]] = []
    for (cp, m, method), vals in sorted(acc.items()):
        out.append(
            {
                "checkpoint": cp,
                "m": m,
                "method": method,
                "mean_oracle_pass": sum(vals) / len(vals),
                "n_problems": len(vals),
            }
        )
    return out


def per_problem_pass_for_gap(
    *,
    rows: list[dict[str, Any]],
    branch_correct: dict[int, list[bool]],
    batch_roots: list[Path],
    checkpoint: int,
    m: int,
    leverage_k: int,
    rng: random.Random,
    random_trials: int,
) -> dict[int, dict[str, float]]:
    """Per global_idx: oracle pass for logprob, leverage, random (mean)."""
    out: dict[int, dict[str, float]] = {}
    for global_idx, row in enumerate(rows):
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
        if checkpoint not in oracle.get("checkpoint_hidden_states", {}):
            continue
        alive = alive_branch_indices(oracle, checkpoint)
        H_full = oracle["checkpoint_hidden_states"][checkpoint]
        clp = oracle.get("checkpoint_cumulative_logprob", {}).get(checkpoint)
        entry: dict[str, float] = {}
        for method in ("logprob", "leverage", "random"):
            v = oracle_pass_one(
                method=method,  # type: ignore[arg-type]
                alive=alive,
                correct=bc,
                m=m,
                H_cp=H_full,
                cum_logprob=clp,
                random_trials=random_trials,
                rng=rng,
                leverage_k=leverage_k,
            )
            if v == v:
                entry[method] = v
        if entry:
            out[global_idx] = entry
    return out


def _max_cluster_concentration(labels: np.ndarray, top_indices: list[int], n_clusters: int, m: int) -> float:
    """Max fraction of top-m branches that fall in a single cluster (1.0 = all in one cluster)."""
    if not top_indices or m <= 0:
        return 0.0
    counts = [0] * n_clusters
    for i in top_indices:
        if 0 <= i < len(labels):
            counts[int(labels[i])] += 1
    return max(counts) / float(len(top_indices))


def run_cluster_mechanism(
    *,
    rows: list[dict[str, Any]],
    branch_correct: dict[int, list[bool]],
    batch_roots: list[Path],
    gap_problems: list[int],
    cluster_checkpoint: int,
    cluster_m: int,
    leverage_k: int,
    plots_dir: Path,
    project_root: Path,
    rng: random.Random,
) -> dict[str, Any]:
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    from sklearn.preprocessing import StandardScaler

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    summaries: list[dict[str, Any]] = []
    conc_logprob: list[float] = []
    conc_leverage: list[float] = []

    for global_idx in gap_problems:
        row = rows[global_idx]
        bid, _ = batch_dir_for_global_index(global_idx, 25)
        pdir = per_problem_dir(batch_roots[bid], row, global_idx)
        pt_path = pdir / "oracle_data.pt"
        if not pt_path.is_file():
            continue
        oracle = load_oracle(pt_path)
        if cluster_checkpoint not in oracle.get("checkpoint_hidden_states", {}):
            continue
        H_full = oracle["checkpoint_hidden_states"][cluster_checkpoint]
        H = H_full.to(torch.float32).numpy()
        bc = branch_correct[global_idx]
        clp = oracle.get("checkpoint_cumulative_logprob", {}).get(cluster_checkpoint)
        alive = alive_branch_indices(oracle, cluster_checkpoint)

        lev_top = top_m_branch_indices(
            method="leverage",
            alive=alive,
            m=cluster_m,
            H_full=H_full,
            cum_logprob=clp,
            leverage_k=leverage_k,
        )
        log_top = top_m_branch_indices(
            method="logprob",
            alive=alive,
            m=cluster_m,
            H_full=H_full,
            cum_logprob=clp,
            leverage_k=leverage_k,
        )

        scaler = StandardScaler()
        X = scaler.fit_transform(H)

        best_k = 2
        best_score = -1.0
        for k in (2, 3, 4):
            if len(X) < k + 1:
                continue
            km_try = KMeans(n_clusters=k, n_init=10, random_state=rng.randint(0, 2**31 - 1))
            labels_try = km_try.fit_predict(X)
            try:
                sil = float(silhouette_score(X, labels_try))
            except Exception:
                sil = -1.0
            if sil > best_score:
                best_score = sil
                best_k = k

        km = KMeans(n_clusters=best_k, n_init=10, random_state=42)
        labels = km.fit_predict(X)

        max_lp_branch = 0
        if clp is not None and alive:
            max_lp_branch = max(alive, key=lambda b: float(clp[b].item()))

        c_log = _max_cluster_concentration(labels, log_top, best_k, cluster_m)
        c_lev = _max_cluster_concentration(labels, lev_top, best_k, cluster_m)
        conc_logprob.append(c_log)
        conc_leverage.append(c_lev)

        cluster_stats: list[dict[str, Any]] = []
        for c in range(best_k):
            idxs = [i for i in range(16) if labels[i] == c]
            if not idxs:
                frac = 0.0
            else:
                frac = sum(1 for i in idxs if bc[i]) / len(idxs)
            lev_in = len(set(lev_top) & set(idxs))
            log_in = len(set(log_top) & set(idxs))
            cluster_stats.append(
                {
                    "cluster": c,
                    "size": len(idxs),
                    "fraction_correct": frac,
                    "contains_max_logprob": int(max_lp_branch in idxs) if alive else 0,
                    "leverage_top_m_branches_in_cluster": lev_in,
                    "logprob_top_m_branches_in_cluster": log_in,
                }
            )

        summaries.append(
            {
                "global_idx": global_idx,
                "problem_id": str(row.get("unique_id", "")),
                "best_k": best_k,
                "silhouette": float(best_score) if best_score >= 0 else None,
                "max_logprob_branch": max_lp_branch,
                "top_m_leverage_indices": lev_top,
                "top_m_logprob_indices": log_top,
                "any_correct_in_leverage_top_m": int(any(bc[i] for i in lev_top)) if lev_top else 0,
                "any_correct_in_logprob_top_m": int(any(bc[i] for i in log_top)) if log_top else 0,
                "max_cluster_concentration_logprob_top_m": c_log,
                "max_cluster_concentration_leverage_top_m": c_lev,
                "clusters": cluster_stats,
            }
        )

    aggregate = {}
    if conc_logprob and conc_leverage:
        aggregate = {
            "mean_max_cluster_concentration_logprob_top_m": float(sum(conc_logprob) / len(conc_logprob)),
            "mean_max_cluster_concentration_leverage_top_m": float(sum(conc_leverage) / len(conc_leverage)),
            "interpretation": (
                "Per problem: among KMeans clusters on z-scored hidden states, "
                "compute max fraction of top-m branches (by logprob vs by leverage) that lie in a single cluster. "
                "Higher = selections more concentrated in one geometry cluster. "
                "Leverage often spreads top-m across clusters vs logprob (lower concentration) when clusters separate wrong high-logprob mass."
            ),
        }

    n_show = min(5, len(summaries))
    plots_dir.mkdir(parents=True, exist_ok=True)
    fig_rel_bars = ""
    fig_rel_cmp = ""

    if n_show > 0:
        fig, axes = plt.subplots(n_show, 1, figsize=(8, 2.2 * n_show), squeeze=False)
        for r in range(n_show):
            s = summaries[r]
            ax = axes[r, 0]
            k = s["best_k"]
            fracs = [c["fraction_correct"] for c in s["clusters"]]
            ax.bar(range(k), fracs, tick_label=[f"C{i}" for i in range(k)])
            ax.set_ylim(0, 1.05)
            ax.set_ylabel("frac correct")
            ax.set_title(f"idx={s['global_idx']} K={k} (z-scored H)")
        plt.tight_layout()
        fig_path = plots_dir / "step_2_1_cluster_bars.png"
        plt.savefig(fig_path, dpi=150)
        plt.close()
        try:
            fig_rel_bars = str(fig_path.resolve().relative_to(project_root.resolve()))
        except ValueError:
            fig_rel_bars = str(fig_path)

    if len(summaries) >= 1:
        fig2, ax2 = plt.subplots(figsize=(7, max(3.0, 0.35 * len(summaries))))
        y_pos = np.arange(len(summaries))
        w = 0.35
        ax2.barh(y_pos - w / 2, conc_logprob, height=w, label="logprob top-m", alpha=0.85)
        ax2.barh(y_pos + w / 2, conc_leverage, height=w, label="leverage top-m", alpha=0.85)
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels([f"idx={s['global_idx']}" for s in summaries])
        ax2.set_xlabel("max fraction of top-m in one cluster")
        ax2.set_xlim(0, 1.05)
        ax2.legend(loc="lower right")
        ax2.set_title(
            f"Cluster concentration (cp={cluster_checkpoint}, m={cluster_m}): "
            "higher = more top-m branches in a single KMeans cluster"
        )
        plt.tight_layout()
        fig_path2 = plots_dir / "step_2_1_cluster_leverage_vs_logprob_concentration.png"
        plt.savefig(fig_path2, dpi=150)
        plt.close()
        try:
            fig_rel_cmp = str(fig_path2.resolve().relative_to(project_root.resolve()))
        except ValueError:
            fig_rel_cmp = str(fig_path2)

    out: dict[str, Any] = {
        "summaries": summaries,
        "aggregate": aggregate,
        "figure_cluster_purity_bars": fig_rel_bars or None,
        "figure_concentration_comparison": fig_rel_cmp or None,
    }
    if n_show == 0:
        out["note"] = "no cluster data"
    return out


def write_policy_md(path: Path) -> None:
    text = """# Oracle Pass@m — alive branch policy (Step 2.1)

- Selection uses only branches with `alive_mask[checkpoint][b] == True`.
- Effective `m_eff = min(m, num_alive)`. Retained set size is `m_eff` (all alive if fewer than `m`).
- If `num_alive == 0`, the (problem, checkpoint) pair is excluded from aggregates unless `--include-no-alive` is set.

## Two metrics (do not conflate)

- **Omniscient (ceiling)** — `method="omniscient"` in aggregates: `1` iff at least one **alive** branch has a correct final answer (same graded labels as Step 1). This is the **policy-free** upper bound for “keep \\(m\\) branches and want ≥1 correct”: if you could choose which branches to keep with full knowledge of correctness, you would succeed whenever such a branch exists among alive. **Does not depend on leverage or logprob.** Implemented as `omniscient_retention_pass` in `src/step2_metrics.py`. Rows are duplicated per \\(m\\) with the same value (ceiling depends only on alive/correct, not on \\(m\\)).

- **Policy retention Pass@m** (historically called “Oracle Pass@m” in CSV): `1` iff at least one **correct** branch appears in the **retained** top-\\(m\\) under the rule (**random** / **logprob** / **exact leverage**). This is **≤** the omniscient rate.

- **Logprob:** sort alive branches by `checkpoint_cumulative_logprob[checkpoint]` descending; take top `m_eff`.
- **Exact leverage:** SVD on rows `H[alive]`; top `m_eff` by `exact_row_leverage_scores` (see `src/leverage.py`).
- **Random:** `random_trials` independent samples of `m_eff` distinct alive branches (uniform without replacement); reported value is the fraction of trials with ≥1 correct branch.

Correctness comes from `solution_texts/branch_*.txt` vs reference (`boxed_only` by default).
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_csv(path: Path, agg: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not agg:
        path.write_text("", encoding="utf-8")
        return
    keys = ["checkpoint", "m", "method", "mean_oracle_pass", "n_problems"]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for row in agg:
            w.writerow({k: row.get(k) for k in keys})


def write_table1_snippet(
    path: Path,
    agg: list[dict[str, Any]],
    hero_cp: int,
    hero_meta: dict[str, Any] | None = None,
) -> None:
    """Markdown + LaTeX-ready lines for Table 1 (tab:leverage-vs-logprob) at one checkpoint."""
    sub = [r for r in agg if int(r["checkpoint"]) == int(hero_cp)]
    if not sub:
        path.write_text(f"(no aggregate rows for checkpoint {hero_cp})\n", encoding="utf-8")
        return

    def mean(m: int, method: str) -> float | None:
        for r in sub:
            if int(r["m"]) == m and r["method"] == method:
                return float(r["mean_oracle_pass"])
        return None

    lines = [
        f"# Table 1 snippet (checkpoint={hero_cp})",
    ]
    if hero_meta and hero_meta.get("rationale"):
        lines.append("")
        lines.append(f"*Hero selection:* {hero_meta['rationale']}")
    lines += [
        "",
        "| Method | Oracle Pass@8 | Oracle Pass@4 | Oracle Pass@2 | Oracle Pass@1 |",
        "|--------|---------------|---------------|---------------|---------------|",
    ]
    o8, o4, o2, o1 = mean(8, "omniscient"), mean(4, "omniscient"), mean(2, "omniscient"), mean(1, "omniscient")
    if o8 is not None and o4 is not None and o2 is not None:
        o1s = f"{o1:.4f}" if o1 is not None else "—"
        lines.append(
            f"| Omniscient (ceiling; any correct among alive) | {o8:.4f} | {o4:.4f} | {o2:.4f} | {o1s} |"
        )
    for method, label in [("random", "Random Retention"), ("logprob", "Top-$m$ Logprob"), ("leverage", "Top-$m$ Leverage (Exact)")]:
        a8 = mean(8, method)
        a4 = mean(4, method)
        a2 = mean(2, method)
        a1 = mean(1, method)
        a1s = f"{a1:.4f}" if a1 is not None else "—"
        lines.append(
            f"| {label} | {a8:.4f} | {a4:.4f} | {a2:.4f} | {a1s} |"
            if a8 is not None and a4 is not None and a2 is not None
            else f"| {label} | — | — | — | {a1s} |"
        )
    lv8, lp8 = mean(8, "leverage"), mean(8, "logprob")
    lv4, lp4 = mean(4, "leverage"), mean(4, "logprob")
    lv2, lp2 = mean(2, "leverage"), mean(2, "logprob")
    lv1, lp1 = mean(1, "leverage"), mean(1, "logprob")
    lines += ["", "**Leverage vs Logprob gap (pp):**"]
    if lv8 is not None and lp8 is not None:
        lines.append(f"- Pass@8: {(lv8 - lp8) * 100:.2f} pp")
    if lv4 is not None and lp4 is not None:
        lines.append(f"- Pass@4: {(lv4 - lp4) * 100:.2f} pp")
    if lv2 is not None and lp2 is not None:
        lines.append(f"- Pass@2: {(lv2 - lp2) * 100:.2f} pp")
    if lv1 is not None and lp1 is not None:
        lines.append(f"- Pass@1: {(lv1 - lp1) * 100:.2f} pp")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_passm_wide_by_checkpoint_md(
    path: Path,
    agg: list[dict[str, Any]],
    checkpoints: list[int],
    m_values: list[int],
) -> None:
    """
    One row per checkpoint: omniscient (any m), then Pass@m for random / logprob / leverage.
    Percentages for readability.
    """
    m_sorted = sorted(set(m_values))
    lookup: dict[tuple[int, int, str], float] = {}
    for r in agg:
        lookup[(int(r["checkpoint"]), int(r["m"]), str(r["method"]))] = float(r["mean_oracle_pass"])

    def get(cp: int, m: int, method: str) -> float | None:
        return lookup.get((cp, m, method))

    hdr = ["checkpoint", "omni %"]
    for m in m_sorted:
        hdr += [f"rnd@{m}", f"lp@{m}", f"lev@{m}"]
    lines = [
        "# Oracle Pass@m — wide by checkpoint",
        "",
        f"m values: {m_sorted}. Random = mean over `--random_trials` samples; logprob / leverage = deterministic top-m.",
        "",
        "| " + " | ".join(hdr) + " |",
        "|" + "|".join(["---"] * len(hdr)) + "|",
    ]
    for cp in sorted(checkpoints):
        o = get(cp, m_sorted[0], "omniscient")
        if o is None:
            continue
        row = [str(cp), f"{o * 100:.1f}"]
        for m in m_sorted:
            for meth in ("random", "logprob", "leverage"):
                v = get(cp, m, meth)
                row.append(f"{v * 100:.1f}" if v is not None else "—")
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_summary_md(path: Path, agg: list[dict[str, Any]], checkpoints: list[int]) -> None:
    lines = [
        "# Step 2.1 — Oracle Pass@m summary",
        "",
        f"Checkpoints: {checkpoints}",
        "",
        "| checkpoint | m | method | mean Oracle Pass@m | n_problems |",
        "|------------|---|--------|---------------------|------------|",
    ]
    for row in agg:
        lines.append(
            f"| {row['checkpoint']} | {row['m']} | {row['method']} | "
            f"{row['mean_oracle_pass']:.4f} | {row['n_problems']} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=PROJECT_ROOT / D.DATA_REL_PATH)
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument(
        "--batch_roots",
        type=Path,
        nargs="*",
        default=None,
        help="Default: results/step_1_oracle_b0 … b7",
    )
    parser.add_argument(
        "--checkpoints",
        type=int,
        nargs="*",
        default=None,
        help="Default: nine checkpoints in [50,750] from config",
    )
    parser.add_argument(
        "--all-step1-checkpoints",
        action="store_true",
        help="Use every checkpoint in config.defaults.STEP1_CHECKPOINTS (overrides default cp list).",
    )
    parser.add_argument("--m", type=int, nargs="*", default=[1, 2, 4, 8])
    parser.add_argument("--random_trials", type=int, default=1000)
    parser.add_argument("--leverage_k", type=int, default=D.LEVERAGE_SVD_RANK)
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
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "step_2_1",
    )
    parser.add_argument("--include_no_alive", action="store_true")
    parser.add_argument("--cluster", action="store_true", help="Run gap-based cluster mechanism analysis")
    parser.add_argument("--cluster_m", type=int, default=4)
    parser.add_argument("--cluster_top_n", type=int, default=30)
    parser.add_argument("--random_seed", type=int, default=None, help="Optional seed for random baseline")
    parser.add_argument(
        "--hero_checkpoint",
        type=int,
        default=None,
        help="Checkpoint for table1_latex_snippet.md; default: auto (max mean leverage@4 and leverage@8)",
    )
    parser.add_argument(
        "--cluster_checkpoint",
        type=int,
        default=None,
        help="Checkpoint for KMeans / gap analysis; default: same as hero checkpoint",
    )
    args = parser.parse_args()

    batch_roots = list(args.batch_roots) if args.batch_roots else default_batch_roots(PROJECT_ROOT)
    if args.all_step1_checkpoints:
        checkpoints = all_step1_checkpoints()
    else:
        checkpoints = list(args.checkpoints) if args.checkpoints else checkpoints_50_750()
    rng = random.Random(args.random_seed)

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

    methods: list[Method] = ["random", "logprob", "leverage"]
    raw_rows, stats = run_coverage_grid(
        rows=rows,
        branch_correct=branch_correct,
        batch_roots=batch_roots,
        checkpoints=checkpoints,
        m_values=list(args.m),
        methods=methods,
        random_trials=args.random_trials,
        leverage_k=args.leverage_k,
        rng=rng,
        include_no_alive=args.include_no_alive,
    )
    agg = aggregate_mean(raw_rows)

    if args.hero_checkpoint is not None:
        hero_cp = int(args.hero_checkpoint)
        hero_meta: dict[str, Any] = {"selected_checkpoint": hero_cp, "rationale": "manual --hero_checkpoint"}
    else:
        hero_cp, hero_meta = pick_hero_checkpoint(agg, checkpoints)

    cluster_cp = int(args.cluster_checkpoint) if args.cluster_checkpoint is not None else hero_cp

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_policy_md(args.out_dir / "oracle_pass_m_policy.md")
    write_csv(args.out_dir / "oracle_pass_m.csv", agg)
    write_summary_md(args.out_dir / "oracle_pass_m_summary.md", agg, checkpoints)
    write_passm_wide_by_checkpoint_md(
        args.out_dir / "oracle_pass_m_wide_by_checkpoint.md",
        agg,
        checkpoints,
        list(args.m),
    )
    (args.out_dir / "hero_checkpoint_selection.json").write_text(
        json.dumps(hero_meta, indent=2),
        encoding="utf-8",
    )
    write_table1_snippet(args.out_dir / "table1_latex_snippet.md", agg, hero_cp, hero_meta)

    with (args.out_dir / "run_stats.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "skip_stats": stats,
                "n_raw_rows": len(raw_rows),
                "checkpoints": checkpoints,
                "m_values": list(args.m),
                "hero_checkpoint": hero_cp,
                "cluster_checkpoint_used": cluster_cp,
            },
            f,
            indent=2,
        )

    if args.cluster:
        per_p = per_problem_pass_for_gap(
            rows=rows,
            branch_correct=branch_correct,
            batch_roots=batch_roots,
            checkpoint=cluster_cp,
            m=args.cluster_m,
            leverage_k=args.leverage_k,
            rng=rng,
            random_trials=args.random_trials,
        )
        gaps: list[tuple[int, float]] = []
        for gid, d in per_p.items():
            lp = d.get("logprob", 0.0)
            lv = d.get("leverage", 0.0)
            gaps.append((gid, lv - lp))
        gaps.sort(key=lambda x: -x[1])
        gap_problems = [g for g, gap in gaps if gap > 0][: args.cluster_top_n]
        if not gap_problems:
            gap_problems = [g for g, _ in gaps[: args.cluster_top_n]]

        cluster_out = run_cluster_mechanism(
            rows=rows,
            branch_correct=branch_correct,
            batch_roots=batch_roots,
            gap_problems=gap_problems,
            cluster_checkpoint=cluster_cp,
            cluster_m=args.cluster_m,
            leverage_k=args.leverage_k,
            plots_dir=PROJECT_ROOT / "results" / "plots",
            project_root=PROJECT_ROOT,
            rng=rng,
        )
        (args.out_dir / "cluster_mechanism.json").write_text(
            json.dumps(cluster_out, indent=2), encoding="utf-8"
        )
        gap_top_fixed = []
        for g in gap_problems:
            d = per_p.get(g, {})
            gap_top_fixed.append(
                {
                    "global_idx": g,
                    "leverage_minus_logprob": float(d.get("leverage", 0) - d.get("logprob", 0)),
                }
            )
        (args.out_dir / "cluster_gap_problems.json").write_text(
            json.dumps(
                {
                    "cluster_checkpoint": cluster_cp,
                    "cluster_m": args.cluster_m,
                    "hero_checkpoint": hero_cp,
                    "problem_global_indices": gap_problems,
                    "gaps_top": gap_top_fixed,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        agg_d = cluster_out.get("aggregate") or {}
        interp = [
            "# Cluster mechanism (contrastive subset)",
            "",
            f"- Checkpoint: **{cluster_cp}** (default = hero checkpoint), m={args.cluster_m}",
            f"- Hero checkpoint for main table: **{hero_cp}** (see `hero_checkpoint_selection.json`)",
            f"- Problems: top {len(gap_problems)} by (per-problem Oracle Pass leverage − logprob)",
            "- KMeans on **z-scored** 1536-D rows (StandardScaler); K ∈ {2,3,4} by best silhouette.",
            "",
            "## Leverage vs logprob (same geometry)",
            "- Each cluster row lists **how many** of the top-m branches by **leverage** vs **logprob** fall in that cluster.",
            "- **max_cluster_concentration_*_top_m**: max over clusters of (branches from that selection in cluster) / m. "
            "Higher = more concentrated in one geometry cluster.",
            "",
        ]
        if agg_d:
            interp.append(
                f"- **Mean concentration (logprob top-m):** {agg_d.get('mean_max_cluster_concentration_logprob_top_m', 0):.3f}"
            )
            interp.append(
                f"- **Mean concentration (leverage top-m):** {agg_d.get('mean_max_cluster_concentration_leverage_top_m', 0):.3f}"
            )
            interp.append("")
        interp += [
            "## Figures",
            f"- Cluster purity (fraction correct per cluster): `{cluster_out.get('figure_cluster_purity_bars')}`",
            f"- Logprob vs leverage concentration: `{cluster_out.get('figure_concentration_comparison')}`",
            "",
        ]
        (args.out_dir / "cluster_mechanism_interpretation.md").write_text(
            "\n".join(interp), encoding="utf-8"
        )
        (args.out_dir / "cluster_leverage_logprob_relationship.md").write_text(
            "\n".join(
                [
                    "# Leverage vs Logprob — cluster relationship (read me)",
                    "",
                    "We use **one** KMeans clustering of the 16 branch hidden states (z-scored) at the same checkpoint.",
                    "Then we compare **two different branch selections** of size m:",
                    "",
                    "1. **Top-m by cumulative logprob** — confidence-ranked.",
                    "2. **Top-m by exact leverage** — geometry-ranked.",
                    "",
                    "For each cluster, `leverage_top_m_branches_in_cluster` vs `logprob_top_m_branches_in_cluster` "
                    "shows **where** each rule places mass. Logprob often piles into one cluster (often dominated by wrong but fluent paths); "
                    "leverage spreads selections across clusters when those clusters correspond to distinct directions in hidden space.",
                    "",
                    "The horizontal bar chart compares **max-cluster concentration**: if leverage’s bar is **lower**, "
                    "its top-m set is **less** concentrated in a single cluster than logprob’s — a diversity signal aligned with the method.",
                    "",
                    f"See `cluster_mechanism.json` for per-problem indices and numeric fields; checkpoint = {cluster_cp}, m = {args.cluster_m}.",
                ]
            ),
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
