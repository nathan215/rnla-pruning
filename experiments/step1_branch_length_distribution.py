"""
Per-branch generation length distributions and per-checkpoint length stats.

For each oracle_data.pt:
  - Final length per branch = valid_lengths[b] (decoded token count; pad excluded by design).

For each scheduled checkpoint step ``cp`` ("chunk"):
  - Effective length = min(cp, valid_lengths[b]) — tokens that exist at or before that depth
    (branch may have finished earlier).

Also runs integrity checks ("corrupt" / inconsistent tensors).

Output: results/step_1_oracle/analysis/branch_length_distribution.json

Run:
  python experiments/step1_branch_length_distribution.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import defaults as D  # noqa: E402


def _percentiles(xs: list[int]) -> dict[str, float]:
    if not xs:
        return {}
    s = sorted(xs)
    n = len(s)

    def pct(p: float) -> float:
        if n == 1:
            return float(s[0])
        k = (n - 1) * p
        f = int(k)
        c = min(f + 1, n - 1)
        return float(s[f] + (k - f) * (s[c] - s[f]))

    return {
        "min": float(s[0]),
        "p25": pct(0.25),
        "p50": pct(0.50),
        "p75": pct(0.75),
        "max": float(s[-1]),
        "mean": float(sum(s) / n),
    }


def _histogram(xs: list[int], bins: list[int]) -> dict[str, int]:
    """Count values in [bins[i], bins[i+1]) except last bucket inclusive of right edge."""
    out = {f"{bins[i]}-{bins[i+1]}": 0 for i in range(len(bins) - 1)}
    for x in xs:
        for i in range(len(bins) - 1):
            lo, hi = bins[i], bins[i + 1]
            if i < len(bins) - 2:
                if lo <= x < hi:
                    out[f"{lo}-{hi}"] += 1
                    break
            else:
                if lo <= x <= hi:
                    out[f"{lo}-{hi}"] += 1
                    break
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "step_1_oracle",
    )
    args = parser.parse_args()

    root = args.out_dir.resolve()
    per_dir = root / "per_problem"
    if not per_dir.is_dir():
        print(f"Missing {per_dir}", file=sys.stderr)
        sys.exit(1)

    all_final_lengths: list[int] = []
    per_problem_rows: list[dict] = []
    per_checkpoint_pooled: dict[int, list[int]] = {}

    integrity_issues: list[dict] = []

    problem_dirs = sorted([d for d in per_dir.iterdir() if d.is_dir()])

    for pdir in problem_dirs:
        pid = pdir.name
        pt_path = pdir / "oracle_data.pt"
        if not pt_path.is_file():
            integrity_issues.append({"problem_id_safe": pid, "issue": "missing oracle_data.pt"})
            continue

        try:
            kw = {"map_location": "cpu"}
            try:
                o = torch.load(pt_path, **kw, weights_only=False)
            except TypeError:
                o = torch.load(pt_path, **kw)
        except Exception as e:
            integrity_issues.append({"problem_id_safe": pid, "issue": f"load: {e}"})
            continue

        vl = o.get("valid_lengths")
        tmax = int(o.get("T_max", 0))
        cps = list(o.get("checkpoints", []))
        if vl is None:
            integrity_issues.append({"problem_id_safe": pid, "issue": "no valid_lengths"})
            continue

        vl = vl.long().view(-1)
        if vl.numel() != D.N_BRANCHES:
            integrity_issues.append(
                {"problem_id_safe": pid, "issue": f"valid_lengths len {vl.numel()} != {D.N_BRANCHES}"}
            )

        lens = [int(vl[b].item()) for b in range(min(D.N_BRANCHES, vl.numel()))]
        all_final_lengths.extend(lens)

        for b, L in enumerate(lens):
            if L < 1 or L > tmax:
                integrity_issues.append(
                    {
                        "problem_id_safe": pid,
                        "branch": b,
                        "issue": f"valid_lengths out of range: {L} not in [1, {tmax}]",
                    }
                )

        per_problem_rows.append(
            {
                "problem_id_safe": pid,
                "shortest_branch_tokens": min(lens),
                "longest_branch_tokens": max(lens),
                "spread_tokens": max(lens) - min(lens),
                "mean_branch_tokens": float(sum(lens) / len(lens)),
                "per_branch_final_tokens": lens,
            }
        )

        for cp in cps:
            eff = [min(cp, L) for L in lens]
            if cp not in per_checkpoint_pooled:
                per_checkpoint_pooled[cp] = []
            per_checkpoint_pooled[cp].extend(eff)

    global_final = _percentiles(all_final_lengths)
    hist_bins = [0, 512, 1024, 2048, 4096, 8192, 10000]
    hist = _histogram(all_final_lengths, hist_bins)

    per_cp_summary: dict[str, dict] = {}
    first_spread_cp: int | None = None
    for cp in sorted(per_checkpoint_pooled.keys()):
        stats = _percentiles(per_checkpoint_pooled[cp])
        stats["spread_tokens"] = float(stats["max"] - stats["min"])
        stats["all_equal_to_cp"] = bool(
            stats["min"] == stats["max"] == float(cp)
        )  # early chunks: every branch still at decode depth cp
        if first_spread_cp is None and stats["spread_tokens"] > 0:
            first_spread_cp = int(cp)
        per_cp_summary[str(cp)] = stats

    out = {
        "description": {
            "final_length": (
                "valid_lengths[b] = number of generated tokens stored for branch b "
                "(EOS stop or hit T_max without EOS)."
            ),
            "per_checkpoint": (
                "At decode step cp (scheduled checkpoint / chunk), effective_length[b] = min(cp, valid_lengths[b]): "
                "how many tokens exist at or before that depth. Early cp: often min=max=cp for all 800 slots "
                "(no branch has finished yet). Late cp: spread reflects branches that ended early vs still decoding."
            ),
            "integrity": (
                "Corrupt / inconsistent: valid_lengths outside [1, T_max], wrong tensor shape, or load errors."
            ),
        },
        "T_max_expected": D.STEP1_MAX_NEW_TOKENS,
        "n_problems": len(per_problem_rows),
        "n_branch_lengths_total": len(all_final_lengths),
        "global_final_length_tokens": global_final,
        "histogram_final_length_tokens": hist,
        "per_problem_shortest_longest": {
            "shortest_across_problems": min(r["shortest_branch_tokens"] for r in per_problem_rows)
            if per_problem_rows
            else None,
            "longest_across_problems": max(r["longest_branch_tokens"] for r in per_problem_rows)
            if per_problem_rows
            else None,
            "mean_of_per_problem_spread": float(
                sum(r["spread_tokens"] for r in per_problem_rows) / max(len(per_problem_rows), 1)
            ),
        },
        "first_scheduled_checkpoint_where_effective_length_spreads": first_spread_cp,
        "note_first_spread": (
            "Spread is computed only at scheduled checkpoint steps in config (50,100,...,8000). "
            "Before the shortest final branch length (dataset min), all branches share effective length = cp at early checkpoints."
        ),
        "per_checkpoint_effective_length_pooled": per_cp_summary,
        "per_problem": per_problem_rows,
        "integrity_flags": integrity_issues,
        "integrity_ok": len(integrity_issues) == 0,
    }

    out_path = root / "analysis" / "branch_length_distribution.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items() if k not in ("per_problem",)}, indent=2))
    print(f"\nWrote {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
