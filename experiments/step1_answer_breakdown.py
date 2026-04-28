"""
Recompute per-problem answer correctness from solution_texts + dataset rows.

Primary headline metric: expected accuracy when each problem independently draws **one**
uniform random branch (equivalently: mean over problems of num_correct/16).

Writes results/step_1_oracle/analysis/answer_breakdown.json.

Run from project root:
  python experiments/step1_answer_breakdown.py
  python experiments/step1_answer_breakdown.py --match_mode legacy_last_line
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import defaults as D  # noqa: E402
from src.data_loader import load_jsonl  # noqa: E402
from src.answer_matching import answers_equal, get_reference_answer, safe_problem_id  # noqa: E402


def _load_coverage_by_index(out_dir: Path) -> dict[int, int]:
    p = out_dir / "analysis" / "coverage_report.json"
    if not p.is_file():
        return {}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}
    out: dict[int, int] = {}
    for row in data.get("valid_checkpoints_per_problem", []):
        out[int(row["problem_offset"])] = int(row["valid_checkpoints"])
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=PROJECT_ROOT / D.DATA_REL_PATH)
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "step_1_oracle",
    )
    parser.add_argument("--limit", type=int, default=D.STEP1_NUM_PROBLEMS)
    parser.add_argument(
        "--match_mode",
        choices=("boxed_only", "legacy_last_line"),
        default="boxed_only",
        help="boxed_only: require \\\\boxed{...} for model output (recommended). "
        "legacy_last_line: last line if no box (old behavior).",
    )
    parser.add_argument(
        "--mc_trials",
        type=int,
        default=5000,
        help="Monte Carlo passes: each pass picks one random branch per problem. "
        "Mean should match analytic random-one-branch rate; stdev shows simulation noise.",
    )
    parser.add_argument("--random_seed", type=int, default=42)
    args = parser.parse_args()

    cov_by_idx = _load_coverage_by_index(args.out_dir)

    rows = load_jsonl(args.data, limit=args.limit)
    per_problem: list[dict[str, Any]] = []
    n_with_ref = 0
    branch_correct_counts = [0] * D.N_BRANCHES

    for i, row in enumerate(rows):
        pid = str(row.get("unique_id", f"problem_{i:04d}"))
        pid_safe = safe_problem_id(pid)
        pdir = args.out_dir / "per_problem" / pid_safe
        sol_dir = pdir / "solution_texts"
        ref = get_reference_answer(row)
        subject = str(row.get("subject", "unknown"))

        entry: dict[str, Any] = {
            "index": i,
            "unique_id": pid,
            "problem_id_safe": pid_safe,
            "subject": subject,
            "has_reference": ref is not None,
        }

        if cov_by_idx:
            entry["checkpoints_with_alive_branch"] = cov_by_idx.get(i)
            entry["checkpoint_coverage_note"] = (
                "Count of scheduled checkpoints where num_alive>0 (see coverage_report.json)."
            )

        if ref is None:
            entry["error"] = "no reference in row"
            per_problem.append(entry)
            continue

        n_with_ref += 1
        flags: list[bool] = []
        n_with_boxed = 0
        for b in range(D.N_BRANCHES):
            fpath = sol_dir / f"branch_{b}.txt"
            if not fpath.is_file():
                flags.append(False)
                continue
            text = fpath.read_text(encoding="utf-8")
            if "\\boxed{" in text:
                n_with_boxed += 1
            flags.append(answers_equal(text, ref, match_mode=args.match_mode))

        n_correct = sum(flags)
        for b, ok in enumerate(flags):
            if ok:
                branch_correct_counts[b] += 1
        b0 = flags[0] if flags else False
        any_ok = any(flags)
        p_random_one = n_correct / float(D.N_BRANCHES)

        entry.update(
            {
                "branch_correct": flags,
                "branches_with_boxed_in_text": n_with_boxed,
                "branch_0_correct": b0,
                "any_branch_correct": any_ok,
                "num_branches_correct": n_correct,
                "prob_random_one_branch_correct": p_random_one,
                "rescue": (not b0) and any_ok,
            }
        )
        per_problem.append(entry)

    pass_at_b0 = sum(1 for p in per_problem if p.get("branch_0_correct")) / max(n_with_ref, 1)
    pass_at_any = sum(1 for p in per_problem if p.get("any_branch_correct")) / max(n_with_ref, 1)
    total_branch_slots = n_with_ref * D.N_BRANCHES
    sum_correct_branch_slots = sum(branch_correct_counts)
    # Primary: E[correct | uniform random branch] = mean_i (k_i / 16)
    expected_random_one_branch = sum_correct_branch_slots / max(total_branch_slots, 1)
    expected_random_one_branch_alt = (
        sum(p.get("num_branches_correct", 0) for p in per_problem if p.get("has_reference"))
        / max(n_with_ref * D.N_BRANCHES, 1)
    )
    per_branch_accuracy = [branch_correct_counts[b] / max(n_with_ref, 1) for b in range(D.N_BRANCHES)]
    n_fail_b0 = sum(1 for p in per_problem if p.get("has_reference") and not p.get("branch_0_correct"))
    n_rescue = sum(1 for p in per_problem if p.get("rescue"))
    rescue_conditional = n_rescue / max(n_fail_b0, 1)

    dist = Counter()
    for p in per_problem:
        if p.get("has_reference"):
            dist[p.get("num_branches_correct", 0)] += 1

    # Stratify: checkpoint coverage vs num_branches_correct
    strat: dict[str, dict[str, float]] = {}
    bucket: dict[int, list[int]] = defaultdict(list)
    for p in per_problem:
        if not p.get("has_reference"):
            continue
        k = int(p.get("num_branches_correct", 0))
        vc = p.get("checkpoints_with_alive_branch")
        if isinstance(vc, int):
            bucket[k].append(vc)
    for k, vals in sorted(bucket.items()):
        if not vals:
            continue
        strat[str(k)] = {
            "n_problems": len(vals),
            "mean_checkpoints_with_alive": float(sum(vals) / len(vals)),
            "min_cp": float(min(vals)),
            "max_cp": float(max(vals)),
        }

    by_subject: dict[str, dict[str, int | float]] = {}
    for p in per_problem:
        if not p.get("has_reference"):
            continue
        sub = p.get("subject", "unknown")
        if sub not in by_subject:
            by_subject[sub] = {"n": 0, "pass_b0": 0, "pass_any": 0, "rescues": 0, "sum_random_p": 0.0}
        by_subject[sub]["n"] += 1  # type: ignore
        if p.get("branch_0_correct"):
            by_subject[sub]["pass_b0"] += 1  # type: ignore
        if p.get("any_branch_correct"):
            by_subject[sub]["pass_any"] += 1  # type: ignore
        if p.get("rescue"):
            by_subject[sub]["rescues"] += 1  # type: ignore
        by_subject[sub]["sum_random_p"] += float(p.get("prob_random_one_branch_correct", 0.0))  # type: ignore

    for sub in by_subject:
        n = int(by_subject[sub]["n"])
        by_subject[sub]["pass_b0_rate"] = float(by_subject[sub]["pass_b0"]) / max(n, 1)
        by_subject[sub]["pass_any_rate"] = float(by_subject[sub]["pass_any"]) / max(n, 1)
        by_subject[sub]["mean_random_one_branch_rate"] = float(by_subject[sub]["sum_random_p"]) / max(n, 1)
        del by_subject[sub]["sum_random_p"]

    # Monte Carlo: whole-dataset passes, each pass picks independent uniform branch per problem
    rng = random.Random(args.random_seed)
    mc_rates: list[float] = []
    for _ in range(max(args.mc_trials, 0)):
        hits = 0
        for p in per_problem:
            bc = p.get("branch_correct")
            if not bc or len(bc) != D.N_BRANCHES:
                continue
            b = rng.randrange(D.N_BRANCHES)
            if bc[b]:
                hits += 1
        mc_rates.append(hits / max(n_with_ref, 1))

    mc_block: dict[str, Any] = {}
    if mc_rates:
        mc_block = {
            "trials": len(mc_rates),
            "seed": args.random_seed,
            "mean_rate": float(statistics.mean(mc_rates)),
            "stdev": float(statistics.stdev(mc_rates)) if len(mc_rates) > 1 else 0.0,
            "note": "Each trial: for every problem, pick one uniform random branch; fraction correct.",
        }

    out = {
        "match_mode": args.match_mode,
        "primary_metric": (
            "expected_accuracy_one_uniform_random_branch_per_problem "
            "(equals mean of num_correct/16, equals total correct slots / (50*16))"
        ),
        "expected_accuracy_random_one_branch": expected_random_one_branch,
        "expected_accuracy_random_one_branch_cross_check": expected_random_one_branch_alt,
        "pass_branch0_only": pass_at_b0,
        "pass_any_of_16": pass_at_any,
        "definitions": {
            "pass_branch0_only": "Fraction of problems where branch index 0 matches reference (batch row 0).",
            "pass_any_of_16": "Fraction of problems where at least one of 16 branches matches.",
            "random_one_branch": (
                "For each problem, draw branch index ~ Uniform({0..15}) independently. "
                "Probability correct = k/16 if k branches match. Dataset-level rate = average of k/16."
            ),
        },
        "num_problems": len(rows),
        "num_with_reference": n_with_ref,
        "per_branch_accuracy": {str(b): per_branch_accuracy[b] for b in range(D.N_BRANCHES)},
        "branch_correct_counts_across_problems": {
            str(b): branch_correct_counts[b] for b in range(D.N_BRANCHES)
        },
        "absolute_gap_any_minus_b0": pass_at_any - pass_at_b0,
        "count_branch0_correct": sum(1 for p in per_problem if p.get("branch_0_correct")),
        "count_any_branch_correct": sum(1 for p in per_problem if p.get("any_branch_correct")),
        "count_rescue_b0_wrong_some_right": n_rescue,
        "rescue_rate_among_b0_failures": rescue_conditional,
        "distribution_num_branches_correct": {str(k): v for k, v in sorted(dist.items())},
        "stratified_checkpoint_coverage_by_num_branches_correct": strat,
        "monte_carlo_random_one_branch": mc_block,
        "by_subject": by_subject,
        "per_problem": per_problem,
        "matching_note": (
            "boxed_only: model must put final answer in \\\\boxed{...}. "
            "legacy_last_line: if no box, last non-empty line is used (can mis-score long CoT)."
        ),
    }

    out_path = args.out_dir / "analysis" / "answer_breakdown.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    slim = {k: v for k, v in out.items() if k != "per_problem"}
    print(json.dumps(slim, indent=2))
    print(f"\nWrote {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
