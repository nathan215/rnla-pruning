"""
Path-length vs correctness for Step 1 oracle (multi-batch 200 problems).

- **Truncated branch:** `finished_at[b] == -1` (no EOS before end of generation run; ran to cap).
  These are almost always unreliable finals; report separately from wrong-but-finished.
- **Length:** `valid_lengths[b]` = generated token count for branch b.

Output: results/step_1_oracle_200_length_correctness.json (default)

Run:
  python experiments/step1_length_correctness_analysis.py
  python experiments/step1_length_correctness_analysis.py --results-root results --limit 200
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import defaults as D  # noqa: E402
from src.answer_matching import answers_equal, get_reference_answer, safe_problem_id  # noqa: E402
from src.data_loader import load_jsonl  # noqa: E402


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
        "p5": pct(0.05),
        "p25": pct(0.25),
        "p50": pct(0.50),
        "p75": pct(0.75),
        "p95": pct(0.95),
        "max": float(s[-1]),
        "mean": float(sum(s) / max(n, 1)),
    }


def _hist(xs: list[int], bins: list[int]) -> dict[str, int]:
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
    parser.add_argument("--data", type=Path, default=PROJECT_ROOT / D.DATA_REL_PATH)
    parser.add_argument("--results-root", type=Path, default=PROJECT_ROOT / "results")
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument(
        "--match_mode",
        choices=("boxed_only", "legacy_last_line"),
        default="boxed_only",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Default: <results-root>/step_1_oracle_200_length_correctness.json",
    )
    args = parser.parse_args()

    root = args.results_root.resolve()
    out_path = args.out or (root / "step_1_oracle_200_length_correctness.json")

    rows = load_jsonl(args.data, limit=args.limit)
    n_rows_with_ref = sum(1 for r in rows if get_reference_answer(r) is not None)
    batch_roots = [root / f"step_1_oracle_b{i}" for i in range(8)]

    # Per-branch records
    lengths_correct: list[int] = []
    lengths_wrong_finished: list[int] = []  # wrong and not truncated
    lengths_wrong_truncated: list[int] = []
    lengths_correct_trunc: list[int] = []  # should be rare

    n_total = 0
    n_correct = 0
    n_wrong = 0
    n_trunc = 0
    cross = Counter()  # (correct, truncated) -> count

    missing_pt: list[str] = []

    for idx, row in enumerate(rows):
        pid = str(row.get("unique_id", f"problem_{idx:04d}"))
        pid_safe = safe_problem_id(pid)
        bdir = batch_roots[idx // 25] / "per_problem" / pid_safe
        pt_path = bdir / "oracle_data.pt"
        sol_dir = bdir / "solution_texts"
        ref = get_reference_answer(row)

        if ref is None:
            continue

        if not pt_path.is_file():
            missing_pt.append(pid_safe)
            continue

        try:
            kw = {"map_location": "cpu"}
            try:
                o = torch.load(pt_path, **kw, weights_only=False)
            except TypeError:
                o = torch.load(pt_path, **kw)
        except Exception as e:
            missing_pt.append(f"{pid_safe}: {e}")
            continue

        vl = o.get("valid_lengths")
        fa = o.get("finished_at")
        tmax = int(o.get("T_max", D.STEP1_MAX_NEW_TOKENS))
        if vl is None or fa is None:
            continue
        vl = vl.long().view(-1)
        fa = fa.long().view(-1)

        for b in range(D.N_BRANCHES):
            n_total += 1
            L = int(vl[b].item())
            fin = int(fa[b].item())
            truncated = fin < 0  # never emitted EOS in stored run

            fp = sol_dir / f"branch_{b}.txt"
            text = fp.read_text(encoding="utf-8") if fp.is_file() else ""
            ok = answers_equal(text, ref, match_mode=args.match_mode)

            if ok:
                n_correct += 1
                lengths_correct.append(L)
                if truncated:
                    lengths_correct_trunc.append(L)
            else:
                n_wrong += 1
                if truncated:
                    n_trunc += 1
                    lengths_wrong_truncated.append(L)
                else:
                    lengths_wrong_finished.append(L)

            cross[(ok, truncated)] += 1

    bins = [0, 512, 1024, 2048, 4096, 8192, 10000]

    payload = {
        "definitions": {
            "length_tokens": "valid_lengths[b] — generated tokens for branch b.",
            "truncated": (
                "finished_at[b] < 0 (no EOS observed). Generation hit the step cap without finishing; "
                "treat as unreliable / wrong for policy analysis."
            ),
            "correct": f"answers_equal(branch_text, ref, match_mode={args.match_mode!r}).",
        },
        "match_mode": args.match_mode,
        "n_branch_slots_analyzed": n_total,
        "n_problems_with_reference": n_rows_with_ref,
        "counts": {
            "correct_branches": n_correct,
            "wrong_branches": n_wrong,
            "truncated_branches": n_trunc,
            "truncated_among_wrong": n_trunc,
            "fraction_truncated_of_all_branches": n_trunc / max(n_total, 1),
            "fraction_truncated_of_wrong_only": n_trunc / max(n_wrong, 1),
            "fraction_correct": n_correct / max(n_total, 1),
        },
        "cross_tab_correct_vs_truncated": {
            "(correct, truncated)": int(cross[(True, True)]),
            "(correct, not_truncated)": int(cross[(True, False)]),
            "(wrong, truncated)": int(cross[(False, True)]),
            "(wrong, not_truncated)": int(cross[(False, False)]),
        },
        "length_percentiles": {
            "correct_branches": _percentiles(lengths_correct),
            "wrong_branches_finished": _percentiles(lengths_wrong_finished),
            "wrong_branches_truncated": _percentiles(lengths_wrong_truncated),
        },
        "length_histograms": {
            "correct_branches": _hist(lengths_correct, bins),
            "wrong_branches_finished": _hist(lengths_wrong_finished, bins),
            "wrong_branches_truncated": _hist(lengths_wrong_truncated, bins),
        },
        "note_correct_truncated": (
            f"Branches marked correct but truncated: {len(lengths_correct_trunc)} "
            "(unusual; possible edge case or grading vs partial output)."
        ),
        "missing_or_failed_loads": missing_pt[:50],
        "output_path": str(out_path),
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in payload.items() if k != "missing_or_failed_loads"}, indent=2))
    if missing_pt:
        print(f"\nWARN: {len(missing_pt)} load/skip issues (see JSON)", file=sys.stderr)
    print(f"\nWrote {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
