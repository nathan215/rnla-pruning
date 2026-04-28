"""
Aggregate Step 1 oracle results across step_1_oracle_b0 … b7 (200 problems).

Writes compact reports only (does not merge per_problem/oracle_data.pt).

  python experiments/step1_aggregate_batches.py
  python experiments/step1_aggregate_batches.py --skip-diversity

Outputs:
  results/step_1_oracle_200_summary.json
  results/step_1_oracle_200_report.md
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import defaults as D  # noqa: E402
from src.answer_matching import (  # noqa: E402
    extract_prediction_for_match,
    answers_equal,
    get_reference_answer,
    safe_problem_id,
)
from src.data_loader import load_jsonl  # noqa: E402


def _read_json(path: Path) -> dict | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def aggregate_pass_at_k_files(batch_dirs: list[Path]) -> tuple[dict | None, list[dict]]:
    per_batch: list[dict] = []
    w1_num = 0.0
    w16_num = 0.0
    w_den = 0

    for i, bd in enumerate(batch_dirs):
        pak = _read_json(bd / "analysis" / "pass_at_k.json")
        meta = _read_json(bd / "metadata.json")
        row: dict = {
            "batch_index": i,
            "dir": bd.name,
            "path": str(bd),
        }
        if pak:
            row["pass_at_k"] = {
                "computed": pak.get("computed"),
                "num_problems_total": pak.get("num_problems_total"),
                "num_problems_with_reference": pak.get("num_problems_with_reference"),
                "pass_at_1": pak.get("pass_at_1"),
                "oracle_pass_at_16": pak.get("oracle_pass_at_16"),
                "answer_match_mode": pak.get("answer_match_mode"),
            }
            if pak.get("computed") and pak.get("num_problems_with_reference"):
                nref = int(pak["num_problems_with_reference"])
                w_den += nref
                w1_num += float(pak["pass_at_1"]) * nref
                w16_num += float(pak["oracle_pass_at_16"]) * nref
        if meta:
            row["run"] = {
                "T_max": meta.get("T_max"),
                "dataset_slice": meta.get("dataset_slice"),
                "total_elapsed_sec": meta.get("total_elapsed_sec"),
                "mean_sec_per_problem": meta.get("mean_sec_per_problem"),
                "skip_completed": meta.get("skip_completed"),
            }
        per_batch.append(row)

    merged = None
    if w_den > 0:
        merged = {
            "n_problems_with_reference": w_den,
            "pass_at_1_reference_weighted": w1_num / w_den,
            "oracle_pass_at_16_reference_weighted": w16_num / w_den,
            "note": "Weighted by num_problems_with_reference in each batch pass_at_k.json.",
        }
    return merged, per_batch


def scan_from_branch_texts(
    batch_dirs: list[Path],
    rows: list[dict],
    match_mode: str,
) -> dict:
    n_ref = 0
    n_b0 = 0
    n_any = 0
    dist_correct = Counter()
    dist_distinct = Counter()

    for idx, row in enumerate(rows):
        pid = str(row.get("unique_id", f"problem_{idx:04d}"))
        pid_safe = safe_problem_id(pid)
        batch_i = idx // 25
        sol = batch_dirs[batch_i] / "per_problem" / pid_safe / "solution_texts"
        ref = get_reference_answer(row)
        if ref is None:
            continue

        texts: list[str] = []
        for b in range(D.N_BRANCHES):
            fp = sol / f"branch_{b}.txt"
            texts.append(fp.read_text(encoding="utf-8") if fp.is_file() else "")

        flags = [answers_equal(t, ref, match_mode=match_mode) for t in texts]
        extracted = [extract_prediction_for_match(t, match_mode=match_mode) for t in texts]
        distinct_n = len(set(extracted))

        k = sum(flags)
        n_ref += 1
        n_b0 += int(flags[0])
        n_any += int(any(flags))
        dist_correct[k] += 1
        dist_distinct[distinct_n] += 1

    return {
        "n_with_reference": n_ref,
        "pass_at_1": n_b0 / max(n_ref, 1),
        "oracle_pass_at_16": n_any / max(n_ref, 1),
        "distribution_num_branches_correct": {str(k): dist_correct[k] for k in sorted(dist_correct)},
        "distribution_num_distinct_extracted_answers": {
            str(k): dist_distinct[k] for k in sorted(dist_distinct)
        },
        "definitions": {
            "pass_at_1": "branch 0 matches reference",
            "oracle_pass_at_16": "at least one of 16 branches matches",
            "num_distinct_extracted_answers": (
                "Distinct strings from extract_prediction_for_match per branch; "
                "empty string counts as one value if present."
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=Path, default=PROJECT_ROOT / "results")
    parser.add_argument(
        "--match_mode",
        choices=("boxed_only", "legacy_last_line"),
        default="boxed_only",
    )
    parser.add_argument(
        "--skip-diversity",
        action="store_true",
        help="Skip scanning all branch_*.txt (only aggregate pass_at_k.json files).",
    )
    parser.add_argument("--limit", type=int, default=200)
    args = parser.parse_args()

    root = args.results_root.resolve()
    batch_dirs = [root / f"step_1_oracle_b{i}" for i in range(8)]

    merged_json, per_batch = aggregate_pass_at_k_files(batch_dirs)
    rows = load_jsonl(PROJECT_ROOT / D.DATA_REL_PATH, limit=args.limit)

    from_branch = None
    if not args.skip_diversity:
        from_branch = scan_from_branch_texts(batch_dirs, rows, args.match_mode)

    total_sec = sum(
        float((b.get("run") or {}).get("total_elapsed_sec") or 0) for b in per_batch
    )

    out = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "match_mode": args.match_mode,
        "data_rows_used": len(rows),
        "batch_dirs": [d.name for d in batch_dirs],
        "merged_pass_at_k_from_json_files": merged_json,
        "per_batch": per_batch,
        "total_elapsed_sec_sum_batches": total_sec if total_sec > 0 else None,
        "recomputed_from_branch_texts_0_199": from_branch,
        "storage_note": (
            "Per-problem data remain under step_1_oracle_b0…b7; "
            "oracle_data.pt not merged (too large)."
        ),
    }

    json_path = root / "step_1_oracle_200_summary.json"
    json_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"Wrote {json_path}", file=sys.stderr)

    # Short markdown report
    lines = [
        "# Step 1 aggregate (200 problems, b0–b7)",
        "",
        f"- Generated: `{out['generated_at']}`",
        f"- Match mode: **{args.match_mode}**",
        "",
        "## Pass@1 / Oracle Pass@16",
        "",
    ]
    lines.extend(
        [
            "| Source | Pass@1 | Oracle Pass@16 |",
            "|--------|--------|----------------|",
        ]
    )
    if merged_json:
        lines.append(
            f"| Weighted from each batch `analysis/pass_at_k.json` | "
            f"{merged_json['pass_at_1_reference_weighted']:.4f} | "
            f"{merged_json['oracle_pass_at_16_reference_weighted']:.4f} |"
        )
    if from_branch:
        lines.append(
            f"| Recomputed from all `branch_*.txt` (n={from_branch['n_with_reference']}) | "
            f"{from_branch['pass_at_1']:.4f} | {from_branch['oracle_pass_at_16']:.4f} |"
        )
    lines.append("")
    if merged_json and from_branch:
        lines.append(
            "Recomputed vs weighted-from-JSON should match when data are consistent; "
            "small drift indicates rounding or a stale `pass_at_k.json`."
        )
        lines.append("")
    if from_branch:
        lines.extend(
            [
                "## Distribution: number of branches correct (vs reference)",
                "",
                "```",
                json.dumps(from_branch["distribution_num_branches_correct"], indent=2),
                "```",
                "",
                "## Distribution: distinct extracted answers per problem (16 branches)",
                "",
                "```",
                json.dumps(from_branch["distribution_num_distinct_extracted_answers"], indent=2),
                "```",
                "",
            ]
        )
    elif args.skip_diversity:
        lines.append("_Branch-level recompute skipped (`--skip-diversity`)._\n")

    lines.extend(
        [
            "## Batch run times (metadata)",
            "",
            "```",
            json.dumps(
                {
                    b["dir"]: (b.get("run") or {}).get("total_elapsed_sec")
                    for b in per_batch
                },
                indent=2,
            ),
            "```",
            "",
            f"**Sum batch wall times:** {total_sec:.1f} s (~{total_sec / 3600:.2f} h) — not de-duplicated if batches overlapped.",
            "",
            f"Full JSON: `{json_path.name}`",
        ]
    )

    md_path = root / "step_1_oracle_200_report.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {md_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
