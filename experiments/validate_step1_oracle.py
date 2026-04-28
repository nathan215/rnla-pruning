"""
Validate Step 1 oracle outputs under results/step_1_oracle/.

Run from project root:
  python experiments/validate_step1_oracle.py
  python experiments/validate_step1_oracle.py --out_dir results/step_1_oracle
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

EXPECTED_KEYS = {
    "N",
    "T_max",
    "alive_mask",
    "all_token_ids",
    "checkpoint_hidden_states",
    "checkpoint_logits",
    "checkpoints",
    "coverage",
    "finished_at",
    "generation_config",
    "problem_id",
    "problem_id_safe",
    "timing",
    "valid_lengths",
}

def validate_one_pt(path: Path, expected_n: int, folder_name: str | None = None) -> dict:
    """Load oracle_data.pt and return {ok, errors, stats}."""
    errors: list[str] = []
    stats: dict = {}

    try:
        kw = {"map_location": "cpu"}
        try:
            o = torch.load(path, **kw, weights_only=False)
        except TypeError:
            o = torch.load(path, **kw)
    except Exception as e:
        return {"ok": False, "errors": [f"torch.load failed: {e}"], "stats": {}}

    missing = EXPECTED_KEYS - set(o.keys())
    if missing:
        errors.append(f"missing keys: {sorted(missing)}")

    n = int(o.get("N", -1))
    tmax = int(o.get("T_max", -1))
    cps = list(o.get("checkpoints", []))
    tok = o.get("all_token_ids")
    vl = o.get("valid_lengths")
    fa = o.get("finished_at")

    stats["N"] = n
    stats["T_max"] = tmax
    stats["n_checkpoints"] = len(cps)
    stats["checkpoint_first_last"] = (cps[0], cps[-1]) if cps else None

    if tok is not None:
        stats["all_token_ids_shape"] = list(tok.shape)
    if vl is not None:
        stats["valid_lengths_min_max"] = [int(vl.min()), int(vl.max())]
    if fa is not None:
        stats["finished_at_min_max"] = [int(fa.min()), int(fa.max())]

    if n != expected_n:
        errors.append(f"N={n} expected {expected_n}")

    p_safe = str(o.get("problem_id_safe", ""))
    stats["problem_id_safe"] = p_safe
    if folder_name is not None and p_safe and p_safe != folder_name:
        errors.append(f"problem_id_safe mismatch: folder={folder_name} pt={p_safe}")

    ch = o.get("checkpoint_hidden_states", {})
    lg = o.get("checkpoint_logits", {})
    am = o.get("alive_mask", {})
    cv = o.get("coverage", {})
    ccl = o.get("checkpoint_cumulative_logprob", {})

    if len(ch) != len(cps) or len(lg) != len(cps) or len(am) != len(cps) or len(cv) != len(cps):
        errors.append(
            f"checkpoint dict sizes mismatch: hidden={len(ch)} logits={len(lg)} "
            f"mask={len(am)} coverage={len(cv)} vs checkpoints={len(cps)}"
        )
    if ccl and (len(ccl) != len(cps) or set(ccl.keys()) != set(cps)):
        errors.append(
            f"checkpoint_cumulative_logprob keys mismatch: {len(ccl)} vs checkpoints={len(cps)}"
        )

    for cp in cps:
        if cp not in ch or cp not in lg or cp not in am or cp not in cv:
            errors.append(f"missing checkpoint {cp}")
            continue
        if ch[cp].shape[0] != n or lg[cp].shape[0] != n or am[cp].shape[0] != n:
            errors.append(f"shape mismatch at cp={cp}")
        alive = int(am[cp].sum())
        cov_alive = int(cv[cp]["num_alive"])
        if alive != cov_alive:
            errors.append(f"alive mismatch at cp={cp}: mask_sum={alive} coverage={cov_alive}")
        if ch[cp].dtype.is_floating_point and torch.isnan(ch[cp]).any():
            errors.append(f"NaN in checkpoint_hidden_states at cp={cp}")
        if lg[cp].dtype.is_floating_point and torch.isnan(lg[cp]).any():
            errors.append(f"NaN in checkpoint_logits at cp={cp}")
        if ccl and cp in ccl and ccl[cp].shape[0] != n:
            errors.append(f"checkpoint_cumulative_logprob shape at cp={cp}")

    psl = o.get("per_step_logprob")
    if psl is not None:
        if psl.shape[0] != n or (tok is not None and psl.shape[1] != tok.shape[1]):
            errors.append("per_step_logprob shape mismatch")
        if psl.dtype.is_floating_point and torch.isnan(psl).any():
            errors.append("NaN in per_step_logprob")

    if tok is not None and vl is not None:
        if tok.shape[0] != n:
            errors.append("all_token_ids batch dim != N")
        if vl.shape[0] != n:
            errors.append("valid_lengths len != N")

    return {"ok": len(errors) == 0, "errors": errors, "stats": stats}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "step_1_oracle",
    )
    parser.add_argument(
        "--expected_problems",
        type=int,
        default=D.STEP1_NUM_PROBLEMS,
        help="Expected number of per_problem folders for a full run (default 50).",
    )
    args = parser.parse_args()

    root = args.out_dir.resolve()
    per_dir = root / "per_problem"

    summary: dict = {
        "out_dir": str(root),
        "expected_N": D.N_BRANCHES,
        "expected_checkpoints": D.STEP1_CHECKPOINTS,
        "expected_T_max_full_run": D.STEP1_MAX_NEW_TOKENS,
        "global_files": {},
        "per_problem": [],
        "counts": {},
        "all_ok": True,
    }

    # Global JSON files
    for name in (
        "metadata.json",
        "analysis/pass_at_k.json",
        "analysis/coverage_report.json",
        "analysis/step1_recommendation_memo.json",
    ):
        p = root / name
        summary["global_files"][name] = {"exists": p.is_file(), "path": str(p)}

    meta_path = root / "metadata.json"
    if meta_path.is_file():
        try:
            summary["run_metadata"] = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception as e:
            summary["run_metadata_error"] = str(e)
            summary["all_ok"] = False

    if not per_dir.is_dir():
        summary["error"] = f"Missing per_problem dir: {per_dir}"
        summary["all_ok"] = False
        out_json = root / "analysis" / "validation_report.json"
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps(summary, indent=2))
        sys.exit(1)

    problem_dirs = sorted([d for d in per_dir.iterdir() if d.is_dir()])
    summary["counts"]["per_problem_folders"] = len(problem_dirs)

    failures: list[str] = []

    for pdir in problem_dirs:
        pid = pdir.name
        row: dict = {"problem_id_safe": pid, "ok": True, "errors": []}

        pt = pdir / "oracle_data.pt"
        meta = pdir / "problem_metadata.json"
        sol = pdir / "solution_texts"

        if not pt.is_file():
            row["ok"] = False
            row["errors"].append("missing oracle_data.pt")
            failures.append(pid)
        if not meta.is_file():
            row["errors"].append("missing problem_metadata.json")
            row["ok"] = False

        branch_files = list(sol.glob("branch_*.txt")) if sol.is_dir() else []
        row["n_branch_txt"] = len(branch_files)
        if len(branch_files) != D.N_BRANCHES:
            row["errors"].append(f"expected {D.N_BRANCHES} branch_*.txt, got {len(branch_files)}")
            row["ok"] = False

        if pt.is_file():
            vr = validate_one_pt(pt, D.N_BRANCHES, folder_name=pid)
            row["oracle_stats"] = vr.get("stats", {})
            if not vr["ok"]:
                row["ok"] = False
                row["errors"].extend(vr["errors"])
                failures.append(pid)

        if not row["ok"]:
            summary["all_ok"] = False

        summary["per_problem"].append(row)

    # Count OK
    n_ok = sum(1 for r in summary["per_problem"] if r["ok"])
    summary["counts"]["validated_ok"] = n_ok
    summary["counts"]["validated_fail"] = len(summary["per_problem"]) - n_ok

    if len(problem_dirs) < args.expected_problems:
        summary["warnings"] = summary.get("warnings", [])
        summary["warnings"].append(
            f"Only {len(problem_dirs)} problem folders (expected {args.expected_problems} for full run)."
        )

    out_json = root / "analysis" / "validation_report.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    if not summary["all_ok"]:
        print("\nFAILURES:", failures[:20], "..." if len(failures) > 20 else "", file=sys.stderr)
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
