"""
Step 1 Part 1: Oracle collection with fixed checkpoints.

Run from project root:
  python experiments/step_1_oracle_collection.py --limit 2
  python experiments/step_1_oracle_collection.py ... --skip_completed   # resume a batch after interrupt
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import torch
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import defaults as D  # noqa: E402
from src.answer_matching import answers_equal, get_reference_answer, safe_problem_id  # noqa: E402
from src.data_loader import load_jsonl, problem_text  # noqa: E402
from src.model_loader import load_model_and_tokenizer  # noqa: E402
from src.step1_oracle import collect_problem_oracle  # noqa: E402


def decode_branches(
    tokenizer,
    generated_token_ids: torch.Tensor,
    valid_lengths: torch.Tensor,
) -> list[str]:
    texts: list[str] = []
    for b in range(generated_token_ids.shape[0]):
        vlen = int(valid_lengths[b].item())
        ids = generated_token_ids[b, :vlen].tolist()
        txt = tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        texts.append(txt)
    return texts


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def summarize_coverage(per_problem_cov: list[dict[int, dict]], checkpoints: list[int]) -> dict:
    per_cp_valid_problem_ratio: dict[str, float] = {}
    per_cp_mean_alive: dict[str, float] = {}
    valid_checkpoints_per_problem: list[dict[str, int | str]] = []
    for idx, cov in enumerate(per_problem_cov):
        valid_cnt = sum(1 for cp in checkpoints if int(cov.get(cp, {"num_alive": 0})["num_alive"]) > 0)
        valid_checkpoints_per_problem.append(
            {"problem_offset": idx, "valid_checkpoints": int(valid_cnt), "total_checkpoints": len(checkpoints)}
        )

    n_prob = max(len(per_problem_cov), 1)
    for cp in checkpoints:
        alive_counts = [int(cov.get(cp, {"num_alive": 0})["num_alive"]) for cov in per_problem_cov]
        valid_prob = sum(1 for x in alive_counts if x > 0)
        per_cp_valid_problem_ratio[str(cp)] = float(valid_prob / n_prob)
        per_cp_mean_alive[str(cp)] = float(sum(alive_counts) / n_prob)

    return {
        "valid_checkpoints_per_problem": valid_checkpoints_per_problem,
        "valid_problem_ratio_by_checkpoint": per_cp_valid_problem_ratio,
        "mean_alive_branches_by_checkpoint": per_cp_mean_alive,
    }


def _integrity_summary(oracle_payload: dict, checkpoints: list[int]) -> dict[str, bool]:
    ok_hidden = all(
        int(oracle_payload["checkpoint_hidden_states"][cp].shape[0]) == D.N_BRANCHES
        for cp in checkpoints
        if cp in oracle_payload["checkpoint_hidden_states"]
    )
    ok_logits = all(
        int(oracle_payload["checkpoint_logits"][cp].shape[0]) == D.N_BRANCHES
        for cp in checkpoints
        if cp in oracle_payload["checkpoint_logits"]
    )
    ok_mask = all(
        int(oracle_payload["alive_mask"][cp].shape[0]) == D.N_BRANCHES
        for cp in checkpoints
        if cp in oracle_payload["alive_mask"]
    )
    ok_cum = True
    if "checkpoint_cumulative_logprob" in oracle_payload:
        ccl = oracle_payload["checkpoint_cumulative_logprob"]
        ok_cum = all(
            int(ccl[cp].shape[0]) == D.N_BRANCHES for cp in checkpoints if cp in ccl
        )
    ok_psl = True
    if "per_step_logprob" in oracle_payload:
        psl = oracle_payload["per_step_logprob"]
        ok_psl = int(psl.shape[0]) == D.N_BRANCHES
    return {
        "hidden_shape_ok": bool(ok_hidden),
        "logits_shape_ok": bool(ok_logits),
        "alive_mask_shape_ok": bool(ok_mask),
        "cumulative_logprob_ok": bool(ok_cum),
        "per_step_logprob_ok": bool(ok_psl),
    }


def _load_oracle_if_skippable(
    pdir: Path,
    sdir: Path,
    checkpoints: list[int],
    effective_tmax: int,
) -> dict | None:
    """Return oracle payload if on-disk artifacts are complete and match this run's lock."""
    pt_path = pdir / "oracle_data.pt"
    if not pt_path.is_file() or not (pdir / "problem_metadata.json").is_file():
        return None
    try:
        kw: dict = {"map_location": "cpu"}
        try:
            o = torch.load(pt_path, **kw, weights_only=False)
        except TypeError:
            o = torch.load(pt_path, **kw)
    except Exception:
        return None
    if not isinstance(o, dict):
        return None
    if int(o.get("T_max", -1)) != int(effective_tmax):
        return None
    if list(o.get("checkpoints", [])) != checkpoints:
        return None
    if not all(integ for integ in _integrity_summary(o, checkpoints).values()):
        return None
    for b in range(D.N_BRANCHES):
        if not (sdir / f"branch_{b}.txt").is_file():
            return None
    return o


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=PROJECT_ROOT / D.DATA_REL_PATH)
    parser.add_argument("--out_dir", type=Path, default=PROJECT_ROOT / "results" / "step_1_oracle")
    parser.add_argument("--limit", type=int, default=1)
    parser.add_argument(
        "--offset",
        type=int,
        default=0,
        help="Start index into the JSONL (0-based). Use with --limit for non-overlapping "
        "batches, e.g. --offset 0 --limit 25, then --offset 25 --limit 25, ...",
    )
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument(
        "--run_stage",
        choices=("smoke", "full"),
        default="smoke",
        help="Run order default: smoke first. Use full for 50-problem run.",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=500,
        help="For smoke only. Full stage enforces Step1 lock T_max=8192.",
    )
    parser.add_argument(
        "--confirm_full_run",
        action="store_true",
        help="Required with --run_stage full to avoid accidental long runs.",
    )
    parser.add_argument(
        "--answer_match_mode",
        choices=("boxed_only", "legacy_last_line"),
        default="boxed_only",
        help=(
            "How to read model answers for pass@k metrics. "
            "'boxed_only' requires \\\\boxed{...} (no last-line fallback). "
            "'legacy_last_line' matches old behavior (last line if no box)."
        ),
    )
    parser.add_argument(
        "--skip_completed",
        action="store_true",
        help=(
            "If per_problem/<id>/oracle_data.pt, problem_metadata.json, and all branch_*.txt "
            "exist and match T_max/checkpoints with passing integrity, skip regeneration for that problem."
        ),
    )
    args = parser.parse_args()

    if args.limit <= 0:
        raise ValueError("--limit must be > 0")
    if args.offset < 0:
        raise ValueError("--offset must be >= 0")
    if args.run_stage == "full" and not args.confirm_full_run:
        raise ValueError("Full run requires --confirm_full_run.")

    if args.run_stage == "full":
        effective_limit = D.STEP1_NUM_PROBLEMS
        effective_tmax = D.STEP1_MAX_NEW_TOKENS
    else:
        effective_limit = int(args.limit)
        effective_tmax = int(args.max_new_tokens)

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    rows = load_jsonl(args.data)
    offset = int(args.offset)
    if offset >= len(rows):
        raise ValueError(f"--offset {offset} >= dataset length {len(rows)}")
    available = len(rows) - offset
    if args.run_stage == "full":
        n_run = min(effective_limit, available, D.STEP1_NUM_PROBLEMS)
    else:
        n_run = min(effective_limit, available)
    if n_run <= 0:
        raise RuntimeError("No problems to run (check --limit and --offset).")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer = load_model_and_tokenizer(D.MODEL_ID, device)

    checkpoints = [int(x) for x in D.STEP1_CHECKPOINTS if x <= effective_tmax]
    per_problem_cov: list[dict[int, dict]] = []
    per_problem_elapsed: list[float] = []
    progress_rows: list[dict] = []
    per_problem_any_correct: list[bool] = []
    per_problem_first_correct: list[bool] = []
    n_with_reference = 0
    run_t0 = time.perf_counter()

    for k in tqdm(range(n_run), desc="Step1 Part1"):
        i = offset + k
        row = rows[i]
        pid = str(row.get("unique_id", f"problem_{i:04d}"))
        pid_safe = safe_problem_id(pid)
        pdir = args.out_dir / "per_problem" / pid_safe
        sdir = pdir / "solution_texts"
        sdir.mkdir(parents=True, exist_ok=True)

        if args.skip_completed:
            cached = _load_oracle_if_skippable(pdir, sdir, checkpoints, effective_tmax)
            if cached is not None:
                tqdm.write(f"[skip_completed] {pid_safe} (dataset index {i})")
                per_problem_cov.append(cached["coverage"])
                per_problem_elapsed.append(
                    float(cached.get("timing", {}).get("elapsed_sec", 0.0))
                )
                texts = [
                    (sdir / f"branch_{b}.txt").read_text(encoding="utf-8")
                    for b in range(D.N_BRANCHES)
                ]
                ref = get_reference_answer(row)
                if ref is not None:
                    n_with_reference += 1
                    flags = [answers_equal(t, ref, match_mode=args.answer_match_mode) for t in texts]
                    per_problem_any_correct.append(any(flags))
                    per_problem_first_correct.append(bool(flags[0]))
                integrity = _integrity_summary(cached, checkpoints)
                progress_rows.append(
                    {
                        "problem_index": i,
                        "problem_id_safe": pid_safe,
                        "elapsed_sec": float(cached.get("timing", {}).get("elapsed_sec", 0.0)),
                        "integrity": integrity,
                        "skipped": True,
                    }
                )
                if ((k + 1) % 10 == 0) or (k + 1 == n_run):
                    write_json(
                        args.out_dir / "analysis" / "progress.json",
                        {
                            "processed": k + 1,
                            "target": n_run,
                            "dataset_offset": offset,
                            "rows": progress_rows,
                        },
                    )
                continue

        problem = problem_text(row)
        result = collect_problem_oracle(
            model,
            tokenizer,
            problem,
            device,
            n_branches=D.N_BRANCHES,
            max_new_tokens=effective_tmax,
            checkpoints=checkpoints,
            temperature=D.TEMPERATURE,
            do_sample=True,
        )
        per_problem_cov.append(result["coverage"])
        per_problem_elapsed.append(float(result["elapsed_sec"]))

        texts = decode_branches(tokenizer, result["generated_token_ids"], result["valid_lengths"])
        for b, txt in enumerate(texts):
            (sdir / f"branch_{b}.txt").write_text(txt, encoding="utf-8")

        ref = get_reference_answer(row)
        if ref is not None:
            n_with_reference += 1
            flags = [answers_equal(t, ref, match_mode=args.answer_match_mode) for t in texts]
            per_problem_any_correct.append(any(flags))
            per_problem_first_correct.append(bool(flags[0]))

        oracle_payload = {
            "problem_id": pid,
            "problem_id_safe": pid_safe,
            "N": D.N_BRANCHES,
            "T_max": effective_tmax,
            "checkpoints": checkpoints,
            "all_token_ids": result["generated_token_ids"],
            "valid_lengths": result["valid_lengths"],
            "finished_at": result["finished_at"],
            "per_step_logprob": result["per_step_logprob"],
            "checkpoint_hidden_states": result["checkpoint_hidden_states"],
            "checkpoint_logits": result["checkpoint_logits"],
            "checkpoint_cumulative_logprob": result["checkpoint_cumulative_logprob"],
            "alive_mask": result["alive_mask"],
            "coverage": result["coverage"],
            "generation_config": {
                "temperature": D.TEMPERATURE,
                "do_sample": True,
                "model_id": D.MODEL_ID,
                "logprob_semantics": (
                    "per_step_logprob[b,t-1]=log p(x_t|x_<t) under softmax(logits_{t-1}/T); "
                    "checkpoint_cumulative_logprob[cp][b]=sum_{k=1}^{cp} per_step_logprob[b,k-1] "
                    "when step cp was reached; after EOS, later checkpoints repeat final sum."
                ),
            },
            "timing": {"elapsed_sec": float(result["elapsed_sec"])},
        }
        torch.save(oracle_payload, pdir / "oracle_data.pt")

        problem_meta = {
            "problem_id": pid,
            "problem_id_safe": pid_safe,
            "index": i,
            "dataset_offset": offset,
            "batch_index": k,
            "checkpoints": checkpoints,
            "elapsed_sec": float(result["elapsed_sec"]),
            "num_alive_final": int(result["coverage"].get(checkpoints[-1], {"num_alive": 0})["num_alive"]),
            "oracle_data_file": str((pdir / "oracle_data.pt").resolve()),
        }
        write_json(pdir / "problem_metadata.json", problem_meta)
        integrity = _integrity_summary(oracle_payload, checkpoints)
        progress_rows.append(
            {
                "problem_index": i,
                "problem_id_safe": pid_safe,
                "elapsed_sec": float(result["elapsed_sec"]),
                "integrity": integrity,
                "skipped": False,
            }
        )
        if ((k + 1) % 10 == 0) or (k + 1 == n_run):
            write_json(
                args.out_dir / "analysis" / "progress.json",
                {
                    "processed": k + 1,
                    "target": n_run,
                    "dataset_offset": offset,
                    "rows": progress_rows,
                },
            )

        del result, oracle_payload
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    total_elapsed = time.perf_counter() - run_t0
    coverage_summary = summarize_coverage(per_problem_cov, checkpoints)

    metadata = {
        "step": 1,
        "num_problems": n_run,
        "dataset_offset": offset,
        "dataset_slice": f"[{offset}:{offset + n_run})",
        "N": D.N_BRANCHES,
        "T_max": effective_tmax,
        "checkpoints": checkpoints,
        "model_id": D.MODEL_ID,
        "seed": args.seed,
        "mean_sec_per_problem": float(sum(per_problem_elapsed) / max(len(per_problem_elapsed), 1)),
        "total_elapsed_sec": float(total_elapsed),
        "pad_token_id": int(tokenizer.pad_token_id) if tokenizer.pad_token_id is not None else None,
        "eos_token_id": int(tokenizer.eos_token_id) if tokenizer.eos_token_id is not None else None,
        "storage_policy": (
            "fixed-shape tensors + explicit alive_mask; per_step_logprob float32; "
            "checkpoint cumulative logprob per branch; no NaN in core tensors"
        ),
        "run_stage": args.run_stage,
        "answer_match_mode": args.answer_match_mode,
        "skip_completed": bool(args.skip_completed),
    }
    write_json(args.out_dir / "metadata.json", metadata)

    if n_with_reference > 0:
        pass_at_k = {
            "computed": True,
            "num_problems_total": n_run,
            "num_problems_with_reference": n_with_reference,
            "N": D.N_BRANCHES,
            "pass_at_1": float(sum(per_problem_first_correct) / n_with_reference),
            "oracle_pass_at_16": float(sum(per_problem_any_correct) / n_with_reference),
            "answer_match_mode": args.answer_match_mode,
            "note": (
                "pass_at_1 = branch 0 only. oracle_pass_at_16 = any branch. "
                "See --answer_match_mode (boxed_only vs legacy_last_line)."
            ),
        }
    else:
        pass_at_k = {
            "computed": False,
            "reason": "No reference answer field found (answer/solution/target).",
            "num_problems_total": n_run,
            "N": D.N_BRANCHES,
        }
    write_json(args.out_dir / "analysis" / "pass_at_k.json", pass_at_k)
    write_json(args.out_dir / "analysis" / "coverage_report.json", coverage_summary)

    memo = {
        "step1_recommendation": {
            "N": D.N_BRANCHES,
            "T_max_policy": "Keep 8192 for full oracle collection; evaluate staged cap post-hoc in Step 2.",
            "checkpoint_plan": checkpoints,
            "notes": [
                "Coverage diagnostics included in analysis/coverage_report.json.",
                "Step 0 Part C Spearman assessed up to 500 tokens; interpret late checkpoints with coverage caveat.",
            ],
        }
    }
    write_json(args.out_dir / "analysis" / "step1_recommendation_memo.json", memo)


if __name__ == "__main__":
    main()
