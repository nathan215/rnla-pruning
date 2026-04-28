"""
Step 0: Parts A (profiling), B (VRAM table + recommendation), C (leverage stability).

Run from project root:
  python experiments/step_0_token_analysis.py --part all
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import spearmanr
from tqdm import tqdm

# Suppress a known noisy Transformers warning in this project setup:
# pad_token_id == eos_token_id with decoder-only models can repeatedly emit
# "right-padding was detected" even when tokenizer padding_side is explicitly left.
warnings.filterwarnings(
    "ignore",
    message=r"A decoder-only architecture is being used, but right-padding was detected!.*",
    category=UserWarning,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import defaults as D  # noqa: E402
from src.data_loader import load_jsonl, problem_text  # noqa: E402
from src.leverage import row_leverage_scores  # noqa: E402
from src.model_loader import load_model_and_tokenizer  # noqa: E402
from src.step0_utils import (  # noqa: E402
    build_chat_inputs,
    cuda_peak_gb,
    cuda_reset_peak,
    free_hidden_states,
    full_sequence_attention_mask,
    generate_chunked_batched,
    logits_forward_sanity_check,
    count_generated_tokens_per_row,
)


def _get_model_device(model: nn.Module) -> torch.device:
    return next(model.parameters()).device


def estimate_vram_gb(T_max: int, N: int) -> float:
    cost_kv = N * T_max * 0.01
    return 3.0 + cost_kv + (N * 1.5) + 1.0


def classify_status(v: float) -> str:
    if v < 30:
        return "SAFE"
    if v < 35:
        return "ACCEPTABLE"
    return "RISKY"


def spearman_safe(a: np.ndarray, b: np.ndarray) -> float:
    r, _ = spearmanr(a, b)
    if r is None or (isinstance(r, float) and np.isnan(r)):
        return 1.0 if np.allclose(a, b) else 0.0
    return float(r)


def run_part_b_from_stats(stats: dict) -> dict:
    pt = stats["part_a_token_statistics"]
    p99 = float(pt["p99_tokens"])
    max_tok = float(pt["max_tokens"])

    t_candidates = sorted(
        {
            int(np.ceil(p99 / 128) * 128),
            896,
            1024,
            min(1024, int(np.ceil(max_tok / 128) * 128)),
        }
    )
    t_candidates = [t for t in t_candidates if t >= 128]
    if not t_candidates:
        t_candidates = [896, 1024]

    rows = []
    for T in t_candidates:
        for N in (8, 16, 32):
            ev = estimate_vram_gb(T, N)
            st = classify_status(ev)
            rows.append(
                {
                    "T_max": T,
                    "N": N,
                    "estimated_vram_gb": round(ev, 2),
                    "status": st,
                    "reason": f"Rough formula (~±15%); estimated {ev:.1f} GB total",
                }
            )

    T_rec, N_rec, rationale = None, None, ""
    for T in sorted(set(t_candidates)):
        for N in (16, 8):
            ev = estimate_vram_gb(T, N)
            if ev < 30:
                T_rec, N_rec = T, N
                rationale = (
                    f"p99≈{p99:.0f} tokens; (T_max,N)=({T},{N}) estimated {ev:.1f} GB, SAFE band"
                )
                break
        if T_rec is not None:
            break
    if T_rec is None:
        T_rec, N_rec = 896, 8
        rationale = "Fallback to conservative T_max=896, N=8"

    return {
        "part_b_feasible_configs": rows,
        "step_1_recommendation": {
            "T_max": T_rec,
            "N": N_rec,
            "rationale": rationale,
        },
    }


def run_part_a(
    model,
    tokenizer,
    device: torch.device,
    rows: list[dict],
    out_path: Path,
    out_texts_path: Path,
    part_a_limit: int | None = None,
    logits_sanity: dict | None = None,
) -> dict:
    lengths: list[float] = []
    times: list[float] = []
    vrams: list[float] = []

    n_limit = part_a_limit if part_a_limit is not None else D.PART_A_NUM_PROBLEMS
    n_prob = min(n_limit, len(rows))
    eos_id = tokenizer.eos_token_id
    print("tokenizer.padding_side =", tokenizer.padding_side)
    print("tokenizer.pad_token_id =", tokenizer.pad_token_id)
    print("tokenizer.eos_token_id =", tokenizer.eos_token_id)
    print("model.config.pad_token_id =", model.config.pad_token_id)
    print("model.generation_config.pad_token_id =", model.generation_config.pad_token_id)
    # Overwrite for each run so results correspond to the specific settings.
    out_texts_path.parent.mkdir(parents=True, exist_ok=True)
    out_texts_path.write_text("", encoding="utf-8")

    for i in tqdm(range(n_prob), desc="Part A"):
        text = problem_text(rows[i])
        single_ids, prompt_len = build_chat_inputs(tokenizer, text, device)
        batched = single_ids.repeat(D.N_BRANCHES, 1)

        cuda_reset_peak()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        out_ids = generate_chunked_batched(
            model,
            tokenizer,
            batched,
            max_new_tokens=D.PART_A_MAX_NEW_TOKENS,
            chunk_size=D.CHUNK_SIZE,
            temperature=D.TEMPERATURE,
        )

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        peak_gb = cuda_peak_gb()

        per_branch = count_generated_tokens_per_row(out_ids, prompt_len, eos_id)
        max_len = max(per_branch)

        # Decode and store each branch's generated suffix (exclude prompt).
        out_ids_cpu = out_ids.detach().cpu()
        branch_texts: list[str] = []
        for b in range(D.N_BRANCHES):
            gen_len = per_branch[b]
            gen_ids = out_ids_cpu[b, prompt_len : prompt_len + gen_len].tolist()
            decoded = tokenizer.decode(
                gen_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )
            branch_texts.append(decoded)

        record = {
            "problem_index": i,
            "unique_id": rows[i].get("unique_id"),
            "generated_token_lengths_per_branch": per_branch,
            "branch_texts": branch_texts,
        }
        with out_texts_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

        lengths.append(float(max_len))
        times.append(float(elapsed))
        vrams.append(float(peak_gb))

        del out_ids, batched, single_ids
        free_hidden_states(None)

    arr = np.array(lengths)
    stats = {
        "n_problems": n_prob,
        "mean_tokens": float(arr.mean()),
        "std_tokens": float(arr.std()),
        "p95_tokens": float(np.percentile(arr, 95)),
        "p99_tokens": float(np.percentile(arr, 99)),
        "max_tokens": float(arr.max()),
        "mean_generation_time_sec": float(np.mean(times)),
        "max_generation_time_sec": float(np.max(times)),
        "mean_peak_vram_gb": float(np.mean(vrams)),
        "max_peak_vram_gb": float(np.max(vrams)),
    }

    payload = {
        "part_a_token_statistics": stats,
        "generation_metadata": {
            "messages_format": "chat_template",
            "math_user_prompt_suffix": D.MATH_USER_PROMPT_SUFFIX,
            "temperature": D.TEMPERATURE,
            "top_p": None,
            "n_branches": D.N_BRANCHES,
            "chunk_size": D.CHUNK_SIZE,
            "model_id": D.MODEL_ID,
            "per_token_logits_logged": False,
            "note": (
                "Step 0 does not persist per-step generation logits (memory). "
                "Sampling uses HuggingFace model.generate internally. "
                "Optional one-shot forward logits sanity check: logits_sanity_check field."
            ),
        },
        "part_a_outputs": {
            "generated_texts_jsonl": str(out_texts_path),
            "saved_per_problem": True,
            "n_problems_profiled": n_prob,
        },
    }
    if logits_sanity is not None:
        payload["logits_sanity_check"] = logits_sanity
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def run_part_c(
    model,
    tokenizer,
    device: torch.device,
    rows: list[dict],
    indices: list[int],
    out_path: Path,
) -> dict:
    eos_id = tokenizer.eos_token_id
    checkpoints = D.PART_C_CHECKPOINTS
    per_problem_rhos: list[float] = []
    detail_spearman: list[list[float]] = []

    for pidx in tqdm(indices, desc="Part C"):
        text = problem_text(rows[pidx])
        single_ids, prompt_len = build_chat_inputs(tokenizer, text, device)
        batched = single_ids.repeat(D.N_BRANCHES, 1)

        out_ids = generate_chunked_batched(
            model,
            tokenizer,
            batched,
            max_new_tokens=D.PART_C_MAX_NEW_TOKENS,
            chunk_size=D.CHUNK_SIZE,
            temperature=D.TEMPERATURE,
        )

        dev = _get_model_device(model)
        full = out_ids.to(dev)
        attn = full_sequence_attention_mask(full)
        with torch.inference_mode():
            out = model(full, attention_mask=attn, output_hidden_states=True)
            hidden = out.hidden_states[-1].to(torch.float32)
        del out

        gen_len = full.shape[1] - prompt_len
        rhos_checkpoint: list[float] = []

        for p in checkpoints:
            if gen_len < p:
                continue

            idx_last = prompt_len + p - 1
            start_b = prompt_len + max(0, p - 10)
            end_b = prompt_len + p
            ha = hidden[:, idx_last, :]
            hb = hidden[:, start_b:end_b, :].mean(dim=1)

            seed = 10_000 + pidx * 1_000 + p
            lev_a = row_leverage_scores(ha, D.LEVERAGE_PROJ_DIM, D.LEVERAGE_SVD_RANK, seed)
            lev_b = row_leverage_scores(hb, D.LEVERAGE_PROJ_DIM, D.LEVERAGE_SVD_RANK, seed)

            rho = spearman_safe(lev_a.cpu().numpy(), lev_b.cpu().numpy())
            rhos_checkpoint.append(rho)

        del hidden, full, out_ids, batched, single_ids
        free_hidden_states(None)

        if rhos_checkpoint:
            per_problem_rhos.append(float(np.mean(rhos_checkpoint)))
            detail_spearman.append(rhos_checkpoint)
        else:
            per_problem_rhos.append(float("nan"))
            detail_spearman.append([])

    mean_corr = float(np.nanmean(per_problem_rhos))
    decision = (
        "Use Method A (single-point)"
        if mean_corr > D.SPEARMAN_THRESHOLD
        else "Use Method B (window-based)"
    )

    payload = {
        "part_c_leverage_stability": {
            "n_test_problems": len(indices),
            "test_problem_indices": indices,
            "per_problem_mean_spearman": per_problem_rhos,
            "per_problem_checkpoint_spearman": detail_spearman,
            "mean_correlation": mean_corr,
            "decision": decision,
            "reasoning": (
                f"Mean Spearman {mean_corr:.3f} vs threshold {D.SPEARMAN_THRESHOLD}; "
                f"single-point vs last-10-mean branch rankings"
            ),
        }
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--part",
        choices=("a", "b", "c", "all"),
        default="all",
        help="Which part to run",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=PROJECT_ROOT / D.DATA_REL_PATH,
        help="MATH-500 style JSONL",
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=PROJECT_ROOT / D.RESULTS_STEP0_DIR,
    )
    parser.add_argument(
        "--part_a_limit",
        type=int,
        default=None,
        help="Only run Part A for the first K problems (e.g. 2 for smoke test).",
    )
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument(
        "--logits_sanity",
        action="store_true",
        help=(
            "Before Part A, run one forward on the first problem prompt and record logits "
            "shape/finiteness (not full vocab dump)."
        ),
    )
    args = parser.parse_args()

    out_a = args.out_dir / "part_a_statistics.json"
    out_b = args.out_dir / "part_b_configurations.json"
    out_c = args.out_dir / "part_c_leverage_decision.json"
    out_a_texts = args.out_dir / "part_a_generated_texts.jsonl"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.part == "b" and not out_a.is_file():
        raise FileNotFoundError(f"Run Part A first or provide {out_a}")

    rows: list = []
    if args.part in ("a", "c", "all"):
        rows = load_jsonl(args.data)
        required = D.PART_A_NUM_PROBLEMS
        if args.part == "a" and args.part_a_limit is not None:
            required = int(args.part_a_limit)
        if len(rows) < required:
            raise FileNotFoundError(
                f"Need at least {required} problems in {args.data}; "
                "download MATH-500 test.jsonl into data/math_500/ (see data/math_500/README.md)."
            )

    model = tokenizer = None
    need_model = args.part in ("a", "c", "all")
    if need_model:
        model, tokenizer = load_model_and_tokenizer(D.MODEL_ID, device)

    if args.part in ("a", "all"):
        sanity: dict | None = None
        if args.logits_sanity:
            text0 = problem_text(rows[0])
            single_ids, _pl = build_chat_inputs(tokenizer, text0, device)
            attn0 = full_sequence_attention_mask(single_ids)
            sanity = logits_forward_sanity_check(model, single_ids, attn0)
            sanity_path = args.out_dir / "part_a_logits_sanity.json"
            args.out_dir.mkdir(parents=True, exist_ok=True)
            sanity_path.write_text(json.dumps(sanity, indent=2), encoding="utf-8")
            sanity = {**sanity, "saved_to": str(sanity_path)}
        run_part_a(
            model,
            tokenizer,
            device,
            rows,
            out_a,
            out_a_texts,
            part_a_limit=args.part_a_limit,
            logits_sanity=sanity,
        )

    if args.part in ("b", "all"):
        stats = json.loads(out_a.read_text(encoding="utf-8"))
        part_b = run_part_b_from_stats(stats)
        args.out_dir.mkdir(parents=True, exist_ok=True)
        out_b.write_text(json.dumps(part_b, indent=2), encoding="utf-8")

    if args.part in ("c", "all"):
        rng = random.Random(args.seed)
        indices = sorted(rng.sample(range(D.PART_A_NUM_PROBLEMS), D.PART_C_NUM_PROBLEMS))
        run_part_c(model, tokenizer, device, rows, indices, out_c)


if __name__ == "__main__":
    main()
