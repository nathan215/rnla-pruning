"""Benchmark exact vs randomized leverage timing at small N (branch count).

Matches ``step2_2_approximation.benchmark_leverage_ms`` (batch timer over
``timing_repeats`` calls) so table numbers are reproducible.

**Why exact SVD can be faster than random projection at our scale:**
For $X\\in\\mathbb{R}^{N\\times D}$ with $N\\ll D$, thin SVD cost is often
$\\mathcal{O}(N^2 D)$. Randomized leverage adds a full GEMM
$\\mathcal{O}(N D d)$ plus SVD on $N\\times d$. With $N{=}16$, $D{=}1536$,
$N^2 D \\ll N D d$ when $d\\gtrsim N$, so wall time need not follow a naive
``exact is always slower'' intuition.

Usage:
  python experiments/benchmark_n16_leverage_timing.py --repeats 10000
  python experiments/benchmark_n16_leverage_timing.py --mode batch --batches 200
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
import sys

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.leverage import exact_row_leverage_scores, row_leverage_scores  # noqa: E402


def _bench_exact(x: torch.Tensor, k: int, warmup: int, repeats: int) -> tuple[float, float]:
    for _ in range(warmup):
        exact_row_leverage_scores(x, k)
    times_ms: list[float] = []
    for _ in range(repeats):
        t0 = time.perf_counter_ns()
        exact_row_leverage_scores(x, k)
        t1 = time.perf_counter_ns()
        times_ms.append((t1 - t0) / 1_000_000.0)
    return statistics.fmean(times_ms), statistics.pstdev(times_ms)


def _bench_random(
    x: torch.Tensor, d: int, k: int, warmup: int, repeats: int, seed_base: int
) -> tuple[float, float]:
    for i in range(warmup):
        row_leverage_scores(x, d, k, seed_base + i)
    times_ms: list[float] = []
    for i in range(repeats):
        t0 = time.perf_counter_ns()
        row_leverage_scores(x, d, k, seed_base + warmup + i)
        t1 = time.perf_counter_ns()
        times_ms.append((t1 - t0) / 1_000_000.0)
    return statistics.fmean(times_ms), statistics.pstdev(times_ms)


def _bench_batch_exact(x: torch.Tensor, k: int, warmup: int, n_repeat: int) -> float:
    """Mean ms per call; same pattern as step2_2 exact timing."""
    for _ in range(warmup):
        exact_row_leverage_scores(x, k)
    t0 = time.perf_counter()
    for _ in range(n_repeat):
        exact_row_leverage_scores(x, k)
    return (time.perf_counter() - t0) / max(n_repeat, 1) * 1000.0


def _bench_batch_random(
    x: torch.Tensor, d: int, k: int, warmup: int, n_repeat: int, seed0: int
) -> float:
    """Mean ms per call; same pattern as step2_2 ``benchmark_leverage_ms``."""
    for i in range(warmup):
        row_leverage_scores(x, d, k, seed0 + i)
    t0 = time.perf_counter()
    for i in range(n_repeat):
        row_leverage_scores(x, d, k, seed0 + warmup + i)
    return (time.perf_counter() - t0) / max(n_repeat, 1) * 1000.0


def _flop_notes(n: int, d_hid: int, d_proj: int, k: int) -> dict[str, float]:
    """Rough leading-order real FLOP counts (multiplies+adds ~2x not separated)."""
    gemm = 2.0 * n * d_hid * d_proj
    svd_small = 2.0 * n * d_proj * d_proj  # crude O(N d^2) proxy for Nxd SVD
    svd_exact = 2.0 * n * n * d_hid  # O(N^2 D) proxy when N < D
    return {
        "approx_gemm_N_D_d": gemm,
        "approx_svd_N_d2": svd_small,
        "approx_svd_N2_D": svd_exact,
        "approx_random_total": gemm + svd_small,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark leverage timing at N=16.")
    parser.add_argument(
        "--mode",
        choices=("per_call", "batch"),
        default="per_call",
        help="per_call: nanosecond timer each call. batch: like step2_2 (50 calls / block).",
    )
    parser.add_argument("--batches", type=int, default=200, help="batch mode: independent batch means.")
    parser.add_argument("--timing_repeats", type=int, default=50, help="Calls per timed block (batch mode).")
    parser.add_argument("--timing_warmup", type=int, default=3, help="Warmup calls before each batch block.")
    parser.add_argument("--repeats", type=int, default=10_000, help="Timed iterations per method (per_call).")
    parser.add_argument("--warmup", type=int, default=200, help="Warmup iterations per method (per_call).")
    parser.add_argument("--n", type=int, default=16, help="Number of rows (branches).")
    parser.add_argument("--hidden", type=int, default=1536, help="Hidden dimension D.")
    parser.add_argument("--k", type=int, default=8, help="Leverage rank k.")
    parser.add_argument("--seed", type=int, default=1234, help="Base random seed.")
    parser.add_argument("--threads", type=int, default=1, help="Torch CPU threads.")
    parser.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Write summary JSON (e.g. results/leverage_timing_microbench.json).",
    )
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    x = torch.randn(args.n, args.hidden, dtype=torch.float32)

    if args.mode == "per_call":
        exact_mean, exact_std = _bench_exact(x, args.k, args.warmup, args.repeats)
        d64_mean, d64_std = _bench_random(x, 64, args.k, args.warmup, args.repeats, args.seed + 1_000)
        d128_mean, d128_std = _bench_random(
            x, 128, args.k, args.warmup, args.repeats, args.seed + 2_000
        )
        d256_mean, d256_std = _bench_random(
            x, 256, args.k, args.warmup, args.repeats, args.seed + 3_000
        )

        print(f"N={args.n}, D={args.hidden}, k={args.k}, repeats={args.repeats}, warmup={args.warmup}")
        print(f"exact_ms   = {exact_mean:.6f} (std={exact_std:.6f})")
        print(f"d=64_ms    = {d64_mean:.6f} (std={d64_std:.6f}, ratio={d64_mean / exact_mean:.3f}x)")
        print(f"d=128_ms   = {d128_mean:.6f} (std={d128_std:.6f}, ratio={d128_mean / exact_mean:.3f}x)")
        print(f"d=256_ms   = {d256_mean:.6f} (std={d256_std:.6f}, ratio={d256_mean / exact_mean:.3f}x)")
        out = {
            "mode": "per_call",
            "n": args.n,
            "hidden": args.hidden,
            "k": args.k,
            "ms": {
                "exact": {"mean": exact_mean, "std": exact_std},
                "d64": {"mean": d64_mean, "std": d64_std},
                "d128": {"mean": d128_mean, "std": d128_std},
                "d256": {"mean": d256_mean, "std": d256_std},
            },
        }
    else:
        wr, tr = args.timing_warmup, args.timing_repeats
        exact_samples = [
            _bench_batch_exact(x, args.k, wr, tr) for _ in range(args.batches)
        ]
        d64_s = [
            _bench_batch_random(x, 64, args.k, wr, tr, args.seed + 1_000 + bi * 10_000)
            for bi in range(args.batches)
        ]
        d128_s = [
            _bench_batch_random(x, 128, args.k, wr, tr, args.seed + 2_000 + bi * 10_000)
            for bi in range(args.batches)
        ]
        d256_s = [
            _bench_batch_random(x, 256, args.k, wr, tr, args.seed + 3_000 + bi * 10_000)
            for bi in range(args.batches)
        ]

        def _m(s: list[float]) -> tuple[float, float]:
            return statistics.fmean(s), statistics.pstdev(s)

        em, es = _m(exact_samples)
        m64, s64 = _m(d64_s)
        m128, s128 = _m(d128_s)
        m256, s256 = _m(d256_s)

        print(
            f"batch mode: N={args.n}, D={args.hidden}, k={args.k}, "
            f"batches={args.batches}, timing_repeats={tr}, warmup={wr}"
        )
        print(f"exact_ms   = {em:.4f} (std={es:.4f})")
        print(f"d=64_ms    = {m64:.4f} (std={s64:.4f}, ratio={m64 / em:.3f}x)")
        print(f"d=128_ms   = {m128:.4f} (std={s128:.4f}, ratio={m128 / em:.3f}x)")
        print(f"d=256_ms   = {m256:.4f} (std={s256:.4f}, ratio={m256 / em:.3f}x)")
        out = {
            "mode": "batch",
            "timing_repeats": tr,
            "timing_warmup": wr,
            "batches": args.batches,
            "n": args.n,
            "hidden": args.hidden,
            "k": args.k,
            "ms_mean_per_call": {
                "exact": em,
                "d64": m64,
                "d128": m128,
                "d256": m256,
            },
            "ms_std_across_batches": {
                "exact": es,
                "d64": s64,
                "d128": s128,
                "d256": s256,
            },
        }

    flops = {
        "d64": _flop_notes(args.n, args.hidden, 64, args.k),
        "d128": _flop_notes(args.n, args.hidden, 128, args.k),
        "d256": _flop_notes(args.n, args.hidden, 256, args.k),
    }
    print(
        "FLOP sketch (leading terms): "
        f"exact~N^2 D = {flops['d128']['approx_svd_N2_D']:.0f}, "
        f"random GEMM N D d (d=128) = {flops['d128']['approx_gemm_N_D_d']:.0f}"
    )

    out["flop_sketch"] = flops
    out["torch_num_threads"] = args.threads

    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(f"Wrote {args.json_out}")


if __name__ == "__main__":
    main()
