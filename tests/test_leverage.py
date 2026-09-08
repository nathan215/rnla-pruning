"""Unit tests for the leverage-score kernel in ``src/leverage.py``.

All tests run on CPU in a few seconds and exercise mathematical properties of
row leverage scores rather than specific numbers, so they are robust across
torch versions and hardware.
"""

from __future__ import annotations

import pytest
import torch

from src.leverage import exact_row_leverage_scores, row_leverage_scores


def _randn(n: int, h: int, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n, h, generator=g)


def test_exact_shape_and_range() -> None:
    X = _randn(16, 1536)
    s = exact_row_leverage_scores(X, k=8)
    assert s.shape == (16,)
    assert torch.all(s >= 0)
    assert torch.all(s <= 1 + 1e-5)  # leverage of a row never exceeds 1


@pytest.mark.parametrize("k", [1, 4, 8])
def test_exact_scores_sum_to_k(k: int) -> None:
    # Row norms of an N x k orthonormal block sum (squared) to k.
    X = _randn(16, 64)
    s = exact_row_leverage_scores(X, k=k)
    assert torch.isclose(s.sum(), torch.tensor(float(k)), atol=1e-4)


def test_k_is_clamped_to_rank() -> None:
    X = _randn(6, 64)  # rank 6
    assert torch.allclose(
        exact_row_leverage_scores(X, k=6), exact_row_leverage_scores(X, k=100)
    )


def test_exact_is_invariant_to_orthogonal_rotation() -> None:
    X = _randn(16, 64)
    Q, _ = torch.linalg.qr(_randn(64, 64, seed=1))
    assert torch.allclose(
        exact_row_leverage_scores(X, k=8),
        exact_row_leverage_scores(X @ Q, k=8),
        atol=1e-4,
    )


def test_full_rank_leverage_survives_sketching() -> None:
    # For a rank-r matrix and k = r, row leverage is the diagonal of the hat
    # matrix, a property of the column space alone. Any generic sketch with
    # d >= r preserves that column space, so the randomized scores must agree
    # with the exact ones.
    A, B = _randn(16, 5, seed=2), _randn(5, 64, seed=3)
    X = A @ B  # rank 5
    exact = exact_row_leverage_scores(X, k=5)
    sketched = row_leverage_scores(X, proj_dim=64, k=5, seed=123)
    assert torch.allclose(exact, sketched, atol=1e-4)
    assert torch.isclose(exact.sum(), torch.tensor(5.0), atol=1e-4)


def test_randomized_is_deterministic_given_seed() -> None:
    X = _randn(16, 256)
    a = row_leverage_scores(X, proj_dim=32, k=8, seed=7)
    b = row_leverage_scores(X, proj_dim=32, k=8, seed=7)
    c = row_leverage_scores(X, proj_dim=32, k=8, seed=8)
    assert torch.equal(a, b)
    assert not torch.allclose(a, c)


def test_randomized_scores_are_nonnegative_and_sum_to_k() -> None:
    X = _randn(16, 256)
    s = row_leverage_scores(X, proj_dim=32, k=8, seed=0)
    assert s.shape == (16,)
    assert torch.all(s >= 0)
    assert torch.isclose(s.sum(), torch.tensor(8.0), atol=1e-4)


def test_dominant_outlier_row_gets_top_score() -> None:
    # One row that is huge along a direction the others barely use must
    # dominate the top singular direction, for both exact and sketched scores.
    X = _randn(16, 64) * 0.01
    X[3] = 0.0
    X[3, 0] = 50.0
    assert int(torch.argmax(exact_row_leverage_scores(X, k=1))) == 3
    assert int(torch.argmax(row_leverage_scores(X, proj_dim=32, k=1, seed=0))) == 3


def test_accepts_bfloat16_input_and_returns_float32() -> None:
    X = _randn(16, 256).to(torch.bfloat16)
    s = row_leverage_scores(X, proj_dim=32, k=8, seed=0)
    assert s.dtype == torch.float32
    assert s.shape == (16,)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_randomized_runs_on_cuda_and_matches_cpu_seed() -> None:
    X = _randn(16, 256)
    cpu = row_leverage_scores(X, proj_dim=32, k=8, seed=5)
    gpu = row_leverage_scores(X.cuda(), proj_dim=32, k=8, seed=5)
    assert gpu.device.type == "cuda"
    assert torch.allclose(cpu, gpu.cpu(), atol=1e-4)
