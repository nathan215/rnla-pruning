"""Shared answer extraction and equality checks for MATH-style problems."""

from __future__ import annotations

import re
from fractions import Fraction
from typing import Any, Literal

MatchMode = Literal["boxed_only", "legacy_last_line"]


def safe_problem_id(raw: str) -> str:
    return raw.replace("\\", "__").replace("/", "__")


def extract_last_boxed(text: str) -> str | None:
    """Return content of the last ``\\boxed{...}`` using brace balancing (handles nested ``{}``)."""
    key = "\\boxed{"
    j = text.rfind(key)
    if j < 0:
        return None
    i = j + len(key)
    depth = 1
    start = i
    while i < len(text) and depth > 0:
        c = text[i]
        if c == "\\":
            i += 2 if i + 1 < len(text) else 1
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
        i += 1
    if depth != 0:
        return None
    return text[start : i - 1]


def extract_prediction_for_match(text: str, *, match_mode: MatchMode) -> str:
    """
    Extract model answer for grading.

    ``boxed_only`` (default for new runs): only ``\\boxed{...}`` counts. No fallback to the
    last line — that heuristic can mis-score long CoT traces (e.g. “shortest/last line” bias).
    ``legacy_last_line``: previous behavior — last non-empty line if no box.
    """
    boxed = extract_last_boxed(text)
    if boxed is not None:
        return boxed.strip()
    if match_mode == "boxed_only":
        return ""
    lines = [x.strip() for x in str(text).splitlines() if x.strip()]
    return lines[-1] if lines else str(text).strip()


def extract_reference_for_match(ref: str) -> str:
    """Dataset reference: use ``\\boxed`` if present, else the raw answer string."""
    boxed = extract_last_boxed(ref)
    if boxed is not None:
        return boxed.strip()
    return str(ref).strip()


def _norm(s: str) -> str:
    return s.replace(" ", "").replace("\n", "").lower()


def answers_equal(
    pred: str,
    ref: str,
    *,
    match_mode: MatchMode = "boxed_only",
    tol: float = 1e-6,
) -> bool:
    p_raw = extract_prediction_for_match(pred, match_mode=match_mode)
    r_raw = extract_reference_for_match(ref)
    if match_mode == "boxed_only" and not p_raw.strip():
        return False
    p = _norm(p_raw)
    r = _norm(r_raw)
    if p == r:
        return True
    try:
        if abs(float(p) - float(r)) < tol:
            return True
    except Exception:
        pass
    try:
        if Fraction(p).limit_denominator() == Fraction(r).limit_denominator():
            return True
    except Exception:
        pass
    return False


def get_reference_answer(row: dict[str, Any]) -> str | None:
    for key in ("answer", "solution", "target"):
        val = row.get(key)
        if val:
            return str(val)
    return None


# Back-compat: old name used regex-only boxed (no nested braces)
def extract_final_answer(text: str) -> str:
    """Deprecated for grading; prefer ``extract_prediction_for_match(..., legacy_last_line)``."""
    boxed = extract_last_boxed(text)
    if boxed is not None:
        return boxed.strip()
    lines = [x.strip() for x in str(text).splitlines() if x.strip()]
    return lines[-1] if lines else str(text).strip()
