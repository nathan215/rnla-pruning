"""Load MATH-style JSONL problems."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_jsonl(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if limit is not None and len(rows) >= limit:
                break
    return rows


def problem_text(row: dict[str, Any]) -> str:
    for key in ("problem", "question", "instruction"):
        if key in row and row[key]:
            return str(row[key]).strip()
    raise KeyError(
        "No problem field found; expected one of: problem, question, instruction"
    )
