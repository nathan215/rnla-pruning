#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from huggingface_hub import hf_hub_download, snapshot_download  # noqa: E402

from config import defaults as D  # noqa: E402

MATH500_DATASET_ID = "HuggingFaceH4/MATH-500"
MATH500_FILENAME = "test.jsonl"


def download_math500(
    out_dir: Path,
    *,
    token: str | None = None,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = hf_hub_download(
        repo_id=MATH500_DATASET_ID,
        repo_type="dataset",
        filename=MATH500_FILENAME,
        local_dir=str(out_dir),
        local_dir_use_symlinks=False,
        token=token,
    )
    return Path(path)


def validate_jsonl(path: Path) -> None:
    with path.open(encoding="utf-8") as f:
        first = f.readline()
    if not first.strip():
        raise ValueError(f"Empty file: {path}")
    row = json.loads(first)
    if not any(k in row for k in ("problem", "question", "instruction")):
        raise ValueError(
            f"First row missing problem/question/instruction keys: keys={list(row.keys())}"
        )


def download_model(
    model_id: str,
    cache_root: Path,
    *,
    token: str | None = None,
) -> Path:
    safe_name = model_id.replace("/", "--")
    local_dir = cache_root / safe_name
    local_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=model_id,
        local_dir=str(local_dir),
        local_dir_use_symlinks=False,
        token=token,
    )
    return local_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--math-only",
        action="store_true"
        )
    parser.add_argument(
        "--model-only",
        action="store_true",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "math_500",
    )
    parser.add_argument(
        "--model-cache",
        type=Path,
        default=PROJECT_ROOT / "models" / "cache",
    )
    parser.add_argument(
        "--model-id",
        default=D.MODEL_ID,
    )
    parser.add_argument(
        "--token",
        default=None,
    )
    args = parser.parse_args()

    do_math = not args.model_only
    do_model = not args.math_only
    
    if do_math:
        print(f"Downloading {MATH500_DATASET_ID}/{MATH500_FILENAME} → {args.data_dir}")
        p = download_math500(args.data_dir, token=args.token)
        validate_jsonl(p)
        with p.open(encoding="utf-8") as f:
            n_lines = sum(1 for _ in f)
        print(f"OK: {p} ({n_lines} lines)")

    if do_model:
        print(f"Downloading model {args.model_id} → {args.model_cache}")
        out = download_model(args.model_id, args.model_cache, token=args.token)
        print(f"OK: {out}")

    print("Done.")


if __name__ == "__main__":
    main()
