"""
Extract Google Drive split zips (…-001.zip + …-002.zip) into results/step_1_oracle_b{N}/.

Run from project root:
  python scripts/extract_step1_gdrive_zips.py
  python scripts/extract_step1_gdrive_zips.py --results-dir results --dry-run
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import zipfile
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def find_zip_groups(results_dir: Path) -> dict[str, list[Path]]:
    groups: dict[str, list[Path]] = defaultdict(list)
    for p in sorted(results_dir.glob("step_1_oracle_b*.zip")):
        m = re.match(r"^(step_1_oracle_b\d+)-", p.name)
        if m:
            groups[m.group(1)].append(p)
    for bid in groups:
        # 001 before 002 (lexicographic works for same prefix)
        groups[bid].sort(key=lambda x: x.name)
    return dict(sorted(groups.items()))


def extract_group(batch_id: str, paths: list[Path], dest: Path, dry_run: bool) -> None:
    if len(paths) != 2:
        print(f"  WARN {batch_id}: expected 2 zips, got {len(paths)}: {[p.name for p in paths]}")
    for zp in paths:
        if dry_run:
            print(f"    would extract: {zp.name} -> {dest}")
            continue
        with zipfile.ZipFile(zp, "r") as zf:
            zf.extractall(dest)
        print(f"    extracted: {zp.name}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=PROJECT_ROOT / "results")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-validate", action="store_true")
    args = parser.parse_args()

    results_dir = args.results_dir.resolve()
    groups = find_zip_groups(results_dir)
    if not groups:
        print(f"No step_1_oracle_b*.zip files under {results_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(groups)} batch zip group(s) under {results_dir}")
    for bid, paths in groups.items():
        print(f"  {bid}: {[p.name for p in paths]}")

    for bid, paths in groups.items():
        print(f"\nExtracting {bid} …", flush=True)
        extract_group(bid, paths, results_dir, args.dry_run)

    if args.dry_run or args.skip_validate:
        return

    # Per-batch validation
    print("\n--- validate_step1_oracle per batch ---")
    failed: list[str] = []
    for i in range(8):
        bid = f"b{i}"
        out = results_dir / f"step_1_oracle_{bid}"
        if not out.is_dir():
            print(f"  SKIP missing dir: {out.name}")
            failed.append(bid)
            continue
        r = subprocess.run(
            [
                sys.executable,
                str(PROJECT_ROOT / "experiments" / "validate_step1_oracle.py"),
                "--out_dir",
                str(out),
                "--expected_problems",
                "25",
            ],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
        )
        if r.returncode != 0:
            print(f"  FAIL {out.name} (exit {r.returncode})")
            print(r.stderr[-2000:] if r.stderr else "")
            failed.append(bid)
        else:
            print(f"  OK  {out.name}")

    # Global index coverage 0..199
    print("\n--- index coverage vs data/math_500/test.jsonl ---")
    data_path = PROJECT_ROOT / "data" / "math_500" / "test.jsonl"
    if not data_path.is_file():
        print(f"  WARN: missing {data_path}; skip index check")
        sys.exit(0 if not failed else 1)

    rows = data_path.read_text(encoding="utf-8").strip().splitlines()
    if len(rows) < 200:
        print(f"  WARN: test.jsonl has only {len(rows)} lines")

    seen: set[int] = set()
    dup: list[int] = []
    missing_meta: list[Path] = []

    for i in range(8):
        out = results_dir / f"step_1_oracle_b{i}"
        per = out / "per_problem"
        if not per.is_dir():
            continue
        for d in sorted(per.iterdir()):
            if not d.is_dir():
                continue
            mp = d / "problem_metadata.json"
            if not mp.is_file():
                missing_meta.append(d)
                continue
            meta = json.loads(mp.read_text(encoding="utf-8"))
            idx = int(meta.get("index", -1))
            if idx in seen:
                dup.append(idx)
            seen.add(idx)

    want = set(range(200))
    missing_idx = sorted(want - seen)
    extra_idx = sorted(seen - want)

    report = {
        "results_dir": str(results_dir),
        "batches_found": list(groups.keys()),
        "unique_indices_count": len(seen),
        "missing_indices_0_199": missing_idx,
        "extra_indices": extra_idx[:50],
        "duplicate_indices": sorted(set(dup)),
        "folders_without_problem_metadata": [str(p) for p in missing_meta[:20]],
        "validation_failed_batches": failed,
    }
    out_report = results_dir / "step_1_oracle_200_manifest.json"
    out_report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nWrote {out_report}")

    if missing_idx or dup or missing_meta or failed:
        print("\nISSUES:")
        if failed:
            print(f"  validation failed batches: {failed}")
        if missing_idx:
            print(f"  missing indices (count={len(missing_idx)}): {missing_idx[:30]}...")
        if dup:
            print(f"  duplicate indices: {sorted(set(dup))[:30]}")
        if missing_meta:
            print(f"  dirs without problem_metadata.json: {len(missing_meta)}")
        sys.exit(1)

    if len(seen) != 200:
        print(f"\nWARN: expected 200 unique indices, got {len(seen)}")
        sys.exit(1)

    print("\nAll OK: 200 problems with unique indices 0..199 and per-batch validation passed.")
    sys.exit(0)


if __name__ == "__main__":
    main()
