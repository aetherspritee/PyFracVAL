#!/usr/bin/env python3
"""Independently re-validate every cluster file in a generated catalog.

Deliberately re-reads the saved ``.dat`` files from disk and recomputes
their geometry, rather than trusting the quality record written at
generation time. Those are different claims: one says "the generator
believed this was fine", the other says "the file on disk is fine". A
previous batch reached downstream consumers carrying severe particle
overlap while being catalogued as successful, so the file is the thing
worth checking.

Usage:
    devenv shell -- uv run python scripts/validate_cluster_catalog.py
    devenv shell -- uv run python scripts/validate_cluster_catalog.py --root cluster_data
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from pyfracval.fractal import (  # noqa: E402
    calculate_rg,
    compute_empirical_rg_polydisperse,
)
from pyfracval.quality import max_self_overlap  # noqa: E402


def read_dat(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read coordinates and radii from a saved aggregate file."""
    rows = []
    with path.open() as fh:
        for line in fh:
            if line.startswith("#") or not line.strip():
                continue
            parts = line.split()
            if len(parts) >= 4:
                rows.append([float(x) for x in parts[:4]])
    arr = np.asarray(rows, dtype=float)
    return arr[:, :3], arr[:, 3]


def parse_header_params(path: Path) -> dict:
    """Pull Df/kf/N out of the YAML header without a YAML dependency."""
    out: dict[str, float] = {}
    with path.open() as fh:
        for line in fh:
            if not line.startswith("#"):
                break
            text = line.lstrip("#").strip()
            for key in ("Df:", "kf:", "N:", "rp_gstd:", "tol_ov:"):
                if text.startswith(key):
                    try:
                        out[key.rstrip(":")] = float(text.split(":", 1)[1])
                    except ValueError:
                        pass
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=PROJECT_ROOT / "cluster_data")
    ap.add_argument(
        "--tol",
        type=float,
        default=None,
        help=(
            "max tolerated overlap fraction. Default: each file's own tol_ov, "
            "read from its header - that is the contract the run promised. "
            "Pass a number to check against a fixed stricter/looser bound."
        ),
    )
    args = ap.parse_args()

    files = sorted(args.root.rglob("fracval_*.dat"))
    if not files:
        raise SystemExit(f"No cluster files under {args.root}")

    print(f"Validating {len(files)} cluster files under {args.root}\n")

    bad_overlap: list[dict] = []
    short: list[dict] = []
    worst = 0.0
    n_within_tol = 0
    rg_errors: list[float] = []
    by_n: dict[int, int] = Counter()
    per_combo: dict[tuple, int] = defaultdict(int)

    for i, path in enumerate(files, 1):
        coords, radii = read_dat(path)
        params = parse_header_params(path)
        n_expected = int(params.get("N", coords.shape[0]))
        df = params.get("Df")
        kf = params.get("kf")

        max_ov, n_pairs = max_self_overlap(coords, radii)
        worst = max(worst, max_ov)
        # The generator accepts contact up to tol_ov, so that is the bound
        # to hold it to. Judging every file at a fixed 1e-9 instead tests a
        # promise nobody made: a run configured with tol_ov=1e-6 is free to
        # place a pair 1e-7 into each other, and whether it happens to is a
        # property of the sticking geometry, not of correctness. The worst
        # overlap seen is printed regardless, so nothing is hidden by this.
        tol = args.tol if args.tol is not None else params.get("tol_ov", 1e-9)
        if max_ov > tol:
            bad_overlap.append(
                {
                    "file": str(path),
                    "max_overlap": max_ov,
                    "n_pairs": n_pairs,
                    "tol": tol,
                }
            )
        elif max_ov > 0.0:
            n_within_tol += 1
        if coords.shape[0] != n_expected:
            short.append(
                {
                    "file": str(path),
                    "expected": n_expected,
                    "actual": int(coords.shape[0]),
                }
            )

        if df and kf:
            measured = compute_empirical_rg_polydisperse(coords, radii)
            target = calculate_rg(radii, coords.shape[0], df, kf)
            if target > 0:
                rg_errors.append((measured - target) / target * 100.0)
            per_combo[(params.get("rp_gstd"), df, n_expected)] += 1
        by_n[coords.shape[0]] += 1

        if i % 200 == 0:
            print(f"  ...{i}/{len(files)}")

    print("\n" + "=" * 66)
    print("CATALOG VALIDATION")
    print("=" * 66)
    print(f"  files checked            : {len(files)}")
    print(f"  files with overlap > tol : {len(bad_overlap)}")
    print(f"  worst overlap anywhere   : {worst:.3e}")
    print(f"  ... of which within tol  : {n_within_tol} file(s) show any contact")
    print(f"  files short of N         : {len(short)}")
    if rg_errors:
        arr = np.array(rg_errors)
        print(
            f"  Rg error vs scaling law  : mean {arr.mean():+.2f}%  "
            f"median {np.median(arr):+.2f}%  "
            f"|err|<=5% for {int((np.abs(arr) <= 5).sum())}/{len(arr)}"
        )
    print(f"  distinct combos covered  : {len(per_combo)}")
    print(f"  particle counts present  : {sorted(by_n)[:6]} ... {sorted(by_n)[-4:]}")

    if bad_overlap:
        print("\n  OVERLAPPING FILES (first 10):")
        for entry in bad_overlap[:10]:
            print(
                f"    {entry['max_overlap']:.3e} over {entry['n_pairs']} pairs  "
                f"{entry['file']}"
            )
    if short:
        print("\n  SHORT FILES (first 10):")
        for entry in short[:10]:
            print(f"    {entry['actual']}/{entry['expected']}  {entry['file']}")

    out = args.root / "validation_report.json"
    out.write_text(
        json.dumps(
            {
                "n_files": len(files),
                "n_overlapping": len(bad_overlap),
                "worst_overlap": worst,
                "n_short": len(short),
                "overlapping": bad_overlap[:200],
                "short": short[:200],
                "rg_error_pct_mean": float(np.mean(rg_errors)) if rg_errors else None,
                "rg_error_pct_median": (
                    float(np.median(rg_errors)) if rg_errors else None
                ),
            },
            indent=2,
        )
    )
    print(f"\n  report: {out}")
    print("=" * 66)

    if bad_overlap:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
