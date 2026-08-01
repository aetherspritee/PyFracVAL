#!/usr/bin/env python3
"""Wall-time ratio between FracVAL and PyFracVAL in the hard regime.

The paper reports that "in the hard regime the two implementations perform
equally (median ratio 1.05)".  ``paper_fortran_analysis.py`` computes the
overall and the ``N >= 512`` medians but never a hard-regime one, so the
figure had no script behind it.  This module supplies one: it applies an
explicit, named cell-selection rule to the stored head-to-head grids and
records the median ratio with its interquartile range, a bootstrap
interval and the cell count.

The rule the paper's own notes state (``PAPER.md``: "in the hard regime
(sigma=1.9, Df>=2.2)") is the one labelled ``paper_hard_regime`` below.
The other rules are reported alongside it as a sensitivity analysis, since
the definition is what the number is most sensitive to.

Inputs (both pre-existing, read-only)
-------------------------------------
``benchmark_results/fortran_headtohead/fortran_grid_summary.csv``
``benchmark_results/boundary_sweep_v3_eventlog/stability_sweeps/
stability_sweep_summary.csv``

A cell contributes when both implementations solve it (success rate
>= 0.5, the same threshold ``paper_fortran_analysis.py`` uses for its
ceilings and its overall ratio) and both median wall times are usable.
The ratio is ``median FracVAL wall time / median PyFracVAL wall time``,
so values above 1 mean PyFracVAL is faster.

Usage
-----
    devenv shell -- uv run python benchmarks/hard_regime_ratio.py

Writes ``benchmark_results/hard_regime_ratio/hard_regime_ratio.json``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import statistics
import subprocess
import time
from pathlib import Path
from typing import Callable, Sequence

REPO_ROOT = Path(__file__).resolve().parent.parent

FORTRAN_CSV = (
    REPO_ROOT / "benchmark_results/fortran_headtohead/fortran_grid_summary.csv"
)
PYFRACVAL_CSV = (
    REPO_ROOT / "benchmark_results/boundary_sweep_v3_eventlog/stability_sweeps/"
    "stability_sweep_summary.csv"
)
# The head-to-head inputs live under a gitignored directory (the repo's
# pre-publication policy, see .gitignore), so the summary is written
# outside it in order to be committable. The input digests below tie a
# stored summary to the exact grids it was computed from.
OUT_JSON = REPO_ROOT / "benchmark_results/hard_regime_ratio/hard_regime_ratio.json"

SOLVED_THRESHOLD = 0.5
BOOTSTRAP_DRAWS = 20000
BOOTSTRAP_SEED = 12345

Cell = tuple[int, float, float, float]  # (N, Df, kf, sigma)

# Named selection rules. "paper_hard_regime" is the definition stated in
# PAPER.md; the rest exist so the sensitivity of the reported median to
# the definition is visible rather than implicit.
RULES: dict[str, Callable[[Cell], bool]] = {
    "all_cells": lambda c: True,
    "paper_hard_regime": lambda c: c[3] == 1.9 and c[1] >= 2.2,
    "sigma1.9_only": lambda c: c[3] == 1.9,
    "df>=2.2_only": lambda c: c[1] >= 2.2,
    "df>=2.0_only": lambda c: c[1] >= 2.0,
    "df>=2.1_only": lambda c: c[1] >= 2.1,
    "sigma>=1.5_and_df>=2.0": lambda c: c[3] >= 1.5 and c[1] >= 2.0,
    "sigma>=1.5_and_df>=2.2": lambda c: c[3] >= 1.5 and c[1] >= 2.2,
    "sigma1.9_df>=2.2_N>=128": lambda c: c[3] == 1.9 and c[1] >= 2.2 and c[0] >= 128,
    "N>=512_only": lambda c: c[0] >= 512,
}


def load(path: Path, rate_key: str, time_key: str) -> dict[Cell, dict]:
    """Read one grid summary CSV keyed by ``(N, Df, kf, sigma)``.

    Mirrors ``paper_fortran_analysis.load`` so the two scripts key cells
    identically.
    """
    out: dict[Cell, dict] = {}
    with path.open() as fh:
        for row in csv.DictReader(fh):
            key: Cell = (
                int(row["N"]),
                round(float(row["Df"]), 1),
                round(float(row["kf"]), 1),
                float(row["rp_gstd"]),
            )
            raw_time = row.get(time_key)
            out[key] = {
                "rate": float(row[rate_key]),
                "time": float(raw_time) if raw_time else float("nan"),
            }
    return out


def bootstrap_median_ci(
    values: Sequence[float], draws: int = BOOTSTRAP_DRAWS, seed: int = BOOTSTRAP_SEED
) -> tuple[float, float] | None:
    """Percentile bootstrap 95% interval for the median.

    Uses a seeded generator so the interval is reproducible.
    """
    if len(values) < 2:
        return None
    import numpy as np

    rng = np.random.default_rng(seed)
    arr = np.asarray(values, dtype=float)
    samples = rng.choice(arr, size=(draws, arr.size), replace=True)
    meds = np.median(samples, axis=1)
    return float(np.percentile(meds, 2.5)), float(np.percentile(meds, 97.5))


def summarise(ratios: Sequence[float]) -> dict:
    """Median, quartiles, range and bootstrap interval of a ratio sample."""
    if not ratios:
        return {"cells": 0}
    ordered = sorted(ratios)
    quartiles = statistics.quantiles(ordered, n=4, method="inclusive")
    ci = bootstrap_median_ci(ordered)
    return {
        "cells": len(ordered),
        "median": statistics.median(ordered),
        "q1": quartiles[0],
        "q3": quartiles[2],
        "min": ordered[0],
        "max": ordered[-1],
        # One Fortran cell is recorded at 0.0 s (below its timer resolution),
        # so a ratio of exactly zero can occur; it is kept in the sample for
        # consistency with paper_fortran_analysis.py's selection, but the
        # geometric mean is then undefined.
        "geometric_mean": (
            math.exp(sum(math.log(r) for r in ordered) / len(ordered))
            if all(r > 0 for r in ordered)
            else None
        ),
        "zero_ratio_cells": sum(1 for r in ordered if r <= 0),
        "median_ci95_bootstrap": list(ci) if ci else None,
    }


def sha256(path: Path) -> str:
    """Hex digest of a file, recorded so a summary can be tied to its input."""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_commit() -> str | None:
    """Return the current HEAD hash, or None outside a git checkout."""
    try:
        out = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return out.stdout.strip()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=OUT_JSON)
    args = parser.parse_args(argv)

    for path in (FORTRAN_CSV, PYFRACVAL_CSV):
        if not path.exists():
            raise SystemExit(f"missing input: {path}")

    fortran = load(FORTRAN_CSV, "success_rate", "median_wall_s")
    pyfracval = load(PYFRACVAL_CSV, "success_rate", "median_runtime_s")
    common = sorted(set(fortran) & set(pyfracval))

    usable: list[Cell] = [
        c
        for c in common
        if fortran[c]["rate"] >= SOLVED_THRESHOLD
        and pyfracval[c]["rate"] >= SOLVED_THRESHOLD
        and pyfracval[c]["time"] > 0
        and math.isfinite(fortran[c]["time"])
        and math.isfinite(pyfracval[c]["time"])
    ]
    ratio_of = {c: fortran[c]["time"] / pyfracval[c]["time"] for c in usable}

    results = {}
    for name, rule in RULES.items():
        selected = [c for c in usable if rule(c)]
        stats = summarise([ratio_of[c] for c in selected])
        stats["cell_keys"] = [
            {"N": c[0], "Df": c[1], "kf": c[2], "sigma": c[3], "ratio": ratio_of[c]}
            for c in selected
        ]
        results[name] = stats

    payload = {
        "experiment": "hard_regime_ratio",
        "description": (
            "Median FracVAL/PyFracVAL wall-time ratio over cells both "
            "implementations solve, under several named definitions of the "
            "hard regime."
        ),
        "environment": {
            "git_commit": git_commit(),
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "python": platform.python_version(),
            "bootstrap_draws": BOOTSTRAP_DRAWS,
            "bootstrap_seed": BOOTSTRAP_SEED,
        },
        "inputs": {
            "fortran_csv": str(FORTRAN_CSV.relative_to(REPO_ROOT)),
            "fortran_csv_sha256": sha256(FORTRAN_CSV),
            "pyfracval_csv": str(PYFRACVAL_CSV.relative_to(REPO_ROOT)),
            "pyfracval_csv_sha256": sha256(PYFRACVAL_CSV),
        },
        "solved_threshold": SOLVED_THRESHOLD,
        "cells_in_both_grids": len(common),
        "cells_both_solve": len(usable),
        "rules": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2))

    print(f"cells in both grids : {len(common)}")
    print(f"cells both solve    : {len(usable)}")
    header = (
        f"{'rule':<26}{'cells':>6}{'median':>9}{'Q1':>9}{'Q3':>9}{'median 95% CI':>24}"
    )
    print()
    print(header)
    print("-" * len(header))
    for name, stats in results.items():
        if not stats["cells"]:
            print(f"{name:<26}{0:>6}")
            continue
        ci = stats["median_ci95_bootstrap"]
        ci_txt = f"[{ci[0]:.3f}, {ci[1]:.3f}]" if ci else "-"
        print(
            f"{name:<26}{stats['cells']:>6}{stats['median']:>9.3f}"
            f"{stats['q1']:>9.3f}{stats['q3']:>9.3f}{ci_txt:>24}"
        )
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
