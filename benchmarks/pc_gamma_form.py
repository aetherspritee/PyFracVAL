#!/usr/bin/env python3
"""Count vs. mass form of the Gamma relation inside PCA.

The paper states that the two aggregation stages *require* different forms
of the contact relation, and supports it by a measurement: applying the
mass form within PC aggregation collapses sub-cluster construction success
"from 62% to 0.7% at sigma = 1.9".  That measurement existed only as a
docstring table in :mod:`pyfracval.pca_agg` (``counts 93/150``,
``masses 1/150``) with no script and no stored output.  This module is its
committed replacement.

Design
------
Two arms differing in exactly one line of code:

``counts``
    the production path, :class:`pyfracval.pca_agg.PCAggregator`, which
    substitutes particle counts for the masses in the Gamma equation
    (Filippov et al. 2000 Eq. 7);
``mass``
    :class:`benchmarks.pc_probe.MassGammaPCAggregator`, which overrides
    ``_gamma_calculation`` to pass ``use_mass=True`` and so solves the true
    mass form (Moran et al. 2019 Eq. 6) that the CCA stage uses.

Both arms see the *same seeds* and therefore the same radius draws, so the
difference between them is attributable to the contact relation alone.
Seed ``i`` is ``SEED_BASE + i`` with ``SEED_BASE = 0``, matching
``benchmarks/pc_probe.py``.

Conditions
----------
``n = 12`` primary particles, ``sigma = 1.9``, single-shot (no retries),
at two (Df, kf) pairs: the pair recorded in the repository's regression
test for this behaviour (``tests/test_densities.py``: 1.79 / 1.40) and the
neighbouring cell of the paper's PC probe table (1.8 / 1.4).

Usage
-----
    devenv shell -- uv run python benchmarks/pc_gamma_form.py --seeds 200

Writes ``benchmark_results/pc_probe/pc_gamma_form_<seeds>seeds.json``.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.pc_probe import (  # noqa: E402
    RESULTS_DIR,
    SEED_BASE,
    environment,
    run_cell,
    wilson_ci,
)

SIGMA = 1.9
N_PARTICLES = 12
DEFAULT_SEEDS = 200

# (Df, kf) conditions. The first is what tests/test_densities.py pins as
# the regression guard for this behaviour; the second is the corresponding
# cell of the paper's tab:pcprobe grid.
CONDITIONS: tuple[tuple[float, float, str], ...] = (
    (1.79, 1.40, "regression-test condition (tests/test_densities.py)"),
    (1.80, 1.40, "tab:pcprobe cell Df=1.8, kf=1.4"),
)

# The original claim was measured over 150 seeds; recomputing the same
# statistic on the first 150 of our seeds makes the comparison exact.
LEGACY_SEED_COUNT = 150


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seeds", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--n", type=int, default=N_PARTICLES)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    logging.disable(logging.CRITICAL)

    seeds = [SEED_BASE + i for i in range(args.seeds)]
    out_path = args.out or (RESULTS_DIR / f"pc_gamma_form_{args.seeds}seeds.json")

    t0 = time.perf_counter()
    records = []
    for df, kf, label in CONDITIONS:
        for arm, use_mass in (("counts", False), ("mass", True)):
            cell = run_cell(df, kf, SIGMA, seeds, n=args.n, use_mass=use_mass)
            cell["arm"] = arm
            cell["condition"] = label
            head = cell["outcomes_by_seed"][:LEGACY_SEED_COUNT]
            if len(head) == LEGACY_SEED_COUNT:
                k = int(sum(head))
                lo, hi = wilson_ci(k, LEGACY_SEED_COUNT)
                cell["first150"] = {
                    "successes": k,
                    "trials": LEGACY_SEED_COUNT,
                    "success_rate": k / LEGACY_SEED_COUNT,
                    "ci95_low": lo,
                    "ci95_high": hi,
                }
            records.append(cell)
            print(
                f"  Df={df} kf={kf} sigma={SIGMA} [{arm}]: "
                f"{cell['successes']}/{cell['trials']} "
                f"({cell['success_rate']:.3f})  [{cell['elapsed_s']:.1f} s]",
                flush=True,
            )
    total = time.perf_counter() - t0

    payload = {
        "experiment": "pc_gamma_form",
        "description": (
            "PCA sub-cluster construction success with the count form vs. the "
            "mass form of the Gamma equation, at sigma=1.9, n=12, single-shot."
        ),
        "environment": environment(seeds, args.n),
        "sigma": SIGMA,
        "conditions": [
            {"df": df, "kf": kf, "label": label} for df, kf, label in CONDITIONS
        ],
        "total_elapsed_s": total,
        "cells": records,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2))

    print()
    header = (
        f"{'Df':<6}{'kf':<6}{'arm':<8}{'successes':<12}{'rate':<8}{'Wilson 95% CI':<20}"
    )
    print(header)
    print("-" * len(header))
    for c in records:
        print(
            f"{c['df']:<6.2f}{c['kf']:<6.2f}{c['arm']:<8}"
            f"{str(c['successes']) + '/' + str(c['trials']):<12}"
            f"{c['success_rate']:<8.3f}"
            f"[{c['ci95_low']:.3f}, {c['ci95_high']:.3f}]"
        )
    print(f"\ntotal wall time: {total:.1f} s")
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
